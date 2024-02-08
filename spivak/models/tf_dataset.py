# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from typing import List, Callable, Optional, Dict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from spivak.data.dataset import Dataset, Task
from spivak.models.assembly.head import TrainerHeadInterface, \
    tf_pad_frames_on_right_2d
from spivak.models.video_start_provider import VideoStartProviderInterface

# Setting the prefetch size to None will end up doing auto-tuning,
# which works well in general, but makes it hard to control memory usage.
# To save memory, we manually set the prefetch size to values which seem to
# speed things up while using slightly less memory.
BATCH_PREFETCH_SIZE = 1
# We prefer to sample one lambda for each example in the mini-batch. In
# contrast, the original mixup paper and implementation use a single lambda for
# the whole mini-batch, which is nice in that it preserves the overall weight
# for each single example (which gets weighed by lambda, then (1 - lambda) in
# the shuffled version). However, the following website notes that it trains
# slower than using a different lambda for each example:
# https://forums.fast.ai/t/mixup-data-augmentation/22764
# In my experiments, the results were similar using either approach, though
# the training loss gets noisier with the single-lambda batches. I guess we can
# stick with the per-example lambda in case there is some situation in which
# it helps to train faster.
BATCH_MIXUP_LAMBDA = False

TFDataset = tf.data.Dataset

"""
brew link python@3.8 --force
"""

def create_tf_merged_batch_dataset(
        batch_datasets: List[TFDataset]) -> TFDataset:
    num_tasks = len(batch_datasets)
    choice_dataset = tf.data.Dataset.range(num_tasks).repeat()
    merged_batch_dataset = tf.data.experimental.choose_from_datasets(
        batch_datasets, choice_dataset)
    # This prefetch lets the pipeline start working on creating the next
    # batch(es) before the current batch is processed by the models.
    return merged_batch_dataset.prefetch(BATCH_PREFETCH_SIZE)


def create_tf_task_batch_dataset(
        tf_task_videos_dataset: TFDataset, tf_get_video_chunks: Callable,
        get_chunks_parallelism: int, repetitions: int, batch_size: int,
        chunk_shuffle_size: int,
        batch_augmentation: Optional[Callable]) -> TFDataset:
    """Shuffling takes up some amount of memory, which is why we make it
    optional."""
    # Setting the number of parallel calls to tf.data.experimental.AUTOTUNE
    # causes a dangerous memory increase, so it's better to avoid using that.
    # When setting it to None (or removing the parameter), map runs
    # sequentially, which indeed slows things down. The map function
    # treats None differently than 0, so we handle that here.
    if get_chunks_parallelism == 0:
        num_parallel_calls = None
    else:
        num_parallel_calls = get_chunks_parallelism
    chunks_dataset = (
        tf_task_videos_dataset.repeat(repetitions).map(
            tf_get_video_chunks, num_parallel_calls=num_parallel_calls)
        # _flat_map maps from chunks grouped by video to ungrouped chunks.
        .flat_map(_flat_map))
    if chunk_shuffle_size:
        # Here, we shuffle the chunks. Note that the shuffling goes across epoch
        # boundaries since repeat is called beforehand. I don't imagine that is
        # very important. If we need to reduce memory usage, we could reduce
        # the chunk_shuffle_size, thought it's hard to know exactly what a good
        # number would be. In general, shuffling should help in training, but
        # the batch size might need to be tuned in tandem (for example,
        # maybe if the batch size is too big, it will be better not to shuffle
        # in order to reduce the diversity within the too-big batches). In my
        # experiments on SoccerNet v2, there actually wasn't  much of a
        # difference whether shuffling or not, but it seems better
        # to keep it.
        chunks_dataset = chunks_dataset.shuffle(chunk_shuffle_size,
                                                reshuffle_each_iteration=True)
    batch_dataset = (
        chunks_dataset
        # This prefetch lets the pipeline start working on the samples (chunks)
        # that will go into the next batch before the current batch is ready.
        .prefetch(batch_size)
        .batch(batch_size)
    )
    if batch_augmentation:
        batch_dataset = batch_dataset.map(batch_augmentation)
    return batch_dataset


def create_tf_mixup_batch_augmentation(heads: List[TrainerHeadInterface],
                                       mixup_alpha: float) -> Callable:
    # TODO: in order to get this to work with deltas, we could one-hot encode
    #  them and use earth mover's distance for the loss, while somehow
    #  ignoring regions without deltas.
    def tf_mixup_batch_augmentation(batch_features, batch_targets):
        # Get the batch size.
        batch_features_shape = tf.shape(batch_features)
        batch_size = batch_features_shape[0]
        # Sample lambdas.
        beta = tfp.distributions.Beta(mixup_alpha, mixup_alpha)
        if BATCH_MIXUP_LAMBDA:
            lambdas = beta.sample(1) * tf.ones(batch_size)
        else:
            lambdas = beta.sample(batch_size)
        # Sample a random permutation for the examples in the batch.
        indices = tf.range(start=0, limit=batch_size, dtype=tf.int32)
        permutation = tf.random.shuffle(indices)
        # Do mixup on features first.
        shuffled_batch_features = tf.gather(batch_features, permutation)
        # Expand the lambdas to match the features size.
        features_lambdas_shape = ((batch_size,) + (1,) *
                                  (len(batch_features_shape) - 1))
        features_lambdas = tf.reshape(lambdas, features_lambdas_shape)
        mixed_features = (features_lambdas * batch_features +
                          (1.0 - features_lambdas) * shuffled_batch_features)
        # Do mixup on targets now.
        mixed_targets = _tf_mixup_batch_targets(heads, batch_targets,
                                                permutation, lambdas,
                                                batch_size)
        return mixed_features, mixed_targets

    return tf_mixup_batch_augmentation


def create_tf_task_videos_datasets(
        dataset: Dataset, trainer_heads: List[TrainerHeadInterface],
        video_start_providers: Dict[Task, VideoStartProviderInterface],
        cache_dataset: bool, shuffle_videos: bool) -> Dict[Task, TFDataset]:
    videos_dataset = _create_tf_videos_dataset(dataset, trainer_heads,
                                               cache_dataset, shuffle_videos)
    # Need to restrict set of videos depending on the task at hand.
    task_videos_datasets = {}
    for task in video_start_providers:
        # This approach of creating the starts using tf.data.Dataset supports
        # streaming different amounts of input videos, as opposed to having a
        # fixed-size dataset loaded in memory. Note that create_tf_starts
        # creates the starts randomly, so we don't cache the result.
        video_start_provider = video_start_providers[task]

        def add_chunk_starts(video_features, video_targets,
                             video_labels_from_task_dict):
            video_starts = video_start_provider.create_tf_starts(
                video_labels_from_task_dict)
            return video_features, video_targets, video_starts

        task_dataset = videos_dataset.map(add_chunk_starts)
        # When working with large datasets, we want to sample very sporadically
        # from each video, so sometimes create_tf_starts will provide an empty
        # set of starts for a given video. It's important to remove these empty
        # sets of starts to simplify later processing.
        task_videos_datasets[task] = task_dataset.filter(_chunk_starts_is_empty)
    return task_videos_datasets


def create_tf_get_video_chunks(heads: List[TrainerHeadInterface], task: Task,
                               num_chunk_frames: int) -> Callable:

    def tf_get_video_chunks(video_features, video_targets, chunk_starts):
        return _tf_video_chunks(video_features, video_targets, chunk_starts,
                                heads, task, num_chunk_frames)

    return tf_get_video_chunks


def _create_tf_videos_dataset(dataset: Dataset,
                              trainer_heads: List[TrainerHeadInterface],
                              cache_dataset: bool,
                              shuffle_videos: bool) -> TFDataset:

    def get_video_features(video_index) -> np.ndarray:
        return dataset.video_data[video_index].features

    def get_task_video_labels(vid_ind: int, task_int: int) -> np.ndarray:
        return dataset.video_data[vid_ind].labels(Task(task_int))

    def get_task_valid_labels(vid_ind: int, task_int: int) -> bool:
        return dataset.video_data[vid_ind].valid_labels(Task(task_int))

    def get_video_data(video_index):
        # Read the video features
        video_features = tf.py_function(func=get_video_features,
                                        inp=[video_index],
                                        Tout=tf.float32)
        video_features.set_shape((None, dataset.num_features))
        # Read the labels (a dictionary from tasks to respective labels)
        # py_function cannot return a dictionary, so we call py_function once
        # for each task, and then add the result to the dictionary.
        video_labels_from_task_dict = {}
        for dataset_task in dataset.tasks:
            task_video_labels = tf.py_function(
                func=get_task_video_labels,
                inp=[video_index, dataset_task.value],
                Tout=tf.float32)
            task_video_labels.set_shape(
                (None, dataset.num_classes_from_task[dataset_task]))
            task_valid_labels = tf.py_function(
                func=get_task_valid_labels,
                inp=[video_index, dataset_task.value],
                Tout=tf.bool)
            video_labels_from_task_dict[dataset_task] = (task_video_labels,
                                                         task_valid_labels)

        # We can't pass a dictionary into numpy_function, so we flatten it and
        # pass the flattened version in as a sequence of arguments. We then
        # reconstruct/pack the dictionary inside get_video_targets(). Reference
        # implementation for flattening and packing:
        # https://github.com/tensorflow/tensorflow/issues/27679#issuecomment-522578000
        def get_video_targets(*flat_input):
            # Reconstruct labels_from_task_dict using the inputs.
            labels_from_task_dict = tf.nest.pack_sequence_as(
                video_labels_from_task_dict, flat_input)
            # Create the targets for each head.
            return tuple(
                th.video_targets(labels_from_task_dict).astype(np.float32)
                for th in trainer_heads)

        video_targets = tf.numpy_function(
            func=get_video_targets,
            inp=tf.nest.flatten(video_labels_from_task_dict),
            Tout=(tf.float32,) * len(trainer_heads))
        for trainer_head, video_target in zip(trainer_heads, video_targets):
            video_target.set_shape(trainer_head.video_targets_shape)
        return video_features, video_targets, video_labels_from_task_dict

    num_videos = dataset.num_videos
    tf_range_dataset = TFDataset.range(num_videos)
    if shuffle_videos:
        tf_range_dataset = tf_range_dataset.shuffle(
            num_videos, reshuffle_each_iteration=True)
    videos_dataset = tf_range_dataset.map(get_video_data)
    if cache_dataset:
        videos_dataset = videos_dataset.cache()
    return videos_dataset


def _tf_video_chunks(video_features, video_targets, chunk_starts, heads, task,
                     num_chunk_frames):

    def get_features_chunk(start):
        sliced_features = video_features[start:start + num_chunk_frames]
        return tf_pad_frames_on_right_2d(sliced_features, num_chunk_frames)

    video_chunks_features = tf.map_fn(fn=get_features_chunk,
                                      elems=chunk_starts,
                                      fn_output_signature=tf.float32)
    # Our inputs want this extra dimension, so it's easier to directly do
    # (degenerate) 2D convolutions on them. Is it worth trying to get rid of it?
    video_chunks_features_expanded = tf.expand_dims(video_chunks_features,
                                                    axis=-1)
    video_len = tf.shape(video_features)[0]
    chunk_masks = _tf_create_chunk_masks(chunk_starts, num_chunk_frames,
                                         video_len)
    video_chunks_targets = tuple(
        tf.map_fn(fn=head.tf_chunk_targets_mapper(video_targets[head_index],
                                                  task),
                  elems=(chunk_starts, chunk_masks),
                  fn_output_signature=tf.float32)
        for head_index, head in enumerate(heads))
    return video_chunks_features_expanded, video_chunks_targets


def _tf_create_chunk_masks(chunk_starts, num_chunk_frames, video_len):

    def tf_map_to_chunk_mask(chunk_start):
        valid_chunk_labels_len = tf.minimum(num_chunk_frames,
                                            video_len - chunk_start)
        return _tf_create_chunk_mask(valid_chunk_labels_len)

    return tf.map_fn(fn=tf_map_to_chunk_mask,
                     elems=chunk_starts,
                     fn_output_signature=tf.bool)


def _tf_create_chunk_mask(valid_chunk_labels_len):
    # We don't mask anything, since we want to compute the loss everywhere.
    # mask == True indicates areas where the loss should NOT be computed.
    return tf.zeros(valid_chunk_labels_len, dtype=tf.dtypes.bool)


def _flat_map(video_chunks_features, video_chunks_targets):
    return TFDataset.from_tensor_slices(
        (video_chunks_features, video_chunks_targets))


def _chunk_starts_is_empty(video_features, video_targets, chunk_starts):
    return tf.not_equal(tf.size(chunk_starts), 0)


def _tf_mixup_batch_targets(heads, batch_targets, permutation, lambdas,
                            batch_size):
    return tuple(
        _tf_mixup_batch_head_targets(batch_targets[head_index], permutation,
                                     lambdas, batch_size, head)
        for head_index, head in enumerate(heads))


def _tf_mixup_batch_head_targets(batch_head_targets, permutation, lambdas,
                                 batch_size, head: TrainerHeadInterface):
    if not head.supports_mixup:
        raise ValueError(f"Head of type {type(head)} does not support mixup")
    shuffled_batch_head_targets = tf.gather(batch_head_targets, permutation)
    # Expand the lambdas to match the targets shape.
    lambdas_shape = (batch_size,) + ((1,) * len(head.video_targets_shape))
    targets_lambdas = tf.reshape(lambdas, lambdas_shape)
    # Note that the loss weights are usually stored as part of
    # batch_head_targets, so the weights will get mixed in the same way that
    # the actual targets/labels are mixed.
    return (targets_lambdas * batch_head_targets +
            (1.0 - targets_lambdas) * shuffled_batch_head_targets)

"""
위 코드는 TensorFlow와 TensorFlow Probability를 사용하여 
    비디오 데이터 처리 및 모델 훈련을 위한 다양한 기능을 구현한 것
주요 내용은 다음과 같습니다:

### 데이터셋 처리 및 가공
- **가변적인 데이터 처리**: 
    여러 비디오 데이터셋에서 정보를 추출하고, 각 비디오에 대한 처리를 수행 
    여기서는 비디오의 특정 부분(청크)을 선택하고, 이를 모델 학습에 사용할 배치로 구성
- **데이터셋 캐싱 및 셔플링**: 
    데이터 로딩 성능을 개선하기 위해 데이터셋을 캐시에 저장하고, 
    학습 데이터의 다양성을 보장하기 위해 셔플링을 적용
- **Mixup 데이터 증강**: 
    Mixup은 학습 데이터의 다양성을 늘리고, 모델의 일반화 성능을 향상시키는 데이터 증강 기법
    코드에서는 배치 단위로 Mixup을 적용하며, 각 예제마다 람다 값을 다르게 적용하는 방식을 사용

### 텐서플로우 데이터 API 활용
- **`tf.data.Dataset` 활용**: 
    TensorFlow의 `tf.data` API를 사용하여 효율적인 데이터 파이프라인을 구축
    이를 통해 입력 데이터를 효율적으로 처리하고, 모델 학습에 적합한 형태로 변환
- **프리페칭과 병렬 처리**: 
    데이터 로딩 및 전처리 과정에서 발생할 수 있는 병목 현상을 줄이기 위해 
    프리페칭과 병렬 처리 기법을 사용
        pre-fetching은 학습에 사용될 데이터를 미리 로딩하여 대기시간을 줄이는 방식
    이는 CPU와 GPU 간의 효율적인 작업 분배를 통해 학습 성능을 개선

### 모델 학습 관련 기능
- **가변적인 배치 생성**: 
    다양한 소스에서 온 비디오 데이터를 처리하여 학습에 사용할 배치를 동적으로 생성
    이 과정에서 `tf.data.experimental.choose_from_datasets` 함수를 사용하여 
        여러 데이터셋 중에서 선택합니다.
- **Mixup 구현**: 
    Mixup 기법을 통해 입력 데이터와 타겟 레이블을 선형적으로 조합합니다. 
    이는 모델의 일반화 능력을 향상시키는 데 도움이 됩니다.

### 구조 및 인터페이스
- 코드는 TensorFlow 기반의 비디오 처리 및 학습 파이프라인을 구성하기 위한 여러 함수와 클래스를 정의
- 특히, `TrainerHeadInterface`와 같은 인터페이스를 통해 다양한 학습 헤드를 관리하고, 
    비디오 데이터 처리 로직을 추상화
- 학습 헤  드
- 머신러닝 모델의 일부로, 특정 작업(예: 분류, 회귀, 객체 감지 등)을 위해 최종 출력을 생성하는 구성요소

전반적으로, 이 코드는 비디오 데이터를 효율적으로 처리하고, 
    TensorFlow를 사용한 딥러닝 모델 학습을 위한 고급 기술을 적용하는 방법을 보여줍니다. 
Mixup과 같은 데이터 증강 기법과, 
    `tf.data` API를 활용한 성능 최적화 방법은 대규모 비디오 데이터셋을 다룰 때 특히 유용
"""
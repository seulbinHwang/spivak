# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.
""" Low memory setup을 위해선, 아래 실시
MEMORY_SETUP = MEMORY_SETUP_64GB
"""
# Options for memory usage setups.
MEMORY_SETUP_256GB = "256"
MEMORY_SETUP_64GB = "64"
# Change the constants below according to your local setup.
FEATURES_DIR = "./data/features/"
BAIDU_FEATURES_DIR = "./data/features/baidu/"
BAIDU_TWO_FEATURES_DIR = "./data/features/baidu_2.0/"
RESNET_FEATURES_DIR = "./data/features/resnet/"
RESNET_NORMALIZED_FEATURES_DIR = "./data/features/resnet_normalized/"
LABELS_DIR = "./data/labels/"
SPLITS_DIR = "./data/splits/"
BASE_CONFIG_DIR = "./configs/"
MODELS_DIR = "YOUR_MODELS_DIR"
RESULTS_DIR = "YOUR_RESULTS_DIR"
RUN_NAME = "first"
MEMORY_SETUP = MEMORY_SETUP_256GB

# You might have your data set up in such a way that all the SoccerNet features
# and labels are under the same directory tree. In that case, each game folder
# would contain its respective Baidu (Combination) features, ResNet features,
# and action spotting labels. For example, if all your data is in a base
# directory called "all_soccernet_features_and_labels/", then a game folder
# might have the following files:
#
# $ ls all_soccernet_features_and_labels/england_epl/2014-2015/2015-02-21\ -\ 18-00\ Chelsea\ 1\ -\ 1\ Burnley/
#   1_baidu_soccer_embeddings.npy  2_baidu_soccer_embeddings.npy
#   1_ResNET_TF2.npy  1_ResNET_TF2_PCA512.npy  2_ResNET_TF2.npy
#   2_ResNET_TF2_PCA512.npy  Labels-cameras.json  Labels-v2.json
#
# In that case, you could just set some particular constants above to the same
# base directory, as follows:
# BAIDU_FEATURES_DIR = "all_soccernet_features_and_labels/"
# BAIDU_TWO_FEATURES_DIR = "all_soccernet_features_and_labels/"
# RESNET_FEATURES_DIR = "all_soccernet_features_and_labels/"
# RESNET_NORMALIZED_FEATURES_DIR = "all_soccernet_features_and_labels/"
# LABELS_DIR = "all_soccernet_features_and_labels/"
#
# We suggest setting FEATURES_DIR to a different folder, like
# "./data/features/". At the same time, you probably won't have to change
# SPLITS_DIR and BASE_CONFIG_DIR.

"""
위 코드는 머신러닝 프로젝트 설정을 위한 구성 파일의 일부입니다. 
이 설정은 비디오 데이터 처리와 액션 스팟팅을 포함한 SoccerNet 데이터셋을 사용하는 프로젝트에 특히 적용
주요 구성 요소와 그 목적은 다음과 같습니다:

### 메모리 설정 옵션
- `MEMORY_SETUP_256GB`와 `MEMORY_SETUP_64GB`는 사용 가능한 시스템 메모리에 따른 두 가지 설정
- 이를 통해 프로젝트에서 필요로 하는 메모리 양에 따라 최적의 설정을 선택할 수 있음

### 디렉토리 및 파일 경로 설정
- `FEATURES_DIR`, `BAIDU_FEATURES_DIR`, `BAIDU_TWO_FEATURES_DIR`, 
- `RESNET_FEATURES_DIR`, `RESNET_NORMALIZED_FEATURES_DIR`, 
- `LABELS_DIR`, `SPLITS_DIR`, `BASE_CONFIG_DIR`, `MODELS_DIR`, `RESULTS_DIR`
  - 프로젝트의 다양한 구성 요소가 저장될 디렉토리의 경로를 지정
  - 예를 들어, `FEATURES_DIR`는 피처 데이터가 저장되는 기본 디렉토리를 가리키고, 
  `LABELS_DIR`는 라벨 데이터가 저장되는 디렉토리를 나타냄
  - `MODELS_DIR`과 `RESULTS_DIR`는 사용자가 모델과 결과물을 저장할 위치를 지정해야 하는 변수
  - 이들은 `"YOUR_MODELS_DIR"`와 `"YOUR_RESULTS_DIR"`로 표시되어 있어, 
  사용자가 실제 경로로 변경해야 함을 나타냄
  

### 데이터 구조 예시
- 주석 부분에서는 `all_soccernet_features_and_labels/` 디렉토리 안에 SoccerNet 데이터셋의 예시 구조를 설명합니다. 각 게임 폴더는 Baidu 피처, ResNet 피처, 액션 스팟팅 라벨 등을 포함합니다.
- 이 부분은 사용자가 데이터를 어떻게 구성할 수 있는지에 대한 예시를 제공하며, 필요에 따라 경로 설정을 단순화하기 위한 방법을 제안합니다.

### 실행 이름 및 메모리 설정 선택
- `RUN_NAME` 변수는 현재 실행에 대한 이름을 설정하는 데 사용됩니다. 이는 결과 파일이나 모델 파일을 구별할 때 유용합니다.
- `MEMORY_SETUP` 변수는 사용할 메모리 설정을 선택합니다. 예를 들어, 시스템이 64GB RAM을 갖고 있다면, `MEMORY_SETUP_64GB`를 사용하는 것이 적절할 수 있습니다.

이 코드는 프로젝트의 기본 구성을 정의하며, 실제 작업 환경에 맞게 사용자가 적절히 수정해야 합니다. 데이터 경로, 모델 저장 위치, 메모리 설정 등은 프로젝트의 요구 사항과 사용자의 시스템 구성에 따라 달라질 수 있습니다.
"""
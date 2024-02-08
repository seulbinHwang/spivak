from SoccerNet.Downloader import SoccerNetDownloader
import argparse

# models:
# results
# reports
# data/features

parser = argparse.ArgumentParser(description="Download SoccerNet data")
parser.add_argument("--download_video",
                    default=True,
                    type=bool,
                    help="Download the videos")
parser.add_argument("--enable_symlinks",
                    default=False,
                    type=bool,
                    help="Enable symbolic links")

args = parser.parse_args()

# soccernet_dataset
mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory="./soccernet_dataset/")

# Download SoccerNet features -> data/features
mySoccerNetDownloader.downloadGames(
    files=["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"],
    split=["train", "valid", "test"])  # download Features
mySoccerNetDownloader.downloadGames(
    files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"],
    split=["train", "valid", "test"])  # download Features reduced with PCA
mySoccerNetDownloader.downloadGames(
    files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"],
    split=["train", "valid", "test"]
)  # download Frame Embeddings from https://github.com/baidu-research/vidpress-sports

# Download SoccerNet labels
mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"],
                                    split=["train", "valid",
                                           "test"])  # download labels SN v2

# Download SoccerNet videos
if args.download_video:
    mySoccerNetDownloader.password = input(
        "Password for videos (received after filling the NDA)")
    # mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train","valid","test","challenge"])
    mySoccerNetDownloader.downloadGames(
        files=["1_224p.mkv", "2_224p.mkv"],
        split=["train", "valid", "test", "challenge"])

import os
"""
cd data/  # This folder will initially just contain the splits/ folder.
ln -s YOUR_LABELS_FOLDER  labels  # For the Labels-v2.json and/or the Labels-cameras.json files.
ln -s YOUR_FEATURES_RESNET_FOLDER  features/resnet  # For the ResNet-based features.
ln -s YOUR_FEATURES_BAIDU_FOLDER  features/baidu  # For the Baidu Combination features.
ln -s YOUR_VIDEOS_224P_FOLDER  videos_224p  # For the low-resolution videos.
"""
# Define the paths for labels, features, and videos
YOUR_LABELS_FOLDER = "/soccernet_dataset/labels"
YOUR_FEATURES_RESNET_FOLDER = "/soccernet_dataset/features/resnet"
YOUR_FEATURES_BAIDU_FOLDER = "/soccernet_dataset/features/baidu"
YOUR_VIDEOS_224P_FOLDER = "/soccernet_dataset/videos_224p"

# Define the target directory
data_directory = "data"

if args.enable_symlinks:
    # Ensure the data directory exists and is a directory
    os.makedirs(data_directory, exist_ok=True)

    # Change to the data directory
    os.chdir(data_directory)

    # Create symbolic links
    os.symlink(YOUR_LABELS_FOLDER, "labels", target_is_directory=True)
    os.makedirs("features", exist_ok=True)
    os.symlink(YOUR_FEATURES_RESNET_FOLDER,
               "features/resnet",
               target_is_directory=True)
    os.symlink(YOUR_FEATURES_BAIDU_FOLDER,
               "features/baidu",
               target_is_directory=True)
    os.symlink(YOUR_VIDEOS_224P_FOLDER, "videos_224p", target_is_directory=True)

    # Change back to the original directory if needed
    # os.chdir("..")

    print("Symbolic links created successfully.")

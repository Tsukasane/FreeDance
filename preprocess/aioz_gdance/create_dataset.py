import argparse
import os
from pathlib import Path

from audio_extraction.baseline_features import \
    extract_folder as baseline_extract
from audio_extraction.jukebox_features import extract_folder as jukebox_extract
from filter_split_data import *
from slice import *


def create_dataset(opt):
    # split the data according to the splits files
    print("Creating train / val / test split")
    split_data(opt.dataset_folder)
    # slice motions/music into sliding windows to create training dataset
    print("Slicing train data")
    slice_aioz(f"AIOZ_Gdance_dataset/train/motions", f"AIOZ_Gdance_dataset/train/wavs")
    print("Slicing val data")
    slice_aioz(f"AIOZ_Gdance_dataset/val/motions", f"AIOZ_Gdance_dataset/val/wavs")
    print("Slicing test data")
    slice_aioz(f"AIOZ_Gdance_dataset/test/motions", f"AIOZ_Gdance_dataset/test/wavs")
    # process dataset to extract audio features
    if opt.extract_baseline:
        print("Extracting baseline features")
        baseline_extract("AIOZ_Gdance_dataset/train/wavs_sliced", "AIOZ_Gdance_dataset/train/baseline_feats")
        baseline_extract("AIOZ_Gdance_dataset/val/wavs_sliced", "AIOZ_Gdance_dataset/val/baseline_feats")
        baseline_extract("AIOZ_Gdance_dataset/test/wavs_sliced", "AIOZ_Gdance_dataset/test/baseline_feats")
    if opt.extract_jukebox:
        print("Extracting jukebox features")
        jukebox_extract("AIOZ_Gdance_dataset/train/wavs_sliced", "AIOZ_Gdance_dataset/train/jukebox_feats")
        jukebox_extract("AIOZ_Gdance_dataset/val/wavs_sliced", "AIOZ_Gdance_dataset/val/jukebox_feats")
        jukebox_extract("AIOZ_Gdance_dataset/test/wavs_sliced", "AIOZ_Gdance_dataset/test/jukebox_feats")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--length", type=float, default=5.0, help="checkpoint")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="edge_aistpp",
        help="folder containing motions and music",
    )
    parser.add_argument("--extract-baseline", action="store_true")
    parser.add_argument("--extract-jukebox", action="store_true")
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    create_dataset(opt)
import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random

import numpy as np
import torch
from tqdm import tqdm

from args import parse_test_opt
from data.slice import slice_audio
from POPDG import POPDG
from preprocess.audio_extraction.baseline_features import extract_audio_features as baseline_extract
# from preprocess.audio_extraction.jukebox_features import extract_audio_features as juke_extract

def extract_slice_number(filename):
    """Extract the numeric part of the slice from a filename."""
    return int(Path(filename).stem.split("_slice")[-1])

def compare_filenames(a, b):
    """Compare two filenames based on their base name and slice number."""
    base_name_a, slice_number_a = Path(a).stem.rsplit('_', 1)
    base_name_b, slice_number_b = Path(b).stem.rsplit('_', 1)

    # Compare the base names
    if base_name_a != base_name_b:
        return -1 if base_name_a < base_name_b else 1

    # Compare the slice numbers
    slice_a = extract_slice_number(a)
    slice_b = extract_slice_number(b)
    return -1 if slice_a < slice_b else (1 if slice_a > slice_b else 0)

# Convert comparison function to a key function for sorting
sort_key = cmp_to_key(compare_filenames)

def parse_test_opt():
    parser = argparse.ArgumentParser(description="Test Configs.")

    # Environment settings
    env_group = parser.add_argument_group('Environment and Path Settings')
    env_group.add_argument("--processed_data_dir", type=str, default="data/dataset_backups/", help="Path where processed dataset backups are stored.")
    env_group.add_argument("--render_dir", type=str, default="renders/", help="Directory where rendered outputs will be saved.")
    env_group.add_argument("--music_dir", type=str, default="data/test/wavs", help="Directory containing input music files for testing.")
    env_group.add_argument("--motion_save_dir", type=str, default="eval/motions", help="Directory where generated motion files will be saved if --save_motions is used.")

    # Testing settings
    test_group = parser.add_argument_group('Test Execution Settings')
    test_group.add_argument("--feature_type", type=str, default="baseline", help="Type of features to use for the model testing.")
    test_group.add_argument("--out_length", type=float, default=10.0, help="Maximum length of the output in seconds.")
    test_group.add_argument("--checkpoint", type=str, default="checkpoint.pt", help="Path to the model checkpoint to be used for testing.")

    # Cache settings 
    cache_group = parser.add_argument_group('Feature and Caching Settings')
    cache_group.add_argument("--save_motions", action="store_true", help="Enable saving the generated motions for further evaluation.")
    cache_group.add_argument("--cache_features", action="store_true", help="Enable caching of computed features for reuse.")
    cache_group.add_argument("--use_cached_features", action="store_true", help="Use precomputed features instead of recalculating.")
    cache_group.add_argument("--feature_cache_dir", type=str, default="cached_features/", help="Directory to save/load cached features.")
    cache_group.add_argument("--no_render", action="store_true", help="Disable video rendering after testing.")

    opt = parser.parse_args()

    return opt

def test(opt):
    feature_func = baseline_extract
    sample_length = opt.out_length
    sample_size = int(sample_length / 2.5) - 1
    
    temp_dir_list = []
    all_cond = []
    all_filenames = []
    if opt.use_cached_features:
        print("Using precomputed features")
        # all subdirectories
        dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))
        for dir in dir_list:
            file_list = sorted(glob.glob(f"{dir}/*.wav"), key=sort_key)
            juke_file_list = sorted(glob.glob(f"{dir}/*.npy"), key=sort_key)
            assert len(file_list) == len(juke_file_list)
            # random chunk after sanity check
            rand_idx = random.randint(0, len(file_list) - sample_size)
            file_list = file_list[rand_idx : rand_idx + sample_size]
            juke_file_list = juke_file_list[rand_idx : rand_idx + sample_size]
            cond_list = [np.load(x) for x in juke_file_list]
            all_filenames.append(file_list)
            all_cond.append(torch.from_numpy(np.array(cond_list)))
    else:
        print("Computing features for input music")
        for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
            # create temp folder (or use the cache folder if specified)
            if opt.cache_features:
                songname = os.path.splitext(os.path.basename(wav_file))[0]
                save_dir = os.path.join(opt.feature_cache_dir, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                temp_dir = TemporaryDirectory()
                temp_dir_list.append(temp_dir)
                dirname = temp_dir.name
            # slice the audio file
            print(f"Slicing {wav_file}")
            slice_audio(wav_file, 2.5, 5.0, dirname)
            file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=sort_key)
            rand_idx = random.randint(0, len(file_list) - sample_size)
            cond_list = []

            # generate juke representations
            print(f"Computing features for {wav_file}")
            for idx, file in enumerate(tqdm(file_list)):
                # if not caching then only calculate for the interested range
                if (not opt.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                    continue
                reps, _ = feature_func(file)
                # save reps
                if opt.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)
                # if in the random range, put it into the list of reps we want
                # to actually use for generation
                if rand_idx <= idx < rand_idx + sample_size:
                    cond_list.append(reps)
            cond_list = torch.from_numpy(np.array(cond_list))
            all_cond.append(cond_list)
            all_filenames.append(file_list[rand_idx : rand_idx + sample_size])

    model = POPDG(opt.feature_type, opt.checkpoint)
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir

    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], all_filenames[i]
        model.render_sample(
            data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render
        )
    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)
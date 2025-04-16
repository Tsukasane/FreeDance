import glob
import os
import pickle

import librosa as lr
import numpy as np
import soundfile as sf
from tqdm import tqdm


def slice_audio(audio_file, stride, length, out_dir):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    # print(sr)
    # print(window)
    # print(stride_step)
    # print(len(audio) )
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]

        #sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        try:
            sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        except Exception as e:
            print(f"Audio slice length: {len(audio_slice)}")
            print(f"Error writing file: {out_dir}/{file_name}_slice{idx}.wav")
            print(f"Audio slice shape: {audio_slice.shape}, dtype: {audio_slice.dtype}")
            print(f"Sampling rate: {sr}")
            raise e

        start_idx += stride_step
        idx += 1
    return idx


def slice_motion(motion_file, stride, length, num_slices, out_dir):
    motion = pickle.load(open(motion_file, "rb"))
    pos, q = motion["pos"], motion["q"]
    #scale = motion["scale"][0]

    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    # normalize root position
    #pos /= scale
    start_idx = 0
    window = int(length * 30)          # ----------【litingw：AIST用的60，AIOZ改为30】
    stride_step = int(stride * 30)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= pos.shape[1] - window and slice_count < num_slices:
        # print(start_idx)
        # print(pos.shape[1] - window)
        pos_slice = pos[:, start_idx : start_idx + window, :]    # [H, T[start_idx : start_idx + window], Trans==3]
        q_slice = q[:, start_idx : start_idx + window, :]    # [H, T[start_idx : start_idx + window], Poses==72]
        out = {"pos": pos_slice, "q": q_slice}
        pickle.dump(out, open(f"{out_dir}/{file_name}_slice{slice_count}.pkl", "wb"))
        start_idx += stride_step
        slice_count += 1
    return slice_count


def slice_aioz(motion_dir, wav_dir, stride=0.5, length=5):
    dropped_audio=0
    dropped_motion=0

    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    motions = sorted(glob.glob(f"{motion_dir}/*.pkl"))
    wav_out = wav_dir + "_sliced"
    motion_out = motion_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(motion_out, exist_ok=True)
    assert len(wavs) == len(motions)
    for wav, motion in tqdm(zip(wavs, motions)):
        # make sure name is matching
        m_name = os.path.splitext(os.path.basename(motion))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        assert m_name == w_name, str((motion, wav))
        audio_slices = slice_audio(wav, stride, length, wav_out)
        motion_slices = slice_motion(motion, stride, length, audio_slices, motion_out)

        # 如果切片数量不一致，丢弃多余的片段并计数
        if audio_slices > motion_slices:
            dropped_audio += audio_slices - motion_slices
            # 删除多余的音频切片
            for idx in range(motion_slices, audio_slices):
                os.remove(f"{wav_out}/{w_name}_slice{idx}.wav")
        elif motion_slices > audio_slices:
            dropped_motion += motion_slices - audio_slices
            # 删除多余的动作切片
            for idx in range(audio_slices, motion_slices):
                os.remove(f"{motion_out}/{m_name}_slice{idx}.pkl")

        # print(f" audio_slices: {audio_slices}") 
        # print(f" motion_slices: {motion_slices}") 
        # make sure the slices line up
        # assert audio_slices == motion_slices, str(
        #     (wav, motion, audio_slices, motion_slices)
        # )
    # 打印丢弃的切片数量
    print(f"Dropped unequal audio slices: {dropped_audio}")
    print(f"Dropped unequal motion slices: {dropped_motion}")

        


def slice_audio_folder(wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    wav_out = wav_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    for wav in tqdm(wavs):
        audio_slices = slice_audio(wav, stride, length, wav_out)

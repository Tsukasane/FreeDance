from absl import app
from absl import flags
from absl import logging

import os
from librosa import beat
import torch
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import scipy.signal as scisignal


def motion_peak_onehot(joints):
    """Calculate motion beats.
    Kwargs:
        joints: [nframes, njoints, 3] seq, 24, 3
    Returns:
        - peak_onhot: motion beats.
    """
    # Calculate velocity.
    velocity = np.zeros_like(joints, dtype=np.float32)
    velocity[1:] = joints[1:] - joints[:-1]
    velocity_norms = np.linalg.norm(velocity, axis=2)
    envelope = np.sum(velocity_norms, axis=1)  # (seq_len,)

    # Find local minima in velocity -- beats
    peak_idxs = scisignal.argrelextrema(envelope, np.less, axis=0, order=5) # 5 for 30FPS
    peak_onehot = np.zeros_like(envelope, dtype=bool)
    peak_onehot[peak_idxs] = 1

    # # Second-derivative of the velocity shows the energy of the beats
    # peak_energy = np.gradient(np.gradient(envelope)) # (seq_len,)
    # # optimize peaks
    # peak_onehot[peak_energy<0.001] = 0
    return peak_onehot


def alignment_score(music_beats, motion_beats, sigma=3):
    """Calculate alignment score between music and motion."""
    if motion_beats.sum() == 0:
        return 0.0
    music_beat_idxs = np.where(music_beats)[0]
    motion_beat_idxs = np.where(motion_beats)[0]
    score_all = []
    for motion_beat_idx in motion_beat_idxs:
        dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
        ind = np.argmin(dists)
        score = np.exp(- dists[ind]**2 / 2 / sigma**2)
        score_all.append(score)
    return sum(score_all) / len(score_all)


def extract_core_name(filename):
    return filename.split('_')[1]

def extract_core_name_2(filename):
    return filename.split('_')[0]


def cal_BAS(motion_results, num_person, music_feats):
    '''
    keypoints: bs, h, t, 24 ,3
    music_feats: bs, t', 35
    '''
    keypoints3d_all = motion_results.detach().cpu().numpy() # positions.view(bs, h, sq, 24, 3)
    bs, h, T, J, Dp = keypoints3d_all.shape
    music_feats = music_feats[:,:T,:] # match motion beat
    beat_scores = []

    cnt = 0   
    for n_id in range(bs): # each element
        for h_id in range(num_person[n_id]): # each person, excluding padding
            cnt+=1
            keypoints3d = keypoints3d_all[n_id][h_id]
            music = music_feats[n_id] 
            
            motion_beats = motion_peak_onehot(keypoints3d)
            audio_beats = music[:,-1]
            beat_score = alignment_score(audio_beats, motion_beats, sigma=3)
            beat_scores.append(beat_score)

    # print ("\nBeat score on generated data: %.3f\n" % (sum(beat_scores) / cnt / 24))
    return sum(beat_scores) / cnt / 24


if __name__=='__main__':

    motion_results = torch.randn(2, 3, 148, 24 ,3)
    music_feats = torch.randn(2, 150, 35)
    num_person = [1,2]

    cal_BAS(motion_results, num_person, music_feats)

    # folder_a = "/home/xingqunqi/AI_dance/AI_dance/dataset/AIST++_dataset/test/motions_sliced"
    # folder_b = "/home/xingqunqi/AI_dance/AI_dance/dataset/AIST++_dataset/test/baseline_feats"
    # beat_scores = []

    # files_a = {extract_core_name(f): os.path.join(folder_a, f) for f in os.listdir(folder_a) if f.endswith('.pkl')}
    # files_b = {extract_core_name_2(f): os.path.join(folder_b, f) for f in os.listdir(folder_b) if f.endswith('.npy')}
    # matched_files = set(files_a.keys()).intersection(files_b.keys())

    # for file_core_name in matched_files:
        
    #     with open(files_a[file_core_name], 'rb') as file:
    #         data_a = pickle.load(file)

    #     data_b = np.load(files_b[file_core_name])

    #     keypoints = data_a["full_pose"] 
    #     keypoints = keypoints[:300,:,:] # seq, 24, 3

    #     motion_beats = motion_peak_onehot(keypoints)
    #     audio_beats = data_b[:,-1]
    #     beat_score = alignment_score(audio_beats, motion_beats, sigma=3)
    #     beat_scores.append(beat_score)

    # print ("\nBeat score on generated data: %.3f\n" % (sum(beat_scores) / 24))


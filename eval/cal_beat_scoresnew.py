import numpy as np
import pickle 
from scipy import linalg
import json, torch, sys
# kinetic, manual
import os, librosa
from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt 
sys.path.append(os.getcwd())


    
def get_music_beat_fromwav(fpath, length):
    '''get music beat from the raw waveform'''
    FPS = 30
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    # EPS = 1e-6
    data, _ = librosa.load(fpath, sr=SR)[:length]
    # print("loaded music data shape", data.shape)
    envelope = librosa.onset.onset_strength(y=data, sr=SR)  # (seq_len,)
    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH
    )
    start_bpm = librosa.beat.tempo(y=data)[0]
    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=SR,
        hop_length=HOP_LENGTH,
        start_bpm=start_bpm,
        tightness=100,
    )
    return beat_idxs


def get_music_beat_from_musicfea35(fpath, length):
    '''get the music beat from the preextracted baseline feature'''
    data = np.load(fpath)[:length]
    beat_idxs = data[-1]

    beats = beats.astype(bool)
    beat_axis = np.arange(len(beats))
    beat_axis = beat_axis[beats]
  
    return beat_idxs


def calc_db(keypoints):
    '''get motion beat from keypoints position after FK'''
    keypoints = np.array(keypoints).reshape(-1, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats, len(kinetic_vel)


def BA(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats))



def cal_BAS_feats(motions, num_person, music_feats, wav_paths): # positions_recons.view(B, H, T, J, D), num_person, music_feats
    '''
    keypoints: bs, h, t, 24 ,3
    music_feats: bs, t', 35
    '''
    if isinstance(motions, torch.Tensor):  
        keypoints3d_all = motions.detach().cpu().numpy() # positions.view(bs, h, sq, 24, 3)
    else:
        keypoints3d_all = motions
    bs, h, T, J, Dp = keypoints3d_all.shape
    music_feats = music_feats[:,:(T-1),:] # match motion beat
    beat_scores = []

    cnt = 0   
    for n_id in range(bs): # each element
        # same music 
        wav_path = wav_paths[n_id]
        for h_id in range(num_person[n_id]): # each person, excluding padding
            cnt+=1
            keypoints3d = keypoints3d_all[n_id][h_id] # T, 24, 3
            
            dance_beats, length = calc_db(keypoints3d)
            music_beats = get_music_beat_fromwav(wav_path, length) # currently must load from the wav
            beat_score = BA(music_beats, dance_beats)
            beat_scores.append(beat_score)

    return beat_scores


def load_data(datapath):
    with open(datapath, "rb") as f:
        data = pickle.load(f)
    return data

if __name__ == '__main__':
    music_root = "./dataset/aamixed_dataset/test/baseline_feats/_P-JWcq1ewI_04_0_1260_slice0.npy"
    pred_root = './temp_vis_split/ps1.pkl'
    
    music_feats = np.load(music_root)
    motion_feats = load_data(pred_root)
    smpl_poses = motion_feats['smpl_poses'].reshape(-1, 24, 3)
    smpl_trans = motion_feats['smpl_trans']

    smpl_poses = smpl_poses[np.newaxis, np.newaxis, :]
    music_feats = music_feats[np.newaxis, :]
    num_person = [1]

    wav_path = "./dataset/aamixed_dataset/test/wavs_sliced/_P-JWcq1ewI_04_0_1260_slice0.wav"
    cal_BAS_feats(smpl_poses, num_person, music_feats, wav_path)

    import pdb
    pdb.set_trace()

  
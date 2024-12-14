import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os
from typing import Any
from torch.utils.data import ConcatDataset

from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)
from dataset.preprocess import Normalizer, vectorize_many
from dataset.quaternion import ax_to_6v

import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate
import pickle

from .vis import SMPLSkeleton
from pathlib import Path
import glob
# for visualization
from smplx import SMPL
import matplotlib.pyplot as plt
import pyrender
import trimesh
os.environ["PYOPENGL_PLATFORM"] = "egl" # headless render mode

"""
Collect aist++ statistics for fid_encoder and vqvae model training
"""

'''For use of training music-2-dance generative model'''
class Music2DanceDataset(data.Dataset):
    def __init__(
        self,
        dataset_name: str, 
        is_test: bool, 
        backup_path: str = "/home/xingqunqi/AI_dance/MMM/dataset/backups/aistpp",
        feature_type: str = "baseline", # music feature type
        normalizer: Any = None, 
        data_len: int = -1, # TODO(yw) check, probably relevant to audio sample rate
        shuffle=True, # NOTE(yw) shuffle, no matter train or test. Because some samples are from the same original sequence.
        include_contacts: bool = True, # heel and toe of each foot, dim+=4
        force_reload: bool = True,
        unit_length: int = 8,): # NOTE(yw) set true to debug dataset and dataloader
        
        # data preprocess 做的slice就已经把audio和motion变成了定长，motion150帧
        self.motion_length = 150
        self.dataset_name = dataset_name
        self.is_test = is_test
        self.normalizer = normalizer
        self.data_len = data_len
        self.shuffle = shuffle
        self.include_contacts = include_contacts
        self.unit_length = unit_length
        self.mean = None
        self.std = None
            
        if dataset_name == 'aistpp':
            self.data_root = '/home/xingqunqi/AI_dance/MMM/dataset/AIST++_dataset' # NOTE(yw) 用绝对路径，fid extractor在导入时相对路径不同
            self.motion_dir = pjoin(self.data_root, 'annotations/motions') # using smpl 72 dim pose representation
            self.audio_dir = pjoin(self.data_root, 'extracted_audios')
            
            self.joints_num = 24 #SMPL 24 joints
            self.raw_fps = 60
            self.data_fps = 30
            assert self.data_fps <= self.raw_fps
            self.data_stride = self.raw_fps // self.data_fps
            self.feature_type = feature_type # music feature type

            pickle_name = "processed_train_data.pkl" if not is_test else "processed_test_data.pkl"

        print("Loading dataset...")
        data = self.load_aistpp()  # Call this last

        # process data, convert to 6dof etc
        pose_input = self.process_dataset(data["pos"], data["q"])
        
        self.data = {
            "pose": pose_input, # 17733, 1, 150, 151
            "filenames": data["filenames"],
            "wavs": data["wavs"],
        }
        assert len(pose_input) == len(data["filenames"])
        self.length = len(pose_input) # num of data
        
             
    def __len__(self):
        return self.length

    def get_all_data(self):
        return self.data

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        feature = torch.from_numpy(np.load(filename_))
        return (self.data["pose"][idx], feature[:148,:], filename_, self.data["wavs"][idx]) #NOTE(yw) also modify audio feature T
        # audio 的T表示和帧数不同，不对audio截取，用conv/mlp mapping
    def load_aistpp(self):
        # open data path
        split_data_path = os.path.join(
            self.data_root, "test" if self.is_test else "train"
        )

        motion_path = os.path.join(split_data_path, "motions_sliced")
        sound_path = os.path.join(split_data_path, f"{self.feature_type}_feats")
        wav_path = os.path.join(split_data_path, f"wavs_sliced")
        # sort motions and sounds
        motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))
        features = sorted(glob.glob(os.path.join(sound_path, "*.npy")))
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))

        # stack the motions and features together
        all_pos = []
        all_q = []
        all_names = []
        all_wavs = []
        assert len(motions) == len(features)
        for motion, feature, wav in zip(motions, features, wavs):
            # make sure name is matching
            m_name = os.path.splitext(os.path.basename(motion))[0]
            f_name = os.path.splitext(os.path.basename(feature))[0]
            w_name = os.path.splitext(os.path.basename(wav))[0]
            assert m_name == f_name == w_name, str((motion, feature, wav))
            # load motion
            data = pickle.load(open(motion, "rb"))
            pos = data["pos"]
            q = data["q"]
            all_pos.append(pos)
            all_q.append(q)
            all_names.append(feature)
            all_wavs.append(wav)

        all_pos = np.array(all_pos)  # N x seq x 3
        all_q = np.array(all_q)  # N x seq x (joint * 3)
        # downsample the motions to the data fps
        print(f'downsample the motion using stride {self.data_stride}')
        all_pos = all_pos[:, :: self.data_stride, :] # ratio between the rawdata and newdata
        all_q = all_q[:, :: self.data_stride, :]
        data = {"pos": all_pos, "q": all_q, "filenames": all_names, "wavs": all_wavs} # filenames是.npy, wavs是.wav
        
        return data

    def process_dataset(self, root_pos, local_q): # aistpp custom process
        # FK skeleton
        smpl = SMPLSkeleton()

        root_pos = torch.Tensor(root_pos)
        local_q = torch.Tensor(local_q)
        # to ax
        bs, sq, c = local_q.shape # B, T, D
        local_q = local_q.reshape((bs, sq, -1, 3)) # B, T, 24, 3  # 3D representation in axis-angel format
        
        # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
        root_q = local_q[:, :, :1, :]  # sequence x 1 x 3
        root_q_quat = axis_angle_to_quaternion(root_q)
        rotation = torch.Tensor(
            [0.7071068, 0.7071068, 0, 0]
        )  # 90 degrees about the x axis
        root_q_quat = quaternion_multiply(rotation, root_q_quat) # 旋转quaternion表示的root坐标
        root_q = quaternion_to_axis_angle(root_q_quat)
        local_q[:, :, :1, :] = root_q

        # don't forget to rotate the root position too 😩
        pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
        root_pos = pos_rotation.transform_points(
            root_pos
        )  # basically (y, z) -> (-z, y), expressed as a rotation for readability

        # do FK
        positions = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
        
        feet = positions[:, :, (7, 8, 10, 11)]
        feetv = torch.zeros(feet.shape[:3])
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(local_q)  # cast to right dtype

        # to 6d
        local_q = ax_to_6v(local_q)

        # now, flatten everything into: batch x sequence x [...]
        l = [contacts, root_pos, local_q] # TODO(yw) 注意顺序，不取contact是去掉前面四个
        global_pose_vec_input = vectorize_many(l).float().detach()

        # 17733, 150, 151 (B, T, D)
        assert not torch.isnan(global_pose_vec_input).any()
        data_name = "Test" if self.is_test else "Train"

        # cut the dataset
        if self.data_len > 0:
            global_pose_vec_input = global_pose_vec_input[: self.data_len]

        # # 17733, 150, 151 (B, T, D) --> (B, H, T, D)
        global_pose_vec_input = global_pose_vec_input.unsqueeze(1)[:,:,:148,:]
        print(f"{data_name} Dataset Motion Features Dim: {global_pose_vec_input.shape}")

        return global_pose_vec_input



if __name__=='__main__':
    aistpp_train = Music2DanceDataset('aistpp', 
                                is_test=False, 
                                shuffle=False, 
                                normalizer=None)

    aistpp_test = Music2DanceDataset('aistpp', 
                                is_test=True, 
                                shuffle=False, 
                                normalizer=None)

    motion_train = aistpp_train.get_all_data()['pose'] # 17733, 1, 148, 151
    motion_test = aistpp_test.get_all_data()['pose'] # 186, 1, 148, 151

    motion_all = torch.cat([motion_train, motion_test], dim=0) # 17919, 1, 148, 151

    EPSILON = 1e-10
    # mean torch.Size([1, 1, 1, 151])  std torch.Size([1, 1, 1, 151])
    mean = motion_all.mean(dim=(0,1,2), keepdim=True) #TODO(yw) 之后多人，check不同人(dim=1)可以混在一起做mean和std
    std = motion_all.std(dim=(0,1,2), keepdim=True)
    std += EPSILON

    # to numpy, squeeze dim for efficient saving
    mean_np = mean.squeeze().numpy()
    std_np = std.squeeze().numpy()
    with open("/home/xingqunqi/AI_dance/MMM/checkpoints/aistpp/meta/mean_std.pkl", "wb") as f:
        pickle.dump({"mean": mean_np, "std": std_np}, f)

    print("Mean and std saved to mean_std.pkl")


    # with open("mean_std.pkl", "rb") as f:
    #     data = pickle.load(f)
    # mean_loaded = data["mean"]
    # std_loaded = data["std"]
    # mean_tensor = torch.tensor(mean_loaded).view(1, 1, 1, -1)
    # std_tensor = torch.tensor(std_loaded).view(1, 1, 1, -1)
    # x_normalized = (x - mean) / std
    # print("Normalized shape:", x_normalized.shape)  # 仍为 (17919, 1, 148, 75)
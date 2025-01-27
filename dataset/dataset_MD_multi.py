import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os
from typing import Any
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)
from dataset.preprocess import Normalizer, vectorize_many_multi
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

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def fileToList(f):
    out = open(f, "r").readlines()
    out = [x.strip() for x in out]
    out = [x for x in out if len(x)]
    return out

def visualize_joints(joints):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(elev=90, azim=-90) # otherwise it will in lay-down view
    # SMPL skeleton: line 
    skeleton = [
        (0, 1), (1, 4), (4, 7), (0, 2), (2, 5), (5, 8), (8, 11), (7, 10), # 腿部
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),      # 躯干
        (12, 13), (13, 16), (12, 14), (14, 17),         # 手臂
        (16, 18), (18, 20), (17, 19), (19, 21), (20, 22), (21, 23)         # 手
    ]
    
    H = joints.shape[0] 
    for h in range(H):
        person_joints = joints[h]  # joint position of each person (T, 24, 3)
        # plot joints
        ax.scatter(
            person_joints[:, 0],  # x
            person_joints[:, 1],  # y
            person_joints[:, 2],  # z
            label=f'Person {h + 1}', 
            s=25  # point size
        )
        
        for joint_start, joint_end in skeleton:
            ax.plot(
                [joints_t[joint_start, 0], joints_t[joint_end, 0]],
                [joints_t[joint_start, 1], joints_t[joint_end, 1]],
                [joints_t[joint_start, 2], joints_t[joint_end, 2]],
                'b-'
            )

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Joint Visualization')
    ax.legend()

    plt.savefig('joint_visualization.png')
    plt.close(fig)
    
    
'''For use of training music-2-dance generative model'''
class Music2DanceDataset(data.Dataset):
    def __init__(
        self,
        dataset_name: str, 
        data_split: str, 
        feature_type: str = "baseline", # music feature type
        normalizer: Any = None, 
        data_len: int = -1, # cut data if originally not in the same length
        shuffle=True,
        include_contacts: bool = True, # heel and toe of each foot, dim+=4
        force_reload: bool = True,
        unit_length: int = 4,
        stats_path_aistpp: str = "/home/xingqunqi/AI_dance/AI_dance/checkpoints/aistpp/meta/mean_std.pkl",
        stats_path_aioz: str = "/home/xingqunqi/AI_dance/AI_dance/checkpoints/aioz/meta/mean_std.pkl",
        stats_path_aamixed: str = "/home/xingqunqi/AI_dance/AI_dance/checkpoints/aamixed/meta/mean_std.pkl",
        ): # TODO(yiwen) modify this to new stats
        
        # data preprocess has already sliced the audio and motion to fixed length
        self.motion_length = 150
        self.dataset_name = dataset_name
        self.data_split = data_split
        self.normalizer = normalizer
        self.data_len = data_len
        self.shuffle = shuffle
        self.include_contacts = include_contacts
        self.unit_length = unit_length

        # for data alignment and stat collection
        self.pos = None

        # TODO(yiwen) incorporate aioz preprocess code to repo
        if dataset_name == 'aamixed':        
            self.data_root = '/home/xingqunqi/AI_dance/AI_dance/dataset/aamixed_dataset' # NOTE(yiwen) please use absolute path here, since this will be called by other scripts
            self.mean, self.std = self.get_stats(stats_path_aamixed) 

        if dataset_name == 'aioz':
            self.data_root = '/home/xingqunqi/AI_dance/AI_dance/dataset/AIOZ_Gdance_dataset'
            self.mean, self.std = self.get_stats(stats_path_aioz)

        if dataset_name == 'aistpp':
            self.data_root = '/home/xingqunqi/AI_dance/AI_dance/dataset/AIST++_dataset'
            self.mean, self.std = self.get_stats(stats_path_aistpp)

        self.joints_num = 24 #SMPL 24 joints
        self.raw_fps_aistpp = 60
        self.data_fps = 30
        assert self.data_fps <= self.raw_fps_aistpp
        self.data_stride = self.raw_fps_aistpp // self.data_fps
        self.feature_type = feature_type 

        print("Loading dataset...") # load raw data 
        data = self.load_data()  

        print(
            f"Loaded {self.dataset_name} Dataset With Dimensions: Pos: {data['pos'].shape}, Q: {data['q'].shape}"
        )
            
        # process data, convert to 6dof etc
        pose_input = self.process_dataset(data["pos"], data["q"])
        
        # normalize the 6d data
        pose_input = (pose_input - self.mean) / self.std # std has already added 1e-10 in preprocessing
        
        self.data = {
            "pose": pose_input, # B, H, 150, 151 
            "filenames": data["filenames"],
            "wavs": data["wavs"],
            "num_person": data["num_person"]
        }
        assert len(pose_input) == len(data["filenames"])
        self.length = len(pose_input) # num of data
        
             
    def get_stats(self, stats_path):
        with open(stats_path, "rb") as f:
            data = pickle.load(f)
        mean_loaded = data["mean"]
        std_loaded = data["std"]
        mean_tensor = torch.tensor(mean_loaded).view(1, 1, 1, -1)
        std_tensor = torch.tensor(std_loaded).view(1, 1, 1, -1)
        return mean_tensor, std_tensor # 1, 1, 1, 75

    # def inv_transform(self, data):
    #     if self.std==None:
    #         return data
    #     return data * self.std + self.mean

    # def forward_transform(self, data):
    #     return (data - self.mean) / self.std

    def __len__(self):
        return self.length

    def get_all_data(self):
        return self.data, self.pos

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        feature = torch.from_numpy(np.load(filename_))
        return (self.data["pose"][idx], feature, filename_, self.data["wavs"][idx], self.data["num_person"][idx]) 
        # do not slice T in audio

    def load_data(self):
        max_person_num = 3 # TODO(yiwen) make this arg
        delta_height = 2.5388 # NOTE(yiwen) from stats_collect

        # open data path
        split_data_path = os.path.join(
            self.data_root, self.data_split
        )

        # Structure:
        # data
        #   |- train
        #   |    |- motion_sliced
        #   |    |- wav_sliced
        #   |    |- baseline_features
        #   |    |- motions
        #   |    |- wavs
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
        all_h = []
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

            if len(pos.shape)==2: # from aistpp
                pos = np.expand_dims(pos, axis=0) # T, 3 --> H, T, 3
                q = np.expand_dims(q, axis=0) # T, 72 --> H, T, 72
                pos = pos[:, :: self.data_stride, :] # sample rate 60 --> 30
                q = q[:, :: self.data_stride, :]

                pos[:,:,1:2] = pos[:,:,1:2] - delta_height # aistpp align aioz
                # TODO(yiwen) modify root_trans, align the height of aistpp to aioz's
    
            H = pos.shape[0]
            
            if H > max_person_num: # main experiment uses H<=3
                continue
            elif H < max_person_num:
                # print(f"Padding sample {motion} from H={H} to H=3")
                pad_shape_pos = (max_person_num - H, pos.shape[1], pos.shape[2])
                pad_shape_q = (max_person_num - H, q.shape[1], q.shape[2])
                pos = np.vstack([pos, np.zeros(pad_shape_pos, dtype=pos.dtype)])
                q = np.vstack([q, np.zeros(pad_shape_q, dtype=q.dtype)])
            
            all_pos.append(pos)
            all_q.append(q)
            all_names.append(feature)
            all_wavs.append(wav)
            all_h.append(H)

        all_pos = np.array(all_pos)  # N x H x T x 3
        all_q = np.array(all_q)  # N x H x T x (joint * 3)
        self.pos = all_pos
        
        data = {"pos": all_pos, "q": all_q, "filenames": all_names, "wavs": all_wavs, "num_person": all_h} 
        
        return data

    def process_dataset(self, root_pos, local_q): 
        # FK skeleton
        smpl = SMPLSkeleton()
        # to Tensor
        root_pos = torch.Tensor(root_pos)  # (B, H, 150, 3)
        local_q = torch.Tensor(local_q)   # (B, H, 150, 72)
        # to ax
        bs, h, sq, c = local_q.shape      # B, H, T, D
        local_q = local_q.reshape((bs, h, sq, -1, 3))  # (B, H, T, 24, 3)    # 3D representation in axis-angel format
        
        # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
        # if h!=1: # aioz TODO(yiwen) check rotation and position of the two dataset(vis)
        root_q = local_q[:, :, :, :1, :]  # (B, H, T, 1, 3)
        root_q_quat = axis_angle_to_quaternion(root_q)
        rotation = torch.Tensor(
            [0.7071068, 0.7071068, 0, 0]
        )  # 90 degrees about the x axis
        root_q_quat = quaternion_multiply(rotation, root_q_quat) 
        root_q = quaternion_to_axis_angle(root_q_quat)
        local_q[:, :, :, :1, :] = root_q

        # don't forget to rotate the root position too 😩
        pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
        
        root_pos = pos_rotation.transform_points(root_pos.reshape(-1, 3))  
        root_pos = root_pos.view(bs, h, sq, 3)  # (B, H, T, 3)
        
        # do FK
        #positions = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
        positions = smpl.forward(local_q.view(bs * h, sq, -1, 3), root_pos.view(bs * h, sq, 3))
        positions = positions.view(bs, h, sq, -1, 3)  # -------- (B, H, T, 24, 3)
        
        ########## NOTE(yiwen) data visualize
        ## for joints
        # visualize_joints(positions[0,:, 0,:,:]) # (H, 24, 3)
        ##########
        
        feet = positions[:, :, :, (7, 8, 10, 11)]
        feetv = torch.zeros(feet.shape[:4])    # (B, H, T-1, 4)
        feetv[:, :, :-1] = (feet[:, :, 1:] - feet[:, :, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(local_q) # cast to right dtype
        # to 6d
        local_q = ax_to_6v(local_q.reshape(bs * h, sq, -1, 3)) 
        local_q = local_q.view(bs, h, sq, -1)  #  (B, H, T, 24 × 6)
        
        
        # now, flatten everything into: batch x sequence x [...]
        # NOTE：global_pose_vec_input: (B, H, T, 4) + (B, H, T, 3) + (B, H, T, 144) -----> (B, H, T, 151)
        l = [contacts, root_pos, local_q] 
        global_pose_vec_input = vectorize_many_multi(l).float().detach()
        # 17733, 150, 151 (B, T, D)
        assert not torch.isnan(global_pose_vec_input).any()
        
        if self.data_len > 0:
            global_pose_vec_input = global_pose_vec_input[: self.data_len] 
        # TODO(yw) check T=148 (后面vqvae的decoder需要是4的倍数), integrate unit_length here
        
        global_pose_vec_input = global_pose_vec_input[:,:,:148,:] # --> (B, H, T==148, 151)
        print(f"{self.data_split} Dataset Motion Features Dim: {global_pose_vec_input.shape}")
        return global_pose_vec_input
    
def DATALoader(dataset_name,
               data_split,
               batch_size,
               num_workers = 8, 
               normalizer = None,
               shuffle=True) : #TODO(yiwen) add unit_length here
    
    data_loader = torch.utils.data.DataLoader(Music2DanceDataset(dataset_name, data_split=data_split, shuffle=shuffle, normalizer=normalizer),
                                              batch_size,
                                              shuffle = shuffle,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              drop_last = True)
    
    return data_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

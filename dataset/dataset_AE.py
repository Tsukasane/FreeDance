import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import os
from typing import Any
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)
from dataset.preprocess import vectorize_many
from torch.utils.data._utils.collate import default_collate
import pickle
from .vis import SMPLSkeleton
import glob
import matplotlib.pyplot as plt

os.environ["PYOPENGL_PLATFORM"] = "egl" # headless render mode

"""
Compare with dataset_MD_multi, no num_person dimension
output is in unnormalized 3D representation D=75
"""

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
class Music2DanceDataset_AE(data.Dataset):
    def __init__(
        self,
        dataset_name: str, 
        data_split: str, 
        feature_type: str = "baseline", # music feature type
        data_len: int = -1, # cut data if originally not in the same length
        shuffle=True,
        ): 
        
        # data preprocess has already sliced the audio and motion to fixed length
        self.motion_length = 150
        self.dataset_name = dataset_name
        self.data_split = data_split
        self.data_len = data_len
        self.shuffle = shuffle

        # for data alignment and stat collection
        self.pos = None

        # TODO(yiwen) incorporate aioz preprocess code to repo
        if dataset_name == 'aamixed':        
            self.data_root = './dataset/aamixed_dataset' # NOTE(yiwen) please use absolute path here, since this will be called by other scripts

        if dataset_name == 'aioz':
            self.data_root = './dataset/AIOZ_Gdance_dataset'

        if dataset_name == 'aistpp':
            self.data_root = './dataset/AIST++_dataset'
       
        self.raw_fps_aistpp = 60
        self.data_fps = 30
        self.max_motion_length = 50 #length of code in one seq
        
        assert self.data_fps <= self.raw_fps_aistpp
        self.data_stride = self.raw_fps_aistpp // self.data_fps
        self.feature_type = feature_type
        
        print("Loading dataset...") # load raw data 
        data = self.load_data()  

        print(
            f"Loaded {self.dataset_name} Dataset With Dimensions: Pos: {data['pos'].shape}, Q: {data['q'].shape}"
        )
        
        # process data, 3d
        pose_input = self.process_dataset(data["pos"], data["q"])
        self.data = {
            "pose": pose_input, # N, 148, 75
        }
        self.length = len(pose_input) # num of data
        

    def __len__(self):
        return self.length

    def get_all_data(self):
        return self.data, self.pos

    def __getitem__(self, idx):
        return (self.data["pose"][idx]) 
        
    def load_data(self):
        max_person_num = 3 # TODO(yiwen) make this arg
        delta_height = 2.5388 # NOTE(yiwen) from stats_collect

        # open data path
        split_data_path = os.path.join(
            self.data_root, self.data_split
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
                # pos = np.expand_dims(pos, axis=0) # T, 3 --> H, T, 3
                # q = np.expand_dims(q, axis=0) # T, 72 --> H, T, 72
                pos = pos[:: self.data_stride, :] # sample rate 60 --> 30
                q = q[:: self.data_stride, :]

                pos[:,1:2] = pos[:,1:2] - delta_height # aistpp align aioz
                all_pos.append(pos)
                all_q.append(q)

            else:
                H = pos.shape[0]
                if H > max_person_num: # main experiment uses H<=3
                    continue

                else: # H=2 or 3
                    for h in range(H):
                        all_pos.append(pos[h]) # T, 3
                        all_q.append(q[h]) # T, 72

        all_pos = np.array(all_pos)  # N x T x 3
        all_q = np.array(all_q)  # N x H x T x (joint * 3)
        self.pos = all_pos

        data = {"pos": all_pos, "q": all_q} 
        
        return data

    def process_dataset(self, root_pos, local_q): # aistpp custom process
        # to Tensor
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
        root_q_quat = quaternion_multiply(rotation, root_q_quat) 
        root_q = quaternion_to_axis_angle(root_q_quat)
        local_q[:, :, :1, :] = root_q 

        # don't forget to rotate the root position too 😩
        pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
        root_pos = pos_rotation.transform_points(
            root_pos
        )  # basically (y, z) -> (-z, y), expressed as a rotation for readability

        l = [root_pos, local_q] 
        global_pose_vec_input = vectorize_many(l).float().detach() # N, T, 75

        assert not torch.isnan(global_pose_vec_input).any()
        print(f"{self.data_split} Dataset Motion Features Dim: {global_pose_vec_input.shape}")
        
        return global_pose_vec_input[:,:148,:] # 17733, 1, 150, 75

    
def DATALoader(dataset_name,
               data_split,
               batch_size,
               load_motion_code = False,
               num_workers = 8, 
               shuffle=True):
    
    data_loader = torch.utils.data.DataLoader(Music2DanceDataset_AE(dataset_name, 
                                                                 data_split=data_split,
                                                                 shuffle=shuffle,
                                                                 load_motion_code=load_motion_code),
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

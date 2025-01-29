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

"""
Collect AIOZ statistics for fid_encoder and vqvae model training
"""



# def collate_fn(batch):
#     batch.sort(key=lambda x: x[3], reverse=True)
#     return default_collate(batch)

# def fileToList(f):
#     out = open(f, "r").readlines()
#     out = [x.strip() for x in out]
#     out = [x for x in out if len(x)]
#     return out


# def visualize_joints(joints):
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     # ax.view_init(elev=90, azim=-90) # otherwise it will in lay-down view

#     # SMPL skeleton: line 
#     skeleton = [
#         (0, 1), (1, 4), (4, 7), (0, 2), (2, 5), (5, 8), (8, 11), (7, 10), # 腿部
#         (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),      # 躯干
#         (12, 13), (13, 16), (12, 14), (14, 17),         # 手臂
#         (16, 18), (18, 20), (17, 19), (19, 21), (20, 22), (21, 23)         # 手
#     ]
    
#     H = joints.shape[0]  # 获取人数
#     for h in range(H):
#         person_joints = joints[h]  # 每个人的关节位置 (T, 24, 3)

#         # 绘制关节点
#         ax.scatter(
#             person_joints[:, 0],  # x 坐标
#             person_joints[:, 1],  # y 坐标
#             person_joints[:, 2],  # z 坐标
#             label=f'Person {h + 1}',  # 每个人的标签
#             s=25  # 点的大小
#         )
        
#         for joint_start, joint_end in skeleton:
#             ax.plot(
#                 [person_joints[joint_start, 0], person_joints[joint_end, 0]],
#                 [person_joints[joint_start, 1], person_joints[joint_end, 1]],
#                 [person_joints[joint_start, 2], person_joints[joint_end, 2]],
#                 'b-'
#             )

#     # 设置坐标轴
#     ax.set_xlabel('X-axis')
#     ax.set_ylabel('Y-axis')
#     ax.set_zlabel('Z-axis')
#     ax.set_title('3D Joint Visualization')
#     ax.legend()

#     # save
#     plt.savefig('joint_visualization.png')
#     plt.close(fig)
    
    
'''For use of training music-2-dance generative model'''
class Music2DanceDataset(data.Dataset):
    def __init__(
        self,
        dataset_name: str, 
        is_test: bool, 
        #backup_path: str = "/home/xingqunqi/AI_dance/MMM/dataset/backups/aistpp",
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

        if dataset_name == 'aioz':        # -------【litingw：修改下面的绝对路径名称】
            self.data_root = '/home/xingqunqi/AI_dance/litingw/Group-Dance/preprocess/aioz_gdance/AIOZ_Gdance_dataset/' # NOTE(yw) 用绝对路径，fid extractor在导入时相对路径不同
            # self.motion_dir = pjoin(self.data_root, 'annotations/motions') # using smpl 72 dim pose representation
            # self.audio_dir = pjoin(self.data_root, 'extracted_audios')
            
            self.joints_num = 24 #SMPL 24 joints
            self.raw_fps = 30          # -------【litingw：AIOZ为30】
            self.data_fps = 30
            assert self.data_fps <= self.raw_fps
            self.data_stride = self.raw_fps // self.data_fps
            self.feature_type = feature_type # music feature type

            #pickle_name = "processed_train_data.pkl" if not is_test else "processed_test_data.pkl"

        print("Loading dataset...")
        data = self.load_aioz()  # Call this last

        # process data, convert to 6dof etc
        pose_input = self.process_dataset(data["pos"], data["q"])
        
        self.data = {
            "pose": pose_input, # 17733, 1, 150, 151   # ------【litingw：TODO 所以是B，H，148，151？？】
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

    def load_aioz(self):
        # open data path
        split_data_path = os.path.join(
            self.data_root, "val" if self.is_test else "train"
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


            # ---------------------------litingw：筛H
            H = pos.shape[0]

            if H > 3:
                # -----跳过 H > 3 的样本
                # print(f"Skipping sample {motion} with H={H}")
                continue
            elif H < 3:
                # -----对 H < 3 的样本补零
                # print(f"Padding sample {motion} from H={H} to H=3")
                pad_shape_pos = (3 - H, pos.shape[1], pos.shape[2])
                pad_shape_q = (3 - H, q.shape[1], q.shape[2])
                pos = np.vstack([pos, np.zeros(pad_shape_pos, dtype=pos.dtype)])
                q = np.vstack([q, np.zeros(pad_shape_q, dtype=q.dtype)])
                # print('========pos')
                # print(pos)
                # print('========q')
                # print(q)


            all_pos.append(pos)
            all_q.append(q)
            all_names.append(feature)
            all_wavs.append(wav)

        all_pos = np.array(all_pos)  # N x H x seq x 3
        all_q = np.array(all_q)  # N x H x seq x (joint * 3)
        # downsample the motions to the data fps              # -------【litingw：AIOZ motion.pkl帧速率为30，无需downsample】
        # print(f'downsample the motion using stride {self.data_stride}')
        # all_pos = all_pos[:, :: self.data_stride, :] # ratio between the rawdata and newdata
        # all_q = all_q[:, :: self.data_stride, :]
        data = {"pos": all_pos, "q": all_q, "filenames": all_names, "wavs": all_wavs} # filenames是.npy, wavs是.wav
        
        return data

    def process_dataset(self, root_pos, local_q): # aistpp custom process
        # FK skeleton
        smpl = SMPLSkeleton()

        root_pos = torch.Tensor(root_pos)  # (B, H, 150, 3)
        local_q = torch.Tensor(local_q)   # (B, H, 150, 72)

        # to ax
        bs, h, sq, c = local_q.shape      # B, H, T, D
        local_q = local_q.reshape((bs, h, sq, -1, 3))  # (B, H, T, 24, 3)    # 3D representation in axis-angel format
        
        # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
        root_q = local_q[:, :, :, :1, :]  # (B, H, T, 1, 3)
        root_q_quat = axis_angle_to_quaternion(root_q)
        rotation = torch.Tensor(
            [0.7071068, 0.7071068, 0, 0]
        )  # 90 degrees about the x axis
        root_q_quat = quaternion_multiply(rotation, root_q_quat) # 旋转quaternion表示的root坐标
        root_q = quaternion_to_axis_angle(root_q_quat)
        local_q[:, :, :, :1, :] = root_q

        # don't forget to rotate the root position too 😩
        pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
        # root_pos = pos_rotation.transform_points(
        #     root_pos
        # )  # basically (y, z) -> (-z, y), expressed as a rotation for readability
        root_pos = pos_rotation.transform_points(root_pos.reshape(-1, 3))  
        root_pos = root_pos.view(bs, h, sq, 3)  # (B, H, T, 3)

        # do FK
        #positions = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
        positions = smpl.forward(local_q.view(bs * h, sq, -1, 3), root_pos.view(bs * h, sq, 3))
        positions = positions.view(bs, h, sq, -1, 3)  # -------- (B, H, T, 24, 3)
        
        ########## NOTE(yw) visualization 
        # for joints
        # visualize_joints(positions[0,:, 105,:,:]) # (H, 24, 3)

        ##########

        feet = positions[:, :, :, (7, 8, 10, 11)]
        feetv = torch.zeros(feet.shape[:4])    # (B, H, T-1, 4)
        feetv[:, :, :-1] = (feet[:, :, 1:] - feet[:, :, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(local_q) # cast to right dtype
       
        # to 6d
        local_q = ax_to_6v(local_q.reshape(bs * h, sq, -1, 3)) 
        local_q = local_q.view(bs, h, sq, -1)  #  (B, H, T, 24 × 6)   
        print('========local_q')
        print(local_q.shape)   # (B, H, T, 144)

        # now, flatten everything into: batch x sequence x [...]
        # NOTE：global_pose_vec_input 尺寸为：(B, H, T, 4) + (B, H, T, 3) + (B, H, T, 144) -----> (B, H, T, 151)
        l = [contacts, root_pos, local_q] # TODO(yw) 注意顺序，不取contact是去掉前面四个
        global_pose_vec_input = vectorize_many_multi(l).float().detach()

        # 17733, 150, 151 (B, T, D)
        assert not torch.isnan(global_pose_vec_input).any()
        data_name = "Val" if self.is_test else "Train"

        # cut the dataset  # -------【LITINGW：TODO 未执行？，目前设置的是data_len==-1）
        if self.data_len > 0:
            global_pose_vec_input = global_pose_vec_input[: self.data_len] 

        # (B, H, T==150, 151) 
        # print('------global_pose_vec_input-------')
        # print(global_pose_vec_input.shape)
        global_pose_vec_input = global_pose_vec_input[:,:,:148,:] # --> (B, H, T==148, 151)
        print(f"{data_name} Dataset Motion Features Dim: {global_pose_vec_input.shape}")

        return global_pose_vec_input



if __name__=='__main__':
    aistpp_train = Music2DanceDataset('aioz',     # -------【litingw：aistpp改为aioz】
                                is_test=False, 
                                shuffle=False, 
                                normalizer=None)

    aistpp_test = Music2DanceDataset('aioz',  # -------【litingw：aistpp改为aioz】
                                is_test=True, 
                                shuffle=False, 
                                normalizer=None)

    motion_train = aistpp_train.get_all_data()['pose'] # B, H, 148, 151
    # print('------motion_train-------')
    # print(motion_train.shape)
    motion_test = aistpp_test.get_all_data()['pose'] # B, H, 148, 151
    # print('------motion_test-------')
    # print(motion_test)

    
    motion_all = torch.cat([motion_train, motion_test], dim=0) # 17919, H, 148, 151


    EPSILON = 1e-10
    # mean torch.Size([1, 1, 1, 151])  std torch.Size([1, 1, 1, 151])
    mean = motion_all.mean(dim=(0,1,2), keepdim=True) #TODO(yw) 之后多人，check不同人(dim=1)可以混在一起做mean和std
    std = motion_all.std(dim=(0,1,2), keepdim=True)
    std += EPSILON

    # to numpy, squeeze dim for efficient saving
    mean_np = mean.squeeze().numpy()
    std_np = std.squeeze().numpy()
    with open("/home/xingqunqi/AI_dance/litingw/Group-Dance/mean_std.pkl", "wb") as f:
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
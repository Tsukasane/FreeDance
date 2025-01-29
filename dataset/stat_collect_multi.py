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
from .dataset_MD_multi import Music2DanceDataset
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
先统计高度
再把高度的stat放进loader，align高度
再拿position求 D dim stats

"""

def cal_mean_std(motion_all):
    # nan_mask = torch.isnan(motion_all)  # 找到所有 NaN 的位置
    # inf_mask = torch.isinf(motion_all)  # 找到所有 Inf 的位置

    # # 打印非法值的统计信息
    # print(f"Number of NaN values: {nan_mask.sum().item()}")
    # print(f"Number of Inf values: {inf_mask.sum().item()}")

    # # 获取具体位置
    # nan_indices = torch.nonzero(nan_mask, as_tuple=True)  # NaN 的索引
    # inf_indices = torch.nonzero(inf_mask, as_tuple=True)  # Inf 的索引

    # # 如果需要打印具体位置，可以输出
    # if nan_indices[0].numel() > 0:
    #     print(f"NaN found at indices: {nan_indices}")
    # if inf_indices[0].numel() > 0:
    #     print(f"Inf found at indices: {inf_indices}")


    if torch.isnan(motion_all).any() or torch.isinf(motion_all).any():
        # TODO(yiwen) aioz 最后12/151 个元素看起来是非法值
        print("motion_all contains NaN or Inf. Cleaning...")
        motion_all = torch.nan_to_num(motion_all, nan=0.0, posinf=1e10, neginf=-1e10)

    non_zero_mask = (motion_all != 0).float()
    
    mean = (motion_all * non_zero_mask).sum(dim=(0, 1, 2), keepdim=True) / (non_zero_mask.sum(dim=(0, 1, 2), keepdim=True) + 1e-8) # 求stat时不考虑padding的0位置
    variance = ((motion_all - mean)**2 * non_zero_mask).sum(dim=(0, 1, 2), keepdim=True) / (non_zero_mask.sum(dim=(0, 1, 2), keepdim=True) + 1e-8)
    std = variance.sqrt()

    # to numpy, squeeze dim for efficient saving
    mean_np = mean.squeeze().numpy()
    std_np = std.squeeze().numpy()

    print(f'mean_np {mean_np}  std_np {std_np}')

    return mean_np, std_np

if __name__=='__main__':

    stage = 2
    dataset_name = 'aioz'

    if stage==1: # collect average height of pelvis joint in aistpp and aioz respectively
        aistpp_train = Music2DanceDataset('aistpp', data_split='train', shuffle=False, normalizer=None)
        aistpp_test = Music2DanceDataset('aistpp', data_split='test', shuffle=False, normalizer=None)

        aioz_train = Music2DanceDataset('aioz', data_split='train', shuffle=False, normalizer=None)
        aioz_test = Music2DanceDataset('aioz', data_split='test', shuffle=False, normalizer=None)
        aioz_val = Music2DanceDataset('aioz', data_split='val', shuffle=False, normalizer=None)

        _, pos_train_aistpp = aistpp_train.get_all_data()
        _, pos_test_aistpp = aistpp_test.get_all_data()

        _, pos_train_aioz = aioz_train.get_all_data()
        _, pos_test_aioz = aioz_test.get_all_data()
        _, pos_val_aioz = aioz_val.get_all_data()

        # 所有人所有帧trans z的平均（除padding）
        pos_aistpp = torch.cat([torch.tensor(pos_train_aistpp), torch.tensor(pos_test_aistpp)], dim=0)
        pos_aioz = torch.cat([torch.tensor(pos_train_aioz), torch.tensor(pos_test_aioz), torch.tensor(pos_val_aioz)], dim=0)

        non_zero_mask_aistpp = pos_aistpp != 0
        non_zero_mask_aioz = pos_aioz != 0
        
        mean_aistpp = (pos_aistpp * non_zero_mask_aistpp).sum(dim=(0, 1, 2), keepdim=True) / (non_zero_mask_aistpp.sum(dim=(0, 1, 2), keepdim=True) + 1e-8) # 求stat时不考虑padding的0位置
        mean_aioz = (pos_aioz * non_zero_mask_aioz).sum(dim=(0, 1, 2), keepdim=True) / (non_zero_mask_aioz.sum(dim=(0, 1, 2), keepdim=True) + 1e-8)
        
        delta_height = mean_aistpp.squeeze()[2] - mean_aioz.squeeze()[2] # 2.5388

        print(f'delta height: {delta_height}')


    elif stage==2:

        # aamixed
        if dataset_name == 'aamixed':
            aamixed_train = Music2DanceDataset('aamixed', data_split='train', shuffle=False, normalizer=None)
            aamixed_test = Music2DanceDataset('aamixed', data_split='test', shuffle=False, normalizer=None)
            aamixed_val = Music2DanceDataset('aamixed', data_split='val', shuffle=False, normalizer=None)
            
            motion_train, pos_train = aamixed_train.get_all_data()
            motion_train = motion_train['pose']
            motion_test, pos_test = aamixed_test.get_all_data()
            motion_test = motion_test['pose']
            motion_val, pos_val = aamixed_val.get_all_data()
            motion_val = motion_val['pose']
            motion_all = torch.cat([motion_train, motion_test, motion_val], dim=0)
            mean_np, std_np = cal_mean_std(motion_all)

            with open("/home/xingqunqi/AI_dance/AI_dance/checkpoints/aamixed/meta/mean_std.pkl", "wb") as f:
                pickle.dump({"mean": mean_np, "std": std_np}, f)

            print("Mean and std of aamixed saved to mean_std.pkl")

        # aistpp
        elif dataset_name == 'aistpp':
            aistpp_train = Music2DanceDataset('aistpp', data_split='train', shuffle=False, normalizer=None)
            aistpp_test = Music2DanceDataset('aistpp', data_split='test', shuffle=False, normalizer=None)

            motion_train, _ = aistpp_train.get_all_data()
            motion_train = motion_train['pose']
            motion_test, _ = aistpp_test.get_all_data()
            motion_test = motion_test['pose']
            motion_all = torch.cat([motion_train, motion_test], dim=0)
            mean_np, std_np = cal_mean_std(motion_all)

            with open("/home/xingqunqi/AI_dance/AI_dance/checkpoints/aistpp/meta/mean_std.pkl", "wb") as f:
                pickle.dump({"mean": mean_np, "std": std_np}, f)

            print("Mean and std of aistpp saved to mean_std.pkl")


        # aioz TODO(yiwen) stats check, 或者可以用旧的
        elif dataset_name == 'aioz':
            # zhe
            aioz_train = Music2DanceDataset('aioz', data_split='train', shuffle=False, normalizer=None)
            aioz_test = Music2DanceDataset('aioz', data_split='test', shuffle=False, normalizer=None)
            aioz_val = Music2DanceDataset('aioz', data_split='val', shuffle=False, normalizer=None)

            motion_train, _ = aioz_train.get_all_data()
            motion_train = motion_train['pose']
            motion_test, _ = aioz_test.get_all_data()
            motion_test = motion_test['pose']
            motion_val, pos_val = aioz_val.get_all_data()
            motion_val = motion_val['pose']
            motion_all = torch.cat([motion_train, motion_test, motion_val], dim=0)
            mean_np, std_np = cal_mean_std(motion_all)

            with open("/home/xingqunqi/AI_dance/AI_dance/checkpoints/aioz/meta/mean_std.pkl", "wb") as f:
                pickle.dump({"mean": mean_np, "std": std_np}, f)

            print("Mean and std of aioz saved to mean_std.pkl")
            print(f'debug -- mean_np {np.any(np.isnan(mean_np))} std_np {np.any(np.isnan(std_np))}')

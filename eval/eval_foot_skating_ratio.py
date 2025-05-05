from cgi import test
from turtle import left
import numpy as np
import torch
import os, sys
sys.path.append(os.getcwd())
import glob
import os
import pickle

import numpy as np
from tqdm import tqdm
import random

from dataset.vis import SMPLSkeleton
import torch
import argparse

import math


def calc_foot_skating_ratio(dir):
    up_dir = 2 # z-up

    it = glob.glob(os.path.join(dir, "*.pkl"))
    if len(it) > 1000:
        it = random.sample(it, 1000)

    left_ratio_list = []
    right_ratio_list = []
    ground_h = 0 # will not influence the results

    for pkl in tqdm(it):
        info = pickle.load(open(pkl, "rb"))

        if 'pos' in info.keys():
            up_dir = 1  # y is up
            pos_aa = torch.tensor(info["pos"]).unsqueeze(0).to('cuda:0')
            local_aa = torch.tensor(info["q"]).reshape(-1, 24, 3).unsqueeze(0).to('cuda:0')
            
        else:
            pos_aa = torch.tensor(info["smpl_trans"]).unsqueeze(0).to('cuda:0')
            local_aa = torch.tensor(info["smpl_poses"]).reshape(-1, 24, 3).unsqueeze(0).to('cuda:0')

        smpl = SMPLSkeleton(device='cuda:0')

        joint3d = smpl.forward(local_aa, pos_aa).squeeze(0).detach().cpu()

        l_toe_h = joint3d[0, 10, up_dir]
        r_toe_h = joint3d[0, 11, up_dir]
        if abs(l_toe_h - r_toe_h) < 0.02:
            height = (l_toe_h + r_toe_h)/2
        else:
            height = min(l_toe_h, r_toe_h)

        joint3d[:,:,up_dir] = joint3d[:,:,up_dir] - (height -  ground_h)
        
        l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
        relevant_joints = [l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx]

        pred_joint_xyz = joint3d[:, relevant_joints]
        # pred_joint_xyz = model_xp[:, relevant_joints, :] 
        pred_vel = torch.zeros_like(pred_joint_xyz)
        pred_vel[:-1] = (
            pred_joint_xyz[1:, :, :] - pred_joint_xyz[:-1, :, :]
        )  # (S-1, 4, 3)
        
        left_foot_y_ankle = joint3d[:, l_ankle_idx, up_dir]
        right_foot_y_ankle = joint3d[:, r_ankle_idx, up_dir]
        left_foot_y_toe = joint3d[:, l_foot_idx, up_dir]
        right_foot_y_toe = joint3d[:, r_foot_idx, up_dir]

        # print("pred_vel.shape", pred_vel.shape)
        left_fc_mask = (left_foot_y_ankle <= (ground_h +0.08)) & (left_foot_y_toe <= (ground_h +0.05))
        right_fc_mask = (right_foot_y_ankle <= (ground_h +0.08)) & (right_foot_y_toe <= (ground_h +0.05))

        left_pred_vel = torch.cat([pred_vel[:, 0:1, :], pred_vel[:, 2:3, :]], dim=1)
        right_pred_vel = torch.cat([pred_vel[:, 1:2, :], pred_vel[:, 3:4, :]], dim=1)

        left_pred_vel[~left_fc_mask] = 0
        right_pred_vel[~right_fc_mask] = 0
        # print("left_fc_mask", left_fc_mask.shape)
        left_static_num = torch.sum(left_fc_mask)
        # print("left_static_num", left_static_num)
        right_static_num = torch.sum(right_fc_mask)
        # print("right_static_num", right_static_num)

        left_velocity_foot_tangent = torch.cat([left_pred_vel[:, :, 0:1], left_pred_vel[:, :, 2:3] ], dim=2)
        left_velocity_foot_tangent = torch.abs(torch.mean(left_velocity_foot_tangent, dim=-1))        # T, 4
        right_velocity_foot_tangent = torch.cat([right_pred_vel[:, :, 0:1], right_pred_vel[:, :, 2:3] ], dim=2)
        right_velocity_foot_tangent = torch.abs(torch.mean(right_velocity_foot_tangent, dim=-1))        # T, 4
        # print("right_velocity_foot_tangent.shape", right_velocity_foot_tangent.shape)

        left_velocity_foot_tangent = left_velocity_foot_tangent > 0.1 #(0.05 / 30)          # 0.025     # 0.1/30
        right_velocity_foot_tangent = right_velocity_foot_tangent > 0.1 #(0.05 / 30)
        left_slide_frames = torch.any(left_velocity_foot_tangent, dim=-1)
        left_slide_num = torch.sum(left_slide_frames)
        left_static_num = torch.tensor(1) if left_static_num.item() == 0 else left_static_num
        left_ratio = left_slide_num / left_static_num
        left_ratio = left_ratio.item()

        right_slide_frames = torch.any(right_velocity_foot_tangent, dim=-1)
        right_slide_num = torch.sum(right_slide_frames)
        right_static_num = torch.tensor(1) if right_static_num.item() == 0 else right_static_num
        right_ratio = right_slide_num / right_static_num
        right_ratio = right_ratio.item()

        left_ratio_list.append(left_ratio)
        right_ratio_list.append(right_ratio)

    print("left_ratio:", np.mean(left_ratio_list))
    print("right_ratio:", np.mean(right_ratio_list))
    print("ratio:", (np.mean(right_ratio_list) + np.mean(left_ratio_list))/2 )



def parse_eval_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="eval/motions",
        help="Where to load saved motions",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":

    '''python eval/eval_pfc.py --motion_path <your_single_dancer_pkl_folder>
    '''
    
    opt = parse_eval_opt()
    calc_foot_skating_ratio(opt.motion_path)


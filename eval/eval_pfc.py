import argparse
import glob
import os
import pickle

import numpy as np
from tqdm import tqdm
import random

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.vis import SMPLSkeleton
import torch

'''
Cite from EDGE:
    1. On the horizontal (xy) plane, any center of mass
    (COM) acceleration must be due to static contact be-
    tween the feet and the ground. Therefore, either at least
    one foot is stationary on the ground or the COM is not
    accelerating.
    2. On the vertical (z) axis, any positive COM acceleration
    must be due to static foot contact.

NOTE(yiwen) 
    There is no concensus made in dance generation to objectively evaluate natural foot movement.
    
    PFC is a metric designed in previous solo dance generation work EDGE.
    In defination, it claims that dance has unique motion patterns, and sliding itself is actually a valid one.
    which is reasonable.

    But in the official implementation of PFC, the score is calculated by 
        horizontal_speed_leftfeet * horizontal_speed_rigntfeet * positive acceleration (z-up)

    meaning if the dancer is jumping up then at least one of their foot should be static.
    
    However, it has a bias to fix positions (without horizental velocity)
    As long as the dancer stand still, this term will approximate zero.
    While in group dance scenario, the switching of formation is important, which requires large position changes.
'''

def calc_physical_score(dir):
    scores = []
    names = []
    accelerations = []
    up_dir = 2  # z is up
    flat_dirs = [i for i in range(3) if i != up_dir]
    DT = 1 / 30

    it = glob.glob(os.path.join(dir, "*.pkl"))
    if len(it) > 1000:
        it = random.sample(it, 1000)
    for pkl in tqdm(it):
        info = pickle.load(open(pkl, "rb"))

        if 'pos' in info.keys():
            up_dir = 1  # y is up
            flat_dirs = [i for i in range(3) if i != up_dir]
            pos_aa = torch.tensor(info["pos"]).unsqueeze(0).to('cuda:0')
            local_aa = torch.tensor(info["q"]).reshape(-1, 24, 3).unsqueeze(0).to('cuda:0')
            
        else:
            pos_aa = torch.tensor(info["smpl_trans"]).unsqueeze(0).to('cuda:0')
            local_aa = torch.tensor(info["smpl_poses"]).reshape(-1, 24, 3).unsqueeze(0).to('cuda:0')

        smpl = SMPLSkeleton(device='cuda:0')

        joint3d = smpl.forward(local_aa, pos_aa).squeeze(0).detach().cpu()

        # root_v = (pos3d[1:,:] - pos3d[:-1,:]) / DT # root velocity (S-1, 3)
        root_v = (joint3d[1:, 0, :] - joint3d[:-1, 0, :]) / DT 

        root_a = (root_v[1:] - root_v[:-1]) / DT  # (S-2, 3) root accelerations
        # clamp the up-direction of root acceleration
        root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)  # (S-2, 3)
        # l2 norm
        root_a = np.linalg.norm(root_a, axis=-1)  # (S-2,)
        scaling = root_a.max()
        root_a /= scaling

        foot_idx = [7, 10, 8, 11]
        feet = joint3d[:, foot_idx]  # foot positions (S, 4, 3)
        foot_v = np.linalg.norm(
            feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1
        )  # (S-2, 4) horizontal velocity

        foot_mins = np.zeros((len(foot_v), 2))
        foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])
        foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])

        # foot horizontal sliding * up acc, lower better
        foot_loss = (
            foot_mins[:, 0] * foot_mins[:, 1] * root_a
        )  # min leftv * min rightv * root_a (S-2,)
        foot_loss = foot_loss.mean()
        scores.append(foot_loss)
        names.append(pkl)
        accelerations.append(foot_mins[:, 0].mean())

    out = np.mean(scores) * 10000
    print(f"{dir} has a mean PFC of {out}")


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
    calc_physical_score(opt.motion_path)

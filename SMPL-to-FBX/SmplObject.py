import glob
import os
import pickle
from typing import Dict, Tuple

import numpy as np


class SmplObjects(object):
    joints = [
        "m_avg_Pelvis",
        "m_avg_L_Hip",
        "m_avg_R_Hip",
        "m_avg_Spine1",
        "m_avg_L_Knee",
        "m_avg_R_Knee",
        "m_avg_Spine2",
        "m_avg_L_Ankle",
        "m_avg_R_Ankle",
        "m_avg_Spine3",
        "m_avg_L_Foot",
        "m_avg_R_Foot",
        "m_avg_Neck",
        "m_avg_L_Collar",
        "m_avg_R_Collar",
        "m_avg_Head",
        "m_avg_L_Shoulder",
        "m_avg_R_Shoulder",
        "m_avg_L_Elbow",
        "m_avg_R_Elbow",
        "m_avg_L_Wrist",
        "m_avg_R_Wrist",
        "m_avg_L_Hand",
        "m_avg_R_Hand",
    ]

    joints_f = [
        "f_avg_Pelvis",
        "f_avg_L_Hip",
        "f_avg_R_Hip",
        "f_avg_Spine1",
        "f_avg_L_Knee",
        "f_avg_R_Knee",
        "f_avg_Spine2",
        "f_avg_L_Ankle",
        "f_avg_R_Ankle",
        "f_avg_Spine3",
        "f_avg_L_Foot",
        "f_avg_R_Foot",
        "f_avg_Neck",
        "f_avg_L_Collar",
        "f_avg_R_Collar",
        "f_avg_Head",
        "f_avg_L_Shoulder",
        "f_avg_R_Shoulder",
        "f_avg_L_Elbow",
        "f_avg_R_Elbow",
        "f_avg_L_Wrist",
        "f_avg_R_Wrist",
        "f_avg_L_Hand",
        "f_avg_R_Hand",
    ]

    def __init__(self, read_path):
        self.files = {}

        paths = sorted(glob.glob(os.path.join(read_path, "*.pkl")))
        for path in paths:
            filename = path.split("/")[-1]
            with open(path, "rb") as fp:
                data = pickle.load(fp)
            try:
                self.files[filename] = {
                    "smpl_poses": data["smpl_poses"],
                    "smpl_trans": data["smpl_trans"],
                }
            except:
                self.files[filename] = {
                    "smpl_poses": data["smpl_poses"],
                    "smpl_trans": data["root_trans"],
                }
        self.keys = [key for key in self.files.keys()]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[str, Dict]:
        key = self.keys[idx]
        return key, self.files[key]

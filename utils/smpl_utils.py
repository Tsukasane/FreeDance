"""
Copied from EDGE and modified.
"""
import torch
import numpy as np

smpl_joints = [
    "root",  # 0
    "lhip",
    "rhip",
    "belly",  # 1 2 3
    "lknee",
    "rknee",
    "spine",  # 4 5 6
    "lankle",
    "rankle",
    "chest",  # 7 8 9
    "ltoes",
    "rtoes",
    "neck",  # 10 11 12
    "linshoulder",
    "rinshoulder",  # 13 14
    "head",
    "lshoulder",
    "rshoulder",  # 15 16 17
    "lelbow",
    "relbow",  # 18 19
    "lwrist",
    "rwrist",  # 20 21
    "lhand",
    "rhand",  # 22 23
]

NUM_SMPL_JOINTS = len(smpl_joints)
SMPL_UPPER_BODY_JOINTS = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
SMPL_LOWER_BODY_JOINTS = [i for i in range(len(smpl_joints)) if i not in SMPL_UPPER_BODY_JOINTS]

SMPL_LOWER_BODY_JOINTS_BINARY = np.array([i in SMPL_LOWER_BODY_JOINTS for i in range(NUM_SMPL_JOINTS)])
SMPL_UPPER_BODY_JOINTS_BINARY = np.array([i in SMPL_UPPER_BODY_JOINTS for i in range(NUM_SMPL_JOINTS)])

SMPL_LOWER_BODY_MASK = np.concatenate(([True]*(1+2+1), #4
                                     SMPL_LOWER_BODY_JOINTS_BINARY[1:].repeat(3), #21*3
                                     SMPL_LOWER_BODY_JOINTS_BINARY[1:].repeat(6), #21*6
                                     SMPL_LOWER_BODY_JOINTS_BINARY.repeat(3), #22*3
                                     [True]*4)) #4
# 4+63+126+66+4 = 263, same as output_dim
# TODO(yw) modify here after deciding the outputdim
SMPL_UPPER_BODY_MASK = ~SMPL_LOWER_BODY_MASK
SMPL_ROOT_BINARY = np.array([True] + [False] * (NUM_SMPL_JOINTS-1))

ALL_JOINT_FALSE = np.full(*SMPL_ROOT_BINARY.shape, False)
HML_UPPER_BODY_JOINTS_BINARY = np.array([i in SMPL_UPPER_BODY_JOINTS for i in range(NUM_SMPL_JOINTS)])

UPPER_JOINT_Y_TRUE = np.array([ALL_JOINT_FALSE[1:], HML_UPPER_BODY_JOINTS_BINARY[1:], ALL_JOINT_FALSE[1:]])
UPPER_JOINT_Y_TRUE = UPPER_JOINT_Y_TRUE.T
UPPER_JOINT_Y_TRUE = UPPER_JOINT_Y_TRUE.reshape(ALL_JOINT_FALSE[1:].shape[0]*3)

UPPER_JOINT_Y_MASK = np.concatenate(([False]*(1+2+1),
                                UPPER_JOINT_Y_TRUE,
                                ALL_JOINT_FALSE[1:].repeat(6),
                                ALL_JOINT_FALSE.repeat(3),
                                [False] * 4))

def joint_indices_to_channel_indices(indices):
    out = []
    for index in indices:
        out += list(range(3 + 3 * index, 3 + 3 * index + 3))
    return out


def get_first_last_mask(posq_batch, start_width=1, end_width=1):
    # an array in batch x seq_len x channels
    # return a mask that is ones in the first and last row (or first/last WIDTH rows) in the sequence direction
    mask = torch.zeros_like(posq_batch)
    mask[..., :start_width, :] = 1
    mask[..., -end_width:, :] = 1
    return mask


def get_first_mask(posq_batch, start_width=1):
    # an array in batch x seq_len x channels
    # return a mask that is ones in the first and last row (or first/last WIDTH rows) in the sequence direction
    mask = torch.zeros_like(posq_batch)
    mask[..., :start_width, :] = 1
    return mask


def get_middle_mask(posq_batch, start=0, end=-1):
    # an array in batch x seq_len x channels
    # return a mask that is ones in the first and last row (or first/last WIDTH rows) in the sequence direction
    mask = torch.zeros_like(posq_batch)
    mask[..., start:end, :] = 1
    return mask


def lowerbody_mask(posq_batch):
    # an array in batch x seq_len x channels
    # return a mask that is ones in the first and last row (or first/last WIDTH rows) in the sequence direction
    mask = torch.zeros_like(posq_batch)
    lowerbody_indices = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    root_traj_indices = [0, 1, 2]
    lowerbody_indices = (
        joint_indices_to_channel_indices(lowerbody_indices) + root_traj_indices
    )  # plus root traj
    mask[..., :, lowerbody_indices] = 1
    return mask


def upperbody_mask(posq_batch):
    # an array in batch x seq_len x channels
    # return a mask that is ones in the first and last row (or first/last WIDTH rows) in the sequence direction
    mask = torch.zeros_like(posq_batch)
    upperbody_indices = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    root_traj_indices = [0, 1, 2]
    upperbody_indices = (
        joint_indices_to_channel_indices(upperbody_indices) + root_traj_indices
    )  # plus root traj
    mask[..., :, upperbody_indices] = 1
    return mask
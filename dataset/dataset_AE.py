import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import os
from typing import Any

from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)
from dataset.preprocess import Normalizer, vectorize_many
from dataset.quaternion import ax_to_6v
from torch.utils.data._utils.collate import default_collate
import pickle

from .vis import SMPLSkeleton
from pathlib import Path
import glob
# for visualization
import matplotlib.pyplot as plt
import pyrender
import trimesh
os.environ["PYOPENGL_PLATFORM"] = "egl" # headless render mode

"""
For fid encoder training
    - data item: B, H*T, D=3+24*3
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(elev=90, azim=-90) # otherwise it will in lay-down view
    
    # joint: dot
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', s=25)

    # SMPL skeleton: line 
    skeleton = [
        (0, 1), (1, 4), (4, 7), (0, 2), (2, 5), (5, 8), (8, 11), (7, 10), # 腿部
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),      # 躯干
        (12, 13), (13, 16), (12, 14), (14, 17),         # 手臂
        (16, 18), (18, 20), (17, 19), (19, 21), (20, 22), (21, 23)         # 手
    ]
    for joint_start, joint_end in skeleton:
        ax.plot(
            [joints[joint_start, 0], joints[joint_end, 0]],
            [joints[joint_start, 1], joints[joint_end, 1]],
            [joints[joint_start, 2], joints[joint_end, 2]],
            'b-'
        )

    plt.savefig('joint_visualization.png')

def visualize_mesh(vertices, smpl_faces, image_size=(800,800)):
    # --> shift to center --> normalize
    vertices = vertices - vertices.mean(axis=0)  # 将模型居中
    vertices = vertices / np.linalg.norm(vertices, axis=1).max()  
    print(f"Vertices min: {vertices.min(axis=0)}, max: {vertices.max(axis=0)}")
    
    # init mesh
    mesh = trimesh.Trimesh(vertices, smpl_faces, process=False)
    mesh = pyrender.Mesh.from_trimesh(mesh)

    # init scene
    scene = pyrender.Scene()
    scene.add(mesh)

    # add camera according to the model position
    center = vertices.mean(axis=0)
    x_range = vertices[:, 0].ptp()
    y_range = vertices[:, 1].ptp()
    z_range = vertices[:, 2].ptp()

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)  # 调整视野角度
    camera_distance = max(x_range, y_range, z_range)  # 相机距离范围的倍数
    # z+向上，y+向后
    camera_pose = np.array([
        [1.0, 0.0, 0.0, center[0]],  
        [0.0, 1.0, 0.0, center[1] - camera_distance/4],  # 垂直 y 方向居中
        [0.0, 0.0, 1.0, center[2] + camera_distance * 3/2],  # z 方向距离中心一定范围
        [0.0, 0.0, 0.0, 1.0]
    ])
   
    angle = np.radians(45)  # 将角度转换为弧度
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # 绕 X 轴旋转的旋转矩阵
    rotation_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],            # x 不变
        [0.0, cos_angle, -sin_angle, 0.0],  # y 和 z 互相变换
        [0.0, sin_angle, cos_angle, 0.0],   # y 和 z 互相变换
        [0.0, 0.0, 0.0, 1.0]             # 齐次坐标
    ])
    
    camera_pose = np.dot(rotation_matrix, camera_pose)

    scene.add(camera, pose=camera_pose)

    # add light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)
        
    renderer = pyrender.OffscreenRenderer(viewport_width=image_size[0], viewport_height=image_size[1])
    color, _ = renderer.render(scene)
 
    renderer.delete()

    plt.figure(figsize=(8, 8))
    plt.imshow(color)
    plt.axis('off')
    plt.savefig('render_img.png')
    
    
'''For use of training music-2-dance generative model'''
class Music2DanceDataset_AE(data.Dataset):
    def __init__(
        self,
        dataset_name: str, 
        is_test: bool, 
        backup_path: str = "/home/xingqunqi/AI_dance/MMM/dataset/backups/aistpp",
        feature_type: str = "baseline", # music feature type
        normalizer: Any = None, 
        data_len: int = -1, # if set the upper bound of data length
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
            backup_path = Path(backup_path)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            if is_test: 
                pickle.dump(
                    normalizer, open(os.path.join(backup_path, "normalizer.pkl"), "wb")
                )
                
        # load raw data
        if not force_reload and pickle_name in os.listdir(backup_path):
            print("Using cached dataset...")
            with open(os.path.join(backup_path, pickle_name), "rb") as f:
                data = pickle.load(f)
        else:
            print("Loading dataset...")
            data = self.load_aistpp()  # Call this last
            with open(os.path.join(backup_path, pickle_name), "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL) # dump and cache the loaded data
                
        print(
            f"Loaded {self.dataset_name} Dataset With Dimensions: Pos: {data['pos'].shape}, Q: {data['q'].shape}"
        )
            
        # process data, convert to 6dof etc
        # TODO(yw) process_dataset customize for different dataset (expand dim)
        pose_input = self.process_dataset(data["pos"], data["q"])
        self.data = {
            "pose": pose_input, # 17733, 1, 150, 75
            "filenames": data["filenames"],
            "wavs": data["wavs"],
        }
        assert len(pose_input) == len(data["filenames"])
        self.length = len(pose_input) # num of data
        
        
        ######## NOTE(yw) if collect the mean and std
        # radius = 240 * 8 #?
        # dim_pose = 251
        # self.max_motion_length = 196
        # kinematic_chain = paramUtil.kit_kinematic_chain
        # self.meta_dir # to get the mean and std of the dataset
        
        # mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        # std = np.load(pjoin(self.meta_dir, 'std.npy')) 
        ########
        
             

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        feature = torch.from_numpy(np.load(filename_))
        return (self.data["pose"][idx], feature[:148,:], filename_, self.data["wavs"][idx]) #NOTE(yw) also modify audio feature T
        #TODO(yw) audio 的T表示和帧数不同，不对audio截取，用conv/mlp mapping

    def load_aistpp(self):
        # open data path
        split_data_path = os.path.join(
            self.data_root, "test" if self.is_test else "train"
        )

        # Structure:
        # data
        #   |- train
        #   |    |- motion_sliced
        #   |    |- wav_sliced
        #   |    |- baseline_features
        #   |    |- jukebox_features
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
        print(all_pos.shape)
        all_pos = all_pos[:, :: self.data_stride, :] # ratio between the rawdata and newdata
        all_q = all_q[:, :: self.data_stride, :]
        data = {"pos": all_pos, "q": all_q, "filenames": all_names, "wavs": all_wavs} # filenames是.npy, wavs是.wav
        
        return data

    def process_dataset(self, root_pos, local_q): # aistpp custom process
        # FK skeleton
        smpl = SMPLSkeleton()
        # to Tensor
        root_pos = torch.Tensor(root_pos)
        local_q = torch.Tensor(local_q)
        # to ax
        bs, sq, c = local_q.shape # B, T, D
        local_q = local_q.reshape((bs, sq, -1, 3)) # B, T, 24, 3  # 3D representation in axis-angel format
        
        # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
        root_q = local_q[:, :, :1, :]  # sequence x 1 x 3
        root_q_quat = axis_angle_to_quaternion(root_q)
        
        # NOTE(yw) for fid encoder, still need to do the rotation
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

        l = [root_pos, local_q] # TODO(yw) 注意顺序
        global_pose_vec_input = vectorize_many(l).float().detach()

        # 17733, 150, 75 (B, T, D) # TODO(yw) use 150 or 148? before stride2 sample or after?
        assert not torch.isnan(global_pose_vec_input).any()
        data_name = "Test" if self.is_test else "Train"

        # cut the dataset
        if self.data_len > 0:
            global_pose_vec_input = global_pose_vec_input[: self.data_len]
       
        print(f"{data_name} Dataset Motion Features Dim: {global_pose_vec_input.shape}")
        
        return global_pose_vec_input.unsqueeze(1)[:,:,:148,:] # # 17733, 1, 150, 75

    


def DATALoader(dataset_name,
               is_test,
               batch_size,
               num_workers = 8, 
               normalizer = None,
               shuffle=True) : #TODO(yw) add unit_length here
    
    data_loader = torch.utils.data.DataLoader(Music2DanceDataset(dataset_name, is_test=is_test, shuffle=shuffle, normalizer=normalizer),
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


if __name__=='__main__':
    aistpp_ae = Music2DanceDataset_AE('aistpp', 
                                is_test=False, 
                                shuffle=False, 
                                normalizer=None)


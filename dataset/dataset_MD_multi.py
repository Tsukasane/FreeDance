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
from eval.features.kinetic import extract_kinetic_features
from eval.features.manual import extract_manual_features
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
        person_joints = joints[h]  # (T, 24, 3)

        ax.scatter(
            person_joints[:, 0], 
            person_joints[:, 1],  
            person_joints[:, 2],  
            label=f'Person {h + 1}', 
            s=25  
        )
        
        for joint_start, joint_end in skeleton:
            ax.plot(
                [person_joints[joint_start, 0], person_joints[joint_end, 0]],
                [person_joints[joint_start, 1], person_joints[joint_end, 1]],
                [person_joints[joint_start, 2], person_joints[joint_end, 2]],
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
        unit_length: int = 4,
        stats_path_aistpp: str = "/home/xingqunqi/AI_dance/AI_dance/checkpoints/aistpp/meta/mean_std.pkl",
        stats_path_aioz: str = "/home/xingqunqi/AI_dance/AI_dance/checkpoints/aioz/meta/mean_std.pkl",
        stats_path_aamixed: str = "/home/xingqunqi/AI_dance/AI_dance/checkpoints/aamixed/meta/mean_std.pkl",
        tokenizer_name: str = "codebook_dir",
        load_motion_code: bool = False,
        codebook_size: int = 1024,
        align_dataset_stage1: bool = False,
        collect_stats_stage2: bool = False,
        max_person_num: int = 3,
        ): 
        
        # data preprocess has already sliced the audio and motion to fixed length
        self.motion_length = 150
        self.dataset_name = dataset_name
        self.data_split = data_split
        self.normalizer = normalizer
        self.data_len = data_len
        self.shuffle = shuffle
        self.include_contacts = include_contacts
        self.unit_length = unit_length
        self.load_motion_code = load_motion_code
        self.max_person_num = max_person_num

        # for data alignment and stat collection
        self.pos = None
        self.align_dataset_stage1 = align_dataset_stage1
        self.collect_stats_stage2 = collect_stats_stage2

        # TODO(yiwen) incorporate aioz preprocess code to repo
        if dataset_name == 'aamixed':        
            self.data_root = './dataset/aamixed_dataset' # NOTE(yiwen) please use absolute path here, since this will be called by other scripts
            self.mean, self.std = self.get_stats(stats_path_aamixed) 

        if dataset_name == 'aioz':
            self.data_root = './dataset/AIOZ_Gdance_dataset'
            self.mean, self.std = self.get_stats(stats_path_aioz)

        if dataset_name == 'aistpp':
            self.data_root = './dataset/AIST++_dataset'
            self.mean, self.std = self.get_stats(stats_path_aistpp)

        self.audio_dir = pjoin(self.data_root, f'{feature_type}_feats')
        self.joints_num = 24 #SMPL 24 joints
        self.raw_fps_aistpp = 60
        self.data_fps = 30
        self.max_motion_length = 50 #length of code in one seq
        
        assert self.data_fps <= self.raw_fps_aistpp
        self.data_stride = self.raw_fps_aistpp // self.data_fps
        self.feature_type = feature_type

        # for motion code
        self.motion_code_path = tokenizer_name # the codebook dir
        self.mot_end_idx = codebook_size # [NEW] end token
        self.mot_pad_idx = codebook_size + 1 # [NEW] pad token
        

        print("Loading dataset...") # load raw data 
        data = self.load_data(align_dataset_stage1, self.max_person_num)  

        print(
            f"Loaded {self.dataset_name} Dataset With Dimensions: Pos: {data['pos'].shape}, Q: {data['q'].shape}"
        )
            
        # process data, convert to 6dof etc
        pose_input = self.process_dataset(data["pos"], data["q"], data["filenames"], data["num_person"])
        
        if not self.collect_stats_stage2: # else return the unnormalized data
            # normalize the 6d data
            pose_input = (pose_input - self.mean) / self.std # std has already added 1e-10 in preprocessing
        
        if self.load_motion_code:
            self.data = {
                "pose": pose_input, # N, H, 148, 151 
                "filenames": data["filenames"],
                "wavs": data["wavs"],
                "num_person": data["num_person"],
                "motion_code_paths": data["motion_codes"]
            }
        else:
            self.data = {
                "pose": pose_input, # N, H, 148, 151 
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

    def __len__(self):
        return self.length

    def get_all_data(self):
        return self.data, self.pos

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        feature = torch.from_numpy(np.load(filename_))
        
        if self.load_motion_code:
            motion_code_name_ = self.data["motion_code_paths"][idx]
            motion_token = torch.from_numpy(np.load(motion_code_name_))
            motion_token_len = motion_token.shape[1]

            end_expand = np.ones((1), dtype=int) * self.mot_end_idx
            end_expand = end_expand[np.newaxis, :, np.newaxis]
            pad_expand = np.ones((self.max_motion_length-1-motion_token_len), dtype=int) * self.mot_pad_idx
            pad_expand = pad_expand[np.newaxis, :, np.newaxis]

            if motion_token_len+1 < self.max_motion_length: # do padding
                # pad with 1s
                motion_token = np.concatenate([motion_token, end_expand, pad_expand], axis=1)
            else:
                motion_token = np.concatenate([motion_token, end_expand], axis=0)

            return (self.data["pose"][idx], feature, filename_, self.data["wavs"][idx], self.data["num_person"][idx], motion_token, motion_token_len) 
    
        else:
            return (self.data["pose"][idx], feature, filename_, self.data["wavs"][idx], self.data["num_person"][idx]) 
            
        # do not slice T in audio

    def load_data(self, align_dataset_stage1=False, max_person_num=3):
        # max_person_num = 5 # TODO(yiwen) make this arg
        delta_height = 2.5388 # NOTE(yiwen) from stats_collect

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
        all_pos_no_pad = []
        all_q = []
        all_names = []
        all_wavs = []
        all_h = []
        all_mc = [] # motion code
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
                
                if not align_dataset_stage1: # when align, do not modify the height
                    pos[:,:,1:2] = pos[:,:,1:2] - delta_height # aistpp align aioz
    
            H = pos.shape[0]
            
            if H > max_person_num: # main experiment uses H<=3
                continue
            elif H < max_person_num:
                if align_dataset_stage1: # when align, recording no pad
                    for h in range(H):
                        all_pos_no_pad.append(pos[h]) # T, 3
                pad_shape_pos = (max_person_num - H, pos.shape[1], pos.shape[2])
                pad_shape_q = (max_person_num - H, q.shape[1], q.shape[2])
                pos = np.vstack([pos, np.zeros(pad_shape_pos, dtype=pos.dtype)])
                q = np.vstack([q, np.zeros(pad_shape_q, dtype=q.dtype)])
            else:
                if align_dataset_stage1: # when align, recording no pad
                    for h in range(H):
                        all_pos_no_pad.append(pos[h]) # T, 3
            all_pos.append(pos)
            all_q.append(q)
            all_names.append(feature) # music token path
            all_wavs.append(wav)
            all_h.append(H)

            if self.load_motion_code:
                file_name = feature.split('/')[-1] 
                motion_code_path = os.path.join(self.motion_code_path, file_name) # motion token path
                all_mc.append(motion_code_path)

        all_pos = np.array(all_pos)  # N x H x T x 3
        all_q = np.array(all_q)  # N x H x T x (joint * 3)

        if align_dataset_stage1:
            all_pos_no_pad = np.array(all_pos_no_pad) # T, 3
            print(f'Single person motion sequence num: {len(all_pos_no_pad)}, without height modification!')
            self.pos = all_pos_no_pad
        else:
            self.pos = all_pos
        
        if self.load_motion_code:
            data = {"pos": all_pos, "q": all_q, "filenames": all_names, "wavs": all_wavs, "num_person": all_h, "motion_codes": all_mc} 
        else:
            data = {"pos": all_pos, "q": all_q, "filenames": all_names, "wavs": all_wavs, "num_person": all_h} 
        
        return data

    def process_dataset(self, root_pos, local_q, filenames, num_person): 
        # FK skeleton
        smpl = SMPLSkeleton()
        # to Tensor
        root_pos = torch.Tensor(root_pos)  # N, H, 150, 3
        local_q = torch.Tensor(local_q)   # N, H, 150, 72
        # to ax
        bs, h, sq, c = local_q.shape     
        local_q = local_q.reshape((bs, h, sq, -1, 3))  # N, H, T, 24, 3 
        
        # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
        root_q = local_q[:, :, :, :1, :]  # N, H, T, 1, 3
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
        root_pos = root_pos.view(bs, h, sq, 3)  # N, H, T, 3
        
        # do FK
        positions = smpl.forward(local_q.view(bs * h, sq, -1, 3), root_pos.view(bs * h, sq, 3))
        positions = positions.view(bs, h, sq, -1, 3)  # -------- (B, H, T, 24, 3)
        
        ## extract statistic features for eval metric
        keypoints3d_all = positions.detach().cpu().numpy() # positions.view(bs*h, sq, 24, 3)
        
        # NOTE(yiwen) manually defined kinetics and geometry feature extraction in the first pass, time consuming!
        new_data_path = '/data/xingqunqi/AI_dance/Group_Dance_output'
        feature_save_dir = os.path.join(new_data_path, self.dataset_name, self.data_split, 'motion_feats')
        os.makedirs(feature_save_dir, exist_ok=True)
        # if len(os.listdir(feature_save_dir)) == 0 and not self.align_dataset_stage1 and not self.collect_stats_stage2:
        #     cnt = 0
        #     for n_id in range(bs): # each element
        #         for h_id in range(num_person[n_id]): # each person, excluding padding
        #             keypoints3d = keypoints3d_all[n_id][h_id] # should be seq, 24, 3
        #             features_manual = extract_manual_features(keypoints3d) # (32,)
        #             features_kinetic = extract_kinetic_features(keypoints3d) # (72,)
        #             cnt+=1
        #             if cnt%100==0:
        #                 print(f'processing data idx {cnt}')
        #             manual_feature_filename = os.path.splitext(filenames[n_id])[0].split('/')[-1] + f'_ps{h_id+1}' + "_manual.npy"
        #             kinetic_feature_filename = os.path.splitext(filenames[n_id])[0].split('/')[-1]+ f'_ps{h_id+1}' + "_kinetic.npy"             
        #             np.save(os.path.join(feature_save_dir, manual_feature_filename), features_manual)
        #             np.save(os.path.join(feature_save_dir, kinetic_feature_filename), features_kinetic)
       
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
               codebook_size = 1024, 
               tokenizer_name = 'codebook_dir', 
               load_motion_code = False,
               align_dataset_stage1 = False,
               collect_stats_stage2 = False,
               unit_length=4,
               num_workers = 8, 
               normalizer = None,
               shuffle=True,
               max_person_num=3) : #TODO(yiwen) add unit_length here
    
    data_loader = torch.utils.data.DataLoader(Music2DanceDataset(dataset_name, 
                                                                 data_split=data_split,
                                                                 codebook_size=codebook_size, 
                                                                 tokenizer_name=tokenizer_name, 
                                                                 unit_length=unit_length,
                                                                 shuffle=shuffle, 
                                                                 normalizer=normalizer,
                                                                 load_motion_code=load_motion_code,
                                                                 align_dataset_stage1=align_dataset_stage1,
                                                                 collect_stats_stage2=collect_stats_stage2,
                                                                 max_person_num=max_person_num),
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

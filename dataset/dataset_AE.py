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
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate
import pickle
from .vis import SMPLSkeleton
from pathlib import Path
import glob
# for visualization
from smplx import SMPL
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
        normalizer: Any = None, 
        data_len: int = -1, # cut data if originally not in the same length
        shuffle=True,
        include_contacts: bool = True, # heel and toe of each foot, dim+=4
        unit_length: int = 4,
        stats_path_aistpp: str = "/home/xingqunqi/AI_dance/AI_dance/checkpoints/aistpp/meta/mean_std.pkl",
        stats_path_aioz: str = "/home/xingqunqi/AI_dance/litingw/Group-Dance/checkpoints/aistpp/meta/mean_std_multi.pkl",
        stats_path_aamixed: str = "/home/xingqunqi/AI_dance/AI_dance/checkpoints/aamixed/meta/mean_std.pkl",
        tokenizer_name: str = "codebook_dir",
        load_motion_code: bool = False,
        codebook_size: int = 1024,
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

        # for data alignment and stat collection
        self.pos = None

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
        data = self.load_data()  

        print(
            f"Loaded {self.dataset_name} Dataset With Dimensions: Pos: {data['pos'].shape}, Q: {data['q'].shape}"
        )
            
        # process data, 3d
        pose_input = self.process_dataset(data["pos"], data["q"])
        
        # # normalize the 6d data
        # pose_input = (pose_input - self.mean) / self.std # std has already added 1e-10 in preprocessing
        
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
                "pose": pose_input, # N, 148, 75
            }
        self.length = len(pose_input) # num of data
        
             
    def get_stats(self, stats_path):
        with open(stats_path, "rb") as f:
            data = pickle.load(f)
        mean_loaded = data["mean"]
        std_loaded = data["std"]
        mean_tensor = torch.tensor(mean_loaded).view(1, 1, -1)
        std_tensor = torch.tensor(std_loaded).view(1, 1, -1)
        return mean_tensor, std_tensor # 1, 1, 1, 75

    def __len__(self):
        return self.length

    def get_all_data(self):
        return self.data, self.pos

    def __getitem__(self, idx):
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
            return (self.data["pose"][idx]) 
            
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

            if self.load_motion_code:
                file_name = feature.split('/')[-1] 
                motion_code_path = os.path.join(self.motion_code_path, file_name) # motion token path
                all_mc.append(motion_code_path)

        all_pos = np.array(all_pos)  # N x H x T x 3
        all_q = np.array(all_q)  # N x H x T x (joint * 3)
        self.pos = all_pos
        
        if self.load_motion_code:
            data = {"pos": all_pos, "q": all_q, "filenames": all_names, "wavs": all_wavs, "num_person": all_h, "motion_codes": all_mc} 

        else:
            data = {"pos": all_pos, "q": all_q} 
        
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
               codebook_size = 1024, 
               tokenizer_name = 'codebook_dir', 
               load_motion_code = False,
               unit_length=4,
               num_workers = 8, 
               normalizer = None,
               shuffle=True):
    
    data_loader = torch.utils.data.DataLoader(Music2DanceDataset_AE(dataset_name, 
                                                                 data_split=data_split,
                                                                 codebook_size=codebook_size, 
                                                                 tokenizer_name=tokenizer_name, 
                                                                 unit_length=unit_length,
                                                                 shuffle=shuffle, 
                                                                 normalizer=normalizer,
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



# import torch
# from torch.utils import data
# import numpy as np
# from os.path import join as pjoin
# import os
# from typing import Any

# from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
#                                   quaternion_multiply,
#                                   quaternion_to_axis_angle)
# from dataset.preprocess import Normalizer, vectorize_many
# from dataset.quaternion import ax_to_6v
# from torch.utils.data._utils.collate import default_collate
# import pickle

# from .vis import SMPLSkeleton
# from pathlib import Path
# import glob
# # for visualization
# import matplotlib.pyplot as plt
# import pyrender
# import trimesh
# os.environ["PYOPENGL_PLATFORM"] = "egl" # headless render mode

# """
# For fid encoder training
#     - data item: B, H*T, D=3+24*3
# """


#         self.normalizer = normalizer
#         self.data_len = data_len
#         self.shuffle = shuffle
#         self.include_contacts = include_contacts
#         self.unit_length = unit_length
#         self.mean = None
#         self.std = None
            
#         if dataset_name == 'aistpp':
#             self.data_root = '/home/xingqunqi/AI_dance/MMM/dataset/AIST++_dataset' # NOTE(yw) 用绝对路径，fid extractor在导入时相对路径不同
#             self.motion_dir = pjoin(self.data_root, 'annotations/motions') # using smpl 72 dim pose representation
#             self.audio_dir = pjoin(self.data_root, 'extracted_audios')
            
#             self.joints_num = 24 #SMPL 24 joints
#             self.raw_fps = 60
#             self.data_fps = 30
#             assert self.data_fps <= self.raw_fps
#             self.data_stride = self.raw_fps // self.data_fps
            
#             self.feature_type = feature_type # music feature type

#             pickle_name = "processed_train_data.pkl" if not is_test else "processed_test_data.pkl"
#             backup_path = Path(backup_path)
#             backup_path.mkdir(parents=True, exist_ok=True)
            
#             if is_test: 
#                 pickle.dump(
#                     normalizer, open(os.path.join(backup_path, "normalizer.pkl"), "wb")
#                 )
                
#         # load raw data
#         if not force_reload and pickle_name in os.listdir(backup_path):
#             print("Using cached dataset...")
#             with open(os.path.join(backup_path, pickle_name), "rb") as f:
#                 data = pickle.load(f)
#         else:
#             print("Loading dataset...")
#             data = self.load_aistpp()  # load each modality from pkl
#             with open(os.path.join(backup_path, pickle_name), "wb") as f:
#                 pickle.dump(data, f, pickle.HIGHEST_PROTOCOL) # dump and cache the loaded data
                
#         print(
#             f"Loaded {self.dataset_name} Dataset With Dimensions: Pos: {data['pos'].shape}, Q: {data['q'].shape}"
#         )
            
#         # process data, convert to 6dof etc
#         # TODO(yw) process_dataset customize for different dataset (expand dim)
#         pose_input = self.process_dataset(data["pos"], data["q"])
#         self.data = {
#             "pose": pose_input, # 17733, 1, 150, 75
#             "filenames": data["filenames"],
#             "wavs": data["wavs"],
#         }
#         assert len(pose_input) == len(data["filenames"])
#         self.length = len(pose_input) # num of data
        
        
#         ######## NOTE(yw) if collect the mean and std
#         # radius = 240 * 8 #?
#         # dim_pose = 251
#         # self.max_motion_length = 196
#         # kinematic_chain = paramUtil.kit_kinematic_chain
#         # self.meta_dir # to get the mean and std of the dataset
        
#         # mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
#         # std = np.load(pjoin(self.meta_dir, 'std.npy')) 
#         ########
        
             

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         filename_ = self.data["filenames"][idx]
#         feature = torch.from_numpy(np.load(filename_))
#         return (self.data["pose"][idx], feature[:148,:], filename_, self.data["wavs"][idx]) #NOTE(yw) also modify audio feature T
#         #TODO(yw) audio 的T表示和帧数不同，不对audio截取，用conv/mlp mapping

#     def load_aistpp(self):
#         # open data path
#         split_data_path = os.path.join(
#             self.data_root, "test" if self.is_test else "train"
#         )

#         # Structure:
#         # data
#         #   |- train
#         #   |    |- motion_sliced
#         #   |    |- wav_sliced
#         #   |    |- baseline_features
#         #   |    |- jukebox_features
#         #   |    |- motions
#         #   |    |- wavs

#         motion_path = os.path.join(split_data_path, "motions_sliced")
#         sound_path = os.path.join(split_data_path, f"{self.feature_type}_feats")
#         wav_path = os.path.join(split_data_path, f"wavs_sliced")
#         # sort motions and sounds
#         motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))
#         features = sorted(glob.glob(os.path.join(sound_path, "*.npy")))
#         wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))

#         # stack the motions and features together
#         all_pos = []
#         all_q = []
#         all_names = []
#         all_wavs = []
#         assert len(motions) == len(features)
#         for motion, feature, wav in zip(motions, features, wavs):
#             # make sure name is matching
#             m_name = os.path.splitext(os.path.basename(motion))[0]
#             f_name = os.path.splitext(os.path.basename(feature))[0]
#             w_name = os.path.splitext(os.path.basename(wav))[0]
#             assert m_name == f_name == w_name, str((motion, feature, wav))
#             # load motion
#             data = pickle.load(open(motion, "rb"))
#             pos = data["pos"]
#             q = data["q"]
#             all_pos.append(pos)
#             all_q.append(q)
#             all_names.append(feature)
#             all_wavs.append(wav)

#         all_pos = np.array(all_pos)  # N x seq x 3
#         all_q = np.array(all_q)  # N x seq x (joint * 3)
#         # downsample the motions to the data fps
#         print(all_pos.shape)
#         all_pos = all_pos[:, :: self.data_stride, :] # ratio between the rawdata and newdata
#         all_q = all_q[:, :: self.data_stride, :]
#         data = {"pos": all_pos, "q": all_q, "filenames": all_names, "wavs": all_wavs} # filenames是.npy, wavs是.wav
        
#         return data

#     def process_dataset(self, root_pos, local_q): # aistpp custom process
#         # FK skeleton
#         smpl = SMPLSkeleton()
#         # to Tensor
#         root_pos = torch.Tensor(root_pos)
#         local_q = torch.Tensor(local_q)
#         # to ax
#         bs, sq, c = local_q.shape # B, T, D
#         local_q = local_q.reshape((bs, sq, -1, 3)) # B, T, 24, 3  # 3D representation in axis-angel format
        
#         # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
#         root_q = local_q[:, :, :1, :]  # sequence x 1 x 3
#         root_q_quat = axis_angle_to_quaternion(root_q)
        
#         # NOTE(yw) for fid encoder, still need to do the rotation
#         rotation = torch.Tensor(
#             [0.7071068, 0.7071068, 0, 0]
#         )  # 90 degrees about the x axis
#         root_q_quat = quaternion_multiply(rotation, root_q_quat) # 旋转quaternion表示的root坐标
#         root_q = quaternion_to_axis_angle(root_q_quat)
#         local_q[:, :, :1, :] = root_q 

#         # don't forget to rotate the root position too 😩
#         pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
#         root_pos = pos_rotation.transform_points(
#             root_pos
#         )  # basically (y, z) -> (-z, y), expressed as a rotation for readability

#         l = [root_pos, local_q] # TODO(yw) 注意顺序
#         global_pose_vec_input = vectorize_many(l).float().detach()

#         # 17733, 150, 75 (B, T, D) # TODO(yw) use 150 or 148? before stride2 sample or after?
#         assert not torch.isnan(global_pose_vec_input).any()
#         data_name = "Test" if self.is_test else "Train"

#         # cut the dataset
#         if self.data_len > 0:
#             global_pose_vec_input = global_pose_vec_input[: self.data_len]
       
#         print(f"{data_name} Dataset Motion Features Dim: {global_pose_vec_input.shape}")
        
#         return global_pose_vec_input.unsqueeze(1)[:,:,:148,:] # 17733, 1, 150, 75

    


# def DATALoader(dataset_name,
#                is_test,
#                batch_size,
#                num_workers = 8, 
#                normalizer = None,
#                shuffle=True) : #TODO(yw) add unit_length here
    
#     data_loader = torch.utils.data.DataLoader(Music2DanceDataset(dataset_name, is_test=is_test, shuffle=shuffle, normalizer=normalizer),
#                                               batch_size,
#                                               shuffle = shuffle,
#                                               num_workers=num_workers,
#                                               pin_memory=True,
#                                               drop_last = True)
    
#     return data_loader


# def cycle(iterable):
#     while True:
#         for x in iterable:
#             yield x


# if __name__=='__main__':
#     aistpp_ae = Music2DanceDataset_AE('aistpp', 
#                                 is_test=False, 
#                                 shuffle=False, 
#                                 normalizer=None)


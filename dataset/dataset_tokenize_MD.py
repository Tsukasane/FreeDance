import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm


import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os
from typing import Any

from torch.utils.data._utils.collate import default_collate
import pickle

import glob
# for visualization
from smplx import SMPL
import matplotlib.pyplot as plt
os.environ["PYOPENGL_PLATFORM"] = "egl" # headless render mode


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def fileToList(f):
    out = open(f, "r").readlines()
    out = [x.strip() for x in out]
    out = [x for x in out if len(x)]
    return out

    
'''For use of training music-2-dance generative model'''
class MDTokenDataset(data.Dataset):
    def __init__(
        self,
        dataset_name: str, 
        is_test: bool = False,
        codebook_size: int = 1024,
        tokenizer_name: str = 'codebook_dir',
        feature_type: str = "baseline", # music feature type
        shuffle=True, # NOTE(yw) shuffle, no matter train or test. Because some samples are from the same original sequence.
        unit_length: int = 8):
        
        # data preprocess 做的slice就已经把audio和motion变成了定长，motion150帧
        self.motion_length = 150
        self.dataset_name = dataset_name
        
        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1
        
        self.tokenizer_name = tokenizer_name
        self.shuffle = shuffle
        self.unit_length = unit_length
        self.mean = None
        self.std = None
        
        if dataset_name =='aistpp':
            min_motion_len = 40
            
            self.data_root = './dataset/AIST++_dataset/test' if is_test else './dataset/AIST++_dataset/train'
            self.audio_dir = pjoin(self.data_root, f'{feature_type}_feats')
            
            self.joints_num = 24 #SMPL 24 joints
            self.raw_fps = 60
            self.data_fps = 30
            assert self.data_fps <= self.raw_fps
            self.data_stride = self.raw_fps // self.data_fps
            self.max_motion_length = 196
            
            self.feature_type = feature_type # music feature type

        ## load motion data from codebook_dir
        self.id_list = os.listdir(tokenizer_name)
        self.id_list.sort()
        new_name_list = []
        data_dict = {}
        for file_name in self.id_list:
            try:
                motion_token = np.load(pjoin(tokenizer_name, file_name))
                music_feats = np.load(pjoin(self.audio_dir, file_name))
                data_dict[file_name] = {'motion_token': motion_token, 
                                        'music_feats':music_feats}
            except:
                pass
            
        self.data_dict = data_dict
                   

    def inv_transform(self, data):
        if self.std==None:
            return data
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item): 
        data = self.data_dict[self.id_list[item]] # check 'item' to be filename
        motion_token, music_feats = data['motion_token'], data['music_feats']
        motion_token_len = motion_token.shape[0]
        # NOTE(yw) also need raw waveform?

        ## NOTE(yw) no padding because the slice motions are in same length
        # if motion_token_len+1 < self.max_motion_length: # do padding
        #     # pad with 1s
        #     # TODO (yw) check dimension
        #     motion_token = np.concatenate([motion_token, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-motion_token_len), dtype=int) * self.mot_pad_idx], axis=0)
        # else:
        #     motion_token = np.concatenate([motion_token, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)
        return (music_feats, motion_token, motion_token_len) # music feats + motion token + motion token length
    

def DATALoader(dataset_name,
               is_test=False,
               batch_size = 1,
               codebook_size = 1024, 
               tokenizer_name = 'codebook_dir', 
               unit_length=4,
                num_workers = 8) : 
    
    train_loader = torch.utils.data.DataLoader(MDTokenDataset(dataset_name, 
                                                              is_test=is_test, 
                                                              codebook_size=codebook_size, 
                                                              tokenizer_name=tokenizer_name, 
                                                              unit_length=unit_length),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              drop_last = True)
    
    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

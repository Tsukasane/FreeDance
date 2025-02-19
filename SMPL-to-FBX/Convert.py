"""
   Copyright (C) 2017 Autodesk, Inc.
   All rights reserved.

   Use of this software is subject to the terms of the Autodesk license agreement
   provided at the time of installation or download, or which otherwise accompanies
   this software in either electronic or hard copy form.
 
"""

import argparse
import os
import sys
sys.path
sys.path.append('.')

from tqdm import tqdm
from FbxReadWriter import FbxReadWrite
from SmplObject import SmplObjects
import pickle

'''
Convert multi-person pkl to single-person subfiles before using this script. 
    <pkl_name>.pkl --> <psID_pkl_name>.pkl
Render the fbx files in blender together afterwise.
'''

def getArg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="SMPL-to-FBX/motions")
    parser.add_argument(
        "--fbx_source_path",
        type=str,
        default="SMPL-to-FBX/characters/SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx" # ["SMPL-to-FBX/characters/ybot.fbx","SMPL-to-FBX/characters/SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx", "SMPL-to-FBX/characters/SMPL_f_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx"]
    )
    parser.add_argument("--output_dir", type=str, default="SMPL-to-FBX/fbx_out")

    return parser.parse_args()

def load_data(datapath):
    with open(datapath, "rb") as f:
        data = pickle.load(f)
    return data

def separate_multi(raw_data_dir):
    raw_multi_pkls = os.listdir(raw_data_dir)
    for pkl_path in raw_multi_pkls:
        raw_multi_pkl = load_data(os.path.join(raw_data_dir, pkl_path))
        # NOTE(yiwen) hard code
        H = 3
        T = 148
        B = 1 # if the pkl sample is generated using generate.py
        smpl_poses=raw_multi_pkl["smpl_poses"]
        smpl_trans=raw_multi_pkl["smpl_trans"]
        
        smpl_poses = smpl_poses.reshape(B, H, T, -1)[0] # first seq in the batch
        smpl_trans = smpl_trans.reshape(B, H, T, 3)[0]
        os.makedirs(f'./temp_vis_split/', exist_ok=True)
        for h in range(H):
            single_pose = smpl_poses[h] # (T, 72)
            single_trans = smpl_trans[h] # (T, 3)

            pickle.dump(
                    {
                        "smpl_poses": single_pose,
                        "smpl_trans": single_trans,
                    },
                    open(f'./temp_vis_split/ps{h+1}.pkl', "wb"),
                ) 

if __name__ == "__main__":
    args = getArg()
    input_dir = args.input_dir # NOTE(yiwen) one pkl in input_dir each time
    fbx_source_path = args.fbx_source_path
    output_dir = args.output_dir

    save_dir = './temp_vis_split/'
    separate_multi(input_dir)

    smplObjects = SmplObjects(save_dir)
    for pkl_name, smpl_params in tqdm(smplObjects):
        try:
            fbxReadWrite = FbxReadWrite(fbx_source_path)
            fbxReadWrite.addAnimation(pkl_name, smpl_params)
            fbxReadWrite.writeFbx(output_dir, pkl_name)
        except Exception as e:
            fbxReadWrite.destroy()
            print("An error was thrown in the FBX conversion process")
            raise e
        finally:
            fbxReadWrite.destroy()
    # convert everything in output folder from ascii to binary
    # this line can be commented out if not directly importing to Blender
    out = os.system(f"wine SMPL-to-FBX/FbxFormatConverter.exe -c {output_dir} -binary")
    

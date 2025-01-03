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


if __name__ == "__main__":
    args = getArg()
    input_dir = args.input_dir
    fbx_source_path = args.fbx_source_path
    output_dir = args.output_dir

    smplObjects = SmplObjects(input_dir)
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
    

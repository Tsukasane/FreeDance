import glob
import os
import pickle
import shutil
from pathlib import Path
import numpy as np

def fileToList(f):
    out = open(f, "r").readlines()
    out = [x.strip() for x in out]
    out = [x for x in out if len(x)]
    return out

train_list = set(fileToList("AIOZ_Gdance_dataset/train_split_sequence_names.txt"))   # ----------【litingw：whole数据集】
test_list = set(fileToList("AIOZ_Gdance_dataset/test_split_sequence_names.txt"))
val_list = set(fileToList("AIOZ_Gdance_dataset/val_split_sequence_names.txt"))

# train_list = set(fileToList("AIOZ_TrialONLY/train_split_sequence_names.txt"))    # ----------【litingw：Trial数据集】
# test_list = set(fileToList("AIOZ_TrialONLY/test_split_sequence_names.txt"))
# val_list = set(fileToList("AIOZ_TrialONLY/val_split_sequence_names.txt"))


def split_data(dataset_path):
    # train - test split
    for split_list, split_name in zip([train_list, val_list, test_list], ["train", "val", "test"]):
        Path(f"AIOZ_Gdance_dataset/{split_name}/motions").mkdir(parents=True, exist_ok=True)
        Path(f"AIOZ_Gdance_dataset/{split_name}/wavs").mkdir(parents=True, exist_ok=True)
        for sequence in split_list:
            # if sequence in filter_list:                          # ----------【litingw：没有ignore.txt，故不进行filter_list】
            #     continue
            motion = f"{dataset_path}/motions/{sequence}.pkl"
            wav = f"{dataset_path}/wavs/{sequence}.wav"
            assert os.path.isfile(motion)
            assert os.path.isfile(wav)
            motion_data = pickle.load(open(motion, "rb")) 
            trans = motion_data["root_trans"]   # [H, T, Trans==3]
            pose = motion_data["smpl_poses"]   # [H, T, Poses==72]
            
            #scale = motion_data["smpl_scaling"]                 # ----------【litingw：没有smpl_scaling，目前设置为1】
            scale = np.array([1])

            out_data = {"pos": trans, "q": pose, "scale": scale}
            pickle.dump(out_data, open(f"AIOZ_Gdance_dataset/{split_name}/motions/{sequence}.pkl", "wb"))
            shutil.copyfile(wav, f"AIOZ_Gdance_dataset/{split_name}/wavs/{sequence}.wav")
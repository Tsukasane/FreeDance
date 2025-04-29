import torch
import numpy as np
import pickle
from .dataset_MD_multi import Music2DanceDataset
import argparse
import os



def get_args_parser():
    parser = argparse.ArgumentParser(description='Options for statistic collection.',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--stage', type=int, default=2, help='stage 1 for dataset alignment, stage 2 for statistic collection.')
    parser.add_argument('--dataset_name', type=str, default='aamixed', help='specifiy the dataset in stage 2')
    
    return parser


def cut_padding(padded_pose, num_person):
    '''
        padded_pose: N, H, T, D
        num_person: list of len(N)
    '''
    motion_all = []
    for n in range(padded_pose.shape[0]):
        for h_id in range(num_person[n]): # each person, excluding padding
            keypoints3d = padded_pose[n][h_id] # T, 151
            motion_all.append(keypoints3d)
    print(f'Single person motion without padding num: {len(motion_all)}')
    motion_all_new = torch.cat(motion_all, dim=0)

    return motion_all_new # nT, 151


if __name__=='__main__':

    parser = get_args_parser()
    args = parser.parse_args()
    stage = args.stage
    dataset_name = args.dataset_name
    EPSILON = 1e-10

    if stage==1: # collect average height of all joints in aistpp and aioz respectively, then align the heights
        aistpp_train = Music2DanceDataset('aistpp', data_split='train', shuffle=False, align_dataset_stage1=True)
        aistpp_test = Music2DanceDataset('aistpp', data_split='test', shuffle=False, align_dataset_stage1=True)

        aioz_train = Music2DanceDataset('aioz', data_split='train', shuffle=False, align_dataset_stage1=True)
        aioz_test = Music2DanceDataset('aioz', data_split='test', shuffle=False, align_dataset_stage1=True)
        aioz_val = Music2DanceDataset('aioz', data_split='val', shuffle=False, align_dataset_stage1=True)

        # NOTE(yiwen) pos should ignore padding person  (seqnum, 150, 3), y-up
        _, pos_train_aistpp = aistpp_train.get_all_data()
        _, pos_test_aistpp = aistpp_test.get_all_data()

        _, pos_train_aioz = aioz_train.get_all_data()
        _, pos_test_aioz = aioz_test.get_all_data()
        _, pos_val_aioz = aioz_val.get_all_data()

        # T, 3
        pos_aistpp = torch.cat([torch.tensor(pos_train_aistpp), torch.tensor(pos_test_aistpp)], dim=0)
        pos_aioz = torch.cat([torch.tensor(pos_train_aioz), torch.tensor(pos_test_aioz), torch.tensor(pos_val_aioz)], dim=0)

        # align mean pelvis
        mean_aistpp = pos_aistpp.mean(dim=(0,1), keepdim=True)
        mean_aioz = pos_aioz.mean(dim=(0,1), keepdim=True)

        delta_height = mean_aistpp.squeeze()[1] - mean_aioz.squeeze()[1]
        
        print(f'delta height: {delta_height}') # 2.0691


    elif stage==2: # collect mean and std for specified dataset
        # aamixed
        if dataset_name == 'aamixed':
            aamixed_train = Music2DanceDataset('aamixed', data_split='train', shuffle=False, collect_stats_stage2=True)
            aamixed_test = Music2DanceDataset('aamixed', data_split='test', shuffle=False, collect_stats_stage2=True)
            aamixed_val = Music2DanceDataset('aamixed', data_split='val', shuffle=False, collect_stats_stage2=True)
            
            motion_train, _ = aamixed_train.get_all_data()
            pose_train, num_person_train = motion_train['pose'], motion_train['num_person']
            motion_test, _ = aamixed_test.get_all_data()
            pose_test, num_person_test = motion_test['pose'], motion_test['num_person']
            motion_val, _ = aamixed_val.get_all_data()
            pose_val, num_person_val = motion_val['pose'], motion_val['num_person']

            motion_all = torch.cat([cut_padding(pose_train, num_person_train), cut_padding(pose_test, num_person_test), cut_padding(pose_val, num_person_val)], dim=0) # nT, 151

        # aistpp
        elif dataset_name == 'aistpp':
            aistpp_train = Music2DanceDataset('aistpp', data_split='train', shuffle=False, collect_stats_stage2=True)
            aistpp_test = Music2DanceDataset('aistpp', data_split='test', shuffle=False, collect_stats_stage2=True)

            motion_train, _ = aistpp_train.get_all_data()
            pose_train, num_person_train = motion_train['pose'], motion_train['num_person']
            motion_test, _ = aistpp_test.get_all_data()
            pose_test, num_person_test = motion_test['pose'], motion_test['num_person']

            motion_all = torch.cat([cut_padding(pose_train, num_person_train), cut_padding(pose_test, num_person_test)], dim=0) # nT, 151

        # aioz
        elif dataset_name == 'aioz':
            aioz_train = Music2DanceDataset('aioz', data_split='train', shuffle=False, collect_stats_stage2=True)
            aioz_test = Music2DanceDataset('aioz', data_split='test', shuffle=False, collect_stats_stage2=True)
            aioz_val = Music2DanceDataset('aioz', data_split='val', shuffle=False, collect_stats_stage2=True)

            motion_train, _ = aioz_train.get_all_data()
            pose_train, num_person_train = motion_train['pose'], motion_train['num_person']
            motion_test, _ = aioz_test.get_all_data()
            pose_test, num_person_test = motion_test['pose'], motion_test['num_person']
            motion_val, _ = aioz_val.get_all_data()
            pose_val, num_person_val = motion_val['pose'], motion_val['num_person']

            motion_all = torch.cat([cut_padding(pose_train, num_person_train), cut_padding(pose_test, num_person_test), cut_padding(pose_val, num_person_val)], dim=0) # nT, 151


        mean_np = motion_all.mean(dim=(0), keepdim=True).numpy().squeeze()
        std_np = motion_all.std(dim=(0), keepdim=True).numpy().squeeze()
        std_np += EPSILON 

        print(f'mean {mean_np}')
        print(f'std {std_np}')

        save_path = f"checkpoints/{dataset_name}/meta/"
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "mean_std.pkl"), "wb") as f:
            pickle.dump({"mean": mean_np, "std": std_np}, f)
        print(f"Mean and std of {dataset_name} saved to {save_path}mean_std.pkl")
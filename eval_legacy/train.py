import os
import numpy as np
import torch
from torch.utils.data import ConcatDataset
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from dataset.vis import SMPLSkeleton
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from dataset.dataset_AE import Music2DanceDataset_AE
import argparse

from .fid_encoder import MovementMotionAutoencoder
from time import time

import matplotlib.pyplot as plt
os.environ["PYOPENGL_PLATFORM"] = "egl" # headless render mode


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=4):
    # decay = decay_rate ** (epoch // decay_epoch)
    if epoch <= 35: # 4
       base_lr = init_lr
    elif epoch >= 35 and epoch <= 70: # 4; 10
       base_lr = init_lr * 0.2 
    elif epoch >= 70 and epoch <= 150: # 10; 50
       base_lr = init_lr * 0.01
    elif epoch >= 150:# and epoch <= 100: # 10; 50
       base_lr = init_lr * 0.005
    # elif epoch >= 101 and epoch <= 500: # 10; 50
    #    base_lr = init_lr * 0.001
    lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def visualize_joints(joints, save_name="vis_joints.png"):
    joints = joints.detach().cpu().numpy()

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

    plt.savefig(save_name)
    plt.close()


def train_epochs(args, train_loader, dataset_name, device):

    if dataset_name == 'aistpp' or dataset_name == 'aioz' or dataset_name == 'aamixed':
        dim_pose = 79 # 24*3+3+4
        dim_movement_hidden = 512
        dim_motion_hidden = 1024
        dim_coemb_hidden = 512
        dim_movement_latent = 512
        motion_seq_len = 148
        
    model = MovementMotionAutoencoder(movement_input_size=dim_pose-4, # 75
                                      movement_hidden_size=dim_movement_hidden,
                                      movement_latent_size=dim_movement_latent,
                                      motion_input_size=dim_movement_latent,
                                      motion_hidden_size=dim_motion_hidden,
                                      motion_latent_size=dim_coemb_hidden,
                                      motion_seq_len=motion_seq_len, #TODO(yw) check here
                                      device=device)
    
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=1e-5)
    
    model.to(device)
    model.train()
    
    loss_fn = nn.L1Loss() # nn.MSELoss()
    print(f"Training for {args.epochs} epochs...")
    
    train_steps = 0
    log_steps = 0
    running_loss = 0 
    start_time = time()
    
    for epoch in range(args.epochs):
        print(f"Beginning epoch {epoch}...")
        adjust_lr(model_optimizer, args.lr, epoch)
        
        for iter_idx, data in enumerate(train_loader, 0):

            gt_motion = data # B, T, D
            pose_seq = gt_motion.cuda().float() 
            out_seq = model(pose_seq) # TODO(yw) 128, 148, 75 check here, reconstruct 算loss时用dim=3？ 

            recons_loss = loss_fn(out_seq, pose_seq)

            model_optimizer.zero_grad()
            recons_loss.backward()
            model_optimizer.step() 

            log_steps += 1
            train_steps += 1
            running_loss += recons_loss.item()
            # print(f'debug -- train_steps {train_steps}')
            
            if train_steps % 100 == 0:
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() 
                print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

                # visualization
                smpl = SMPLSkeleton(device=device) # root_pos, local_q

                root_pos = gt_motion[:,:,:3].squeeze().to(device)
                local_q = gt_motion[:,:,3:].squeeze().view(root_pos.shape[0], root_pos.shape[1], -1, 3).to(device)
                positions = smpl.forward(local_q, root_pos) # 128, 148, 24, 3
                visualize_joints(positions[0,0,:,:], save_name='./eval_legacy/ae_gt.png') # (24, 3)

                root_pos_recons = out_seq[:,:,:3].to(device)
                local_q_recons = out_seq[:,:,3:].view(root_pos_recons.shape[0], root_pos_recons.shape[1], -1, 3).to(device)
                positions_recons = smpl.forward(local_q_recons, root_pos_recons) # 128, 148, 24, 3
                visualize_joints(positions_recons[0,0,:,:], save_name='./eval_legacy/ae_recons.png') # (24, 3)

        # Save checkpoint to dict:
        if (epoch+1) % args.ckpt_every == 0 and epoch>0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint_path = f"{args.checkpoint_dir}/epoch_{epoch+1}.pth"
            model.save_weights(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")   
            
                
def main(args):
    print(f'training motion AE on {args.dataset_name}')
    dataset_name = args.dataset_name
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = Music2DanceDataset_AE(dataset_name, 
                                        data_split='train',
                                        shuffle=True)

    if args.dataset_name != 'aistpp':
        val_dataset = Music2DanceDataset_AE(dataset_name, 
                                            data_split='val',
                                            shuffle=True)

    test_dataset = Music2DanceDataset_AE(dataset_name, 
                                        data_split='test',
                                        shuffle=True)

    if args.dataset_name == 'aistpp':
        Full_dataset = ConcatDataset([train_dataset, test_dataset])
    else:
        Full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    
    train_loader = DataLoader(dataset=Full_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True)
    
    train_epochs(args, train_loader, dataset_name, device)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default='aamixed', type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--loader_workers", default=4, type=int)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--ckpt_every", type=int, default=100) 
    parser.add_argument("--lr", type=float, default = 1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--checkpoint_dir", type=str, default='./eval_legacy/checkpoints')
       

    args = parser.parse_args()  
    
    main(args)

import os
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import models.vqvae as vqvae
import utils.losses as losses 
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_VQ, dataset_TM_eval, dataset_MD, dataset_MD_multi       
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
from models.evaluator_wrapper_dance import EvaluatorModelWrapper_Dance
import warnings
warnings.filterwarnings('ignore')
from utils.word_vectorizer import WordVectorizer
from tqdm import tqdm
from exit.utils import get_model, generate_src_mask, init_save_folder
from models.vqvae_sep import VQVAE_SEP
#from eval.train import visualize_joints
import matplotlib.pyplot as plt
from dataset.quaternion import ax_from_6v
from dataset.vis import SMPLSkeleton

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


def unnormalized6D_to_3Daa(motion_6D):    # -------litingw: multi版
    B, H, T, D = motion_6D.shape
    motion_6D = motion_6D.view(B, H*T, D) # 32, 148, 151    
    root_pos_eval = motion_6D[:,:,4:7] # 151 = 4 contacts + 3 root_pos + 144 local_q(6D)

    local_q_eval = motion_6D[:,:,7:].view(root_pos_eval.shape[0], root_pos_eval.shape[1], -1, 6)
    local_q_eval_aa = ax_from_6v(local_q_eval) # 32, 148, 24, 3   # (B, H*T, Joints, 3)

    local_q_eval_aa = local_q_eval_aa.view(B, H, T, -1)  # 恢复 (B, H, T, 72)
    motion_3D = torch.cat([root_pos_eval.view(B, H, T, 3), local_q_eval_aa], dim=-1)  

    return motion_3D


def visualize_motion3D(motion_3D, vis_dir='./vq', save_name="visualization_3d_motion.png", device='cuda:0'): # -------litingw: multi版
    # motion_3D (B, H, 148, 75)
    smpl = SMPLSkeleton(device=device)   # root_pos, local_q

    root_pos = motion_3D[:, :, :, :3].to(device)  # (B, H, T, 3)
    local_q = motion_3D[:, :, :, 3:].view(root_pos.shape[0], root_pos.shape[1], root_pos.shape[2], -1, 3).to(device)  # (B, H, T, Joints, 3)

    for t in range(root_pos.shape[2]):  # each Frame
        extend_name = f't{t}_' + save_name
        save_path = os.path.join(vis_dir, extend_name)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for h in range(root_pos.shape[1]):  # each Person
            current_root_pos = root_pos[0, h, t, :].unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
            current_local_q = local_q[0, h, t, :, :].unsqueeze(0).unsqueeze(0)  # (1, 1, Joints, 3)

            # smpl.forward: input：current_local_q (B, T, Joints,3)+ current_root_pos (B, T, 3) ➡️ output：positions (B, T, Joints,3)
            positions = smpl.forward(current_local_q, current_root_pos)  # (1, 1, Joints, 3) 
            joints = positions[0, 0, :, :].detach().cpu().numpy()  

            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', s=25)
            skeleton = [
                (0, 1), (1, 4), (4, 7), (0, 2), (2, 5), (5, 8), (8, 11), (7, 10),  
                (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),                     
                (12, 13), (13, 16), (12, 14), (14, 17),                         
                (16, 18), (18, 20), (17, 19), (19, 21), (20, 22), (21, 23)       
            ]
            for joint_start, joint_end in skeleton:
                ax.plot(
                    [joints[joint_start, 0], joints[joint_end, 0]],
                    [joints[joint_start, 1], joints[joint_end, 1]],
                    [joints[joint_start, 2], joints[joint_end, 2]],
                    'b-'
                )
        plt.savefig(save_path)
        plt.close()


##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'vq') # /{args.exp_name}
# os.makedirs(args.out_dir, exist_ok = True)
init_save_folder(args)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


#w_vectorizer = WordVectorizer('./glove', 'our_vab')

if args.dataname == 'kit' : 
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'  
    args.nb_joints = 21
    
elif args.dataname == 't2m' :
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22
    
elif args.dataname == 'aistpp':
    #TODO(yw) check the datasetopt, (relevant to eval trans)
    dataset_opt_path = 'checkpoints/aistpp/opt.txt' # NOTE(yw) the above two are roughly the same, is_continue=True/False
    # dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 24

elif args.dataname == 'aioz':
    #TODO(yw) check the datasetopt, (relevant to eval trans)
    dataset_opt_path = 'checkpoints/aioz/opt.txt' # NOTE(yw) the above two are roughly the same, is_continue=True/False
    # dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 24

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
if args.dataname == 'aistpp':
    eval_wrapper = EvaluatorModelWrapper_Dance(wrapper_opt)
elif args.dataname == 'aioz':
    pass
else:
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Dataloader ---- #####
if args.dataname == 'aistpp':
    train_loader = dataset_MD.DATALoader(dataset_name=args.dataname,
                                         is_test=False,
                                         batch_size=args.batch_size)
    train_loader_iter = dataset_MD.cycle(train_loader)
    
    val_loader = dataset_MD.DATALoader(dataset_name=args.dataname,
                                        is_test=True, # TODO (yw) del optimizer in data loader
                                        batch_size=32) # use the testset, since aistpp has no val set
elif args.dataname == 'aioz':
    train_loader = dataset_MD_multi.DATALoader(dataset_name=args.dataname,
                                         is_test=False,
                                         batch_size=args.batch_size)
    train_loader_iter = dataset_MD_multi.cycle(train_loader)
    
    val_loader = dataset_MD_multi.DATALoader(dataset_name=args.dataname,
                                        is_test=True, # TODO (yw) del optimizer in data loader
                                        batch_size=32) # use the testset, since aistpp has no val set                                      
else:  
    train_loader = dataset_VQ.DATALoader(args.dataname,
                                         args.batch_size,
                                         window_size=args.window_size,
                                         unit_length=2**args.down_t)

    train_loader_iter = dataset_VQ.cycle(train_loader)

    val_loader = dataset_TM_eval.DATALoader(args.dataname, False,
                                            32,
                                            w_vectorizer,
                                            unit_length=2**args.down_t)

data_mean = val_loader.dataset.mean
data_std = val_loader.dataset.std

##### ---- Network ---- #####
if args.dataname == 'aistpp':
    args.sep_uplow = False

if args.dataname == 'aioz':
    args.sep_uplow = False
    
if args.sep_uplow:
    net = VQVAE_SEP(args, ## use args to define different parameters in different quantizers
                        args.nb_code,
                        args.code_dim,
                        args.output_emb_width,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate,
                        args.vq_act,
                        args.vq_norm,
                        {'mean': torch.from_numpy(train_loader.dataset.mean).cuda().float(), 
                        'std': torch.from_numpy(train_loader.dataset.std).cuda().float()},
                        True)
else:
    net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                        args.nb_code,
                        args.code_dim,
                        args.output_emb_width,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate,
                        args.vq_act,
                        args.vq_norm)


if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
net.train()
net.cuda() #TODO(yw) add multi GPU parallel here

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
  
if args.dataname=='aistpp':
    Loss = losses.DanceReConsLoss(args.recons_loss, args.nb_joints)
elif args.dataname=='aioz':
    Loss = losses.DanceReConsLoss(args.recons_loss, args.nb_joints)
else:
    Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)

##### ------ warm-up ------- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
vis_dir = './vq_image'

for nb_iter in range(1, args.warm_up_iter):
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
    
    if args.dataname=='aistpp':
        # NOTE (yw) then check utils/losses.py
        gt_motion, features, filenames, wavs = next(train_loader_iter)  
        # motion(256, 148, 151), audio_feats(256, 148, 35) # TODO(yw) check whether need to slice the audio here
    elif args.dataname=='aioz':
        # NOTE (yw) then check utils/losses.py
        gt_motion, features, filenames, wavs = next(train_loader_iter)  
        # motion(256, 148, 151), audio_feats(256, 148, 35) # TODO(yw) check whether need to slice the audio here
    else:
        gt_motion = next(train_loader_iter) # if kit dataset, 256, 64, 251 (B, T(window_size), D)

    gt_motion = gt_motion.cuda().float() # (bs, 64, dim) or 256, 1, 150, 151
    
    pred_motion, loss_commit, perplexity = net(gt_motion) # ---------TODO Litingw：net（vqvae）的输入输出均为(B,H,T,D),只是在net内部计算的时候，H乘到D上

    loss_motion = Loss(pred_motion, gt_motion) # 256, 1, 148, 151  in 6d

    ############ NOTE(yiwen) add predicted motion reconstruction visualization
    if nb_iter==1:
        data_std = data_std.to(pred_motion.device)
        data_mean = data_mean.to(pred_motion.device)
        unnormalized_pred_motion_6D = pred_motion * data_std + data_mean
        unnormalized_gt_motion_6D = gt_motion * data_std + data_mean

        pred_motion_3D = unnormalized6D_to_3Daa(unnormalized_pred_motion_6D)
        gt_motion_3D = unnormalized6D_to_3Daa(unnormalized_gt_motion_6D)
        

        os.makedirs(vis_dir, exist_ok=True)
        visualize_motion3D(pred_motion_3D, vis_dir, "vqvae_recons_init.png", pred_motion_3D.device)
        visualize_motion3D(gt_motion_3D, vis_dir, "vqvae_gt_init.png", pred_motion_3D.device)

    
    if args.dataname=='t2m' or args.dataname=='kit':
        loss_vel = Loss.forward_joint(pred_motion, gt_motion) # 3 vel xyz 除根节点之外的速度xyz
        loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
    else:
        # NOTE(yiwen) no velocity prediction here
        loss = loss_motion + args.commit * loss_commit
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_recons += loss_motion.item() # motion reconstruction
    avg_perplexity += perplexity.item() # codebook utilization
    avg_commit += loss_commit.item() # vq loss, encoded motion 能在codebook中找到匹配
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        
        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

##### ---- Training ---- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

# TODO(yw)
if args.dataname=='t2m' or args.dataname=='kit':
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper)
elif args.dataname=='aistpp':
    best_fid, best_iter, best_div, writer, logger = eval_trans.evaluation_vqvae_dance(args.out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, eval_wrapper=eval_wrapper)
# elif args.dataname=='aioz':
#     best_fid, best_iter, best_div, writer, logger = eval_trans.evaluation_vqvae_dance(args.out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, eval_wrapper=eval_wrapper)

for nb_iter in tqdm(range(1, args.total_iter + 1)):
    if args.dataname=='aistpp':
        gt_motion, features, filenames, wavs = next(train_loader_iter)  
        # motion(256, 148, 151), audio_feats(256, 148, 35)
    elif args.dataname=='aioz':
        gt_motion, features, filenames, wavs = next(train_loader_iter)  
        # motion(256, 148, 151), audio_feats(256, 148, 35)
    else:
        gt_motion = next(train_loader_iter)
    gt_motion = gt_motion.cuda().float() # bs, nb_joints, joints_dim, seq_len
    
    if args.sep_uplow:
        pred_motion, loss_commit, perplexity = net(gt_motion, idx_noise=0)
    else:
        pred_motion, loss_commit, perplexity = net(gt_motion)

    loss_motion = Loss(pred_motion, gt_motion)
    
    ### NOTE(yiwen) visualize the gt and reconstructed results
    if nb_iter%100000==0:
        unnormalized_pred_motion_6D = pred_motion * data_std + data_mean
        unnormalized_gt_motion_6D = gt_motion * data_std + data_mean

        pred_motion_3D = unnormalized6D_to_3Daa(unnormalized_pred_motion_6D)
        gt_motion_3D = unnormalized6D_to_3Daa(unnormalized_gt_motion_6D)
        
        visualize_motion3D(pred_motion_3D, vis_dir, f"vqvae_recons_iter{nb_iter}.png", pred_motion_3D.device)
        visualize_motion3D(gt_motion_3D, vis_dir, f"vqvae_gt_iter{nb_iter}.png", pred_motion_3D.device)
    
    if args.dataname=='t2m' or args.dataname=='kit':
        loss_vel = Loss.forward_joint(pred_motion, gt_motion) # 3 vel xyz 除根节点之外的速度xyz
        loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
    else:
        loss = loss_motion + args.commit * loss_commit 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        
        writer.add_scalar('./Train/L1', avg_recons, nb_iter)
        writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
        writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
        
        logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.,

    # NOTE(yw) temp
    if nb_iter==args.total_iter:
        torch.save({'net' : net.state_dict()}, os.path.join(args.out_dir, 'net_last.pth'))
    
    # NOTE (yw) train+test合成一个数据集，train motion feature extractor ae, 算feature fid
    if nb_iter % args.eval_iter==0 :
        if args.dataname=='t2m' or args.dataname=='kit':
            best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper=eval_wrapper)
        elif args.dataname=='aistpp':
            best_fid, best_iter, best_div, writer, logger = eval_trans.evaluation_vqvae_dance(args.out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, eval_wrapper=eval_wrapper)
        # elif args.dataname=='aioz':
        #     best_fid, best_iter, best_div, writer, logger = eval_trans.evaluation_vqvae_dance(args.out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, eval_wrapper=eval_wrapper)
        
import os
import json
import torch
import models.vqvae as vqvae
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_MD_multi
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper_dance import EvaluatorModelWrapper_Dance
import warnings
warnings.filterwarnings('ignore')
from exit.utils import get_model, init_save_folder
from dataset.quaternion import ax_from_6v
from torch.utils.tensorboard import SummaryWriter

from dataset.vis import SMPLSkeleton
from eval.calculate_scores import extract_features_multi, calculate_FID_DIST, extract_features_tofiles
from eval.calculate_beat_scores import cal_BAS



##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
args.ref_dataname = 'aamixed'

checkpoint_path = '/data/xingqunqi/AI_dance/Group_Dance_output/output/vq/temp_save/net_last.pth'

if args.dataname == 'aamixed': 
    dataset_opt_path = 'checkpoints/aamixed/opt.txt' 

elif args.dataname == 'aistpp':
    dataset_opt_path = 'checkpoints/aistpp/opt.txt' 

elif args.dataname == 'aioz':
    dataset_opt_path = 'checkpoints/aioz/opt.txt'

args.nb_joints = 24

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
if args.dataname == 'aistpp':
    eval_wrapper = EvaluatorModelWrapper_Dance(wrapper_opt)
elif args.dataname == 'aioz':
    eval_wrapper = EvaluatorModelWrapper_Dance(wrapper_opt)
elif args.dataname == 'aamixed':
    eval_wrapper = EvaluatorModelWrapper_Dance(wrapper_opt)


##### ---- Dataloader ---- #####
if args.dataname == 'aistpp':
    val_loader = dataset_MD_multi.DATALoader(dataset_name=args.dataname,
                                        data_split='test',
                                        batch_size=32) # use the testset, since aistpp has no val set, only eval no training here

elif args.dataname == 'aioz':
    val_loader = dataset_MD_multi.DATALoader(dataset_name=args.dataname,
                                        data_split='test',
                                        batch_size=32)          

elif args.dataname == 'aamixed':
    val_loader = dataset_MD_multi.DATALoader(dataset_name=args.dataname,
                                        data_split='test',
                                        batch_size=64)     
    
val_loader_iter = dataset_MD_multi.cycle(val_loader)
    
data_mean = val_loader.dataset.mean # NOTE(yiwen) train, val, test use the same stats.
data_std = val_loader.dataset.std

##### ---- Network ---- #####
net = vqvae.HumanVQVAE(args,
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


print('loading vqvae checkpoint from {}'.format(checkpoint_path))
checkpoint = torch.load(checkpoint_path, map_location='cpu')
net = get_model(net)
net.load_state_dict(checkpoint['net'], strict=True)

net.eval()
net.cuda() 

smpl = SMPLSkeleton(device='cuda:0')


results_features_dic = {"kinetic": [], "manual": []}
results_features_dic_gt = {"kinetic": [], "manual": []}

avg_l2_distance = 0
for batch in val_loader: 
    motion, music_feats, filenames, wavs, num_person = next(val_loader_iter) # normalized 6d motion
    
    motion = motion.cuda()
    B, H, T, D = motion.shape
    
    data_std = data_std.to(motion.device)
    data_mean = data_mean.to(motion.device)

    ########### NOTE(yiwen) unnormalize gt 6d-->3d, getting feature distribution
    unnormalized_motion = motion * data_std + data_mean
    motion_copy = unnormalized_motion.view(B*H, T, D) # BH, 148, 151
    ### 151 = contacts, root_pos, local_q
    root_pos_gt = motion_copy[:,:,4:7] # BH, T=148, 3 # TODO(yiwen) check whether contact force is the last several dims
    local_q_gt = motion_copy[:,:,7:].view(root_pos_gt.shape[0], root_pos_gt.shape[1], -1, 6) # BH, T, 24, 6
    local_q_gt_aa = ax_from_6v(local_q_gt) # BH, 148, 24, 3 b,

    BH, T, J, D = local_q_gt_aa.shape
    positions_gt = smpl.forward(local_q_gt_aa, root_pos_gt)
    
    # NOTE(yiwen) new FID DIST metrics in eval
    results_features_gt = extract_features_multi(positions_gt.view(B, H, T, J, D), num_person)
    results_features_dic_gt['kinetic'].extend(results_features_gt['kinetic'])
    results_features_dic_gt['manual'].extend(results_features_gt['manual'])

    local_q_gt_aa = local_q_gt_aa.view(BH, T, -1) # 32, 148, 72  BH, T, 72

    ########### NOTE(yiwen) predict motion using normalized 6d
    bs, num_ps, seq = motion.shape[0], motion.shape[1], motion.shape[2] # B, H, T
    if motion.shape[-1] == 251:
        num_joints = 21 
    elif motion.shape[-1] == 263:
        num_joints = 22
    else:
        num_joints = 24      
    feature_dim = num_joints*6 + 3 + 4
    pred_pose_eval = torch.zeros((bs, num_ps, seq, feature_dim)).cuda()

    with torch.no_grad():
        for i in range(bs): 
            pred_pose, loss_commit, perplexity = net(motion[i:i+1, :, :, :], num_person[i:i+1]) # put single motion to the net

            # unnormalize motion using data statistic
            unnormalized_pred_pose = pred_pose.clone() * data_std + data_mean
            pred_pose_eval[i:i+1, :, :, :] = unnormalized_pred_pose # 32, 1, 148, 151

    # music_feats 32, 148, 35
    B, H, T, D = pred_pose_eval.shape
    pred_pose_eval = pred_pose_eval.view(B*H, T, D)

    ########### NOTE (yw) unnormalized 6D-->3D
    root_pos_eval = pred_pose_eval[:,:,4:7]
    local_q_eval = pred_pose_eval[:,:,7:].view(root_pos_eval.shape[0], root_pos_eval.shape[1], -1, 6)
    local_q_eval_aa = ax_from_6v(local_q_eval) # BH, T=148, 24, 3
    
    BH, T, J, D = local_q_eval_aa.shape
    
    positions_recons = smpl.forward(local_q_eval_aa, root_pos_eval) # 128, 148, 24, 3
    
    # L2 joint norm
    l2_distance = torch.norm(positions_recons - positions_gt, dim=-1) # BH, T, J (padding also includes)
    avg_l2_distance += l2_distance.mean() 

    # NOTE(yiwen) new FID DIST metrics in eval
    # results_features = extract_features_tofiles(positions_recons.view(B, H, T, J, D), num_person, filenames, args.out_dir)
    results_features = extract_features_multi(positions_recons.view(B, H, T, J, D), num_person)
    results_features_dic['kinetic'].extend(results_features['kinetic'])
    results_features_dic['manual'].extend(results_features['manual'])


FID_k, FID_g, Dist_k, Dist_g = calculate_FID_DIST(results_features_dic, args.ref_dataname) # output the scores
FID_k_gt, FID_g_gt, Dist_k_gt, Dist_g_gt = calculate_FID_DIST(results_features_dic_gt, args.ref_dataname)

msg = f"--> \t Eva. Scores:, \n\
            FID_k. {FID_k:.4f} , \n\
            FID_g. {FID_g:.4f} , \n\
            Dist_k. {Dist_k:.4f},  Dist_k_refs. {Dist_k_gt:.4f},\n\
            Dist_g. {Dist_g:.4f},  Dist_g_refs. {Dist_g_gt:.4f}"
print(msg)


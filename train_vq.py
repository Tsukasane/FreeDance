import os
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import models.vqvae as vqvae
import utils.losses as losses 
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_VQ, dataset_TM_eval, dataset_MD
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

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

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


w_vectorizer = WordVectorizer('./glove', 'our_vab')

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

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
if args.dataname == 'aistpp':
    eval_wrapper = EvaluatorModelWrapper_Dance(wrapper_opt)
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

##### ---- Network ---- #####
if args.dataname == 'aistpp':
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
else:
    Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)

##### ------ warm-up ------- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

for nb_iter in range(1, args.warm_up_iter):
    
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
    
    if args.dataname=='aistpp':
        # NOTE (yw) then check utils/losses.py
        gt_motion, features, filenames, wavs = next(train_loader_iter)  
        # motion(256, 148, 151), audio_feats(256, 148, 35) # TODO(yw) check whether need to slice the audio here
    else:
        gt_motion = next(train_loader_iter) # if kit dataset, 256, 64, 251 (B, T(window_size), D)

    gt_motion = gt_motion.cuda().float() # (bs, 64, dim) or 256, 1, 150, 151
    
    pred_motion, loss_commit, perplexity = net(gt_motion)

    loss_motion = Loss(pred_motion, gt_motion)
    if args.dataname=='t2m' or args.dataname=='kit':
        loss_vel = Loss.forward_joint(pred_motion, gt_motion) # 3 vel xyz 除根节点之外的速度xyz
        loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
    else:
        # NOTE(yw) no velocity prediction here
        loss = loss_motion + args.commit * loss_commit
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    
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

for nb_iter in tqdm(range(1, args.total_iter + 1)):
    if args.dataname=='aistpp':
        gt_motion, features, filenames, wavs = next(train_loader_iter)  
        # motion(256, 148, 151), audio_feats(256, 148, 35)
    else:
        gt_motion = next(train_loader_iter)
    gt_motion = gt_motion.cuda().float() # bs, nb_joints, joints_dim, seq_len
    
    if args.sep_uplow:
        pred_motion, loss_commit, perplexity = net(gt_motion, idx_noise=0)
    else:
        pred_motion, loss_commit, perplexity = net(gt_motion)

    loss_motion = Loss(pred_motion, gt_motion) # TODO(yw) check loss 还是用normlizaed 6D
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
        
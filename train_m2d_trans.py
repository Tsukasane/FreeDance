import os 
import torch
import numpy as np
import torch.nn as nn
from typing import Any, Callable, List, Optional, Union
from torch import Tensor

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
# import clip

import options.option_transformer_dance as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans

from dataset import dataset_MD, dataset_MD_multi
from dataset import dataset_tokenize_MD
import models.m2d_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper_dance import EvaluatorModelWrapper_Dance
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from exit.utils import get_model, visualize_2motions, generate_src_mask, init_save_folder, uniform, cosine_schedule
from einops import rearrange, repeat
import torch.nn.functional as F
from exit.utils import base_dir
import shutil


"""
Inject music condition to interactive motion sequences
"""

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

init_save_folder(args)

codebook_dir = f'{args.vq_dir}/codebook/'
args.resume_pth = f'{args.vq_dir}/net_last.pth'
os.makedirs(args.vq_dir, exist_ok = True)
os.makedirs(codebook_dir, exist_ok = True)
os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.out_dir+'/html', exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

# NOTE(yiwen) use untokenized data
val_loader = dataset_MD_multi.DATALoader(dataset_name=args.dataname,
                                    data_split='val', 
                                    batch_size=32,
                                    normalizer=None)

if args.dataname == 'aamixed': 
    dataset_opt_path = 'checkpoints/aamixed/opt.txt' 

elif args.dataname == 'aistpp':
    dataset_opt_path = 'checkpoints/aistpp/opt.txt' 

elif args.dataname == 'aioz':
    dataset_opt_path = 'checkpoints/aioz/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper_Dance(wrapper_opt) # TODO(yiwen) check this new wrapper

##### ---- Network ---- #####
from models.modules import MusicTransformerEncoder
musicFeatsEncoder = MusicTransformerEncoder(cond_feature_dim=35)     

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code, # 8192
                       args.code_dim, # 32
                       args.output_emb_width, # 512
                       args.down_t, # 2
                       args.stride_t, # 2
                       args.width, # 512
                       args.depth, # 3
                       args.dilation_growth_rate) # 3
 
trans_encoder = trans.Music2Dance_Transformer(vqvae=net,
                                num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                music_dim=args.music_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                num_local_layer=args.num_local_layer, 
                                n_head=args.n_head_gpt, # do not use multi head self attention here.
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)


## load pretrained vq
print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')

net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

iter_start = 1

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.resume_trans is not None: 
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    # ckpt_trans = torch.load(args.resume_trans, map_location='cpu')
    # trans_encoder.load_state_dict(ckpt_trans['trans'], strict=True)

    checkpoint = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder = get_model(trans_encoder) 
    trans_encoder.load_state_dict(checkpoint['trans'], strict=True)
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        if isinstance(state, torch.Tensor):
            state.data = state.data.to(device)
        elif isinstance(state, dict):
            for key, val in state.items():
                if isinstance(val, torch.Tensor):
                    state[key] = val.to(device)

    scheduler.load_state_dict(checkpoint['scheduler'])
    iter_start = checkpoint['iters']

trans_encoder.train()
trans_encoder.cuda()
trans_encoder = torch.nn.DataParallel(trans_encoder)


##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss(reduction='none')

##### ---- get code ---- #####
##### ---- Dataloader ---- #####

## NOTE(yiwen) offline converting motion sequence to codebook
## NOTE(yiwen) first time running will take long time here
if len(os.listdir(codebook_dir)) == 0:
    train_loader_token = dataset_MD_multi.DATALoader(
                                    dataset_name=args.dataname,  
                                    batch_size=1,
                                    data_split='train') 

    for batch in train_loader_token:
        pose, _, name, _, num_person = batch 
        # pose.shape 1, 3, 148, 151 
        bs, seq = pose.shape[0], pose.shape[2]
        pose = pose.cuda().float() # bs, nb_joints, joints_dim, seq_len

        target = net(pose, num_person, type='encode')
        target = target.cpu().numpy() # (1, 37, 1) 37 = 148(seq length)/4(unit_length)
        
        prefix = name[0].split('/')[-1]
        np.save(pjoin(codebook_dir, prefix), target) 


# NOTE(yiwen) a new dataloader with both raw motions and music & motion tokens
train_loader = dataset_MD_multi.DATALoader(dataset_name=args.dataname,  
                                    batch_size=args.batch_size,
                                    data_split='train',
                                    codebook_size=args.nb_code, # 8192
                                    tokenizer_name=codebook_dir,
                                    load_motion_code=True) 

train_loader_iter = dataset_MD_multi.cycle(train_loader)

# NOTE(yiwen) a dataloader that providing codebook data
# train_loader = dataset_tokenize_MD.DATALoader(dataset_name=args.dataname, 
#                                      data_split='train',
#                                      batch_size=args.batch_size,
#                                      codebook_size=args.nb_code, # 8192
#                                      tokenizer_name=codebook_dir)

# train_loader_iter = dataset_tokenize_MD.cycle(train_loader)

        
##### ---- Training ---- #####
best_fid=5000 
best_iter=0 
best_div=100 
best_matching=100 

# TODO(yiwen) check and implement new eval metrics
pred_pose_eval, pose, m_length, music_feature, best_fid, best_iter, best_div, writer, logger = eval_trans.evaluation_transformer_dance(args.out_dir, 
                                                                                                                                        val_loader, 
                                                                                                                                        net, 
                                                                                                                                        trans_encoder, 
                                                                                                                                        logger, 
                                                                                                                                        writer, 
                                                                                                                                        0, 
                                                                                                                                        best_fid=5000, 
                                                                                                                                        best_iter=0, 
                                                                                                                                        best_div=100, 
                                                                                                                                        music_encoder=musicFeatsEncoder, 
                                                                                                                                        eval_wrapper=eval_wrapper,
                                                                                                                                        exp_name=args.exp_name)


def get_acc(cls_pred, target, mask):
    cls_pred = torch.masked_select(cls_pred, mask.unsqueeze(-1)).view(-1, cls_pred.shape[-1])
    target_all = torch.masked_select(target, mask)
    probs = torch.softmax(cls_pred, dim=-1)
    _, cls_pred_index = torch.max(probs, dim=-1)
    right_num = (cls_pred_index == target_all).sum()
    return right_num*100/mask.sum()


# while nb_iter <= args.total_iter:
for nb_iter in tqdm(range(iter_start, args.total_iter + 1), position=0, leave=True):
    batch = next(train_loader_iter)

    gt_motion, music_feats, filenames, wavs, num_person, motion_token, motion_token_len = batch

    # music_feats, motion_token, motion_token_len = batch 
    # B, T, Mutok 128, 150, 35   B, H, T, Motok 128, 1, 37, 1   128  

    motion_token = motion_token.cuda()
    batch_size = motion_token.shape[0]
    target = motion_token.squeeze()  
    target = target.cuda()
    max_len = target.shape[1] # TMutok 37

    ######### NOTE(yiwen) music features --> music embeddings
    music_feats_emb = musicFeatsEncoder(music_feats) # B, T, Muemb 128, 150, 256

    ######### NOTE(yiwen) mask motion features(mask token modeling)
    # [INFO] Swap input tokens
    if args.pkeep == -1:
        proba = np.random.rand(1)[0] # random a probability
        mask = torch.bernoulli(proba * torch.ones(target.shape,
                                                device=target.device)) # randomly mask that much tokens
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(target.shape,
                                                device=target.device)) # B, 50
    # random only motion token (not pad token). To prevent pad token got mixed up.
    seq_mask_no_end = generate_src_mask(max_len, motion_token_len).to(target.device) # B, 50

    mask = torch.logical_or(mask, ~seq_mask_no_end).int() 
    r_indices = torch.randint_like(target, args.nb_code) # 
    input_indices = mask*target+(1-mask)*r_indices # random init only motion tokens

    ###### Time step masking (using special id)
    mask_id = get_model(net).vqvae.num_code + 2 # end_id = vqvae.num_code; pad_id = vqvae.num_code + 1; mask_id = vqvae.num_code + 2

    rand_mask_probs = torch.zeros(batch_size, device = motion_token_len.device).float().uniform_(0.5, 1)
    num_token_masked = (motion_token_len * rand_mask_probs).round().clamp(min = 1).to(target.device)
    
    seq_mask = generate_src_mask(max_len, motion_token_len+1) 
    seq_mask = torch.cat([seq_mask]*args.max_person, dim=-1)
    
    batch_randperm = torch.rand((batch_size, max_len), device = target.device) - seq_mask_no_end.int()

    batch_randperm = batch_randperm.argsort(dim = -1) 
    mask_token = batch_randperm < rearrange(num_token_masked, 'b -> b 1') 

    # masked_target = torch.where(mask_token, input=input_indices, other=-1)
    masked_input_indices = torch.where(mask_token, mask_id, input_indices) 

    ####### NOTE(yiwen) load transformer to predict masked tokens
    cls_pred = trans_encoder(masked_input_indices, # B, 50
                             src_mask=seq_mask, # B, T(padded)H
                             word_emb=music_feats_emb)  
    # B, T', code_dim

    ###### NOTE(yiwen) 在music condition下，predict正确的codebook class
    # [INFO] Compute xent loss as a batch
    weights = seq_mask_no_end / (seq_mask_no_end.sum(-1).unsqueeze(-1) * seq_mask_no_end.shape[0])
    cls_pred_seq_masked = cls_pred[seq_mask_no_end, :].view(-1, cls_pred.shape[-1])
    target_seq_masked = target[seq_mask_no_end]
    weight_seq_masked = weights[seq_mask_no_end]
    loss_cls = F.cross_entropy(cls_pred_seq_masked, target_seq_masked, reduction = 'none')
    loss_cls = (loss_cls * weight_seq_masked).sum()


    ###### TODO(yiwen) add decoder from net(freezed) and add recons loss
    # bs, num_ps, seq, feature_dim = gt_motion.shape[0], gt_motion.shape[1], gt_motion.shape[2] # B, H, T
    # feature_dim = 24*6 + 3 + 4
    # pred_pose_eval = torch.zeros((bs, num_ps, seq, feature_dim)).cuda()
    # m_length = torch.tensor([148 for i in range(batch_size)])
    # m_tokens_len = torch.tensor([37 for i in range(batch_size)])
    # pred_len = m_length.cuda()
    # pred_tok_len = m_tokens_len

    # for k in range(batch_size):
    #     # NOTE(yiwen) use the decoder side of the pretrained codebook
    #     pred_pose = net(cls_pred[k:k+1, :int(pred_tok_len[k].item())], num_person, type='decode') # decode([1, 37])
    #     pred_pose = pred_pose[:,:num_ps,:,:motion.shape[-1]]
    #     # 1, 3, 148, 151 
    #     pred_pose_eval[k:k+1,:int(pred_len[k].item())] = pred_pose

    # # TODO(yiwen) debug here, add element-wise loss, add weight
    # loss_recons = nn.MSELoss()(pred_pose_eval, gt_motions)

    # loss_all = loss_cls + loss_recons

    ## global loss
    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    if nb_iter % args.print_iter ==  0 :
        probs_seq_masked = torch.softmax(cls_pred_seq_masked, dim=-1)
        _, cls_pred_seq_masked_index = torch.max(probs_seq_masked, dim=-1)
        target_seq_masked = torch.masked_select(target, seq_mask_no_end)
        right_seq_masked = (cls_pred_seq_masked_index == target_seq_masked).sum()

        # TODO(yiwen) add recons schedular
        writer.add_scalar('./Loss/all', loss_cls, nb_iter)
        writer.add_scalar('./ACC/every_token', right_seq_masked*100/seq_mask_no_end.sum(), nb_iter)
        
        # NOTE log mask/nomask separately
        no_mask_token = ~mask_token * seq_mask_no_end
        writer.add_scalar('./ACC/masked', get_acc(cls_pred, target, mask_token), nb_iter)
        writer.add_scalar('./ACC/no_masked', get_acc(cls_pred, target, no_mask_token), nb_iter)

        # msg = f"Train. Iter {nb_iter} : Loss. {loss_cls:.5f}, ACC. {get_acc(cls_pred, target, mask_token):.4f}"
        # logger.info(msg)


    if nb_iter % 100==0:
        src = os.path.join(args.out_dir, 'net_last.pth')
        dst = os.path.join(args.out_dir, 'net_last_save.pth') # the one before last one
        if os.path.exists(src):
            shutil.copy(src, dst)
        print(f'Saving checkpoint of iter {nb_iter}')
        checkpoint = {
            'trans': get_model(trans_encoder).state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'iters': nb_iter,
        }
        torch.save(checkpoint, os.path.join(args.out_dir, 'net_last.pth'))


    if nb_iter==0 or nb_iter % args.eval_iter ==  0 or nb_iter == args.total_iter:
        num_repeat = 1
        rand_pos = False
        if nb_iter == args.total_iter:
            num_repeat = -30
            rand_pos = True
            if args.dataset_name=='aistpp':
                val_loader = dataset_MD_multi.DATALoader(args.dataname, 'test', 32)
            else:
                val_loader = dataset_MD_multi.DATALoader(args.dataname, 'val', 32)

    
        pred_pose_eval, pose, m_length, music_feature, best_fid, best_iter, best_div, writer, logger = eval_trans.evaluation_transformer_dance(args.out_dir, 
                                                                                                                                                val_loader, 
                                                                                                                                                net, 
                                                                                                                                                trans_encoder, 
                                                                                                                                                logger, 
                                                                                                                                                writer, 
                                                                                                                                                nb_iter, 
                                                                                                                                                best_fid, 
                                                                                                                                                best_iter, 
                                                                                                                                                best_div, 
                                                                                                                                                music_encoder=musicFeatsEncoder, 
                                                                                                                                                eval_wrapper=eval_wrapper,
                                                                                                                                                exp_name=args.exp_name)

    if nb_iter == args.total_iter: 
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}"
        logger.info(msg_final)
        break            
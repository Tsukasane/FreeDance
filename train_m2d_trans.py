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
import clip

import options.option_transformer_dance as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_train
from dataset import dataset_TM_eval
from dataset import dataset_MD
from dataset import dataset_tokenize_MD
import models.m2d_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper_dance import EvaluatorModelWrapper_Dance
import warnings
warnings.filterwarnings('ignore')
from exit.utils import get_model, visualize_2motions
from tqdm import tqdm
from exit.utils import get_model, visualize_2motions, generate_src_mask, init_save_folder, uniform, cosine_schedule
from einops import rearrange, repeat
import torch.nn.functional as F
from exit.utils import base_dir


"""
Inject music condition to interactive motion sequences
"""

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

# args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
init_save_folder(args)

# [TODO] make the 'output/' folder as arg
args.vq_dir = f'./output/vq/{args.vq_name}' #os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{args.vq_name}')
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


# from utils.word_vectorizer import WordVectorizer
# w_vectorizer = WordVectorizer('./glove', 'our_vab')
# val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer)

# NOTE(yiwen) use untokenized data
val_loader = dataset_MD.DATALoader(dataset_name=args.dataname,
                                    is_test=False, 
                                    batch_size=32,
                                    normalizer=None)

dataset_opt_path = '/home/xingqunqi/AI_dance/MMM/checkpoints/aistpp/opt.txt'
wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper_Dance(wrapper_opt) # TODO(yiwen) check this new wrapper

##### ---- Network ---- #####

#TODO(yw) add music encoder model initialization
########## NOTE(yw) These are borrowed from EDGE
from models.modules import MusicTransformerEncoder

musicFeatsEncoder = MusicTransformerEncoder(cond_feature_dim=35)     
###########

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code, # 8192
                       args.code_dim, # 32
                       args.output_emb_width, # 512
                       args.down_t, # 2
                       args.stride_t, # 2
                       args.width, # 512
                       args.depth, # 3
                       args.dilation_growth_rate) # 3


# TODO(yiwen) 
trans_encoder = trans.Music2Dance_Transformer(vqvae=net,
                                num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                music_dim=args.music_dim, #TODO music encoder output dim
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                num_local_layer=args.num_local_layer, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)


print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')

net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()
trans_encoder = torch.nn.DataParallel(trans_encoder)

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss(reduction='none')

##### ---- get code ---- #####
##### ---- Dataloader ---- #####

## NOTE(yiwen) offline converting motion sequence to codebook
if len(os.listdir(codebook_dir)) == 0:
    train_loader_token = dataset_MD.DATALoader(
                                    dataset_name=args.dataname,  
                                    batch_size=1,
                                    is_test=False) #TODO(yw) unit_length=2**args.down_t check unit_length here
    for batch in train_loader_token:
        pose, _, name, _ = batch # 1, 1, 148, 151
        # pose, name = batch
        bs, seq = pose.shape[0], pose.shape[2]
        pose = pose.cuda().float() # bs, nb_joints, joints_dim, seq_len
        target = net(pose, type='encode')
        target = target.cpu().numpy() # (1, 1, 37) 37 = 148(seq length)/4(unit_length)
        # TODO(yiwen) codebook 在加人数之后需要specific design
        prefix = name[0].split('/')[-1]
        np.save(pjoin(codebook_dir, prefix), target) 


# NOTE(yiwen) a dataloader that providing codebook data
train_loader = dataset_tokenize_MD.DATALoader(dataset_name=args.dataname, 
                                     is_test=False,
                                     batch_size=args.batch_size,
                                     codebook_size=args.nb_code, # 8192
                                     tokenizer_name=codebook_dir)

train_loader_iter = dataset_tokenize_MD.cycle(train_loader)

        
##### ---- Training ---- #####
best_fid=1000 
best_iter=0 
best_div=100 
best_top1=0 
best_top2=0 
best_top3=0 
best_matching=100 

# TODO(yiwen) the initialization of eval transformer, using raw data
pred_pose_eval, pose, m_length, music_feature, best_fid, best_iter, best_div, best_multi, writer, logger = eval_trans.evaluation_transformer_dance(args.out_dir, 
                                                                                                                                                   val_loader, 
                                                                                                                                                   net, 
                                                                                                                                                   trans_encoder, 
                                                                                                                                                   logger, 
                                                                                                                                                   writer, 
                                                                                                                                                   0, 
                                                                                                                                                   best_fid=1000, 
                                                                                                                                                   best_iter=0, 
                                                                                                                                                   best_div=100, 
                                                                                                                                                   music_encoder=musicFeatsEncoder, 
                                                                                                                                                   eval_wrapper=eval_wrapper)


def get_acc(cls_pred, target, mask):
    cls_pred = torch.masked_select(cls_pred, mask.unsqueeze(-1)).view(-1, cls_pred.shape[-1])
    target_all = torch.masked_select(target, mask)
    probs = torch.softmax(cls_pred, dim=-1)
    _, cls_pred_index = torch.max(probs, dim=-1)
    right_num = (cls_pred_index == target_all).sum()
    return right_num*100/mask.sum()


# while nb_iter <= args.total_iter:
for nb_iter in tqdm(range(1, args.total_iter + 1), position=0, leave=True):
    batch = next(train_loader_iter)
    music_feats, motion_token, motion_token_len = batch 
    # B, T, Mutok 128, 150, 35   B, H, T, Motok 128, 1, 1, 37   128  
    
    motion_token = motion_token.cuda()
    batch_size = motion_token.shape[0]
    target = motion_token.squeeze()  # (bs, 26) # NOTE(yiwen) should I squeeze here?
    target = target.cuda()
    max_len = target.shape[1] # TMutok 37

    # batch_size, max_len = target.shape[:2]

    ######### NOTE(yiwen) music features --> music embeddings
    # TODO(yiwen) Random Drop Music feats here
    # text_mask = np.random.random(len(clip_text)) > .05
    # clip_text = np.array(clip_text)
    # clip_text[~text_mask] = ''
    music_feats_emb = musicFeatsEncoder(music_feats) # B, T, Muemb 128, 150, 256

    ######### NOTE(yiwen) mask motion features
    # [INFO] Swap input tokens
    if args.pkeep == -1:
        proba = np.random.rand(1)[0] # random a probability
        mask = torch.bernoulli(proba * torch.ones(target.shape,
                                                device=target.device)) # randomly mask that much tokens
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(target.shape,
                                                device=target.device))
    # random only motion token (not pad token). To prevent pad token got mixed up.
    seq_mask_no_end = generate_src_mask(max_len, motion_token_len).to(target.device)
    mask = torch.logical_or(mask, ~seq_mask_no_end).int()
    r_indices = torch.randint_like(target, args.nb_code)
    input_indices = mask*target+(1-mask)*r_indices # 128, 1, 128, 37

    # Time step masking
    mask_id = get_model(net).vqvae.num_code + 2
    # rand_time = uniform((batch_size,), device = target.device)
    # rand_mask_probs = cosine_schedule(rand_time)
    rand_mask_probs = torch.zeros(batch_size, device = motion_token_len.device).float().uniform_(0.5, 1)
    # rand_mask_probs = cosine_schedule(rand_mask_probs)
    num_token_masked = (motion_token_len * rand_mask_probs).round().clamp(min = 1).to(target.device)
    seq_mask = generate_src_mask(max_len, motion_token_len+1) # no padding token, only motion token, so this mask is actually no use
    batch_randperm = torch.rand((batch_size, max_len), device = target.device) - seq_mask_no_end.int()
    batch_randperm = batch_randperm.argsort(dim = -1)
    mask_token = batch_randperm < rearrange(num_token_masked, 'b -> b 1') # 128, 37

    # masked_target = torch.where(mask_token, input=input_indices, other=-1)
    masked_input_indices = torch.where(mask_token, mask_id, input_indices) # 128, 1, 128, 37

    ####### NOTE(yiwen) should combine music and motion information here
    att_txt = None # CFG: torch.rand((seq_mask.shape[0], 1)) > 0.1
    sentence_style = music_feats_emb.mean(dim=1) # 128, 256

    cls_pred = trans_encoder(masked_input_indices, # 128, 1, 128, 37
                             sentence_style, # 128, 256 整个sentence所有frames的平均
                             src_mask = seq_mask, # 128, 37
                             att_txt=att_txt,
                             word_emb=music_feats_emb)[:, 1:] # 128, 150, 256 带着T的维度
                             # TODO(yiwen) DEBUG here temp, abandon the word_emb, 【or can use sliced music feats(along time dim)】

    ## NOTE(yiwen) librosa feature should be audio segment level (sentence level feature) --> music_feats_emb
    ## TODO(yiwen) add discrete music features(e.g. hubert) for better combining nearby information (word level feature)

    ###### NOTE(yiwen) 在music condition下，predict正确的codebook class
    # [INFO] Compute xent loss as a batch
    weights = seq_mask_no_end / (seq_mask_no_end.sum(-1).unsqueeze(-1) * seq_mask_no_end.shape[0])
    cls_pred_seq_masked = cls_pred[seq_mask_no_end, :].view(-1, cls_pred.shape[-1])
    target_seq_masked = target[seq_mask_no_end]
    weight_seq_masked = weights[seq_mask_no_end]
    loss_cls = F.cross_entropy(cls_pred_seq_masked, target_seq_masked, reduction = 'none')
    loss_cls = (loss_cls * weight_seq_masked).sum()

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

        writer.add_scalar('./Loss/all', loss_cls, nb_iter)
        writer.add_scalar('./ACC/every_token', right_seq_masked*100/seq_mask_no_end.sum(), nb_iter)
        
        # [INFO] log mask/nomask separately
        no_mask_token = ~mask_token * seq_mask_no_end
        writer.add_scalar('./ACC/masked', get_acc(cls_pred, target, mask_token), nb_iter)
        writer.add_scalar('./ACC/no_masked', get_acc(cls_pred, target, no_mask_token), nb_iter)

        # msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
        # logger.info(msg)

    if nb_iter==0 or nb_iter % args.eval_iter ==  0 or nb_iter == args.total_iter:
        num_repeat = 1
        rand_pos = False
        if nb_iter == args.total_iter:
            num_repeat = -30
            rand_pos = True
            val_loader = dataset_MD.DATALoader(args.dataname, True, 32)
        
        ## TODO(yiwen)
        ## pred_pose_eval, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper, dataname=args.dataname, num_repeat=num_repeat, rand_pos=rand_pos) 
        pred_pose_eval, pose, m_length, music_feature, best_fid, best_iter, best_div, best_multi, writer, logger = eval_trans.evaluation_transformer_dance(args.out_dir, 
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
                                                                                                                                                   eval_wrapper=eval_wrapper)

        # for i in range(4):
        #     x = pose[i].detach().cpu().numpy()
        #     y = pred_pose_eval[i].detach().cpu().numpy()
        #     l = m_length[i]
        #     caption = clip_text[i]
        #     cleaned_name = '-'.join(caption[:200].split('/'))

            # TODO(yiwen) check and update visualization here
            # visualize_2motions(x, val_loader.dataset.std, val_loader.dataset.mean, args.dataname, l, y, save_path=f'{args.out_dir}/html/{str(nb_iter)}_{cleaned_name}_{l}.html')

    if nb_iter == args.total_iter: 
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}"
        logger.info(msg_final)
        break            
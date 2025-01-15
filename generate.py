import torch
import clip
import models.vqvae as vqvae
from models.vqvae_sep import VQVAE_SEP
import models.m2d_trans as trans
import numpy as np
import glob
import os
import random
import pickle

from functools import cmp_to_key
from tqdm import tqdm
from pathlib import Path
from tempfile import TemporaryDirectory
from preprocess.aistpp.slice import slice_audio
# from exit.utils import visualize_2motions
from dataset.quaternion import ax_from_6v
from dataset.vis import skeleton_render, SMPLSkeleton
import options.option_transformer_dance as option_trans
from preprocess.aistpp.audio_extraction.baseline_features import extract as baseline_extract

## TODO(yiwen) adapt to m2d

##### ---- CLIP ---- #####
# clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
# clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
# clip_model.eval()
# for p in clip_model.parameters():
#     p.requires_grad = False

# # https://github.com/openai/CLIP/issues/111
# class TextCLIP(torch.nn.Module):
#     def __init__(self, model) :
#         super(TextCLIP, self).__init__()
#         self.model = model
        
#     def forward(self,text):
#         with torch.no_grad():
#             word_emb = self.model.token_embedding(text).type(self.model.dtype)
#             word_emb = word_emb + self.model.positional_embedding.type(self.model.dtype)
#             word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
#             word_emb = self.model.transformer(word_emb)
#             word_emb = self.model.ln_final(word_emb).permute(1, 0, 2).float()
#             enctxt = self.model.encode_text(text).float()
#         return enctxt, word_emb
# clip_model = TextCLIP(clip_model)

def extract_slice_number(filename):
    """Extract the numeric part of the slice from a filename."""
    return int(Path(filename).stem.split("_slice")[-1])


def compare_filenames(a, b):
    """Compare two filenames based on their base name and slice number."""
    base_name_a, slice_number_a = Path(a).stem.rsplit('_', 1)
    base_name_b, slice_number_b = Path(b).stem.rsplit('_', 1)

    # Compare the base names
    if base_name_a != base_name_b:
        return -1 if base_name_a < base_name_b else 1

    # Compare the slice numbers
    slice_a = extract_slice_number(a)
    slice_b = extract_slice_number(b)
    return -1 if slice_a < slice_b else (1 if slice_a > slice_b else 0)


sort_key = cmp_to_key(compare_filenames)

from models.modules import MusicTransformerEncoder
musicFeatsEncoder = MusicTransformerEncoder(cond_feature_dim=35)   
# music_feats_emb = musicFeatsEncoder(music_feats) TODO(yiwen) save to use later


def get_vqvae(args):
    return vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                        args.nb_code,
                        args.code_dim,
                        args.output_emb_width,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate)


def get_maskdecoder(args, vqvae):
    return trans.Music2Dance_Transformer(vqvae=vqvae,
                                num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                music_dim=args.music_dim, #TODO add to config
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                num_local_layer=args.num_local_layer, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)

class GroupDance(torch.nn.Module):
    def __init__(self, args=None):
        super().__init__()

        args.dataname = 'aistpp'

        self.vqvae = get_vqvae(args)
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        self.vqvae.load_state_dict(ckpt['net'], strict=True)
    
        # checkpoint = torch.load(args.resume_pth, map_location='cpu')
        # net = get_model(net)
        # net.load_state_dict(checkpoint['net'], strict=True)

        self.vqvae.eval()
        self.vqvae.cuda()

        self.maskdecoder = get_maskdecoder(args, self.vqvae)

        ckpt = torch.load(args.resume_trans, map_location='cpu')
        self.maskdecoder.load_state_dict(ckpt['trans'], strict=True)

        # checkpoint = torch.load(args.resume_trans, map_location='cpu')
        # trans_encoder = get_model(trans_encoder) 
        # trans_encoder.load_state_dict(checkpoint['trans'], strict=True)

        self.maskdecoder.eval()
        self.maskdecoder.cuda()

    def forward(self, music_feats, lengths=-1, rand_pos=True, num_ps=3):
        b = len(music_feats)
        seq = 148
        num_joints = 24  

        feature_dim = num_joints*6 + 3 + 4
        # feat_clip_text = clip.tokenize(text, truncate=True).cuda()
        # feat_clip_text, word_emb = clip_model(feat_clip_text)
        music_feats_emb = musicFeatsEncoder(music_feats).cuda()
        sentence_style = music_feats_emb.mean(dim=1)

        m_length = torch.tensor([148 for i in range(b)])
        m_tokens_len = torch.tensor([37 for i in range(b)])

        pred_len = m_length.cuda()
        pred_tok_len = m_tokens_len

        index_motion = self.maskdecoder(music_feature=sentence_style, 
                                 type="sample", 
                                 m_length=pred_len, 
                                 rand_pos=rand_pos, 
                                 CFG=-1,
                                 word_emb=music_feats_emb)

        # index_motion = self.maskdecoder(feat_clip_text, word_emb, type="sample", m_length=lengths, rand_pos=rand_pos, if_test=False)
        # m_token_length = torch.ceil((lengths)/4).int()
        # pred_pose_all = torch.zeros((b, 196, 263)).cuda()
        
        pred_pose_eval = torch.zeros((b, num_ps, seq, feature_dim)).cuda() 

        for k in range(b):
        ######### [INFO] Eval by m_length
            # NOTE(yiwen) use the decoder side of the pretrained codebook
            pred_pose = self.vqvae(index_motion[k:k+1, :int(pred_tok_len[k].item())], type='decode') # decode([1, 37])
            pred_pose = pred_pose[:,:num_ps,:,:feature_dim]
            # 1, 3, 148, 151 
            pred_pose_eval[k:k+1,:int(pred_len[k].item())] = pred_pose

        return pred_pose_eval

    

def get_stats(stats_path):
    with open(stats_path, "rb") as f:
        data = pickle.load(f)
    mean_loaded = data["mean"]
    std_loaded = data["std"]
    mean_tensor = torch.tensor(mean_loaded).view(1, 1, 1, -1)
    std_tensor = torch.tensor(std_loaded).view(1, 1, 1, -1)

    return mean_tensor, std_tensor


if __name__ == '__main__':
    args = option_trans.get_args_parser()
    '''
    CUDA_VISIBLE_DEVICES=0 python generate.py \
        --resume-pth './output/vq/2025-01-01-10-47-58_vq_dance2d_train/net_last.pth' \
        --resume-trans './output/m2d/2025-01-01-04-28-09_trans_2d/net_last.pth' \
        --music_dir './demos/group-dance-demo/resources/' \
        --cache_features \
        --feature_cache_dir '/home/xingqunqi/AI_dance/AI_dance/inference' \
        --use_cached_features
    '''
    group_dance = GroupDance(args).cuda()

    ### Process music input
    feature_func = baseline_extract
    sample_length = 5
    sample_size = int(sample_length / 2.5) - 1
    temp_dir_list = []
    all_cond = []
    all_filenames = []
    if args.use_cached_features:
        print("Using precomputed features")
        # all subdirectories
        dir_list = glob.glob(os.path.join(args.feature_cache_dir, "*/"))
        for dir in dir_list:
            file_list = sorted(glob.glob(f"{dir}/*.wav"), key=sort_key)
            feat_file_list = sorted(glob.glob(f"{dir}/*.npy"), key=sort_key)
            assert len(file_list) == len(feat_file_list)
            # random chunk after sanity check        
            rand_idx = random.randint(0, len(file_list) - sample_size)
            file_list = file_list[rand_idx : rand_idx + sample_size]
            feat_file_list = feat_file_list[rand_idx : rand_idx + sample_size]
            cond_list = [np.load(x) for x in feat_file_list]
            all_filenames.append(file_list)
            all_cond.append(torch.from_numpy(np.array(cond_list)))
        cond_list = torch.from_numpy(cond_list[0]).unsqueeze(0)

    else:
        print("Computing features for input music")
        for wav_file in glob.glob(os.path.join(args.music_dir, "*.wav")):
            # create temp folder (or use the cache folder if specified)
            if args.cache_features:
                songname = os.path.splitext(os.path.basename(wav_file))[0]
                save_dir = os.path.join(args.feature_cache_dir, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                temp_dir = TemporaryDirectory()
                temp_dir_list.append(temp_dir)
                dirname = temp_dir.name
            # slice the audio file
            print(f"Slicing {wav_file}")
            slice_audio(wav_file, 2.5, 5.0, dirname)
            file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=sort_key)
            rand_idx = random.randint(0, len(file_list) - sample_size)
            cond_list = []

            # generate juke representations
            print(f"Computing features for {wav_file}")
            for idx, file in enumerate(tqdm(file_list)):
                # if not caching then only calculate for the interested range
                if (not args.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                    continue
                reps, _ = feature_func(file)
                # save reps
                if args.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)
                # if in the random range, put it into the list of reps we want
                # to actually use for generation
                if rand_idx <= idx < rand_idx + sample_size:
                    cond_list.append(reps)
            cond_list = torch.from_numpy(np.array(cond_list))
            all_cond.append(cond_list)
            all_filenames.append(file_list[rand_idx : rand_idx + sample_size])


    music_feat_ls = all_cond
    stats_path = './checkpoints/aistpp/meta/mean_std.pkl'
    data_mean, data_std = get_stats(stats_path)

    # sample one slice from each music
    for mf in range(len(music_feat_ls)):
        music_feats = music_feat_ls[mf] # music feature for file mf
        filename = all_filenames[mf][0]
        prefix = filename[:-4].split('/')[-1] # filepath without suffix

        ### Estimate pose by inputing music feats to pretrained models
        pred_pose_eval = group_dance(music_feats, torch.tensor([args.length]).cuda(), rand_pos=False, num_ps=3)
        # 1, 3, 148, 151

        ### postprocess
        # TODO(yiwen) make it dim=3 temporarily
        data_mean = data_mean.to(pred_pose_eval.device)
        data_std = data_std.to(pred_pose_eval.device)

        video_flag_recons = True
        smpl = SMPLSkeleton(device='cuda:0')
        fk_out = './inference_out/pickle' # NOTE(yiwen) store .pkl for blender visualization

        pred_pose_eval = pred_pose_eval * data_std + data_mean  
        B, H, T, D = pred_pose_eval.shape
        
        # TODO(yiwen) check H 
        pred_pose_eval = pred_pose_eval.view(B*H, T, D) # TODO(yiwen) check blender rendering changes when H>1

        # NOTE (yiwen) unnormalized 6D-->3D This is for blender rendering
        root_pos_eval = pred_pose_eval[:,:,4:7]
        local_q_eval = pred_pose_eval[:,:,7:].view(root_pos_eval.shape[0], root_pos_eval.shape[1], -1, 6)
        local_q_eval_aa = ax_from_6v(local_q_eval) # 32, 148, 24, 3

        _, T, J, Dp = local_q_eval_aa.shape # TODO(yiwen) check blender rendering changes when H>1
        
        # TODO(yiwen) check inference all set B=1?
        # local_q_eval_aa = local_q_eval_aa.view(B, H, T, J, Dp)
        # root_pos_eval = root_pos_eval.view(B, H, T, 3)

        positions_recons = []

        for h in range(H):
            pos_recons = smpl.forward(local_q_eval_aa[h:(h+1)], root_pos_eval[h:(h+1)]) # 128, 148, 24, 3
            pos_recons = pos_recons.squeeze(0).detach().cpu()
            positions_recons.append(pos_recons)

        ### Save and visualize the results
        if video_flag_recons and fk_out is not None: 
            outname = f'{prefix}.pkl' #f'{nb_iter}_recons_{"_".join(os.path.splitext(os.path.basename(filenames[0]))[0].split("_")[:-1])}.pkl'
            Path(fk_out).mkdir(parents=True, exist_ok=True)
            pickle.dump(
                {
                    "smpl_poses": local_q_eval_aa.squeeze(0).reshape((H, -1, 72)).detach().cpu().numpy(),
                    "smpl_trans": root_pos_eval.squeeze(0).detach().cpu().numpy(),
                    "full_pose": np.stack(positions_recons), # 3, 148, 24, 3
                },
                open(os.path.join(fk_out, outname), "wb"),
            ) 


        # local_q_eval_aa = local_q_eval_aa.view(B, T, -1) # 32, 148, 72
        # pred_pose_eval_aa = torch.cat([root_pos_eval, local_q_eval_aa], dim=-1)

        # TODO(yiwen) only render the first person here
        if video_flag_recons:
            # render to gif, w/ sound
            skeleton_render(
                positions_recons[0], # 148, 24, 3
                epoch='0',
                out="./inference_out/render",
                name=all_filenames[mf], # list wav name
                sound=True, # bool
                stitch=True,
                sound_folder=args.music_dir,
                render=True
            )
            
        import pdb
        pdb.set_trace()
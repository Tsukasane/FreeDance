import torch
import models.vqvae as vqvae
import models.m2d_trans as trans
import numpy as np
import os
import pickle
from dataset.quaternion import ax_from_6v
from dataset.vis import SMPLSkeleton
import options.option_transformer_dance as option_trans

from dataset import dataset_MD_multi
from eval.cal_beat_scoresnew import cal_BAS_feats
from options.get_eval_option import get_opt
from models.evaluator_wrapper_dance import EvaluatorModelWrapper_Dance
from utils.eval_trans import calculate_activation_statistics, calculate_diversity, calculate_frechet_distance


os.environ["PYOPENGL_PLATFORM"] = "egl" # offscreen render
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from models.modules import MusicTransformerEncoder
musicFeatsEncoder = MusicTransformerEncoder(cond_feature_dim=35).to(device)


def get_vqvae(args):
    return vqvae.HumanVQVAE(args, 
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
                                fc_rate=args.ff_rate,
                                max_person=3) 

class FreeDance(torch.nn.Module):
    def __init__(self, args=None):
        super().__init__()

        self.vqvae = get_vqvae(args)
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        self.vqvae.load_state_dict(ckpt['net'], strict=True)
        self.vqvae.eval()
        self.vqvae.cuda()

        self.maskdecoder = get_maskdecoder(args, self.vqvae)
        ckpt = torch.load(args.resume_trans, map_location='cpu')
        self.maskdecoder.load_state_dict(ckpt['trans'], strict=True)

        self.maskdecoder.eval()
        self.maskdecoder.cuda()


    def forward(self, music_feats, num_person=None, rand_pos=True, num_ps=3):
        b = len(music_feats) # num of music = num of motion
        seq = 148
        num_joints = 24  
        feature_dim = num_joints*6 + 3 + 4
        
        music_feats_emb = musicFeatsEncoder(music_feats)

        m_length = torch.tensor([148 for i in range(b)])
        m_tokens_len = torch.tensor([37 for i in range(b)])
        pred_len = m_length.cuda()
        pred_tok_len = m_tokens_len

        # TODO(yiwen) use a dataloader to do batch inference, save the results to single person pkl
        index_motion = self.maskdecoder(type="sample", 
                                        m_length=pred_len, 
                                        rand_pos=rand_pos, 
                                        word_emb=music_feats_emb,
                                        real_num_person=num_person) #need a constrain of id range based on ps num
        
        pred_pose_eval = torch.zeros((b, num_ps, seq, feature_dim)).to(device)

        for k in range(b):
        ######### [INFO] Eval by m_length
            # NOTE(yiwen) use the decoder side of the pretrained codebook
            pred_pose = self.vqvae(index_motion[k:k+1, :int(pred_tok_len[k].item())], num_ps, type='decode') # decode([1, 37])
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
    '''
    save pred motion and eval
    '''
    # args
    args = option_trans.get_args_parser()
    args.dataname = 'aamixed'
    args.batch_size = 16
    if args.dataname == 'aamixed': 
        dataset_opt_path = 'checkpoints/aamixed/opt.txt' 
    elif args.dataname == 'aistpp':
        dataset_opt_path = 'checkpoints/aistpp/opt.txt' 
    elif args.dataname == 'aioz':
        dataset_opt_path = 'checkpoints/aioz/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper_Dance(wrapper_opt)
    motion_pred_list = []
    motion_annotation_list = []
    nb_sample = 0
    batch_cnt = 0
    save_dir = f'./dataset/{args.dataname}_dataset/test_for_eval2048'

    # score
    BAS_score = []
    diversity = 0
    fid = 0

    # stage
    stage1 = True
    stage2 = True

    if stage1:
        # model
        freedance = FreeDance(args).to(device)
        smpl = SMPLSkeleton(device='cuda:0')

        # data
        stats_path = f'./checkpoints/{args.dataname}/meta/mean_std.pkl'
        data_mean, data_std = get_stats(stats_path)
        data_mean = data_mean.to(device)
        data_std = data_std.to(device)
        test_loader = dataset_MD_multi.DATALoader(dataset_name=args.dataname,
                                        data_split='test', 
                                        batch_size=args.batch_size,
                                        normalizer=None,
                                        max_person_num=3)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f'{save_dir}/gt_aa', exist_ok=True) # for fid
        os.makedirs(f'{save_dir}/pred_aa', exist_ok=True) # for fid and div

        for batch in test_loader:
            batch_cnt+=1

            if batch_cnt%50==0:
                print(f'Processed {batch_cnt * args.batch_size} samples')
            motion, music_feats, filenames, wavs, num_person = batch 
            
            ## gt, only for distribution calculation
            motion = motion.to(device) # 32, 1, 148, 151
            B, H, T, D = motion.shape
            unnormalized_motion = motion * data_std + data_mean
            motion_copy = unnormalized_motion.view(B*H, T, D) # BH, 148, 151
            root_pos_gt = motion_copy[:,:,4:7] # BH, 148, 3
            local_q_gt = motion_copy[:,:,7:].view(root_pos_gt.shape[0], root_pos_gt.shape[1], -1, 6) # 32, 148, 24, 6
            local_q_gt_aa = ax_from_6v(local_q_gt) # BH, 148, 24, 3
            BH, T, J, D = local_q_gt_aa.shape
            positions_gt = smpl.forward(local_q_gt_aa, root_pos_gt) # 128, 148, 24, 3
            local_q_gt_aa = local_q_gt_aa.view(BH, T, -1) # 32, 148, 72
            pose_gt_aa = torch.cat([root_pos_gt, local_q_gt_aa], dim=-1) # BH, T, 75
            
            pose_gt_aa = pose_gt_aa.reshape(B, H, T, 75)
            for n_id in range(B):
                for h_id in range(num_person[n_id]):
                    filename = os.path.splitext(filenames[n_id])[0].split('/')[-1] + f'_ps{h_id+1}' + '.npy'
                    save_aa = pose_gt_aa[n_id, h_id, :,:].detach().cpu().numpy() # T=148, 75
                    np.save(os.path.join(save_dir, 'gt_aa', filename), save_aa)

            ## pred
            music_feats = music_feats.to(device)
            pred_pose_eval = freedance(music_feats, num_person, rand_pos=False, num_ps=3) # 1, 3, 148, 151
            pred_pose_eval = pred_pose_eval * data_std + data_mean  
            B, H, T, D = pred_pose_eval.shape
            pred_pose_eval = pred_pose_eval.view(B*H, T, D)
            root_pos_eval = pred_pose_eval[:,:,4:7]
            local_q_eval = pred_pose_eval[:,:,7:].view(root_pos_eval.shape[0], root_pos_eval.shape[1], -1, 6)
            local_q_eval_aa = ax_from_6v(local_q_eval) # 32, 148, 24, 3
            BH, T, J, Dp = local_q_eval_aa.shape 
            
            
            
            positions_recons = smpl.forward(local_q_eval_aa, root_pos_eval).detach().cpu() # 128, 148, 24, 3
            # positions_recons = positions_recons.view(B, H, T, J, D)

            # cal scores
            beat_score = cal_BAS_feats(positions_recons.view(B, H, T, J, Dp), num_person, music_feats, wavs)
            BAS_score.extend(beat_score)
            local_q_eval_aa = local_q_eval_aa.view(BH, T, -1) # BH, 148, 72
            pred_pose_eval_aa = torch.cat([root_pos_eval, local_q_eval_aa], dim=-1)
            
            pred_pose_eval_aa = pred_pose_eval_aa.reshape(B, H, T, 75)
            for n_id in range(B):
                for h_id in range(num_person[n_id]):
                    filename = os.path.splitext(filenames[n_id])[0].split('/')[-1] + f'_ps{h_id+1}' + '.npy'
                    save_aa = pred_pose_eval_aa[n_id, h_id, :,:].detach().cpu().numpy() # T, 75
                    np.save(os.path.join(save_dir, 'pred_aa', filename), save_aa)

        all_BAS = np.mean(BAS_score)
        print(f'The final bas result is: {all_BAS}') 
        with open(os.path.join(save_dir, "output.txt"), "a", encoding="utf-8") as file:
            file.write(f"BAS: {all_BAS}")


    if stage2:
        dataset_root = save_dir
        gt_motion = os.listdir(os.path.join(dataset_root,'gt_aa'))
        pred_motion = os.listdir(os.path.join(dataset_root,'pred_aa'))
        gt_motion.sort()
        pred_motion.sort()

        assert len(gt_motion) == len(pred_motion)

        for mf in range(len(pred_motion)):
            if mf%1000==0:
                print(f'{mf} data here.')
            pose_gt_aa = np.load(os.path.join(dataset_root,'gt_aa', gt_motion[mf])) # np
            pred_pose_eval_aa = np.load(os.path.join(dataset_root,'pred_aa', pred_motion[mf])) # np
            
            _, em = eval_wrapper.get_co_embeddings(motions=torch.tensor(pose_gt_aa).unsqueeze(0)) # use only pose relevant dim to calculate fid
            _, em_pred = eval_wrapper.get_co_embeddings(motions=torch.tensor(pred_pose_eval_aa).unsqueeze(0))

            motion_annotation_list.append(em) 
            motion_pred_list.append(em_pred) # B, 512
    
        motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
        motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
        gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
        mu, cov= calculate_activation_statistics(motion_pred_np)
        diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    
        print(f'FID: {fid}, Div: {diversity}')
        with open(os.path.join(save_dir, "output.txt"), "a", encoding="utf-8") as file:
            file.write(f"\nFID: {fid} \nDiv: {diversity}")

    with open(os.path.join(save_dir, "output.txt"), "a", encoding="utf-8") as file:
        file.write(f"\n{args.resume_trans}")

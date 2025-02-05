import os
import numpy as np
import torch
from scipy import linalg
from utils.motion_process import recover_from_ric
from exit.utils import get_model, visualize_2motions, generate_src_mask
from dataset.quaternion import ax_from_6v
from dataset.vis import skeleton_render, SMPLSkeleton
from pathlib import Path
import pickle
from eval.calculate_scores import extract_features_multi, calculate_FID_DIST
from eval.calculate_beat_scores import cal_BAS


@torch.no_grad()        
def evaluation_vqvae_dance(out_dir, 
                           val_loader, 
                           net, 
                           logger, 
                           writer, 
                           nb_iter, 
                           best_fid, 
                           best_iter, 
                           best_div, 
                           eval_wrapper, 
                           dataset_name = 'aamixed',
                           draw = True) : 
    net.eval()

    smpl = SMPLSkeleton(device='cuda:0')

    # normalize predicted motion (for cal fid later)
    data_mean = val_loader.dataset.mean
    data_std = val_loader.dataset.std

    motion_annotation_list = []
    motion_pred_list = []

    nb_sample = 0

    results_features_dic = {"kinetic": [], "manual": []}
    results_features_dic_gt = {"kinetic": [], "manual": []}
    cnt = 0 # NOTE(yiwen) here use a subset of val to show the trend, but will use full set for eval.
    avg_l2_distance = 0
    for batch in val_loader: 
        cnt+=1
        if cnt>=10:
            break
        motion, music_feats, filenames, wavs, num_person = batch # normalized 6d motion
        
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

        pose_gt_aa = torch.cat([root_pos_gt, local_q_gt_aa], dim=-1) # BH, T, 75
        et, em = eval_wrapper.get_co_embeddings(music_feats, pose_gt_aa) # use only pose relevant dim to calculate fid

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
                pose = motion[i:i+1, :, :, :].detach().cpu().numpy()
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
        
        local_q_eval_aa = local_q_eval_aa.view(BH, T, -1) # BH, 148, 72
        pred_pose_eval_aa = torch.cat([root_pos_eval, local_q_eval_aa], dim=-1)
        et_pred, em_pred = eval_wrapper.get_co_embeddings(music_feats, pred_pose_eval_aa)

        motion_pred_list.append(em_pred) # 32, 512
        motion_annotation_list.append(em) 

        nb_sample += bs

        
        # L2 joint norm
        l2_distance = torch.norm(positions_recons - positions_gt, dim=-1) # BH, T, J (padding also includes)
        avg_l2_distance += l2_distance.mean() 

        # NOTE(yiwen) new FID DIST metrics in eval
        results_features = extract_features_multi(positions_recons.view(B, H, T, J, D), num_person)
        results_features_dic['kinetic'].extend(results_features['kinetic'])
        results_features_dic['manual'].extend(results_features['manual'])

    # NOTE(yiwen) motion eval metrics based on AE
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)
    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)


    # NOTE(yiwen) dance eval metrics based on kinetic and geometry features
    FID_k, FID_g, Dist_k, Dist_g = calculate_FID_DIST(results_features_dic, dataset_name) # output the scores
    FID_k_gt, FID_g_gt, Dist_k_gt, Dist_g_gt = calculate_FID_DIST(results_features_dic_gt, dataset_name)
    avg_l2_distance /= cnt
    
    msg = f"--> \t Eva. Iter {nb_iter} :, \n\
                FID_ae. {fid:.4f}, \n\
                Div_real. {diversity_real:.4f}, Div. {diversity:.4f},\n\
                FID_k. {FID_k:.4f}, \n\
                FID_g. {FID_g:.4f}, \n\
                Dist_k. {Dist_k:.4f},  Dist_k_refs. {Dist_k_gt:.4f},\n\
                Dist_g. {Dist_g:.4f},  Dist_g_refs. {Dist_g_gt:.4f},\n\
                Average_Joint_L2. {avg_l2_distance:.4f}"
    logger.info(msg)
    
    if draw:
        writer.add_scalar('./Test/FID_ae', fid, nb_iter)
        writer.add_scalar('./Test/Div_real', diversity_real, nb_iter)
        writer.add_scalar('./Test/Div', diversity, nb_iter)

        writer.add_scalar('./Test/FID_k', FID_k, nb_iter)
        writer.add_scalar('./Test/FID_g', FID_g, nb_iter)
        writer.add_scalar('./Test/Dist_k', Dist_k, nb_iter)
        writer.add_scalar('./Test/Dist_g', Dist_g, nb_iter)
        writer.add_scalar('./Test/Average_Joint_L2', avg_l2_distance, nb_iter)
    
    if fid < best_fid : 
        msg = f"--> --> \t FID_ae Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        # save the checkpoint only for inference
        torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if abs(Dist_k - Dist_k_gt) < best_div: # the difference
        msg = f"--> --> \t Dist_k difference decreased from {best_div:.5f} to {abs(Dist_k - Dist_k_gt):.5f} !!!"
        logger.info(msg)
        best_div = abs(Dist_k - Dist_k_gt)
        # save the checkpoint only for inference
        torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_dist.pth'))
    
    # if save:
    #     # torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))
    #     checkpoint = {
    #         'trans': get_model(trans).state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'scheduler': scheduler.state_dict(),
    #         'iters': nb_iter,
    #     }
    #     torch.save(checkpoint, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_iter, best_div, writer, logger



@torch.no_grad()        
def evaluation_transformer_dance(out_dir, 
                                 val_loader, 
                                 net, 
                                 trans, 
                                 logger, 
                                 writer, 
                                 nb_iter, 
                                 best_fid, 
                                 best_iter, 
                                 best_div, 
                                 music_encoder,
                                 eval_wrapper, 
                                #  dataname='aistpp', 
                                 draw = True, 
                                #  save = True, 
                                #  savegif=False, 
                                 num_repeat=1, 
                                 rand_pos=False,
                                 exp_name='trans_multi') : 
    if num_repeat < 0:
        is_avg_all = True
        num_repeat = -num_repeat
    else:
        is_avg_all = False


    trans.eval()
    nb_sample = 0

    motion_annotation_list = []
    motion_pred_list = []

    nb_sample = 0
    blank_id = get_model(trans).num_vq

    # normalize predicted motion (for cal fid later)
    data_mean = val_loader.dataset.mean
    data_std = val_loader.dataset.std

    video_flag_gt = True
    video_flag_recons = True
    smpl = SMPLSkeleton(device='cuda:0')
    results_features_dic = {"kinetic": [], "manual": []}
    results_features_dic_gt = {"kinetic": [], "manual": []}
    batch_BAS = []
    cnt = 0

    fk_out = f'/data/xingqunqi/AI_dance/Group_Dance_output/output/fk_out_{exp_name}' # NOTE(yiwen) store .pkl for blender visualization
    for batch in val_loader:
        cnt+=1
        if cnt>=10: # NOTE(yiwen) here use a subset of val to show the trend, but will use full set for eval.
            break

        motion, music_feats, filenames, wavs, num_person = batch # normalized 6d motion
        
        motion = motion.cuda() # 32, 1, 148, 151
        B, H, T, D = motion.shape
        
        data_std = data_std.to(motion.device)
        data_mean = data_mean.to(motion.device)
        
        ########### NOTE(yiwen) unnormalize gt 6d-->3d, getting feature distribution
        unnormalized_motion = motion * data_std + data_mean
        motion_copy = unnormalized_motion.view(B*H, T, D) # BH, 148, 151
        root_pos_gt = motion_copy[:,:,4:7] # BH, 148, 3
        local_q_gt = motion_copy[:,:,7:].view(root_pos_gt.shape[0], root_pos_gt.shape[1], -1, 6) # 32, 148, 24, 6
        local_q_gt_aa = ax_from_6v(local_q_gt) # BH, 148, 24, 3

        BH, T, J, D = local_q_gt_aa.shape

        positions_gt = smpl.forward(local_q_gt_aa, root_pos_gt) # 128, 148, 24, 3
        
        # NOTE(yiwen) new FID DIST metrics in eval
        results_features_gt = extract_features_multi(positions_gt.view(B, H, T, J, D), num_person)
        results_features_dic_gt['kinetic'].extend(results_features_gt['kinetic'])
        results_features_dic_gt['manual'].extend(results_features_gt['manual'])

        if video_flag_gt and fk_out is not None:
            outname = f'{nb_iter}_gt_{"_".join(os.path.splitext(os.path.basename(filenames[0]))[0].split("_")[:-1])}.pkl'
            Path(fk_out).mkdir(parents=True, exist_ok=True)

            # TODO(yiwen) reorganize the dim
            pickle.dump(
                {
                    "smpl_poses": local_q_gt_aa.squeeze(0).reshape((BH*T, 72)).cpu().numpy(), # BHT, 72, B=1 --> save them in three different pkls (H*  B, T, 72)
                    "smpl_trans": root_pos_gt.squeeze(0).cpu().numpy(), # 128, 148, 24, 3
                    "full_pose": positions_gt[0], # TODO(yiwen) check here
                },
                open(os.path.join(fk_out, outname), "wb"),
            ) 

        local_q_gt_aa = local_q_gt_aa.view(BH, T, -1) # 32, 148, 72
        pose_gt_aa = torch.cat([root_pos_gt, local_q_gt_aa], dim=-1) # BH, T, 75

        if video_flag_gt:
            # render to gif, w/ sound
            skeleton_render( 
                positions_gt[0:3], # TODO(yiwen) the input should be H, 148, 24, 3, make it to --> # 148, 24, 3
                epoch=f"{nb_iter}",
                out=f"/data/xingqunqi/AI_dance/Group_Dance_output/output/renders_gt_{exp_name}",
                name=filenames, # list wav name
                sound=True, # bool
                stitch=True,
                render=True
            )
            video_flag_gt = False

        _, em = eval_wrapper.get_co_embeddings(music_feats, pose_gt_aa) # use only pose relevant dim to calculate fid

        ########### NOTE(yiwen) predict motion using normalized 6d
        bs, num_ps, seq = motion.shape[0], motion.shape[1], motion.shape[2] # B, H, T
        if motion.shape[-1] == 251:
            num_joints = 21 
        elif motion.shape[-1] == 263:
            num_joints = 22
        else:
            num_joints = 24  
        feature_dim = num_joints*6 + 3 + 4

        music_feats_emb = music_encoder(music_feats)
        # sentence_style = music_feats_emb.mean(dim=1)

        # m_tokens_len = torch.ceil((m_length)/4)
        m_length = torch.tensor([148 for i in range(motion.shape[0])])
        m_tokens_len = torch.tensor([37 for i in range(motion.shape[0])])

        pred_len = m_length.cuda()
        pred_tok_len = m_tokens_len

        with torch.no_grad():
            for i in range(num_repeat):
                pred_pose_eval = torch.zeros((bs, num_ps, seq, feature_dim)).cuda() #NOTE(yiwen) only use valid H to eval
                index_motion = trans(type="sample", 
                                    m_length=pred_len, 
                                    rand_pos=rand_pos, 
                                    word_emb=music_feats_emb)
                # 32, 50

                # [INFO] 1. this get the last index of blank_id
                # pred_length = (index_motion == blank_id).int().argmax(1).float()
                # [INFO] 2. this get the first index of blank_id
                pred_length = (index_motion >= blank_id).int()
                pred_length = torch.topk(pred_length, k=1, dim=1).indices.squeeze().float()
                
                for k in range(bs):
                    # NOTE(yiwen) use the decoder side of the pretrained codebook
                    pred_pose = net(index_motion[k:k+1, :int(pred_tok_len[k].item())], num_person, type='decode') # decode([1, 37])
                    pred_pose = pred_pose[:,:num_ps,:,:motion.shape[-1]]
                    # 1, 3, 148, 151 

                    pred_pose_eval[k:k+1,:int(pred_len[k].item())] = pred_pose

                pred_pose_eval = pred_pose_eval * data_std + data_mean  
                B, H, T, D = pred_pose_eval.shape
                pred_pose_eval = pred_pose_eval.view(B*H, T, D) # TODO(yiwen) check blender rendering changes when H>1
                
                ########### NOTE (yiwen) unnormalized 6D-->3D This is for blender rendering
                root_pos_eval = pred_pose_eval[:,:,4:7]
                local_q_eval = pred_pose_eval[:,:,7:].view(root_pos_eval.shape[0], root_pos_eval.shape[1], -1, 6)
                local_q_eval_aa = ax_from_6v(local_q_eval) # 32, 148, 24, 3
                
                BH, T, J, D = local_q_eval_aa.shape # TODO(yiwen) check blender rendering changes when H>1

                positions_recons = smpl.forward(local_q_eval_aa, root_pos_eval) # 128, 148, 24, 3

                # NOTE(yiwen) new FID DIST metrics in eval
                results_features = extract_features_multi(positions_recons.view(B, H, T, J, D), num_person)
                
                results_features_dic['kinetic'].extend(results_features['kinetic'])
                results_features_dic['manual'].extend(results_features['manual'])

                batch_BAS.append(cal_BAS(positions_recons.view(B, H, T, J, D), num_person, music_feats))

                if video_flag_recons and fk_out is not None: 
                    outname = f'{nb_iter}_recons_{"_".join(os.path.splitext(os.path.basename(filenames[0]))[0].split("_")[:-1])}.pkl'
                    Path(fk_out).mkdir(parents=True, exist_ok=True)
                    
                    # TODO(yiwen) reorganize the dim
                    pickle.dump(
                        {
                            "smpl_poses": local_q_eval_aa.squeeze(0).reshape((BH*T, 72)).cpu().numpy(),
                            "smpl_trans": root_pos_eval.squeeze(0).cpu().numpy(),
                            "full_pose": positions_recons[0],
                        },
                        open(os.path.join(fk_out, outname), "wb"),
                    ) 

                local_q_eval_aa = local_q_eval_aa.view(BH, T, -1) # 32, 148, 72
                pred_pose_eval_aa = torch.cat([root_pos_eval, local_q_eval_aa], dim=-1)

                if video_flag_recons:
                    # render to gif, w/ sound
                    skeleton_render(
                        positions_recons[0:3], # 148, 24, 3
                        epoch=f"{nb_iter}",
                        out=f"/data/xingqunqi/AI_dance/Group_Dance_output/output/renders_recons_{exp_name}",
                        name=filenames, # list wav name
                        sound=True, # bool
                        stitch=True,
                        render=True
                    )
                    video_flag_recons = False

                _, em_pred = eval_wrapper.get_co_embeddings(music_feats, pred_pose_eval_aa)
            
                if i == 0 or is_avg_all:
                    motion = motion.cuda().float()
                    
                    _, em = eval_wrapper.get_co_embeddings(music_feats, pose_gt_aa)
                    motion_annotation_list.append(em)
                    motion_pred_list.append(em_pred)

                    nb_sample += bs

    # NOTE(yiwen) dance eval metrics based on kinetic and geometry features
    FID_k, FID_g, Dist_k, Dist_g = calculate_FID_DIST(results_features_dic)
    FID_k_gt, FID_g_gt, Dist_k_gt, Dist_g_gt = calculate_FID_DIST(results_features_dic_gt)

    all_BAS = sum(batch_BAS) / cnt

    # NOTE(yiwen) motion eval metrics based on AE
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)
    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, \n\
                FID_ae. {fid:.4f}, \n\
                Div_real. {diversity_real:.4f}, Div. {diversity:.4f},\n\
                FID_k. {FID_k:.4f} , \n\
                FID_g. {FID_g:.4f} , \n\
                Dist_k. {Dist_k:.4f},  Dist_k_refs. {Dist_k_gt:.4f},\n\
                Dist_g. {Dist_g:.4f},  Dist_g_refs. {Dist_g_gt:.4f},\n\
                Diversity. {diversity:.4f}, \n\
                BAS. {all_BAS:.4f},"
    logger.info(msg)

    # NOTE(yiwen) FID lower better; Dist similar to gt better; Diversity higher better
    if draw:
        writer.add_scalar('./Test/FID_ae', fid, nb_iter)
        writer.add_scalar('./Test/Div_real', diversity_real, nb_iter)
        writer.add_scalar('./Test/Div', diversity, nb_iter)

        writer.add_scalar('./Test/FID_k', FID_k, nb_iter)
        writer.add_scalar('./Test/FID_g', FID_g, nb_iter)
        writer.add_scalar('./Test/Dist_k', Dist_k, nb_iter)
        writer.add_scalar('./Test/Dist_g', Dist_g, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/BAS', all_BAS, nb_iter)
    
    if fid < best_fid : 
        msg = f"--> --> \t FID_ae Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        # save the checkpoint only for inference
        torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))
    
    if diversity > best_div: 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity
        # save the checkpoint only for inference
        torch.save({'trans' : get_model(trans).state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    trans.train()
    return pred_pose_eval, motion, m_length, music_feats, best_fid, best_iter, best_div, writer, logger



# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_multimodality(activation, multimodality_times): # TODO(yiwen) remove this
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist
import torch
import models.vqvae as vqvae
import models.m2d_trans as trans
import numpy as np
import glob
import os
import pickle

from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
from dataset.quaternion import ax_from_6v
from dataset.vis import skeleton_render, SMPLSkeleton
import options.option_transformer_dance as option_trans
from preprocess.aistpp.audio_extraction.baseline_features import extract as baseline_extract

import matplotlib.pyplot as plt
from smplx import SMPL
import pyrender
import trimesh
import cv2
import os
import librosa as lr
import soundfile as sf

os.environ["PYOPENGL_PLATFORM"] = "egl" # offscreen render

# SMPL model
smpl_model_path = 'dataset/AIST++_dataset/SMPL_models/smpl/SMPL_FEMALE.pkl'
SMPL_model = SMPL(model_path=smpl_model_path, gender='female')  # gender: male, female, neutral
smpl_faces = SMPL_model.faces

from models.modules import MusicTransformerEncoder
musicFeatsEncoder = MusicTransformerEncoder(cond_feature_dim=35)   


def multi_mesh_render(all_mesh, colors=None, save_name=None, music_name=None, stitch=True):
    """
    Render multiple 3D meshes with different colors.

    Args:
        meshes (list of trimesh.Trimesh): List of 3D meshes to render.
        colors (list of tuples): List of RGB colors (0-255) for each mesh.
    """
    scene = pyrender.Scene(ambient_light=np.array([0.3, 0.3, 0.3, 1.0]))

    if colors is None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)] 

    # light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=np.eye(4))

    light = pyrender.PointLight(color=np.ones(3), intensity=10.0)
    scene.add(light, pose=np.array([[1, 0, 0, 1],  # (1, 1, 1)
                                [0, 1, 0, 2],
                                [0, 0, 1, 2],
                                [0, 0, 0, 1]]))
    
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)  # perspective angle
    camera_distance = 1.5 
    center = [0,0,0]
    # z+up，y+back
    camera_pose = np.array([
        [1.0, 0.0, 0.0, center[0]],  
        [0.0, 1.0, 0.0, center[1] - camera_distance/4], 
        [0.0, 0.0, 1.0, center[2] + camera_distance * 5/2],  
        [0.0, 0.0, 0.0, 1.0]
    ])
   
    angle = np.radians(60)  # angle --> radians
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # rotate around X-axis
    rotation_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],           
        [0.0, cos_angle, -sin_angle, 0.0],  # switch y&z
        [0.0, sin_angle, cos_angle, 0.0],   # switch y&z
        [0.0, 0.0, 0.0, 1.0]        
    ])
    camera_pose = np.dot(rotation_matrix, camera_pose)
    scene.add(camera, pose=camera_pose)

    ground = trimesh.creation.box(extents=(5, 5, 0.01))
    ground.visual.vertex_colors = [128, 128, 128, 255]
    ground_mesh = pyrender.Mesh.from_trimesh(ground)

    video_fps = 30    
    video_size = (800, 600)  
    
    out_dir = "./results/mesh_ws"
    os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 encode
    video_writer = cv2.VideoWriter(f'{save_name}.mp4', fourcc, video_fps, video_size)
    renderer = pyrender.OffscreenRenderer(viewport_width=video_size[0], viewport_height=video_size[1])

    for meshes in all_mesh:
        scene.add(ground_mesh, pose=np.array([
                                    [1, 0, 0, 0], # x- left
                                    [0, 1, 0, 3.0], # y- front
                                    [0, 0, 1, -1.2], # z- down
                                    [0, 0, 0, 1]]))
        for i, mesh in enumerate(meshes):
            color = colors[i % len(colors)]
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=np.array([color[0]/255, color[1]/255, color[2]/255, 1.0])
            )
            mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh_pyrender)

        color, _ = renderer.render(scene)

        # RGB->BGR for cv2
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        video_writer.write(color_bgr)

        # clean meshes for one frame
        for node in list(scene.mesh_nodes):
            scene.remove_node(node)

    video_writer.release()
    renderer.delete()

    if stitch:
        temp_dir = TemporaryDirectory()
        assert type(music_name) == list  # must be a list of names to do stitching
        nametemp_ = [os.path.splitext(x)[0] + ".wav" for x in music_name]
        name_ = [x.replace("baseline_feats", "wavs_sliced") for x in nametemp_]
        audio, sr = lr.load(name_[0], sr=None)
        ll, half = len(audio), len(audio) // 2
        total_wav = np.zeros(ll + half * (len(name_) - 1))
        total_wav[:ll] = audio
        idx = ll
        for n_ in name_[1:]:
            audio, sr = lr.load(n_, sr=None)
            total_wav[idx : idx + half] = audio[half:]
            idx += half

        # save a dummy spliced audio
        audioname = f"{temp_dir.name}/tempsound.wav" 
        sf.write(audioname, total_wav, sr)

        # filename cannot start with '-'
        outname = music_name[0].split('/')[-1][:-4]+'.mp4'
        outpath = os.path.join(out_dir,outname)
    
        out = os.system(
            # f"ffmpeg -loglevel error -y -i {f'{save_name}.mp4'} -i {audioname} -shortest -c:v copy -c:a aac -q:a 4 {outname}"
            f"ffmpeg -loglevel error -stream_loop 0 -y -i {f'{save_name}.mp4'} -i {audioname} -shortest -vb 20M -vcodec mpeg4 -c:a aac -q:a 4 {outpath}"
        )


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
    args.block_size = args.block_size  # * args.max_person # FIXME(yiwen) unify in different version model extend the block size
    return trans.Music2Dance_Transformer(vqvae=vqvae,
                                num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                music_dim=args.music_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                num_local_layer=args.num_local_layer, 
                                n_head=args.n_head_gpt,
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate,
                                max_person=args.max_person)

class FreeDance(torch.nn.Module):
    def __init__(self, args=None):
        super().__init__()

        args.dataname = 'aamixed'
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
        self.num_person = args.num_person


    def forward(self, music_feats, lengths=-1, rand_pos=True, num_ps=3):
        b = len(music_feats) # num of music = num of motion
        seq = 148
        num_joints = 24  
        feature_dim = num_joints*6 + 3 + 4
        music_feats_emb = musicFeatsEncoder(music_feats).cuda()

        m_length = torch.tensor([148 for i in range(b)])
        m_tokens_len = torch.tensor([37 for i in range(b)])
        pred_len = m_length.cuda()
        pred_tok_len = m_tokens_len

        index_motion = self.maskdecoder(type="sample", 
                                        m_length=pred_len, 
                                        rand_pos=rand_pos, 
                                        mus_emb=music_feats_emb,
                                        real_num_person=self.num_person) #need a constrain of id range based on ps num
        
        pred_pose_eval = torch.zeros((b, num_ps, seq, feature_dim)).cuda() 

        for k in range(b):
            pred_pose = self.vqvae(index_motion[k:k+1, :int(pred_tok_len[k].item())], num_ps, type='decode') # decode([1, 37])
            pred_pose = pred_pose[:,:num_ps,:,:feature_dim] # 1, 3, 148, 151 
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
    Inference motion from custom music 
    The results will be saved to
        - results/mesh
        - results/pickle
        - results/skeleton
        - results/mesh_ws (mesh with sound)
    '''

    args = option_trans.get_args_parser()

    # NOTE(yiwen) len >= num music files
    args.num_person = [3 for i in range(10)] 

    # NOTE(yiwen) must specify at least one of them
    music_dir = "./../sample_music"
    music_feats_dir = "dataset/AIOZ_Gdance_dataset/test/baseline_feats"

    # music_dir = args.music_dir
    # music_feats_dir = args.feature_cache_dir

    print(f'Num_person: {args.num_person}')

    freedance = FreeDance(args).cuda()

    ### Process music input
    feature_func = baseline_extract
    all_cond = []
    all_filenames = []
    stats_path = './checkpoints/aamixed/meta/mean_std.pkl'
    data_mean, data_std = get_stats(stats_path)


    if args.use_cached_features and music_feats_dir!=None:
        print(f"Use precomputed features")
        for feats_file in glob.glob(os.path.join(music_feats_dir, "*.npy")):   
            cond_list = []
            reps = np.load(feats_file) # (150, 35)
            cond_list.append(reps)

            cond_list = torch.from_numpy(np.array(cond_list))
            all_cond.append(cond_list) 
            all_filenames.append(feats_file) # sample the same range from audio

    else:
        print("Computing features for input music")
        for wav_file in glob.glob(os.path.join(music_dir, "*.wav")):   
            cond_list = []
            reps, _ = feature_func(wav_file)  
            cond_list.append(reps)

            cond_list = torch.from_numpy(np.array(cond_list))
            all_cond.append(cond_list)
            all_filenames.append(wav_file) # sample the same range from audio

    music_feat_ls = all_cond


    ### sample one slice from each music
    for mf in range(len(music_feat_ls)):
        music_feats = music_feat_ls[mf] # music feature for file mf
        filename = all_filenames[mf]
        prefix = filename[:-4].split('/')[-1] # filepath without suffix

        # music_feats --> dance_seq
        pred_pose_eval = freedance(music_feats, torch.tensor([args.length]).cuda(), rand_pos=False, num_ps=args.max_person)
        # 1, 3, 148, 151

        # postprocess
        data_mean = data_mean.to(pred_pose_eval.device)
        data_std = data_std.to(pred_pose_eval.device)

        video_flag_recons = True
        smpl = SMPLSkeleton(device='cuda:0')
        fk_out = './results/pickle' 
        os.makedirs(fk_out, exist_ok=True)

        pred_pose_eval = pred_pose_eval * data_std + data_mean  
        B, H, T, D = pred_pose_eval.shape
        pred_pose_eval = pred_pose_eval.view(B*H, T, D) 
        root_pos_eval = pred_pose_eval[:,:,4:7]
        local_q_eval = pred_pose_eval[:,:,7:].view(root_pos_eval.shape[0], root_pos_eval.shape[1], -1, 6)
        local_q_eval_aa = ax_from_6v(local_q_eval) # 32, 148, 24, 3

        BH, T, J, Dp = local_q_eval_aa.shape 

        positions_recons = smpl.forward(local_q_eval_aa, root_pos_eval).detach().cpu() # 3, 148, 24, 3


        ### Save the pkl results for blender
        print(f'Saving pkl for {all_filenames[mf]}')
        if video_flag_recons and fk_out is not None: 
            outname = f'{prefix}.pkl'
            Path(fk_out).mkdir(parents=True, exist_ok=True)
            pickle.dump(
                {
                        "smpl_poses": local_q_eval_aa.squeeze(0).reshape((BH, T, 72)).detach().cpu().numpy(),
                        "smpl_trans": root_pos_eval.squeeze(0).detach().cpu().numpy(),
                        "full_pose": positions_recons[0],
                },
                open(os.path.join(fk_out, outname), "wb"),
            )


        ### Visualization
        ### Mesh video
        print(f'Rendering mesh for {all_filenames[mf]}')
        all_mesh = []
        os.makedirs('./results/mesh', exist_ok=True)
        for f_id in range(local_q_eval_aa.shape[1]):
            mesh_ls = []
            for h_id in range(freedance.num_person[mf]):
                pose_tensor = local_q_eval_aa[h_id,f_id,:,:].reshape(1, 72)
                trans_tensor = root_pos_eval[h_id,f_id,:].reshape(1, 3)
                output = SMPL_model.forward(body_pose=pose_tensor[:, 3:].detach().cpu(), global_orient=pose_tensor[:, :3].detach().cpu(), transl=trans_tensor.detach().cpu())
                vertices = output.vertices.detach().cpu().numpy()[0]  # (6890, 3)
 
                mesh = trimesh.Trimesh(vertices, smpl_faces, process=False)
                mesh_ls.append(mesh)
            all_mesh.append(mesh_ls)
        multi_mesh_render(
            all_mesh, 
            colors=[(239, 211, 171), (23, 49, 224), (250, 99, 72), (183, 29, 235)], 
            save_name=f'./results/mesh/render{prefix}_ps{freedance.num_person[mf]}',
            music_name=[all_filenames[mf]])

        ### Skeleton video
        print(f'Rendering skeleton for {all_filenames[mf]}')
        os.makedirs('./results/skeleton', exist_ok=True)
        if video_flag_recons:
            skeleton_render(
                positions_recons[0:args.num_person[mf]], # 148, 24, 3
                epoch='0',
                out="./results/skeleton",
                name=[all_filenames[mf]], # list wav name
                sound=True, # bool
                stitch=True,
                render=True,
                num_person=freedance.num_person[mf],
            )
import torch
from os.path import join as pjoin
import numpy as np
from models.modules import MovementConvEncoder, MotionEncoderBiGRUCo
from utils.word_vectorizer import POS_enumerator

def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose-4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    
    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, 'epoch_300.pth'),
                            map_location=opt.device) 
    
    # dict_keys(['text_encoder', 'motion_encoder', 'movement_encoder', 'opt_text_encoder', 'opt_motion_encoder', 'epoch', 'iter'])
    
    movement_enc.load_state_dict(checkpoint['movement_encoder']) 
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    return motion_enc, movement_enc


class EvaluatorModelWrapper_Dance(object):

    def __init__(self, opt):

        if opt.dataset_name == 'aistpp' or opt.dataset_name == 'aioz' or opt.dataset_name == 'aamixed':
            opt.dim_pose = 79  # 24*3+3+4
        else:
            raise KeyError('Dataset not Recognized!!!')

        opt.dim_motion_hidden = 1024
        opt.dim_coemb_hidden = 512

        # print(opt)

        self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.motion_encoder.eval()
        self.movement_encoder.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, music_feats, motions): # NOTE(yiwen) this is for pretrained fid extractor 
        with torch.no_grad():
            music_feats = music_feats.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float() # BH, T, D
            
            unit_lens = motions.shape[1] # T TODO(yw) check this dim
            m_lens = torch.tensor([unit_lens for i in range(motions.shape[0])])

            '''Movement Encoding'''
            movements = self.movement_encoder(motions).detach() # motion D=75
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            '''Music Encoding'''
            music_embedding = music_feats 
        return music_embedding, motion_embedding

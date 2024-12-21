import torch
from os.path import join as pjoin
import numpy as np
from models.modules import MovementConvEncoder, TextEncoderBiGRUCo, MotionEncoderBiGRUCo, MusicTransformerEncoder
from utils.word_vectorizer import POS_enumerator

def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose-4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    # text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word,
    #                               pos_size=opt.dim_pos_ohot,
    #                               hidden_size=opt.dim_text_hidden,
    #                               output_size=opt.dim_coemb_hidden,
    #                               device=opt.device)
    # TODO music_enc = MusicTransformerEncoder(latent_dim=256) 
    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, 'epoch_200.pth'),
                            map_location=opt.device) 
    
    # dict_keys(['text_encoder', 'motion_encoder', 'movement_encoder', 'opt_text_encoder', 'opt_motion_encoder', 'epoch', 'iter'])
    
    movement_enc.load_state_dict(checkpoint['movement_encoder']) #NOTE(yw) 而eval时候的模型结构也由state_dict决定
    # text_enc.load_state_dict(checkpoint['text_encoder']) 
    # TODO music_enc.load_state_dict(checkpoint['music_encoder']) 有没有music-motion align的model？
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    # print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return motion_enc, movement_enc


class EvaluatorModelWrapper_Dance(object):

    def __init__(self, opt):

        if opt.dataset_name == 't2m':
            opt.dim_pose = 263
        elif opt.dataset_name == 'kit':
            opt.dim_pose = 251
        elif opt.dataset_name == 'aistpp':
            opt.dim_pose = 79
        else:
            raise KeyError('Dataset not Recognized!!!')

        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512

        # print(opt)

        self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        # TODO self.music_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        # TODO self.music_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, music_feats, motions): # NOTE(yw) 这里的co-embedding也是pretrain model得到的co-embedding
        with torch.no_grad():
            music_feats = music_feats.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()
            
            unit_lens = motions.shape[2] # T TODO(yw) check this dim
            m_lens = torch.tensor([unit_lens for i in range(motions.shape[0])])

            '''Movement Encoding'''
            movements = self.movement_encoder(motions).detach() # motion D=75
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            '''Music Encoding'''
            # music_embedding = self.music_encoder(cond_embed=music_feats) # TODO (yw) map music feats to music embedding
            music_embedding = music_feats # NOTE (yiwen) temporarily using the features extracted by librosa
            # text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
        return music_embedding, motion_embedding

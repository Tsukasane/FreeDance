import torch.nn as nn
import torch
from models.encdec import Encoder, Decoder, Encoder2D, Decoder2D
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset, QuantizeEMAReset2D
from models.t2m_trans import Decoder_Transformer, Encoder_Transformer
from exit.utils import generate_src_mask
import numpy as np

class VQVAE_251(nn.Module): # TODO(yiwen) del class
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer
        # if args.dataname == 'kit':
        #     output_dim = 251  
        # elif args.dataname == 't2m':
        #     output_dim = 263
        if args.dataname == 'aistpp'or args.dataname == 'aioz' or args.dataname == 'aamixed':
            output_dim = 151
        self.encoder = Encoder(output_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        
        # Transformer Encoder
        # self.encoder = Encoder_Transformer(
        #     input_feats=output_dim,
        #     embed_dim=512, # 1024
        #     output_dim=512,
        #     block_size=4,
        #     num_layers=6,
        #     n_head=16
        # )

         # Transformer Encoder 4 frames
        # from exit.motiontransformer import MotionTransformerEncoder
        # in_feature = 251 if args.dataname == 'kit' else 263
        # self.encoder2 = MotionTransformerEncoder(in_feature, args.code_dim, num_frames=4, num_layers=2)

        self.decoder = Decoder(output_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)        
        # self.decoder = Decoder_Transformer(
        #     code_dim=512,
        #     embed_dim=512, # 1024
        #     output_dim=output_dim,
        #     block_size=49,
        #     num_layers=6,
        #     n_head=8
        # )
        if args.quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        elif args.quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx


    def forward(self, x):
        
        x_in = self.preprocess(x)
        # Encode
        # _x_in = x_in.reshape( int(x_in.shape[0]*4), x_in.shape[1], 16)
        # x_encoder = self.encoder(_x_in)
        # x_encoder = x_encoder.reshape(x_in.shape[0], -1, int(x_in.shape[2]/4))

        # [Transformer Encoder]
        # _x_in = x_in.reshape( int(x_in.shape[0]*x_in.shape[2]/4), x_in.shape[1], 4)
        # _x_in = _x_in.permute(0,2,1)
        # x_encoder = self.encoder2(_x_in)
        # x_encoder = x_encoder.permute(0,2,1)
        # x_encoder = x_encoder.reshape(x_in.shape[0], -1, int(x_in.shape[2]/4))

        x_encoder = self.encoder(x_in) # 256, 32, 16
        
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)
        # 256, 32, 16
        
        ## decoder
        x_decoder = self.decoder(x_quantized) # 256, 251, 64
        x_out = self.postprocess(x_decoder) # 256, 64, 251
    
        return x_out, loss, perplexity


    def forward_decoder(self, x):
        # x = x.clone()
        # pad_mask = x >= self.code_dim
        # x[pad_mask] = 0

        x_d = self.quantizer.dequantize(x)
        x_d = x_d.permute(0, 2, 1).contiguous()

        # pad_mask = pad_mask.unsqueeze(1)
        # x_d = x_d * ~pad_mask
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class VQVAE_DANCE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer

        if args.dataname == 'aistpp'or args.dataname == 'aioz' or args.dataname == 'aamixed':
            output_dim = 151
        self.encoder = Encoder(output_dim, 
                               output_emb_width, 
                               down_t, 
                               stride_t, 
                               width, 
                               depth, 
                               dilation_growth_rate, 
                               activation=activation, 
                               norm=norm)
        
        # Transformer Encoder
        # self.encoder = Encoder_Transformer(
        #     input_feats=output_dim,
        #     embed_dim=512, # 1024
        #     output_dim=512,
        #     block_size=4,
        #     num_layers=6,
        #     n_head=16
        # )

         # Transformer Encoder 4 frames
        # from exit.motiontransformer import MotionTransformerEncoder
        # in_feature = 251 if args.dataname == 'kit' else 263
        # self.encoder2 = MotionTransformerEncoder(in_feature, args.code_dim, num_frames=4, num_layers=2)

        self.decoder = Decoder(output_dim, 
                               output_emb_width, 
                               down_t, 
                               stride_t, 
                               width, 
                               depth, 
                               dilation_growth_rate, 
                               activation=activation, 
                               norm=norm)        
        # self.decoder = Decoder_Transformer(
        #     code_dim=512,
        #     embed_dim=512, # 1024
        #     output_dim=output_dim,
        #     block_size=49,
        #     num_layers=6,
        #     n_head=8
        # )
        if args.quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        elif args.quantizer == "ema_reset2d":
            self.quantizer = QuantizeEMAReset2D(nb_code, args.max_person, code_dim, args)
        elif args.quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)


    def preprocess(self, x):
        # (B, H*T, D) -> (B, D, H*T) 
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (B, D, H*T) -> (B, H*T, D) D = Jx3
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        B, H, T, D = x.shape
        x = x.view(B, H*T, D)
        x_in = self.preprocess(x) # (B, H*T, D) -> (B, D, H*T) 
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder) # (B, D, H*T) -> (B, H*T, D)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (BHT, D)

        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(B, H, -1)
        return code_idx


    def forward(self, x):
        B, H, T, D = x.shape
        x = x.view(B, H*T, D)
        x_in = self.preprocess(x) # 256, 151, 150 NOTE(yw) maybe need to cut T to be 148
        # Encode
        # _x_in = x_in.reshape( int(x_in.shape[0]*4), x_in.shape[1], 16)
        # x_encoder = self.encoder(_x_in)
        # x_encoder = x_encoder.reshape(x_in.shape[0], -1, int(x_in.shape[2]/4))

        # [Transformer Encoder]
        # _x_in = x_in.reshape( int(x_in.shape[0]*x_in.shape[2]/4), x_in.shape[1], 4)
        # _x_in = _x_in.permute(0,2,1)
        # x_encoder = self.encoder2(_x_in)
        # x_encoder = x_encoder.permute(0,2,1)
        # x_encoder = x_encoder.reshape(x_in.shape[0], -1, int(x_in.shape[2]/4))
        
        x_encoder = self.encoder(x_in) # 256, 32, 37

        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)
        # 256, 32, 37
        
        ## decoder
        x_decoder = self.decoder(x_quantized) # 256, 151, 148
        x_out = self.postprocess(x_decoder) # 256, 148, 151
        
        x_out = x_out.view(B, H, -1, D) # B, H*T, D -> B, H, T, D
        return x_out, loss, perplexity # reconstructed x, 


    def forward_decoder(self, x):
        # x = x.clone()
        # pad_mask = x >= self.code_dim
        # x[pad_mask] = 0

        x_d = self.quantizer.dequantize(x)
        x_d = x_d.permute(0, 2, 1).contiguous()

        # pad_mask = pad_mask.unsqueeze(1)
        # x_d = x_d * ~pad_mask
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class VQVAE_DANCE2D(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024, #8192
                 code_dim=512, #32
                 output_emb_width=512,#512
                 down_t=3,
                 stride_t=2,
                 width=512, #512
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer

        if args.dataname == 'aistpp' or args.dataname == 'aioz' or args.dataname == 'aamixed':
            output_dim = 3 # H
        self.encoder = Encoder2D(output_dim, 
                               output_emb_width, 
                               down_t, 
                               stride_t, 
                               width, 
                               depth, 
                               dilation_growth_rate, 
                               activation=activation, 
                               norm=norm)

        self.decoder = Decoder2D(output_dim, 
                               output_emb_width, 
                               down_t, 
                               stride_t, 
                               width, 
                               depth, 
                               dilation_growth_rate, 
                               activation=activation, 
                               norm=norm)        

        if args.quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        elif args.quantizer == "ema_reset2d":
            self.quantizer = QuantizeEMAReset2D(nb_code, args.max_person, code_dim, args)
        elif args.quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)

        self.max_person = args.max_person

    def preprocess(self, x, real_num_person): # TODO(yiwen) add num_person to arg here
        # init
        B, H, T, D = x.shape # 64, H_data, 148, 151
        x_in = x[0] # H, T, D
        pad_D = 0
        pad_H = 0
        rand_insert_ls = [] # padding index for H==2
        rand_insert = -1
        
        if D % 4!=0: # 64, 3, 148, 151 --> 64, 3, 148, 152
            pad_D = (4 - D % 4)
            pad_D_tensor = torch.zeros(B, H, T, pad_D, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad_D_tensor], dim=-1)
            D_new = D + pad_D
        
        new_x = x
        # TODO(yiwen) test the modification here
        pad_tensor = torch.zeros(1, T, D_new, device=x.device, dtype=x.dtype)
        for b in range(B):
            real_H = real_num_person[b]
            assert real_H <= self.max_person
            x_in = x[b,:real_H,:,:] # dim=3

            # TODO(yiwen) think about how to pad H>3 in ablation
            if real_H == 1:
                x_in = torch.cat([x_in, pad_tensor, pad_tensor], dim=0)
                rand_insert_ls.append(-1)
            elif real_H == 2:
                x_in = torch.cat([x_in, pad_tensor], dim=0)
                # rand_insert = np.random.randint(3)
                # if rand_insert == 0:  
                #     x_in = torch.cat([pad_tensor, x_in], dim=0)
                # elif rand_insert == 1:
                #     x_in = torch.cat([x_in[:1,:,:], pad_tensor, x_in[1:,:,:]], dim=0)
                # else:
                #     x_in = torch.cat([x_in, pad_tensor], dim=0)
                rand_insert_ls.append(rand_insert)
            else:
                # H==3, no padding
                rand_insert_ls.append(-1)

            new_x[b,:,:,:] = x_in
        # zero padding the H dimension
        # H=1, pad the last 2 dim
        # H=2, pad 1 random dim
        # H=3, no pad
            
        return new_x, rand_insert_ls


    def postprocess(self, x_decoder, rand_insert_ls, real_num_person, ori_D):
        # (B, D, H*T) -> (B, H*T, D) D = Jx3
        B, H, T, D = x_decoder.shape # 64, H_data, 148, 151
        x_output = x_decoder[:,:,:,:ori_D]
        assert len(rand_insert_ls)==B

        for b in range(B):
            real_H = real_num_person[b]
            rand_insert = rand_insert_ls[b]
            x_decoder_this = x_decoder[b,:,:,:ori_D]

            if real_H == 1:
                x_output_this = x_decoder_this[:1,:,:ori_D]
            elif real_H == 2:
                x_output_this = x_decoder_this[:2,:,:ori_D]
                # assert rand_insert!=-1
                # if rand_insert == 0:  
                #     x_output_this = x_decoder_this[1:,:,:ori_D]
                # elif rand_insert == 1:
                #     x_output_this = torch.cat([x_decoder_this[:1,:,:ori_D], x_decoder_this[2:,:,:ori_D]], dim=0)
                # else:
                #     x_output_this = x_decoder_this[:2,:,:ori_D]
            else:
                x_output_this = x_decoder_this
            
            x_output[b,:real_H,:,:] = x_output_this # TODO(yiwen) 算loss的时候需要把pad的地方去掉(?)

        return x_output


    def encode(self, x, real_num_person):
        B, H, T, D = x.shape
        x_in, rand_insert_ls = self.preprocess(x, real_num_person)
        x_encoder = self.encoder(x_in) # 1, 3, 37, 32

        x_encoder = x_encoder.permute(0,2,1,3) # B, T, H, D'
        dp = x_encoder.shape[3] # 32
        tp = x_encoder.shape[1]
        x_encoder = x_encoder.reshape(B*tp, -1, dp) # B*T', H, D'

        code_idx = self.quantizer.quantize(x_encoder) # NT

        code_idx = code_idx.view(B, tp, -1) # 1, 37, 1
        return code_idx


    def forward(self, x, real_num_person):
        B, H, T, D = x.shape # 64, H_data, 148, 151
        x_in, rand_insert_ls = self.preprocess(x, real_num_person)
 
        x_encoder = self.encoder(x_in) # B, H_pad, T', D' 64, 3, 37, 32

        x_quantized, loss, perplexity = self.quantizer(x_encoder, real_num_person) # TODO(yiwen) 这部分的loss也要去除padding的影响
        # B, H_pad, T', D'

        x_decoder = self.decoder(x_quantized) # B, H_pad, T, D_pad

        x_output = self.postprocess(x_decoder, rand_insert_ls, real_num_person, D)
        # TODO(yiwen) check the postprocess

        return x_output, loss, perplexity # reconstructed x, 


    def forward_decoder(self, x):
        # x = x.clone()
        # pad_mask = x >= self.code_dim
        # x[pad_mask] = 0

        x_d = self.quantizer.dequantize(x)
        x_d = x_d.permute(0, 2, 1, 3).contiguous()

        # B, H=3, T'=37, D'=32
    
        x_decoder = self.decoder(x_d)

        # TODO(yiwen) use info from encode and add rand_insert_ls, etc.
        # x_output = self.postprocess(x_decoder, rand_insert_ls, real_num_person, D)

        return x_decoder


class HumanVQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        
        if args.dataname == 'aistpp' or args.dataname == 'aioz' or args.dataname == 'aamixed':
            self.nb_joints = 24
            self.vqvae = VQVAE_DANCE2D(args, nb_code, code_dim, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        

    def forward(self, x, num_person, type='full'):
        '''type=[full, encode, decode]'''
        if type=='full':
            x_out, loss, perplexity = self.vqvae(x, num_person)
            return x_out, loss, perplexity
        elif type=='encode':
            # b, t, c = x.size()
            quants = self.vqvae.encode(x, num_person) # (N, T)
            return quants
        elif type=='decode':
            x_out = self.vqvae.forward_decoder(x)
            return x_out
        else:
            raise ValueError(f'Unknown "{type}" type')
        
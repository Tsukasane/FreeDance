import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding
from exit.utils import cosine_schedule, uniform, top_k, gumbel_sample, top_p
from tqdm import tqdm
from einops import rearrange, repeat
from exit.utils import get_model, generate_src_mask
from models.reaction_attention import Reaction_Attention, AdaIN

class PatchUpSampling(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.up_sampling = nn.Linear(dim, 4 * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, F, C
        """
        x = self.norm(x)
        x = self.up_sampling(x)
        x0 = x[:, :, 0::4]  
        x1 = x[:, :, 1::4]
        x2 = x[:, :, 2::4]
        x3 = x[:, :, 3::4]
        x = torch.cat([x0, x1, x2, x3], 1)  
        return x



# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L342C9-L343C33
class PatchMerging(nn.Module):
    def __init__(self, input_feats, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * input_feats, dim, bias=False)
        self.norm = norm_layer(4 * input_feats)

    def forward(self, x):
        """
        x: B, F, C
        """
        x0 = x[:, 0::4, :]  # B F/2 C
        x1 = x[:, 1::4, :]
        x2 = x[:, 2::4, :]  # B F/2 C
        x3 = x[:, 3::4, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B F/2 2*C
        x = self.norm(x)
        x = self.reduction(x)
        return x



class Music2Dance_Transformer(nn.Module):

    def __init__(self, 
                vqvae,
                num_vq=1024, 
                embed_dim=512, 
                music_dim=512, 
                block_size=16, 
                num_layers=2, 
                num_local_layer=0, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                max_person=3): # B, T, 1/3codebook
        super().__init__()
        self.n_head = n_head
        self.num_vq = num_vq
        self.max_person = max_person
        self.person_num_cbsize = num_vq // self.max_person # codebook size for each person
        self.trans_base = CrossCondTransBase(vqvae, num_vq, embed_dim, music_dim, block_size, num_layers, num_local_layer, n_head, drop_out_rate, fc_rate, max_person)
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, self.person_num_cbsize, self.max_person)
        self.block_size = block_size
        self.sample_block_size = 38

        # self.skip_trans = Skip_Connection_Transformer(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)

    def get_block_size(self):
        return self.block_size

    def get_attn_mask(self, src_mask):
        B, T = src_mask.shape
        src_mask = src_mask.view(B, 1, 1, T).repeat(1, self.n_head, T, 1) # motion self-atten, multi head
        return src_mask

    def forward(self, *args, type='forward', **kwargs):
        '''type=[forward, sample]'''
        if type=='forward':
            return self.forward_function(*args, **kwargs)
        elif type=='sample':
            return self.sample(*args, **kwargs)
        else:
            raise ValueError(f'Unknown "{type}" type')

    def forward_function(self, idxs, src_mask, mus_emb=None, real_num_person=None):
        if src_mask is not None:
            src_mask = self.get_attn_mask(src_mask) # 16, 16, 38, 38
        feat = self.trans_base(idxs, src_mask, mus_emb) 
        logits = self.trans_head(feat, src_mask, real_num_person)

        return logits

    def sample(self, 
               m_length=None, 
               if_test=False, 
               rand_pos=True, 
               token_cond=None, 
               max_steps = 10,
               mus_emb=None,
               real_num_person=None):

        max_length = 49
        batch_size = mus_emb.shape[0]
        mask_id = self.num_vq + 2
        pad_id = self.num_vq + 1
        end_id = self.num_vq
        topk_filter_thres = .9
        starting_temperature = 1.0
        block_size = 50
        
        m_tokens_len = torch.ceil((m_length)/4).long()
        src_token_mask = generate_src_mask(block_size, m_tokens_len+1) # with end token
        src_token_mask_noend = generate_src_mask(block_size, m_tokens_len) # without end token

        if token_cond is not None:
            ids = token_cond.clone()
            ids[~src_token_mask_noend] = pad_id
            num_token_cond = (ids==mask_id).sum(-1)
        else:
            # estimate B, T ids; scores represent confidence
            shape = (batch_size, block_size)
            ids = torch.full(shape, mask_id, dtype = torch.long, device = mus_emb.device) # full (B, T+1) with mask_id
        
        scores = torch.ones_like(ids, dtype=torch.float32)
        sample_max_steps = torch.round(max_steps/max_length*m_tokens_len) + 1e-8 # B

        for step in range(max_steps):
            timestep = torch.clip(step/(sample_max_steps), max=1)
            if len(m_tokens_len)==1 and step > 0 and torch.clip(step-1/(sample_max_steps), max=1).cpu().item() == timestep:
                break
            rand_mask_prob = cosine_schedule(timestep) # INFERENCE prob=1
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1) # INFERENCE mask all

            if token_cond is not None:
                num_token_masked = (rand_mask_prob * num_token_cond).long().clip(min=1)
                scores[token_cond!=mask_id] = 0
            
            # [INFO] rm no motion frames
            scores[~src_token_mask_noend] = 0 # end token
            scores = scores/scores.sum(-1)[:, None] # normalize only unmasked token
            
            sorted, sorted_score_indices = scores.sort(descending=True)
            
            ids[~src_token_mask] = pad_id # padding token
            ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # add end-id to the end of each motion token seq
            
            # Replace "mask_id" to "ids" that have highest "num_token_masked" "scores" 
            select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
            
            # Repeat last_id to make it scatter_ the existing last ids.
            last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1)-1)
            sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index*~select_masked_indices)
            ids.scatter_(-1, sorted_score_indices, mask_id)
            trans_src_mask = torch.cat([src_token_mask]*self.max_person, dim=-1)
            logits = self.forward(ids, trans_src_mask, mus_emb=mus_emb, real_num_person=real_num_person)
            
            filtered_logits = logits #top_p(logits, .5) # #top_k(logits, topk_filter_thres)
            if rand_pos:
                temperature = 1 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed
            else:
                temperature = 0 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            # NOTE if temperature==0: is equal to argmax (filtered_logits.argmax(dim = -1))
            # pred_ids = filtered_logits.argmax(dim = -1)
            pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1) # B, T
        
            is_mask = ids == mask_id
            ids = torch.where(is_mask, pred_ids, ids)
            
            probs_without_temperature = logits.softmax(dim = -1)
            scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')
            scores = scores.masked_fill(~is_mask, 0)

        return ids



class Attention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x, src_mask):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        if src_mask is not None: # no att to the pad-id, end-id
            att[~src_mask] = float('-inf')

        att = F.softmax(att, dim=-1) 
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) then add residual to the original input
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y)) 
        return y

class Block(nn.Module): # self attention block
    '''
    adain injects style after each self attention layer.
    style is obtained by reaction attention.
    '''
    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, block_size, n_head, drop_out_rate)
        self.react_attn = Reaction_Attention(embed_dim).to(self.device)
        self.adaIN = AdaIN()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, src_mask, use_moduleA=True):
        x = x + self.attn(self.ln1(x), src_mask) # self-attn
        # assitant matrix
        if use_moduleA:
            style = self.react_attn(x) # B, HT, D
            x = self.adaIN(x, style) # transformed.shape = motion.shape

        x = x + self.mlp(self.ln2(x))
        return x

class CrossAttention(nn.Module): 

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x, mus_emb):
        '''
         - mus_emb: the music_feature_embedding  B, N, D
         - x: the motion token B, T, C
        '''
        B, T, C = x.size() # 32, 50, 1024
        B, N, D = mus_emb.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(mus_emb).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, N, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(mus_emb).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, N, hs)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # k.size(-1) each head dim=hs
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, N) x (B, nh, N, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y



class TemporalCoherentCrossAttention(nn.Module): 
    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head
        self.att_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x, compressed_music_emb):
        '''
         - x: motion emb B, T, 3072
         - compressed_music_emb: B, T, 3072

         m1: x - raw motion code
         m2: motion_residual - vector difference
         m3: compressed_music_emb - music feature
         
        '''
        B, T, HD = x.size() # 32, 37, 3072

        # cal m2
        motion_residual = x[:,1:,:] - x[:,:-1,:] # B, T-1, HD
        pad_res = torch.zeros(B, 1, HD).to(motion_residual.device)
        motion_residual = torch.cat([pad_res, motion_residual], dim=1)

        # cross atten for m1 & m3 
        k = self.key(compressed_music_emb) # (B, T, HD) 
        q = self.query(x) 
        v = self.value(compressed_music_emb) 

        # NOTE(yiwen) TxT residual metrix (A1: m2 & m3)  32, 37, 37 
        res_met = (motion_residual @ compressed_music_emb.transpose(-2, -1)) * (1.0 / math.sqrt(compressed_music_emb.size(-1)))
        # res_w = F.softmax(res_met, dim=-1) # use res_met similarity itself not the probability

        # NOTE(yiwen) similarity metrix (A2: m1 & m3)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) 
        att = F.softmax(att, dim=-1) 

        att = self.attn_drop(att) 
        cat_w = torch.stack([att, res_met], dim=1) # B, 2, T, T
        
        out_w = self.att_conv(cat_w).squeeze(1)        
        y = out_w @ v
        y = y.transpose(1, 2).contiguous().view(B, T, HD) # re-assemble all head outputs side by side

        y = self.resid_drop(self.proj(y)) 
        return y


class Block_crossatt(nn.Module): # cross attention block

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.attn = CrossAttention(embed_dim, block_size, n_head, drop_out_rate)
        use_moduleB = True
        if use_moduleB:
            self.temporal_co_attn = TemporalCoherentCrossAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, mus_emb, compressed_music_emb=None, use_moduleB=True):
        if use_moduleB:
            x = x + self.temporal_co_attn(self.ln1(x), self.ln3(compressed_music_emb)) # temporal coherent cross-attention
        else:
            x = x + self.attn(self.ln1(x), self.ln3(mus_emb)) # NOTE(yiwen) ablation for cross-atten
        x = x + self.mlp(self.ln2(x))
        return x


class CrossCondTransBase(nn.Module):

    def __init__(self, 
                vqvae,
                num_vq=1024,   
                embed_dim=512, 
                music_dim=256, 
                block_size=16, 
                num_layers=2, 
                num_local_layer = 1,
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                max_person=3):
        super().__init__()
        self.vqvae = vqvae
        self.block_size2 = block_size # same as T(padded)
        block_size = max_person * block_size # T(padded) * H
        
        # self.tok_emb = nn.Embedding(num_vq + 3, embed_dim).requires_grad_(False) 
        self.learn_tok_emb = nn.Embedding(3, self.vqvae.vqvae.code_dim * self.vqvae.vqvae.max_person)# [INFO] 3 = [end_id, blank_id, mask_id] 
        self.to_emb = nn.Linear(self.vqvae.vqvae.code_dim, embed_dim)

        self.cond_emb = nn.Linear(music_dim, embed_dim) # motion and cond to the same dim 
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        

        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers-num_local_layer)]) # 先self-atten
        self.pos_embed1 = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False) # TODO(yiwen) check whether need to be 37*3+1
        self.pos_embed2 = pos_encoding.PositionEmbedding(self.block_size2, embed_dim*self.vqvae.vqvae.max_person, 0.0, False) # TODO(yiwen) check whether need to be 37+1

        self.num_local_layer = num_local_layer
        if num_local_layer > 0:
            self.music_to_emb = nn.Linear(music_dim, embed_dim*self.vqvae.vqvae.max_person)
            self.music_linear = nn.Linear(150, self.block_size2) # 150=T
            self.cross_att = nn.Sequential(*[Block_crossatt(embed_dim*self.vqvae.vqvae.max_person, self.block_size2, 1, drop_out_rate, fc_rate) for _ in range(num_local_layer)]) # nhead=1 here
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, src_mask, mus_emb, use_moduleB=True):
        
        b, t = idx.size() # 32, 50
        idx = idx[:,:self.block_size2]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        not_learn_idx = idx<self.vqvae.vqvae.num_code # motion
        learn_idx = ~not_learn_idx # end-id, pad-id, mask-id

        token_embeddings = torch.empty((*idx.shape, self.vqvae.vqvae.max_person, self.vqvae.vqvae.code_dim), device=idx.device)
        # B, T(padded), H, D

        token_embeddings[not_learn_idx] = self.vqvae.vqvae.quantizer.dequantize(idx[not_learn_idx]).requires_grad_(False) # NOTE(yiwen) freeze vq decoder
        token_embeddings[learn_idx] = self.learn_tok_emb(idx[learn_idx]-self.vqvae.vqvae.num_code).view(-1, self.vqvae.vqvae.max_person, self.vqvae.vqvae.code_dim) 
        token_embeddings = self.to_emb(token_embeddings) # B, T(pad), H, D --> B, T(pad), H, emb_dim

        B, T, H, D = token_embeddings.shape
        token_embeddings = token_embeddings.view(B, T, H*D) 
        
        if self.num_local_layer > 0: 
            mus_emb = self.music_to_emb(mus_emb) # 32, 150, 3072
            compressed_music_emb = self.music_linear(mus_emb.permute(0,2,1)).permute(0,2,1) # 32, 37, 3072
            token_embeddings = self.pos_embed2(token_embeddings) # add positional encoding to motion tokens B, T, H*D'' 32, 37, 3*1024

            for module in self.cross_att: # modality fusion
                if use_moduleB:
                    token_embeddings = module(token_embeddings, mus_emb, compressed_music_emb, use_moduleB) # temporal coherent cross-atten
                else:
                    token_embeddings = module(token_embeddings, mus_emb) # NOTE(yiwen) ablation for cross-atten

        token_embeddings = token_embeddings.view(B, T*H, D)  
        x = token_embeddings
        for block in self.blocks: # motion self-atten
            x = block(x, src_mask)

        return x


def masked_logits(logits, valid_ids, mask_value=-1e6, eps=1e-8):
    '''
    logits: B, T, C
    valid_ids, B, T, C//num_person
    '''
    B, T, C = logits.shape
    mask = torch.full((B, T, C), mask_value, device=logits.device)
    mask.scatter_(2, valid_ids, logits.gather(2, valid_ids))

    return mask + eps


class CrossCondTransHead(nn.Module):
    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                person_num_cbsize=-1,
                max_person=3,):
        super().__init__()

        self.max_person = max_person
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(self.max_person*embed_dim, num_vq, bias=False)
        self.block_size = block_size
        self.person_num_cbsize = person_num_cbsize

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, src_mask, real_num_persons):
        for block in self.blocks:
            x = block(x, src_mask)  
        x = self.ln_f(x)
        x = x.view(x.shape[0], -1, self.max_person*x.shape[-1]) # B, T, HD

        logits = self.head(x) # B, T, codebook_cls 
        
        mask_logits = True
        if mask_logits:
            B, T, _ = logits.shape
            valid_ids = torch.zeros((B, T, self.person_num_cbsize), device=logits.device, dtype=torch.long)
            for n_id in range(B):
                num_person = real_num_persons[n_id]
                valid_id = torch.arange((num_person-1)*self.person_num_cbsize, num_person * self.person_num_cbsize, device=logits.device, dtype=torch.long)
                valid_id = valid_id.repeat(T, 1) # T, person_num_cbsize
                valid_ids[n_id] = valid_id
            # whole codebook --> 1/H codebook according to real_num_person
            probs = masked_logits(logits, valid_ids)
            return probs 
        else:
            return logits

    


        


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
                fc_rate=4):
        super().__init__()
        self.n_head = n_head
        self.trans_base = CrossCondTransBase(vqvae, num_vq, embed_dim, music_dim, block_size, num_layers, num_local_layer, n_head, drop_out_rate, fc_rate)
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.sample_block_size = 38
        self.num_vq = num_vq
        self.max_person = 3

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
        # elif type=='inpaint':
        #     return self.inpaint(*args, **kwargs)
        else:
            raise ValueError(f'Unknown "{type}" type')

    def forward_function(self, idxs, src_mask, word_emb=None):
        if src_mask is not None:
            src_mask = self.get_attn_mask(src_mask) # 16, 16, 38, 38
        feat = self.trans_base(idxs, src_mask, word_emb) 
        logits = self.trans_head(feat, src_mask)

        return logits

    def sample(self, 
               m_length=None, 
               if_test=False, 
               rand_pos=True, 
               token_cond=None, 
               max_steps = 10,
               word_emb=None):

        # TODO(yiwen) check details here
        max_length = 49
        batch_size = word_emb.shape[0]
        mask_id = self.num_vq + 2
        pad_id = self.num_vq + 1
        end_id = self.num_vq
        topk_filter_thres = .9
        starting_temperature = 1.0
        block_size = 50
        
        m_tokens_len = torch.ceil((m_length)/4).long()
        src_token_mask = generate_src_mask(block_size, m_tokens_len+1) # with end token
        src_token_mask_noend = generate_src_mask(block_size, m_tokens_len) # without end token

        # estimate B, T ids; scores represent confidence
        shape = (batch_size, block_size)
        ids = torch.full(shape, mask_id, dtype = torch.long, device = word_emb.device) # full (B, T+1) with mask_id
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
            
            sorted, sorted_score_indices = scores.sort(descending=True) # deterministic
            
            ids[~src_token_mask] = pad_id # padding token
            ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # add end-id to the end of each motion token seq
            
            ## [INFO] Replace "mask_id" to "ids" that have highest "num_token_masked" "scores" 
            select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
            
            # [INFO] repeat last_id to make it scatter_ the existing last ids.
            last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1)-1)
            sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index*~select_masked_indices)
            ids.scatter_(-1, sorted_score_indices, mask_id)
            trans_src_mask = torch.cat([src_token_mask]*self.max_person, dim=-1)
            logits = self.forward(ids, trans_src_mask, word_emb=word_emb) # NOTE(yiwen) feel not necessary to add the end-id
            
            filtered_logits = logits #top_p(logits, .5) # #top_k(logits, topk_filter_thres)
            if rand_pos:
                temperature = 1 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed
            else:
                temperature = 0 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            # [INFO] if temperature==0: is equal to argmax (filtered_logits.argmax(dim = -1))
            # pred_ids = filtered_logits.argmax(dim = -1)
            pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1) # B, T
        
            is_mask = ids == mask_id
            ids = torch.where(is_mask, pred_ids, ids)
            
            probs_without_temperature = logits.softmax(dim = -1)
            scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')
            scores = scores.masked_fill(~is_mask, 0)

        if if_test:
            return ids # 32, 37 B, T
        return ids
    
    # def inpaint(self, first_tokens, last_tokens, music_feature=None, inpaint_len=2, rand_pos=False):
    #     # support only one sample
    #     assert first_tokens.shape[0] == 1
    #     assert last_tokens.shape[0] == 1
    #     max_steps = 20
    #     max_length = 49
    #     batch_size = first_tokens.shape[0]
    #     mask_id = self.num_vq + 2
    #     pad_id = self.num_vq + 1
    #     end_id = self.num_vq
    #     shape = (batch_size, self.block_size - 1)
    #     scores = torch.ones(shape, dtype = torch.float32, device = first_tokens.device)
        
    #     # force add first / last tokens
    #     first_partition_pos_idx = first_tokens.shape[1]
    #     second_partition_pos_idx = first_partition_pos_idx + inpaint_len
    #     end_pos_idx = second_partition_pos_idx + last_tokens.shape[1]

    #     m_tokens_len = torch.ones(batch_size, device = first_tokens.device)*end_pos_idx

    #     src_token_mask = generate_src_mask(self.block_size-1, m_tokens_len+1)
    #     src_token_mask_noend = generate_src_mask(self.block_size-1, m_tokens_len)
    #     ids = torch.full(shape, mask_id, dtype = torch.long, device = first_tokens.device)
        
    #     ids[:, :first_partition_pos_idx] = first_tokens
    #     ids[:, second_partition_pos_idx:end_pos_idx] = last_tokens
    #     src_token_mask_noend[:, :first_partition_pos_idx] = False
    #     src_token_mask_noend[:, second_partition_pos_idx:end_pos_idx] = False
        
    #     # [TODO] confirm that these 2 lines are not neccessary (repeated below and maybe don't need them at all)
    #     ids[~src_token_mask] = pad_id # [INFO] replace with pad id
    #     ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # [INFO] replace with end id

    #     temp = []
    #     sample_max_steps = torch.round(max_steps/max_length*m_tokens_len) + 1e-8

    #     if music_feature is None:
    #         music_feature = torch.zeros(1, 512).to(first_tokens.device)
    #         att_txt = torch.zeros((batch_size,1), dtype=torch.bool, device = first_tokens.device)
    #     else:
    #         att_txt = torch.ones((batch_size,1), dtype=torch.bool, device = first_tokens.device)

    #     for step in range(max_steps):
    #         timestep = torch.clip(step/(sample_max_steps), max=1)
    #         rand_mask_prob = cosine_schedule(timestep) # timestep #
    #         num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)
    #         # [INFO] rm no motion frames
    #         scores[~src_token_mask_noend] = 0
    #         # [INFO] rm begin and end frames
    #         scores[:, :first_partition_pos_idx] = 0
    #         scores[:, second_partition_pos_idx:end_pos_idx] = 0
    #         scores = scores/scores.sum(-1)[:, None] # normalize only unmasked token
            
    #         sorted, sorted_score_indices = scores.sort(descending=True) # deterministic
            
    #         ids[~src_token_mask] = pad_id # [INFO] replace with pad id
    #         ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # [INFO] replace with end id
    #         ## [INFO] Replace "mask_id" to "ids" that have highest "num_token_masked" "scores" 
    #         select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
    #         # [INFO] repeat last_id to make it scatter_ the existing last ids.
    #         last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1)-1)
    #         sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index*~select_masked_indices)
    #         ids.scatter_(-1, sorted_score_indices, mask_id)

    #         # [TODO] force replace begin/end tokens b/c the num mask will be more than actual inpainting frames
    #         ids[:, :first_partition_pos_idx] = first_tokens
    #         ids[:, second_partition_pos_idx:end_pos_idx] = last_tokens
            
    #         logits = self.forward(ids, music_feature, src_token_mask)[:,1:]
    #         filtered_logits = logits #top_k(logits, topk_filter_thres)
    #         if rand_pos:
    #             temperature = 1 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed
    #         else:
    #             temperature = 0 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

    #         # [INFO] if temperature==0: is equal to argmax (filtered_logits.argmax(dim = -1))
    #         # pred_ids = filtered_logits.argmax(dim = -1)
    #         pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
    #         is_mask = ids == mask_id
    #         temp.append(is_mask[:1])
            
    #         ids = torch.where(
    #                     is_mask,
    #                     pred_ids,
    #                     ids
    #                 )
            
    #         probs_without_temperature = logits.softmax(dim = -1)
    #         scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
    #         scores = rearrange(scores, '... 1 -> ...')
    #         scores = scores.masked_fill(~is_mask, 0)
    #     return ids


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
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
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

    def forward(self, x, src_mask):
        x = x + self.attn(self.ln1(x), src_mask) # self-attn
        # assitant matrix
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
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(block_size, 77)).view(1, 1, block_size, 77)) 
        # NOTE(yiwen) didn't use the mask since all motion can see all music (not a online inference)
        self.n_head = n_head

    def forward(self, x, word_emb):
        '''
         - word_emb: the music_feature_embedding  B, T, Muemb 128, 150, 256
        '''
        B, T, C = x.size() # 32, 50, 1024 # TODO(yiwen) find H
        B, N, D = word_emb.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(word_emb).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, N, hs) 每个head关注一部分空间特征
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) NOTE(yiwen) query 的motion特征在输入之前
        v = self.value(word_emb).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, N, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, N) -> (B, nh, T, N)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # k.size(-1) 每个head的维度hs, 这里的map表示motion的时序和music的时序之间的关系
        att = F.softmax(att, dim=-1) # --> probability distribution 对t=1～T的每一个motion token，算N个music token和motion token的相关度
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, N) x (B, nh, N, hs) -> (B, nh, T, hs) 每个motion token受到自己最相关的music token的影响
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
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(block_size, 77)).view(1, 1, block_size, 77)) 
        # NOTE(yiwen) didn't use the mask since all motion can see all music (not a online inference)
        self.n_head = n_head
        self.att_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x, compressed_music_emb):
        '''
         - x: motion emb 32, 37, 3072
         - compressed_music_emb: 32, 37, 3072

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
        res_met = (x @ compressed_music_emb.transpose(-2, -1)) * (1.0 / math.sqrt(compressed_music_emb.size(-1)))
        res_w = F.softmax(res_met, dim=-1)

        # NOTE(yiwen) similarity metrix (A2: m1 & m3)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) 
        att = F.softmax(att, dim=-1) 

        # att = self.attn_drop(att) # TODO(yiwen) check whether need attn_drop here
        cat_w = torch.stack([att, res_met], dim=1) # B, 2, T, T
        
        out_w = self.att_conv(cat_w).squeeze(1)        
        y = out_w @ v # 
        y = y.transpose(1, 2).contiguous().view(B, T, HD) # re-assemble all head outputs side by side

        y = self.resid_drop(self.proj(y)) # TODO(yiwen) check whether need resid_drop here
        return y


class Block_crossatt(nn.Module): # cross attention block

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.attn = CrossAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.temporal_co_attn = TemporalCoherentCrossAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, word_emb, compressed_music_emb):
        # x = x + self.attn(self.ln1(x), self.ln3(word_emb)) # NOTE(yiwen) ablation for cross-atten
        x = x + self.temporal_co_attn(self.ln1(x), self.ln3(compressed_music_emb)) # temporal coherent cross-attention
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
                fc_rate=4):
        super().__init__()
        self.vqvae = vqvae
        
        # self.tok_emb = nn.Embedding(num_vq + 3, embed_dim).requires_grad_(False) 
        self.learn_tok_emb = nn.Embedding(3, self.vqvae.vqvae.code_dim * self.vqvae.vqvae.max_person)# [INFO] 3 = [end_id, blank_id, mask_id] 
        self.to_emb = nn.Linear(self.vqvae.vqvae.code_dim, embed_dim) # motion code的维数->embedding维数

        self.cond_emb = nn.Linear(music_dim, embed_dim) # motion and cond to the same dim 
        self.pos_embedding = nn.Embedding(block_size, embed_dim) # pos总数，维数
        self.drop = nn.Dropout(drop_out_rate)
        
        self.block_size2 = 50 # same as T(padded)

        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers-num_local_layer)]) # 先self-atten
        self.pos_embed1 = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False) # TODO(yiwen) check whether need to be 37*3+1
        self.pos_embed2 = pos_encoding.PositionEmbedding(self.block_size2, embed_dim*self.vqvae.vqvae.max_person, 0.0, False) # TODO(yiwen) check whether need to be 37+1

        self.num_local_layer = num_local_layer
        if num_local_layer > 0:
            self.word_emb = nn.Linear(music_dim, embed_dim*self.vqvae.vqvae.max_person)
            self.music_linear = nn.Linear(150, self.block_size2)
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
    
    def forward(self, idx, src_mask, word_emb):
        
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
            word_emb = self.word_emb(word_emb) # 32, 150, 3072
            compressed_music_emb = self.music_linear(word_emb.permute(0,2,1)).permute(0,2,1) # 32, 37, 3072
            token_embeddings = self.pos_embed2(token_embeddings) # add positional encoding to motion tokens B, T, H*D'' 32, 37, 3*1024

            for module in self.cross_att: # modality fusion
                # token_embeddings = module(token_embeddings, word_emb) # NOTE(yiwen) ablation for cross-atten
                token_embeddings = module(token_embeddings, word_emb, compressed_music_emb) # temporal coherent cross-atten

        token_embeddings = token_embeddings.view(B, T*H, D)  
        x = self.pos_embed1(token_embeddings) 
        
        for block in self.blocks: # motion self-atten
            x = block(x, src_mask)

        return x


class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.max_person = 3 # TODO(yiwen) add to args
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(self.max_person*embed_dim, num_vq, bias=False)
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

    def forward(self, x, src_mask):
        for block in self.blocks:
            x = block(x, src_mask)  
        x = self.ln_f(x)
        x = x.view(x.shape[0], -1, self.max_person*x.shape[-1]) # B, T, HD
        logits = self.head(x) # B, T, codebook_cls   2D codebook --> index class
        return logits

    


        


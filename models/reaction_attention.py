'''
compute global reaction weight matrix
    (seq_length x max_person) x (seq_length x max_person)

get max response 1D tensor by argmax on each column
    seq_length x max_person

inject max response style guidance to the original motion seq through AdaIN
'''

import torch
import torch.nn as nn

import numpy as np

def compute_timeCorr_matrix(T, alpha=0.1):
    """
    Compute a T x T weight matrix where W[t_i, t_j] = exp(-alpha * |t_i - t_j|).
    """
    t = np.arange(1, T + 1) 
    distance_m = np.abs(t[:, None] - t[None, :])  # Broadcasting to compute pairwise differences
    weight_matrix = np.exp(-alpha * distance_m)
    
    return torch.tensor(weight_matrix, dtype=torch.float32)


class Reaction_Attention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x, alpha=0.1):
        B, H, T, D = x.size() # B, 3, 37, 32
        x_in = x.view(B, H*T, D)
        HT = H*T
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, HT, D)
        q = self.query(x).view(B, HT, D)
        v = self.value(x).view(B, HT, D) # TODO(yiwen) 这里的作为value的x还需要通过linear吗
        # causal self-attention; 
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # TODO(yiwen) Spatial Correlation Matrix input B, H, T', D' loader中额外需要拿trans的三维
        # cal 3x3 weight matrix using trans: W[i,j]=1/(1+distance[i,j]) 
        # if H_i or H_j is padded, set distance[i,j]=inf 

        # return person_corr = T' x (HxH) weight
    
        # TODO(yiwen) Temporal Correlation Matrix
        time_corr = compute_timeCorr_matrix(T, alpha) # TxT

        st_corr = np.zeros((T * H, T * H))

        for i in range(T):
            for j in range(T):
                weight = time_corr[i, j]
                scaled_block = weight * person_corr[i]
                st_corr[i * H:(i + 1) * H, j * H:(j + 1) * H] = scaled_block

        # mask the lower triangular matrix = 0, only can see former frames
        valid_att = torch.triu(att)
        
        # TODO(yiwen) mask the same person positions

        max_corr = # TODO(yiwen) argmax to get the max correspondence in each column (weight, index)

        style = max_corr * v

        return style



class AdaIN(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon

    def forward(self, content, style):
        """
        Perform AdaIN on the motion sequence.
        Args:
            content (torch.Tensor): Input motion sequence of shape (B, HT, D).
            style_mean (torch.Tensor): Target style mean of shape (B, D).
            style_std (torch.Tensor): Target style std of shape (B, D).
        Returns:
            torch.Tensor: Transformed motion sequence of shape (B, T, D).
        """
        style_mean = style.mean(dim=1, keepdim=True)  # (B, 1, D)
        style_std = style.std(dim=1, keepdim=True) + self.epsilon # (B, 1, D)

        # Compute mean and std for content along the time axis (T)
        content_mean = content.mean(dim=1, keepdim=True)  # (B, 1, D)
        content_std = content.std(dim=1, keepdim=True) + self.epsilon  # (B, 1, D)

        normalized_content = (content - content_mean) / content_std  # (B, T, D)

        transformed_content = normalized_content * style_std.unsqueeze(1) + style_mean.unsqueeze(1)

        return transformed_content


if __name__ == "__main__":
    
    motion = torch.rand(2, 148, 3)  # B, T, D

    reaction_attn = Reaction_Attention()

    style = reaction_attn(motion)

    # style_mean = torch.tensor([[0.5, 0.6, 0.7], [0.2, 0.3, 0.4]])  # B, D
    # style_std = torch.tensor([[1.0, 1.2, 1.1], [0.9, 1.1, 1.3]])  # B, D

    adain = AdaIN()
    transformed = adain(motion, style) # transformed.shape = motion.shape


    import pdb
    pdb.set_trace()

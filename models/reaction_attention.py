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
import math


class Reaction_Attention(nn.Module):

    def __init__(self, embed_dim=1024):
        super().__init__()
        assert embed_dim % 8 == 0
  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Linear(embed_dim, embed_dim).to(self.device)
        self.linear2 = nn.Linear(embed_dim, embed_dim).to(self.device)

        self.proj = nn.Linear(embed_dim, embed_dim).to(self.device)

    def forward(self, x_in, alpha=0.1):
        B, HT, D = x_in.size()
        T = 37 # TODO(yiwen) optimize format
        H = HT // T
        
        motionf1 = self.linear1(x_in) # B, HT, D  2, 111, 1024
        motionf2 = self.linear2(x_in)

        # spatial-temporal correlation B, HT, HT
        st_corr = (motionf1 @ motionf2.transpose(-2, -1)) * (1.0 / math.sqrt(motionf2.size(-1))) # the similarity matrix

        valid_corr = st_corr

        # NOTE(yiwen) mask the same person positions (3 TxT metrix in diagonal)
        all_mask = torch.ones((HT, HT), device=self.device)
        mask = torch.zeros((T, T), device=self.device)
        for h in range(H):
            all_mask[h*T:(h+1)*T,h*T:(h+1)*T] = mask
        valid_corr = all_mask * valid_corr

        # maxpooling to get the most significant reaction
        max_corr, idx = torch.max(valid_corr, dim=-1) # idx is the max correspondense index for each motion token
 
        save_feat = torch.zeros([B, HT, D], device=self.device)
        for b in range(B):
            save_feat[b] = motionf2[b, idx[b]]    

        return save_feat # as style



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
        style_std = style.std(dim=1, keepdim=True) + self.epsilon  # (B, 1, D)

        # Compute mean and std for content along the temporal dimension (HT)
        content_mean = content.mean(dim=1, keepdim=True)  # (B, 1, D)
        content_std = content.std(dim=1, keepdim=True) + self.epsilon  # (B, 1, D)

        # Normalize content and apply style statistics
        normalized_content = (content - content_mean) / content_std  # (B, HT, D)
        transformed_content = normalized_content * style_std + style_mean  # (B, HT, D)

        return transformed_content

if __name__ == "__main__":
    
    motion = torch.rand(2, 3, 37, 1024)  # B, H, T', D'
    reaction_attn = Reaction_Attention()

    B, H, T, D = motion.shape
    motion = motion.view(B, H*T, D)
    
    style = reaction_attn(motion) # B, HT, D

    adain = AdaIN()
    transformed = adain(motion, style) # transformed.shape = motion.shape
    transformed = transformed.view(B, H, T, D)


"""
Autoencoder motion feature extractor network
    - movement encoder --> motion encoder --> motion decoder --> movement decoder
    - 2 encoders will be kept as feature extractors in calculating FID
"""
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence

class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)


    def forward(self, inputs): 
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return self.out_net(outputs)


class MovementConvDecoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(MovementConvDecoder, self).__init__()
        self.main = nn.Sequential(
            # keep time dimension unchanged
            # latent_size -> hidden_size
            nn.ConvTranspose1d(latent_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            # hidden_size -> output_size
            nn.ConvTranspose1d(hidden_size, output_size, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)

    def forward(self, latent_inputs):
        latent_inputs = latent_inputs.permute(0, 2, 1)  # (batch, channels, seq_len)
        outputs = self.main(latent_inputs)  
        outputs = outputs.permute(0, 2, 1) 
        return self.out_net(outputs)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(MotionEncoderBiGRUCo, self).__init__()
        self.device = device

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True, enforce_sorted=False)

        gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MotionDecoderGRU(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size, seq_len, device):
        super(MotionDecoderGRU, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        self.gru = nn.GRU(hidden_size//2, hidden_size, batch_first=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, latent_inputs):
        hidden = self.latent_to_hidden(latent_inputs).unsqueeze(0)
        repeated_latent = latent_inputs.unsqueeze(1).repeat(1, self.seq_len, 1) # 128, 148, 512
        gru_outputs, _ = self.gru(repeated_latent, hidden) # 128, 148, 1024

        outputs = self.output_net(gru_outputs)
        return outputs
# B，H*T, D (24*6+3) 输入三维算fid

class MovementMotionAutoencoder(nn.Module):
    def __init__(self, 
                 movement_input_size, 
                 movement_hidden_size, 
                 movement_latent_size, 
                 motion_input_size,
                 motion_hidden_size,
                 motion_latent_size,
                 motion_seq_len, 
                 device):
        super(MovementMotionAutoencoder, self).__init__()
        self.movement_encoder = MovementConvEncoder(movement_input_size, 
                                                    movement_hidden_size, 
                                                    movement_latent_size)
        self.motion_encoder = MotionEncoderBiGRUCo(motion_input_size, 
                                                   motion_hidden_size, 
                                                   motion_latent_size, 
                                                   device)
        self.motion_decoder = MotionDecoderGRU(motion_latent_size, 
                                               motion_hidden_size, 
                                               motion_input_size, 
                                               motion_seq_len, 
                                               device)
        self.movement_decoder = MovementConvDecoder(movement_latent_size, 
                                                    movement_hidden_size, 
                                                    movement_input_size)
        self.unit_length = 4
        self.motion_seq_len = motion_seq_len
        self.m_lens = None
        
    def forward(self, inputs):
        latent = self.movement_encoder(inputs) # input dim = D-4   128, 37, 512
        # print(f"debug1 -- latent shape {latent.shape}")
        self.m_lens = torch.tensor([self.motion_seq_len for i in range(inputs.shape[0])])
        self.m_lens = self.m_lens // self.unit_length # NOTE(yw) num of tokens
        motion_embedding = self.motion_encoder(latent, self.m_lens)
        # print(f"debug2 -- motion shape {motion_embedding.shape}")
        
        reconstructed_latent = self.motion_decoder(motion_embedding)
        # print(f"debug3 -- rec motion shape {reconstructed_latent.shape}")
        reconstructed = self.movement_decoder(reconstructed_latent)
        # print(f"debug4 -- rec movement shape {reconstructed.shape}")
        return reconstructed
    
    def save_weights(self, filepath):
        # 保存权重到字典
        weights_dict = {
            "movement_encoder": self.movement_encoder.state_dict(),
            "movement_decoder": self.movement_decoder.state_dict(),
            "motion_encoder": self.motion_encoder.state_dict(),
            "motion_decoder": self.motion_decoder.state_dict(),
        }
        torch.save(weights_dict, filepath)



if __name__ == '__main__':
    
    dataset_name = 'aistpp'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unit_length = 4
    
    if dataset_name == 'aistpp':
        dim_pose = 151
        max_motion_length = 196
        dim_movement_dec_hidden: 512
        dim_movement_enc_hidden: 512
        dim_motion_hidden = 1024
        dim_text_hidden = 512
        dim_coemb_hidden = 512
        dim_movement_latent = 512
    
    movement_enc = MovementConvEncoder(input_size=dim_pose-4, 
                                       hidden_size=dim_movement_enc_hidden, 
                                       output_size=dim_movement_latent)

    movement_dec = MovementConvDecoder(latent_size=dim_movement_latent, 
                                       hidden_size=dim_movement_enc_hidden, 
                                       output_size=dim_pose-4)
    
    motion_enc = MotionEncoderBiGRUCo(input_size=dim_movement_latent,
                                      hidden_size=dim_motion_hidden,
                                      output_size=dim_coemb_hidden,
                                      device=device)
    
    motion_dec = MotionDecoderGRU(latent_size=dim_coemb_hidden,
                                  hidden_size=dim_motion_hidden,
                                  output_size=dim_movement_latent)
    

    
    inputs = torch.rand(32, 1, 148, 151)  # TODO from dataloader
    m_lens = inputs.shape[2] 
    
    latent = movement_enc(inputs) # input dim = D-4
    m_lens = m_lens // unit_length # NOTE(yw) num of tokens
    motion_embedding = motion_enc(latent, m_lens)
    
    reconstructed_latent = motion_dec(motion_embedding)
    reconstructed = movement_dec(reconstructed_latent)
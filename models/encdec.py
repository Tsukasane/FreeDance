import torch.nn as nn
from models.resnet import Resnet1D, Resnet2D

class PrintModule(nn.Module):
    def __init__(self, me=''):
        super().__init__()
        self.me = me

    def forward(self, x):
        print(self.me, x.shape)
        return x
    
class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()
        
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
    


class Encoder2D(nn.Module):
    def __init__(self,
                 input_emb_width=3, 
                 output_emb_width=512, # 512
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        
        # Initial convolution layer
        blocks.append(nn.Conv2d(input_emb_width, width, kernel_size=3, stride=1, padding=1))
        blocks.append(nn.ReLU())

        # Downsampling layers
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv2d(input_dim, width, kernel_size=filter_t, stride=stride_t, padding=pad_t),
                Resnet2D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)

        # dim H: 3-->512-->3
        blocks.append(nn.Conv2d(width, input_emb_width, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*blocks)

        # add a linear layer, to project the spatial dim to codebook dim.
        self.D = 152 # T
        dp_dim = self.D // 2**(down_t)# D'=38
        self.projection = nn.Linear(dp_dim, output_emb_width)
        

    def forward(self, x):
        x = self.model(x)
        B, H, Tp, Dp = x.shape
        x = self.projection(x.view(B*H*Tp, Dp))
        x = x.view(B, H, Tp, -1)

        return x


class Decoder2D(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2

        # add a linear layer, to project the spatial dim to codebook dim.
        dp_dim = 38# D'=38
        self.projection = nn.Linear(output_emb_width, dp_dim)

        # Initial convolution layer
        blocks.append(nn.Conv2d(input_emb_width, width, kernel_size=3, stride=1, padding=1))
        blocks.append(nn.ReLU())

        # Upsampling layers
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet2D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(width, out_dim, kernel_size=3, stride=1, padding=1)
            )
            blocks.append(block)

        # Final layers to produce the output embedding
        blocks.append(nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv2d(width, input_emb_width, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*blocks)

        # real_D = 151 # TODO(yiwen) 上采样之后只能恢复到148，或者在进入encoder之前padding一维到152
        # self.projection1 = nn.Linear(output_emb_width, dp_dim)

    def forward(self, x):
        B, H, Tp, Dp = x.shape
        x = self.projection(x.view(B*H*Tp, Dp))

        x = x.view(B, H, Tp, -1)
        x = self.model(x)
        return x


if __name__ == "__main__":
    # Testing the Decoder2D
    import torch
    # Hyperparameters
    input_emb_width = 3
    output_emb_width = 32
    batch_size = 4
    H, Tp, Dp = 3, 37, 32  # Input dimensions

    # Initialize the decoder
    decoder = Decoder2D(input_emb_width=input_emb_width, output_emb_width=output_emb_width, down_t=2)

    # Create random input tensor
    x = torch.randn(batch_size, H, Tp, Dp)

    # Forward pass
    try:
        output = decoder(x)
        print("Output shape:", output.shape)
    except Exception as e:
        print("Error during forward pass:", e)

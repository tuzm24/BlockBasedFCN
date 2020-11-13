import torch.nn as nn



class Conv2d_Recycle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, bias=True, padding_mode='zeros'):
        super(Conv2d_Recycle, self).__init__()



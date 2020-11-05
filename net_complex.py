import torch.nn as nn
from mtorchsummary import summary
def Conv3x3Bn(in_channels, out_channels, stride, non_linear='ReLU', padding = 1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class PlainNetwork(nn.Sequential):
    def __init__(self, num_input_features, output_dim):
        super(PlainNetwork, self).__init__()
        convlist = []
        convlist.append(Conv3x3Bn(num_input_features, 32, 1, 'ReLU',0))
        convlist.append(Conv3x3Bn(32, 32, 1, 'ReLU',0))
        convlist.append(Conv3x3Bn(32, 32, 1))
        convlist.append(Conv3x3Bn(32,32,1))
        convlist.append(Conv3x3Bn(32,32,1))
        convlist.append(Conv3x3Bn(32,32,1))
        convlist.append(Conv3x3Bn(32,32,1))
        convlist.append(Conv3x3Bn(32,output_dim,1))
        self.layers = nn.Sequential(*convlist)

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__=='__main__':
    m = PlainNetwork(3, 3).cuda()
    summary(m, (3, 1920, 1080), 1)
import torch.nn as nn
from mtorchsummary import summary
# from torchvision.models.densenet import *
from model.densenet import *
from model.resnet import *
from model.inception import *
from model.vgg import *
import torch
from option import args
from model.edsr import EDSR
from model.rdn import RDN
from model.rcan import RCAN
from model.ddbpn import DDBPN
from model.srcnn import SRCNN
from model.FCNSS import FCN32s
from model.rdn_denoise import RDN_DENOISE
from model.didn import DIDN
from model.dncnn import DnCNN

from stastic_tool.ptflops.flops_counter import get_model_complexity_info

def Conv3x3Bn(in_channels, out_channels, stride, non_linear='ReLU', padding = 1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class PlainNetwork(nn.Sequential):
    def __init__(self, num_input_features, output_dim):
        super(PlainNetwork, self).__init__()
        self.convlist = []
        self.convlist.append(Conv3x3Bn(num_input_features, 32, 1, 'ReLU'))
        self.convlist.append(Conv3x3Bn(32, 32, 1, 'ReLU'))
        self.convlist.append(Conv3x3Bn(32, 32, 1))
        self.convlist.append(Conv3x3Bn(32,32,1))
        self.convlist.append(Conv3x3Bn(32,32,1))
        self.convlist.append(Conv3x3Bn(32,32,1))
        self.convlist.append(Conv3x3Bn(32,32,1))
        self.convlist.append(Conv3x3Bn(32,output_dim,1))
        self.layers = nn.Sequential(*self.convlist)

    def forward(self, x):
        x = self.layers(x)

        return x

if __name__=='__main__':
    torch.set_grad_enabled(False)
    # m = PlainNetwork(3, 3).cuda()
    # m = densenet161().cuda()
    # m = EDSR(args).cuda()
    m = resnet101().cuda()
    # m = RDN(args).cuda()
    # m = RCAN(args).cuda()
    # m = SRCNN(args).cuda()
    # m = FCN32s().cuda()
    # m = RDN_DENOISE().cuda()
    # m = DIDN().cuda() # Deep Iterative Down-Up CNN for Image Denoising (http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Yu_Deep_Iterative_Down-Up_CNN_for_Image_Denoising_CVPRW_2019_paper.pdf)
    # m = DnCNN().cuda()
    with torch.no_grad():
        summary(m, (3, 226, 226), 1)
        macs, params = get_model_complexity_info(m, (3, 226, 226))
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
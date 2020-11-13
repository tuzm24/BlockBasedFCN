from torch import nn


class SRCNN(nn.Module):
    def __init__(self, args):
        super(SRCNN, self).__init__()
        self.scale = args.scale[0]
        self.conv1 = nn.Conv2d(args.srcnn_input_channel, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, args.srcnn_input_channel, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = nn.Upsample(scale_factor=self.scale)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
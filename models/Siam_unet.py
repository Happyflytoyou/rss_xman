import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

# For nested 3 channels are required

class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


# Nested Unet

class SiamUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(SiamUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.bn1 = nn.BatchNorm2d(filters[0])
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.bn4 = nn.BatchNorm2d(filters[3])
        self.bn5 = nn.BatchNorm2d(filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final1 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final2 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final3 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final4 = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x1, x2):
        x1_0_0 = self.conv0_0(x1)
        x2_0_0 = self.conv0_0(x2)
        x0_0 = x2_0_0 - x1_0_0
        x0_0 = self.bn1(x0_0)

        x1_1_0 = self.conv1_0(self.pool(x1_0_0))
        x2_1_0 = self.conv1_0(self.pool(x2_0_0))
        x1_0 = x1_1_0 - x2_1_0
        x1_0 = self.bn2(x1_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x1_2_0 = self.conv2_0(self.pool(x1_1_0))
        x2_2_0 = self.conv2_0(self.pool(x2_1_0))
        x2_0 = x2_2_0 - x1_2_0
        x2_0 = self.bn3(x2_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x1_3_0 = self.conv3_0(self.pool(x1_2_0))
        x2_3_0 = self.conv3_0(self.pool(x2_2_0))
        x3_0 = x2_3_0 - x1_3_0
        x3_0 = self.bn4(x3_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x1_4_0 = self.conv4_0(self.pool(x1_3_0))
        x2_4_0 = self.conv4_0(self.pool(x2_3_0))
        x4_0 = x2_4_0 - x1_4_0
        x4_0 = self.bn5(x4_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)
        # output = (output1 + output2 + output3 +output4) / 4

        return output1, output2, output3, output4
class SiamUNetU(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(SiamUNetU, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv1 = conv_block_nested(filters[0] * 2, filters[0], filters[0])
        self.conv2 = conv_block_nested(filters[1] * 2, filters[1], filters[1])
        self.conv3 = conv_block_nested(filters[2] * 2, filters[2], filters[2])
        self.conv4 = conv_block_nested(filters[3] * 2, filters[3], filters[3])
        self.conv5 = conv_block_nested(filters[4] * 2, filters[4], filters[4])


        self.bn1 = nn.BatchNorm2d(filters[0])
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.bn4 = nn.BatchNorm2d(filters[3])
        self.bn5 = nn.BatchNorm2d(filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x1, x2):
        x1_0_0 = self.conv0_0(x1)
        x2_0_0 = self.conv0_0(x2)
        x0_0 = self.conv1(torch.cat([x2_0_0, x1_0_0], 1))

        x0_0 = self.bn1(x0_0)

        x1_1_0 = self.conv1_0(self.pool(x1_0_0))
        x2_1_0 = self.conv1_0(self.pool(x2_0_0))
        x1_0 = self.conv2(torch.cat([x2_1_0, x1_1_0], 1))
        x1_0 = self.bn2(x1_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x1_2_0 = self.conv2_0(self.pool(x1_1_0))
        x2_2_0 = self.conv2_0(self.pool(x2_1_0))
        x2_0 = self.conv3(torch.cat([x2_2_0, x1_2_0], 1))
        x2_0 = self.bn3(x2_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x1_3_0 = self.conv3_0(self.pool(x1_2_0))
        x2_3_0 = self.conv3_0(self.pool(x2_2_0))
        x3_0 = self.conv4(torch.cat([x2_3_0, x1_3_0], 1))
        x3_0 = self.bn4(x3_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x1_4_0 = self.conv4_0(self.pool(x1_3_0))
        x2_4_0 = self.conv4_0(self.pool(x2_3_0))
        x4_0 = self.conv5(torch.cat([x2_4_0, x1_4_0], 1))
        x4_0 = self.bn5(x4_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output

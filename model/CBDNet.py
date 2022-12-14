import torch
import torch.nn as nn
import torch.nn.functional as F

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, features):
        super(UNet, self).__init__()

        self.inc = nn.Sequential(
            single_conv(1, features),
            single_conv(features, features)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(features, features*2),
            single_conv(features*2, features*2),
            single_conv(features*2, features*2)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(features*2, features*4),
            single_conv(features*4, features*4),
            single_conv(features*4, features*4),
            single_conv(features*4, features*4),
            single_conv(features*4, features*4),
            single_conv(features*4, features*4)
        )

        self.up1 = up(features * 4)
        self.conv3 = nn.Sequential(
            single_conv(features*2, features*2),
            single_conv(features*2, features*2),
            single_conv(features*2, features*2)
        )

        self.up2 = up(features*2)
        self.conv4 = nn.Sequential(
            single_conv(features, features),
            single_conv(features, features)
        )

        self.outc = outconv(features, 1)

    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out


class CBDNet(nn.Module):
    def __init__(self, features = 8):
        super(CBDNet, self).__init__()
        self.unet = UNet(features)

    def forward(self, image):
        x = self.unet(image) + image
        return x
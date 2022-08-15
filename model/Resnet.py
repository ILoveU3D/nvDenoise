import torch.nn as nn
from torchvision.models.vgg import vgg13
from torchsummary import summary

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.begin = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        net = vgg13(pretrained=False).cuda()
        self.backbone = list(net.children())[0]
        self.fcn = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=(2, 2), padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=(2, 2), padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=(2, 2), padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=(2, 2), padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=(2, 2), padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 1, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

    def forward(self, image):
        x = self.begin(image)
        x = self.backbone(x)
        x = image - self.fcn(x)
        return x

    def initiate(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.zeros_(m.weight.data)
import torch
import torch.nn as nn

class Pix2PixHD(nn.Module):
    def __init__(self, features = 32):
        super(Pix2PixHD, self).__init__()
        self.G11 = nn.Sequential(
            nn.Conv2d(1, features * 2, kernel_size=3,stride=(2,2),padding=1),
            nn.ReLU(True),
        )
        self.G21 = nn.Sequential(
            nn.Conv2d(1, features * 2, kernel_size=3,stride=(2,2),padding=1),
            nn.ReLU(True),
        )
        self.G22 = nn.Sequential(
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        self.G23 = nn.Sequential(
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        self.G24 = nn.Sequential(
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        self.G25 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, features * 2, kernel_size=2, stride=(2, 2), padding=0),
            nn.ReLU(True),
        )
        self.G12 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=(2, 2), padding=0),
            nn.ReLU(True),
            nn.Conv2d(features, 1, kernel_size=1, padding=0),
            nn.ReLU(True),
        )
        self.G31 = nn.Sequential(
            nn.Conv2d(1, features, kernel_size=3, stride=(2, 2), padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(features, 1, kernel_size=4, stride=(2, 2), padding=1),
            nn.ReLU(True),
        )

    def forward(self, image):
        x = self.G11(image)
        z = nn.functional.avg_pool2d(image, kernel_size=3, stride=2, padding=1)
        z = self.G21(z)
        z = self.G22(z) + z
        z = self.G23(z) + z
        z = self.G24(z) + z
        z = self.G25(z)
        return self.G12(x), self.G12(x+z), self.G31(image)

    def initiate(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.zeros_(m.weight.data)
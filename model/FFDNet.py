import torch
import torch.nn as nn

class FFDNet(nn.Module):
    def __init__(self, features = 16):
        super(FFDNet, self).__init__()
        self.DnCNN_B = nn.Sequential(
            nn.Conv2d(1, features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, 1, kernel_size=3, padding=1),
        )
        self.DnCNN_L = nn.Sequential(
            nn.Conv2d(1, features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, 1, kernel_size=3, padding=1),
            nn.Softmax(dim=1),
        )
        self.DnCNN_F = nn.Sequential(
            nn.Conv2d(2, features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, 1, kernel_size=3, padding=1),
        )

    def forward(self, image):
        v = self.DnCNN_B(image)
        l = self.DnCNN_L(image)
        v = image - v
        l = image * l
        t = torch.cat([v,l], dim=1)
        t = self.DnCNN_F(t)
        return v - t
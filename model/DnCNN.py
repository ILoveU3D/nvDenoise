import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, features = 16):
        super(DnCNN, self).__init__()
        self.DnCNN = nn.Sequential(
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

    def forward(self, image):
        t = self.DnCNN(image)
        return image - t
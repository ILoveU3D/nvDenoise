import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from options import size

class NoiseSet(Dataset):
    def __init__(self, root):
        self.items = []
        for dir in os.listdir(root):
            clean = os.listdir(os.path.join(root, dir, "clean"))[0]
            for item in os.listdir(os.path.join(root, dir, "noise")):
                self.items.append({"noise":os.path.join(root, dir, "noise", item),
                                   "clean":os.path.join(root, dir, "clean", clean)})
        self.normalization = transforms.Normalize(0, 1)

    def __getitem__(self,index):
        noise = np.fromfile(self.items[index]["noise"], "float32")
        clean = np.fromfile(self.items[index]["clean"], "float32")
        noise = torch.from_numpy(np.reshape(noise, size))
        clean = torch.from_numpy(np.reshape(clean, size))
        return noise, clean

    def __len__(self):
        return len(self.items)
import os
import torch
import numpy as np
import torchvision.transforms

from options import checkpointPath, model, outputPath, size, root

net = torch.load(os.path.join(checkpointPath, model))
net.eval()

# temp = np.zeros([24,288,3072], dtype="float32")
# for i,dir in enumerate(os.listdir(root)):
#     p = os.path.join(root, str(i+1), "noise")
#     noise = np.fromfile(os.path.join(p, os.listdir(p)[0]), "float32")
#     noise = torch.from_numpy(np.reshape(noise, size))
#     noise = noise.unsqueeze(0)
#     output = net(noise.cuda())
#     output = output.squeeze(0).squeeze(0)
#     temp[i,:,:] = output.detach().cpu().numpy()
#     print(i)
# temp.tofile(os.path.join(outputPath, "output.raw"))

input = torch.from_numpy(np.reshape(np.fromfile(os.path.join(outputPath, "test.raw"), "float32"),size))
input = input.unsqueeze(0).cuda()
output = net(input)
output.detach().cpu().numpy().tofile(os.path.join(outputPath, "output.raw"))
# print(torch.sum(output.cpu() - input))
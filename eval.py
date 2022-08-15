import os
import torch
import numpy as np
from options import checkpointPath, model, outputPath, size

net = torch.load(os.path.join(checkpointPath, model))
net.eval()
input = torch.from_numpy(np.reshape(np.fromfile(os.path.join(outputPath, "test.raw"), "float32"),size))
output = net(input.cuda())
output.detach().cpu().numpy().tofile(os.path.join(outputPath, "output.raw"))
print(torch.sum(output.cpu() - input))
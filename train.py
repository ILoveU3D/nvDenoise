import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary
from visdom import Visdom
from dataloader.loader import NoiseSet
from model.Pix2PixHD import Pix2PixHD
from model.FFDNet import FFDNet
from options import checkpointPath

trainSet = NoiseSet()
trainLoader = DataLoader(trainSet, batch_size=4, shuffle=True)
net = FFDNet().cuda()
# summary(net, (1, 288, 3072))
optimizer = torch.optim.Adam(net.parameters(), lr=10e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
lossFunction = torch.nn.MSELoss(reduction="sum")
viz = Visdom()
viz.line([[0,0]],[0],win="train",opts={"title":"Train Loss","legend":["trainLoss","validLoss"]})

epoch = 500

for i in range(epoch):
    net.train()
    trainLoss = []
    with tqdm(trainLoader) as iterator:
        iterator.set_description("Epoch {}".format(i))
        for idx,data in enumerate(iterator):
            noise, clean = data
            noise, clean = noise.cuda(),clean.cuda()
            output = net(noise)
            optimizer.zero_grad()
            loss = lossFunction(output, clean)
            loss.backward()
            optimizer.step()
            trainLoss.append(loss.item())
            iterator.set_postfix_str("loss:{:.2f}".format(np.mean(np.array(trainLoss))))
    viz.line([[np.mean(np.array(trainLoss)),0]], [i], win="train", update="append")
    scheduler.step()
    if i % 2 == 0:
        torch.save(net, "{}/vae_{}_{:.10f}.pth".format(checkpointPath, i, np.mean(np.array(trainLoss))))
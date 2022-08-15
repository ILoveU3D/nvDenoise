import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary
from visdom import Visdom
from dataloader.loader import NoiseSet
from model.Pix2PixHD import Pix2PixHD
from model.Resnet import Resnet
from options import checkpointPath
from loss import doubleLoss as lossFunction

trainSet = NoiseSet()
trainLoader = DataLoader(trainSet, batch_size=12, shuffle=True)
net = Pix2PixHD().cuda()
net.initiate()
# summary(net, (1, 288, 3072))
optimizer = torch.optim.Adam(net.parameters(), lr=10e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
sigma = 1
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
            output1, output2 = net(noise)
            optimizer.zero_grad()
            loss, loss1, loss2 = lossFunction(output1, output2, clean, sigma)
            loss.backward()
            optimizer.step()
            trainLoss.append(loss.item())
            iterator.set_postfix_str("loss:{:.2f}({:.2f},{:.2f})".format(np.mean(np.array(trainLoss)),loss1,loss2))
    viz.line([[np.mean(np.array(trainLoss)),0]], [i], win="train", update="append")
    scheduler.step()
    sigma = loss1 / loss2
    if i % 10 == 0:
        torch.save(net, "{}/vae_{:.10f}.pth".format(checkpointPath, np.mean(np.array(trainLoss))))
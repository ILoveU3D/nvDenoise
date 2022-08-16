import numpy as np
import torch.nn as nn

def doubleLoss(outputs,label,sigmas):
    loss = np.zeros_like(sigmas)
    lossSum = 0
    for i,output in enumerate(outputs):
        loss[i] = nn.MSELoss(reduction="sum")(output, label)
        lossSum += loss[i] * sigmas[i]
        loss[i] = loss[i].item()
    return lossSum, loss
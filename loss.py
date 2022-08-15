import torch.nn as nn

def doubleLoss(output1,output2,label,sigma):
    loss1 = nn.MSELoss(reduction="sum")(output1, label)
    loss2 = nn.MSELoss(reduction="sum")(output2, label)
    return loss1 * sigma + loss2, loss1.item(), loss2.item()
import torch.nn as nn

class Loss_UNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, labels):
        """
        :param pred: (batchsize,30,7,7)的网络输出数据
        :param labels: (batchsize,30,7,7)的样本标签数据
        :return: 当前批次样本的平均损失
        """

        return loss/n_batch
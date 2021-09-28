
from torch.utils.data.dataset import Dataset
from Dataset import SEGData
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Transforms import ImageTransform
from UNet import UNet

#DATASET_PATH = 'C:/Users/97090/Desktop/dataset/VOCdevkit/VOC2012/'
DATASET_PATH = 'D:/DATASETS/VOC2012/'

if __name__ == '__main__':

    epoch = 50
    batchsize = 15
    lr = 0.001

    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    train_data = SEGData(dataset_dir = DATASET_PATH, is_train = True, transform = ImageTransform(), target_transform = ImageTransform(needCrop = True))
    train_dataloader = DataLoader(SEGData(dataset_dir = DATASET_PATH, is_train = True, transform = ImageTransform(), target_transform = ImageTransform(needCrop = True)), batch_size = batchsize, shuffle = True)
    model = UNet(in_channel = 3, out_channel = 3).to(device = device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = 0.0005)
    for e in range(epoch):
        model.train()
        for i, (inputs, label) in enumerate(train_dataloader):
            inputs = inputs.to(device = device)
            label = label.to(device = device)
            pred = model(inputs)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch %d/%d| Step %d/%d| Loss: %.2f"%(e, epoch, i, len(train_data) // batchsize, loss))
        if (e + 1) % 10 == 0:
            torch.save(model, "C:/Users/97090/Desktop/dataset/VOCdevkit/VOC2012/models_pkl/YOLOv1_epoch" + str(e + 1) + ".pkl")
            print('saved one model')
            #compute_val_map(model)
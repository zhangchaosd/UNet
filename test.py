from Dataset import SEGData
from LossFunction import Loss_UNet
import torch
from torch.utils.data import DataLoader
from Transforms import ImageTransform
from UNet import UNet
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

#from torchvision.io import image

DATASET_PATH = 'C:/Users/97090/Desktop/dataset/VOCdevkit/VOC2012/'
#DATASET_PATH = 'D:/DATASETS/VOC2012/'

def ss(image):
    print(image.dtype)
    print(image.shape)
    plt.axis("off")
    image = image.transpose(1,2,0)
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    train_data = SEGData(dataset_dir = DATASET_PATH, is_train = False, transform = ImageTransform(), target_transform = ImageTransform(needCrop = True))
    train_dataloader = DataLoader(SEGData(dataset_dir = DATASET_PATH, is_train = False, transform = ImageTransform(), target_transform = ImageTransform(needCrop = True)), batch_size = 1, shuffle = True)
    #model = UNet(in_channel = 3, out_channel = 1).to(device = device)
    model = torch.load(DATASET_PATH + 'models_pkl/unet_epoch10.pkl').to(device = device)
    for i, (inputs, label) in enumerate(train_dataloader):
        inputs = inputs.to(device = device)
        pred = model(inputs).cpu()


        inputs = inputs.squeeze(dim = 0)
        pred = pred * 255.
        pred = torch.cat((pred,label),dim = 3).detach()
        pred = pred.squeeze(dim = 0).numpy()
        pred = pred.astype(np.ubyte) #1,w,h
        rgbpred = np.zeros((3, pred.shape[1], pred.shape[2]))
        rgbpred[0,:,:] = pred[0,:,:] >> 5
        rgbpred[1,:,:] = (pred[0,:,:] << 3) >> 5
        rgbpred[2,:,:] = (pred[0,:,:] << 6) >> 6
        tp = torch.tensor(rgbpred)
        inputs = transforms.functional.center_crop(inputs, tp.shape[1]) / 255
        tp = torch.cat((inputs.cpu(),tp),dim = 2).detach()
        ss(tp.numpy())
        input()
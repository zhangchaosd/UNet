from Dataset import SEGData
from LossFunction import Loss_UNet
import torch
from torch.utils.data import DataLoader
from Transforms import ImageTransform
from UNet import UNet

DATASET_PATH = 'C:/Users/97090/Desktop/dataset/VOCdevkit/VOC2012/'
#DATASET_PATH = 'D:/DATASETS/VOC2012/'

if __name__ == '__main__':
    epoch = 50
    batchsize = 3
    lr = 0.0002

    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    train_data = SEGData(dataset_dir = DATASET_PATH, is_train = True, transform = ImageTransform(), target_transform = ImageTransform(needCrop = True))
    train_dataloader = DataLoader(SEGData(dataset_dir = DATASET_PATH, is_train = True, transform = ImageTransform(), target_transform = ImageTransform(needCrop = True)), batch_size = batchsize, shuffle = True)
    #model = UNet(in_channel = 3, out_channel = 1).to(device = device)
    model = torch.load(DATASET_PATH + 'models_pkl/YOLOv1_epoch50.pkl').to(device = device)
    loss_function = Loss_UNet()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.99, weight_decay = 0.0005)
    #optimizer = torch.optim.Adam(model.parameters())
    for e in range(epoch):
        model.train()
        for i, (inputs, label) in enumerate(train_dataloader):
            inputs = inputs.to(device = device)
            label = label.to(device = device)
            label = label / 255.0
            pred = model(inputs)
            loss = loss_function(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch %d/%d| Step %d/%d| Loss: %.2f"%(e, epoch, i, len(train_data) // batchsize, loss))
        if (e + 1) % 10 == 0:
            torch.save(model, DATASET_PATH + 'models_pkl/UNet_epoch' + str(e + 1) + '.pkl')
            print('saved one model')
        #compute_val_map(model)
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.io import read_image

class SEGData(Dataset):
    def __init__(self, dataset_dir, is_train = True, transform = None, target_transform = None):
        self.filenames = []
        if is_train:
            with open(dataset_dir + "ImageSets/Segmentation/train.txt", 'r') as f:
                self.filenames = [x.strip() for x in f]
        else:
            with open(dataset_dir + "ImageSets/Segmentation/val.txt", 'r') as f:
                self.filenames = [x.strip() for x in f]
        self.img_dir = dataset_dir + "JPEGImages/"
        self.img_labels = dataset_dir + "SegmentationClass/"
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return  len(self.filenames)

    def __getitem__(self, idx):
        img = read_image(self.img_dir + self.filenames[idx] + ".jpg").type(torch.float)  #TO cuda
        label = read_image(self.img_labels + self.filenames[idx] + ".jpg").type(torch.float)  #TO cuda
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            labels = self.target_transform(label)
        return img, label

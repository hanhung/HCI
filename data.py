import os
import torch
import numpy as np

from torch.utils.data import Dataset

class HandWritingDataset(object):
    def __init__(self, root='./dataset/', split='train'):
        self.root = root
        self.split = split
        self.imgs = np.load(root + split + '_images.npy')
        self.bbox = np.load(root + split + '_bbox.npy')
        self.label = np.load(root + split + '_label.npy')

        randomize = np.arange(self.imgs.shape[0])
        np.random.shuffle(randomize)
        self.imgs = self.imgs[randomize]
        self.bbox = self.bbox[randomize]
        self.label = self.label[randomize]

    def __getitem__(self, idx):
        img = torch.from_numpy(self.imgs[idx] / 255).float().permute(2, 0, 1)
        temp = self.bbox[idx]
        bbox = [temp[0], temp[2], temp[1] - temp[0], temp[3] - temp[2]]
        bbox = torch.from_numpy(np.array(bbox)).float()
        label = torch.from_numpy(np.array(self.label[idx])).long()

        target = {}
        target["img"] = img
        target["boxes"] = bbox
        target["labels"] = label
        
        return target

    def __len__(self):
        return self.imgs.shape[0]
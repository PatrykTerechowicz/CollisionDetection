from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import cv2
from torch import tensor
import numpy as np
import os.path
from torchvision.transforms import ToPILImage, ToTensor, Normalize, Resize, RandomHorizontalFlip
import time
from random import random

class CollisionDataset(Dataset):
    def __init__(self, datafolder):
        classFolders = ['free', 'blocked_left', 'blocked_right', 'blocked_all']
        print(f'Found classes: {classFolders}')
        self.len = 0
        self.dataset = []
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.topil = ToPILImage()
        self.totensor = ToTensor()
        self.resize = Resize((224, 224))
        self.randomflip = RandomHorizontalFlip(0.5)
        self.flip = RandomHorizontalFlip(1.0)
        for i in range(len(classFolders)):
            path = os.path.join(datafolder, classFolders[i])
            folder = os.listdir(path)
            for file in folder:
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path):
                    self.dataset.append([file_path, i])
                    self.len += 1



    def __len__(self):
        return self.len

    def __getitem__(self, key):
        if type(key) is not int:
            key = key.item()
        img_path = self.dataset[key][0]
        label = self.dataset[key][1]
        img = cv2.imread(img_path)
        img = self.topil(img)
        img = self.resize(img)
        if label == 0 or label == 3:
            img = self.randomflip(img)
        img = self.totensor(img)
        return [img, label]



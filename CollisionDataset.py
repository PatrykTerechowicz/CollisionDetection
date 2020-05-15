from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import cv2
from torch import tensor
import numpy as np
import os.path
class CollisionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, len=None):
        self.dataset = pd.read_csv(os.path.join(root_dir, csv_file))
        if(len!=None):
            assert type(len) is int
            self.dataset = self.dataset[0:len]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if type(key) is not int:
            key = key.item()
        img_name = self.dataset.iloc[key, 1]
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path)

        if self.transform:
            image = self.transform(image)
        labels = self.dataset.iloc[key, 2:]
        labels = tensor(labels.values.astype('double'))
        return [image, labels]

import cv2
import torch
import torchvision.models as models
import alexnetVisualise
import torchvision.transforms as transforms
import os
import pandas as pd
import Utils
import ClassifyingDataset as d
import numpy as np
root_dir = 'dataset/'

dataset = d.CollisionDataset(root_dir)

model = alexnetVisualise.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(4096, 4, bias=4)
device = torch.device('cpu')
model = model.to(device)
model = model.eval()

BEST_MODEL_PATH = 'best_model.pth'
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location='cpu'))


def get_image_stats(img):
    (means, stds) = cv2.meanStdDev(img)
    stats = np.concatenate([means, stds]).flatten()
    return stats

stats = np.zeros(6)
for i in range(len(dataset)):
    img, label = dataset[i]
    img = img.permute((1,2,0))
    img = img.numpy()
    cv2.imshow('XD', img)
    cv2.waitKey(0)

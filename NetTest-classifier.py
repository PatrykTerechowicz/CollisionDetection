import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
import pandas as pd
import Utils
import ClassifyingDataset as d

root_dir = 'dataset/'

dataset = d.CollisionDataset(root_dir)

model = models.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(4096, 4, bias=4)
device = torch.device('cpu')
model = model.to(device)
model = model.eval()

BEST_MODEL_PATH = 'best_model.pth'
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location='cpu'))

wrong = 0
labels = []
for i in range(len(dataset)):
    if (i % 25 == 0):
        print(labels)
        labels = []
        print(f'{i}/{len(dataset)}')
    img, label = dataset[i]
    out = model(img.unsqueeze(0).cpu())
    out = out.squeeze(0)
    out = out.argmax(0)
    labels.append(out.item())
    cv2.waitKey(0)
    if out.item() != label:
        wrong += 1

acc = 1.0 - wrong / len(dataset)
print(f'Siec osiagnela {acc * 100}% poprawnosci.')

cv2.destroyAllWindows()
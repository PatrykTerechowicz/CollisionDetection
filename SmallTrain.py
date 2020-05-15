import torch
import torchvision
import pandas  as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from CollisionDataset import CollisionDataset
from torchvision.transforms import Compose, Resize, RandomGrayscale, ToTensor, ToPILImage, Normalize
import torch.utils.data
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import time
import torchvision.transforms as transforms
import Utils
import torch.optim.lr_scheduler as lr_scheduler
#imgname,left,center,right,nocollision
csv_file = 'collision_labels.csv'
root_dir = 'data/collision/'

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CollisionDataset(csv_file, root_dir, len=50, transform=preprocess)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 30, 30])
batchsize = 1
train_loader = torch.utils.data.DataLoader(
    train_dataset,
	batch_size=batchsize,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
	batch_size=batchsize,
    shuffle=False,
    num_workers=0
)

model = models.alexnet(pretrained=True)

model.classifier[6] = torch.nn.Linear(4096, 4, bias=4)

device = torch.device('cpu')
model = model.to(device)
torch.autograd.set_grad_enabled(True)

NUM_EPOCHS = 50
BEST_MODEL_PATH = 'small_model_hammingloss.pth'
best_accuracy = 0.0

optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
LOSS_TRAIN = []
LOSS_TEST = []
def train():
    global model, best_accuracy
    print('Starting')
    for epoch in range(NUM_EPOCHS):
        scheduler.step()
        model.train()
        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            for i in range(3): #obliczamy 3x gradient xD
                optimizer.zero_grad()
                outputs = model(images)
                loss = nn.BCEWithLogitsLoss()
                loss = loss(outputs, labels.float())
                loss.backward()
                optimizer.step()

        #Testujemy efektywnosc uczenia za pomoca strictEvaluation
        model.eval()
        wrong = 0
        for images, labels in iter(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            for i in range(len(outputs)):
                desired = labels[i].squeeze().tolist()
                output = torch.reshape(outputs[i], (4, 1))
                for j in range(4):
                    if output[j] > 0.5:
                        output[j] = torch.tensor(1.0)
                    else:
                        output[j] = torch.tensor(0.0)
                hamming = Utils.hamming(output, desired)
                if hamming!=0:
                    wrong = wrong + 1
        n = len(test_dataset)

        test_accuracy = 1-wrong/n
        if test_accuracy >= best_accuracy:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_accuracy = test_accuracy
            print(f'Test Accuracy = *{test_accuracy*100}%* at Epoch: {epoch}')
        else:
            print(f'Test Accuracy = {test_accuracy*100}% at Epoch: {epoch}')


if __name__ == "__main__":
    train()
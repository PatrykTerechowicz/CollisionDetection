import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
import pandas as pd
import Utils
csv_file = 'data/collision_labels_clean.csv'
root_dir = 'data/collision/'


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = pd.read_csv(csv_file)

model = models.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(4096, 3, bias=3)
device = torch.device('cpu')
model = model.to(device)
model = model.eval()


BEST_MODEL_PATH = 'best_model_strictLoss.pth'
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location='cpu'))
n = 0
k = 0
for i in range(len(dataset)):
    if(i%100==0):
        print(i)
    img_name =  dataset.iloc[i, 1]
    img_path = os.path.join(root_dir, img_name)
    frame = cv2.imread(img_path)
    image = preprocess(frame)
    image = image.unsqueeze(0)
    image.to(device)
    y = model(image)
    y = torch.sigmoid(y)
    #dyskretyzacja
    for j in range(len(y)):
        output = torch.reshape(y[j], (3, 1))
        for k in range(3):
            if output[k] > 0.5:
                output[k] = torch.tensor(1.0)
            else:
                output[k] = torch.tensor(0.0)
    labels = dataset.iloc[i,2:5].values
    if Utils.hamming(y[0], labels) == 0:
        n += 1


print(f'Acc: {n/len(dataset)}')
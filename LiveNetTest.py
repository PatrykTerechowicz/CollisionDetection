import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from os import system, name
import Utils

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = models.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(4096, 4)
device = torch.device('cpu')
model = model.to(device)
model = model.eval()

NUM_EPOCHS = 50
BEST_MODEL_PATH = 'best_model_hammingloss.pth'
model.load_state_dict(torch.load(BEST_MODEL_PATH))
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frameTest = cv2.resize(frame, (224, 224))
    frameTest = preprocess(frameTest)
    frameTest = frameTest.type('torch.FloatTensor')
    frameTest = frameTest.unsqueeze(0)
    frameTest.to(device)
    y = model(frameTest)
    y = torch.sigmoid(y)
    print(y[0])
    _ = system('cls')


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.relase()
cv2.destroyAllWindows()
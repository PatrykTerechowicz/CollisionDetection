import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import ClassifyingDataset as d
import cv2

root_dir = 'dataset/'



dataset = d.CollisionDataset(
    root_dir
)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 399, 399])
batchsize = 16
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

NUM_EPOCHS = 100
BEST_MODEL_PATH = 'best_model.pth'
best_accuracy = 0.0
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train():
    global model, best_accuracy
    print('Starting')
    for epoch in range(NUM_EPOCHS):
        model.train()
        i = 0
        for images, labels in iter(train_loader):
            print(f'{i}/{int(len(train_dataset)/batchsize)}')
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            i += 1


        #Testujemy efektywnosc uczenia za pomoca strict hammingloss
        model.eval()
        test_error_count = 0.0
        for images, labels in iter(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            for i in range(len(outputs)):
                out = outputs[i].argmax(0)
                if labels[i] != out:
                    test_error_count += 1
        test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
        print('%d: %f' % (epoch, test_accuracy))
        if test_accuracy > best_accuracy:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_accuracy = test_accuracy


if __name__ == "__main__":
    train()
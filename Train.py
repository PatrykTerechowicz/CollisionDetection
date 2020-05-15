import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
csv_file = 'collision_labels_clean.csv'
root_dir = 'data/collision/'



dataset = datasets.ImageFolder(
    'dataset',
    transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)


train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 50, 50])
batchsize = 16
train_loader = torch.utils.data.DataLoader(
    train_dataset,
	batch_size=batchsize,
    shuffle=True,
    num_workers=8
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
	batch_size=batchsize,
    shuffle=False,
    num_workers=8
)

model = models.alexnet(pretrained=True)

model.classifier[6] = torch.nn.Linear(4096, 4, bias=4)
device = torch.device('cuda')
model = model.to(device)
torch.autograd.set_grad_enabled(True)

NUM_EPOCHS = 50
BEST_MODEL_PATH = 'best_model.pth'
best_accuracy = 0.0
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)


def train():
    global model, best_accuracy
    print('Starting')
    for epoch in range(NUM_EPOCHS):
        model.train()
        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            print(f'Outputs: {outputs}, Labels: {labels}')
            loss.backward()
            optimizer.step()


        #Testujemy efektywnosc uczenia za pomoca strict hammingloss
        model.eval()
        test_error_count = 0.0
        for images, labels in iter(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))

        test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
        print('%d: %f' % (epoch, test_accuracy))
        if test_accuracy > best_accuracy:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_accuracy = test_accuracy


if __name__ == "__main__":
    train()
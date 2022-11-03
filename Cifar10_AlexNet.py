import os
import torch
from torch._C import _CudaDeviceProperties
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("/home/jjunhee98/lab/Nobember/AlexNet/AlexNet[0.01,256]_0.01_regular2_transform3_small2_batchnorm4")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "6" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32)]
)

batch_size = 256

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
valid_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_set, test_set = torch.utils.data.random_split(train_set,[(int)(len(train_set)*0.9), (int)(len(train_set)*0.1)])


trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

print(len(train_set))
print(len(valid_set))
print(len(test_set))


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,96,4)
        self.conv2 = nn.Conv2d(96,256,5,padding=2)
        self.conv3 = nn.Conv2d(256,384,3,padding=1)
        self.conv4 = nn.Conv2d(384,384,3,padding=1)
        self.conv5 = nn.Conv2d(384,256,3,padding=1)
        self.fc1 = nn.Linear(256*2*2,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)
        self.pool = nn.MaxPool2d(3,2)
        self.dropout = nn.Dropout(0.5)
        self.LRN = nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75, k=2)

    def forward(self, x):
        x = self.pool(self.LRN(F.relu(self.conv1(x))))
        x = self.pool(self.LRN(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = F.softmax(self.fc3(x),dim=1)
        return x

class AlexNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,96,4)
        self.conv2 = nn.Conv2d(96,256,5,padding=2)
        self.conv3 = nn.Conv2d(256,384,3,padding=1)
        self.conv4 = nn.Conv2d(384,384,3,padding=1)
        self.conv5 = nn.Conv2d(384,256,3,padding=1)
        self.fc1 = nn.Linear(256*6*6,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)
        self.pool = nn.MaxPool2d(3,2)
        self.dropout = nn.Dropout(0.5)
        self.LRN = nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75, k=2)

    def forward(self, x):
        x = self.LRN(F.relu(self.conv1(x),inplace=True))
        x = self.pool(self.LRN(F.relu(self.conv2(x),inplace=True)))
        x = F.relu(self.conv3(x),inplace=True)
        x = F.relu(self.conv4(x),inplace=True)
        x = self.pool(F.relu(self.conv5(x),inplace=True))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(self.dropout(x)),inplace=True)
        x = F.relu(self.fc2(self.dropout(x)),inplace=True)
        x = F.softmax(self.fc3(x),dim=1)
        return x

class AlexNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,96,4)
        self.conv2 = nn.Conv2d(96,256,5,padding=2)
        self.conv3 = nn.Conv2d(256,384,3,padding=1)
        self.conv4 = nn.Conv2d(384,384,3,padding=1)
        self.conv5 = nn.Conv2d(384,256,3,padding=1)
        self.fc1 = nn.Linear(256*6*6,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)
        self.pool = nn.MaxPool2d(3,2)
        self.dropout = nn.Dropout(0.5)
        self.LRN = nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75, k=2)

    def forward(self, x):
        x = F.relu(self.conv1(x),inplace=True)
        x = self.pool(F.relu(self.conv2(x),inplace=True))
        x = F.relu(self.conv3(x),inplace=True)
        x = F.relu(self.conv4(x),inplace=True)
        x = self.pool(F.relu(self.conv5(x),inplace=True))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(self.dropout(x)),inplace=True)
        x = F.relu(self.fc2(self.dropout(x)),inplace=True)
        x = F.softmax(self.fc3(x),dim=1)
        return x

class AlexNet4(nn.Module): #82%,83%,80%,81%,82%,81%,81%,82%
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,96,4)
        self.conv2 = nn.Conv2d(96,256,5,padding=2)
        self.conv3 = nn.Conv2d(256,384,3,padding=1)
        self.conv4 = nn.Conv2d(384,384,3,padding=1)
        self.conv5 = nn.Conv2d(384,256,3,padding=1)
        self.fc1 = nn.Linear(256*2*2,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,10)
        self.pool = nn.MaxPool2d(3,2)
        self.dropout = nn.Dropout(0.5)
        self.LRN = nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75, k=2)

    def forward(self, x):
        x = self.pool(self.LRN(F.relu(self.conv1(x),inplace=True)))
        x = self.pool(self.LRN(F.relu(self.conv2(x),inplace=True)))
        x = F.relu(self.conv3(x),inplace=True)
        x = F.relu(self.conv4(x),inplace=True)
        x = self.pool(F.relu(self.conv5(x),inplace=True))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(self.dropout(x)),inplace=True)
        x = F.relu(self.fc2(self.dropout(x)),inplace=True)
        x = F.softmax(self.fc3(x),dim=1)
        return x

class AlexNet5(nn.Module): #84%,84%,83%
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,96,4)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96,256,5,padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256,384,3,padding=1)
        self.conv4 = nn.Conv2d(384,384,3,padding=1)
        self.conv5 = nn.Conv2d(384,256,3,padding=1)
        self.fc1 = nn.Linear(256*2*2,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,10)
        self.pool = nn.MaxPool2d(3,2)
        self.dropout = nn.Dropout(0.5)
        self.LRN = nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75, k=2)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x),inplace=True)))
        x = self.pool(self.bn2(F.relu(self.conv2(x),inplace=True)))
        x = F.relu(self.conv3(x),inplace=True)
        x = F.relu(self.conv4(x),inplace=True)
        x = self.pool(F.relu(self.conv5(x),inplace=True))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(self.dropout(x)),inplace=True)
        x = F.relu(self.fc2(self.dropout(x)),inplace=True)
        x = F.softmax(self.fc3(x),dim=1)
        return x

#net = AlexNet()
#net = AlexNet2()
#net = AlexNet3()
#net = AlexNet4()
#net = AlexNet5()
net.to(device)
#net = nn.parallel.DataParallel(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9, weight_decay=0.005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.9)

for epoch in range(100):
    running_loss = 0.0
    total = 0
    correct = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    writer.add_scalar("Loss/train", running_loss / total, epoch+1)
    writer.add_scalar("Accuracy/train", correct / total, epoch+1)
    print('%d loss: %.3f ,Accuracy of the network on the 45000 TRAIN images: %d %%' % (epoch + 1, running_loss / total, 100 * correct / total))        
    running_loss = 0.0

    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validloader:
            images, labels = data[0].to(device), data[1].to(device)
    
            outputs = net(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)

            running_loss += loss.item()
    
    writer.add_scalar("Loss/Valid", running_loss / total, epoch+1)
    writer.add_scalar("Accuracy/Valid", correct / total, epoch+1)
    print('%d loss: %.3f ,Accuracy of the network on the 10000 VALID images: %d %%' % (epoch + 1, running_loss / total, 100 * correct / total))
    running_loss = 0.0
    print("\n")
    scheduler.step()        

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

writer.add_scalar("Accuracy/test", correct / total, epoch+1)
print('Accuracy of the network on the 5000 TEST images: %d %%' % (100 * correct / total))

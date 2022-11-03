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
writer = SummaryWriter("/home/jjunhee98/lab/Nobember/LeNet/LeNet_v2[0.01,256]_weight_decay_modify(0.01)_2048_150_5")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))]
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

class LeNet(nn.Module): #(0.01,256) = 52% , (0.01,128) = 55%
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv3 = nn.Conv2d(6,16,5)
        self.conv5 = nn.Conv2d(16,120,5)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)
        self.pool = nn.AvgPool2d(2,2)

    def forward(self, x):
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv3(x)))
        x = F.tanh(self.conv5(x))
        x = torch.flatten(x,1)
        x = F.tanh(self.fc1(x))
        x = F.softmax(self.fc2(x),dim=1)
        return x

class LeNet_MaxPool(nn.Module):#(0.01,256) = 58% ,(0.01,128) = 61% , (0.01,64) = 60%
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv3 = nn.Conv2d(6,16,5)
        self.conv5 = nn.Conv2d(16,120,5)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)
        #self.pool = nn.AvgPool2d(2) 
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv3(x)))
        x = F.tanh(self.conv5(x))
        x = torch.flatten(x,1)
        x = F.tanh(self.fc1(x))
        x = F.softmax(self.fc2(x),dim=1)
        return x

class LeNet_MaxPool_Relu(nn.Module): #(0.01,256) = 55%, (0.01,128) = 57%
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv3 = nn.Conv2d(6,16,5)
        self.conv5 = nn.Conv2d(16,120,5)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)
        #self.pool = nn.AvgPool2d(2) 
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x),dim=1)
        return x

class LeNet_Relu(nn.Module): #(0.01,256) = 51%, (0.01,128) = 58%
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv3 = nn.Conv2d(6,16,5)
        self.conv5 = nn.Conv2d(16,120,5)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)
        self.pool = nn.AvgPool2d(2,2) 
        #self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x),dim=1)
        return x

class LeNet_BatchNorm(nn.Module): # (0.01,256) = 59%, (0.01,128) = 60%
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6,16,5)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16,120,5)
        self.bn5 = nn.BatchNorm2d(120)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)
        self.pool = nn.AvgPool2d(2,2)

    def forward(self, x):
        x = self.pool(F.tanh(self.bn1(self.conv1(x))))
        x = self.pool(F.tanh(self.bn3(self.conv3(x))))
        x = F.tanh(self.bn5(self.conv5(x)))
        x = torch.flatten(x,1)
        x = F.tanh(self.fc1(x))
        x = F.softmax(self.fc2(x),dim=1)
        return x

class LeNet_BatchNorm_Dropout(nn.Module): # (0.01,256) = 57%, (0.01,128) = 54%
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6,16,5)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16,120,5)
        self.bn5 = nn.BatchNorm2d(120)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)
        self.pool = nn.AvgPool2d(2,2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.tanh(self.bn1(self.conv1(x))))
        x = self.pool(F.tanh(self.bn3(self.conv3(x))))
        x = F.tanh(self.bn5(self.conv5(x)))
        x = torch.flatten(x,1)
        x = self.dropout(F.tanh(self.fc1(x)))
        x = F.softmax(self.fc2(x),dim=1)
        return x

class LeNet_MaxPool_BatchNorm_Dropout(nn.Module): #(0.01,256) = 59%, (0.01,128) = 62%
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6,16,5)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16,120,5)
        self.bn5 = nn.BatchNorm2d(120)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.tanh(self.bn1(self.conv1(x))))
        x = self.pool(F.tanh(self.bn3(self.conv3(x))))
        x = F.tanh(self.bn5(self.conv5(x)))
        x = torch.flatten(x,1)
        x = self.dropout(F.tanh(self.fc1(x)))
        x = F.softmax(self.fc2(x),dim=1)
        return x

class LeNet_MaxPool_Relu_BatchNorm_Dropout(nn.Module): #(0.01,256) = 59%, (0.01,128) = 62%
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6,16,5)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16,120,5)
        self.bn5 = nn.BatchNorm2d(120)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = torch.flatten(x,1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.softmax(self.fc2(x),dim=1)
        return x


class LeNet_v2(nn.Module): #(0.01,256) = 71%/72%/62%/67%(2048)/62%(1024), (0.01,128) = 60%
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3,padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6,16,3,padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16,120,3,padding=1)
        self.bn5 = nn.BatchNorm2d(120)
        self.fc1 = nn.Linear(120*8*8,2048)
        self.fc2 = nn.Linear(2048,10)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = torch.flatten(x,1)
        #x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.softmax(self.fc2(x),dim=1)
        return x

#net = LeNet()
#net = LeNet_MaxPool()
#net = LeNet_MaxPool_Relu()
#net = LeNet_Relu()
#net = LeNet_BatchNorm()
#net = LeNet_BatchNorm_Dropout()
#net = LeNet_MaxPool_BatchNorm_Dropout()
#net = LeNet_MaxPool_Relu_BatchNorm_Dropout()
net = LeNet_v2()

net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.9)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

for epoch in range(150):
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

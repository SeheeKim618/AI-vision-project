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
writer = SummaryWriter("/home/jjunhee98/lab/Nobember/VGGNet/VGGNET16(256,0.1)_5e-4_stepLR_final")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "9" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
#print(device)


transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomCrop(32,padding=4)]
)

transform_val_test = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)

batch_size = 256
valid_batch_size = 256
test_batch_size = 256

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
valid_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val_test)

train_set, test_set = torch.utils.data.random_split(train_set,[(int)(len(train_set)*0.9), (int)(len(train_set)*0.1)])


trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(valid_set, batch_size=valid_batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=True, num_workers=2)


#classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(len(train_set))
print(len(valid_set))
print(len(test_set))


class VGGNET(nn.Module):#87%,88%
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(2,2)
            self.conv1_1 = nn.Conv2d(3,64,3,padding=1)
            self.conv1_2 = nn.Conv2d(64,64,3,padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2_1 = nn.Conv2d(64,128,3,padding=1)
            self.conv2_2 = nn.Conv2d(128,128,3,padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3_1 = nn.Conv2d(128,256,3,padding=1)
            self.conv3_2 = nn.Conv2d(256,256,3,padding=1)
            self.conv3_3 = nn.Conv2d(256,256,3,padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.conv4_1 = nn.Conv2d(256,512,3,padding=1)   
            self.conv4_2 = nn.Conv2d(512,512,3,padding=1)
            self.conv4_3 = nn.Conv2d(512,512,3,padding=1)
            self.bn4 = nn.BatchNorm2d(512)
            self.conv5_1 = nn.Conv2d(512,512,3,padding=1)
            self.conv5_2 = nn.Conv2d(512,512,3,padding=1)
            self.conv5_3 = nn.Conv2d(512,512,3,padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            #self.fc1 = nn.Linear(512*2*2,4096)
            self.fc1 = nn.Linear(512*1*1,512)
            #self.fc2 = nn.Linear(512,512)
            self.fc3 = nn.Linear(512,100)
            self.dropout = nn.Dropout(0.5)
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1_1(x)),inplace = True)
            x = self.pool(F.relu(self.bn1(self.conv1_2(x)),inplace = True))
            x = F.relu(self.bn2(self.conv2_1(x)),inplace = True)
            x = self.pool(F.relu(self.bn2(self.conv2_2(x)),inplace = True))
            x = F.relu(self.bn3(self.conv3_1(x)),inplace = True)
            x = F.relu(self.bn3(self.conv3_2(x)),inplace = True)
            x = self.pool(F.relu(self.bn3(self.conv3_3(x)),inplace = True))
            x = F.relu(self.bn4(self.conv4_1(x)),inplace = True)
            x = F.relu(self.bn4(self.conv4_2(x)),inplace = True)
            x = self.pool(F.relu(self.bn4(self.conv4_3(x)),inplace = True))
            x = F.relu(self.bn5(self.conv5_1(x)),inplace = True)
            x = F.relu(self.bn5(self.conv5_2(x)),inplace = True)
            x = self.pool(F.relu(self.bn5(self.conv5_3(x)),inplace = True))
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.dropout(self.fc1(x)),inplace=True)
            #x = F.relu(self.dropout(self.fc2(x)))
            #x = F.softmax(self.fc3(x),dim=1)
            x = self.fc3(x)
            return x


class VGGNET2(nn.Module):#87%,88%
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(2,2)
            self.conv1_1 = nn.Conv2d(3,64,3,padding=1)
            self.conv1_2 = nn.Conv2d(64,64,3,padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2_1 = nn.Conv2d(64,128,3,padding=1)
            self.conv2_2 = nn.Conv2d(128,128,3,padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3_1 = nn.Conv2d(128,256,3,padding=1)
            self.conv3_2 = nn.Conv2d(256,256,3,padding=1)
            self.conv3_3 = nn.Conv2d(256,256,3,padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.conv4_1 = nn.Conv2d(256,512,3,padding=1)   
            self.conv4_2 = nn.Conv2d(512,512,3,padding=1)
            self.conv4_3 = nn.Conv2d(512,512,3,padding=1)
            self.bn4 = nn.BatchNorm2d(512)
            self.conv5_1 = nn.Conv2d(512,512,3,padding=1)
            self.conv5_2 = nn.Conv2d(512,512,3,padding=1)
            self.conv5_3 = nn.Conv2d(512,512,3,padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            #self.fc1 = nn.Linear(512*2*2,4096)
            self.fc1 = nn.Linear(512*1*1,512)
            #self.fc2 = nn.Linear(4096,4096)
            self.fc3 = nn.Linear(512,100)
            self.dropout = nn.Dropout(0.5)
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        def forward(self, x):
            x = self.dropout(self.bn1(F.relu(self.conv1_1(x),inplace=True)))
            x = self.pool(self.bn1(F.relu(self.conv1_2(x),inplace=True)))
            x = self.dropout(self.bn2(F.relu(self.conv2_1(x),inplace=True)))
            x = self.pool(self.bn2(F.relu(self.conv2_2(x),inplace=True)))
            x = self.dropout(self.bn3(F.relu(self.conv3_1(x),inplace=True)))
            x = self.dropout(self.bn3(F.relu(self.conv3_2(x),inplace=True)))
            x = self.pool(self.bn3(F.relu(self.conv3_3(x),inplace=True)))
            x = self.dropout(self.bn4(F.relu(self.conv4_1(x),inplace=True)))
            x = self.dropout(self.bn4(F.relu(self.conv4_2(x),inplace=True)))
            x = self.pool(self.bn4(F.relu(self.conv4_3(x),inplace=True)))
            x = self.dropout(self.bn5(F.relu(self.conv5_1(x),inplace=True)))
            x = self.dropout(self.bn5(F.relu(self.conv5_2(x),inplace=True)))
            x = self.dropout(self.pool(self.bn5(F.relu(self.conv5_3(x),inplace=True))))
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(F.relu(self.fc1(x),inplace=True))
            #x = self.dropout(F.relu(self.fc2(x),inplace=True))
            x = self.fc3(x)
            return x


class VGGNET19(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(2,2)
            self.conv1_1 = nn.Conv2d(3,64,3,padding=1)
            self.conv1_2 = nn.Conv2d(64,64,3,padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2_1 = nn.Conv2d(64,128,3,padding=1)
            self.conv2_2 = nn.Conv2d(128,128,3,padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3_1 = nn.Conv2d(128,256,3,padding=1)
            self.conv3_2 = nn.Conv2d(256,256,3,padding=1)
            self.conv3_3 = nn.Conv2d(256,256,3,padding=1)
            self.conv3_4 = nn.Conv2d(256,256,3,padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.conv4_1 = nn.Conv2d(256,512,3,padding=1)
            self.conv4_2 = nn.Conv2d(512,512,3,padding=1)
            self.conv4_3 = nn.Conv2d(512,512,3,padding=1)
            self.conv4_4 = nn.Conv2d(512,512,3,padding=1)
            self.bn4 = nn.BatchNorm2d(512)
            self.conv5_1 = nn.Conv2d(512,512,3,padding=1)
            self.conv5_2 = nn.Conv2d(512,512,3,padding=1)
            self.conv5_3 = nn.Conv2d(512,512,3,padding=1)
            self.conv5_4 = nn.Conv2d(512,512,3,padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            #self.fc1 = nn.Linear(512*2*2,4096)
            self.fc1 = nn.Linear(512*1*1,1024)
            self.fc2 = nn.Linear(1024,1024)
            self.fc3 = nn.Linear(1024,100)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1_1(x)))
            x = self.pool(F.relu(self.bn1(self.conv1_2(x))))
            x = F.relu(self.bn2(self.conv2_1(x)))
            x = self.pool(F.relu(self.bn2(self.conv2_2(x))))
            x = F.relu(self.bn3(self.conv3_1(x)))
            x = F.relu(self.bn3(self.conv3_2(x)))
            x = F.relu(self.bn3(self.conv3_3(x)))
            x = self.pool(F.relu(self.bn3(self.conv3_4(x))))
            x = F.relu(self.bn4(self.conv4_1(x)))
            x = F.relu(self.bn4(self.conv4_2(x)))
            x = F.relu(self.bn4(self.conv4_3(x)))
            x = self.pool(F.relu(self.bn4(self.conv4_4(x))))
            x = F.relu(self.bn5(self.conv5_1(x)))
            x = F.relu(self.bn5(self.conv5_2(x)))
            x = F.relu(self.bn5(self.conv5_3(x)))
            x = self.pool(F.relu(self.bn5(self.conv5_4(x))))
            x = torch.flatten(x, 1)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
            x = F.softmax(self.fc3(x),dim=1)
            return x


net = VGGNET()
#net = VGGNET2()
#net = VGGNET3()
#net = VGGNET19()
net.to(device)
#net = nn.DataParallel(net)





criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(net.parameters(),lr=0.01)
optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum=0.9, weight_decay=5e-4)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
#scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch: 0.9 ** epoch,last_epoch=-1,verbose=False)
for epoch in range(180):
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
#writer.flush()
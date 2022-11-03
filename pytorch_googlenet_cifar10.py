import os
from re import X
import torch
from torch._C import _CudaDeviceProperties
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("/home/jjunhee98/lab/Nobember/GoogleNet/GoogleNet(128,0.001)_5e-4_stepLR3")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3,5" 

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

batch_size = 128
valid_batch_size = 128
test_batch_size = 128

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


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(True),
            nn.Conv2d(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch (alternative two 3x3 conv branch for cifar dataset)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(True),
            nn.Conv2d(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(True),
            nn.Conv2d(in_channels=ch5x5, out_channels=ch5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(True),
        )

        #3x3 pool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding = 1),
            nn.Conv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(True),
        )

    def forward(self,x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)

        return torch.cat([y1,y2,y3,y4],1)

class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)
    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

net = GoogLeNet()
net.to(device)
#net = nn.DataParallel(net)





criterion = nn.CrossEntropyLoss()

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
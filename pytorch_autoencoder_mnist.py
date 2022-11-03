import os
from tqdm import tqdm
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
#print(device)

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-2

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MNIST('./data', download=True, transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True), 
            nn.Linear(128, 64), 
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True), 
            nn.Linear(16, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True), 
            nn.Linear(256, 28 * 28), 
            nn.Tanh())

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(32, 16, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 32, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

model = conv_autoencoder()
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    running_loss = 0.0
    total = 0
    for i, data in enumerate(dataloader, 0):
        img, _ = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(img)
        outputs = torch.flatten(outputs, 1)
        loss = criterion(outputs, torch.flatten(img, 1))
        loss.backward()
        optimizer.step()

        total += _.size(0)
        running_loss += loss.item()
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, running_loss/total))
    running_loss = 0.0
    total = 0

    if epoch % 10 == 0:
        origin_picture = to_img(img.cpu().data)
        picture = to_img(outputs.cpu().data)
        save_image(origin_picture,'./mlp_img/conv5_origin_image_{}.png'.format(epoch))
        save_image(picture, './mlp_img/conv5_image_{}.png'.format(epoch))

torch.save(model.state_dict(), './basic_autoencoder.pth')

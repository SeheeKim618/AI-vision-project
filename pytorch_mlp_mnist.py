import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

training_data = datasets.MNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
)

training_data, valid_data = torch.utils.data.random_split(training_data, [50000, 10000])

test_data = datasets.MNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size = batch_size)
valid_dataloader = DataLoader(valid_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of x: ", X.shape)
    print("Shape of y: ",y.shape, y.dtype)
    break

for X, y in valid_dataloader:
    print("Shape of x: ", X.shape)
    print("Shape of y: ",y.shape, y.dtype)
    break

#Model make

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))

# GPU 할당 변경하기
GPU_NUM = 9 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

# Additional Infos
if device.type == 'cuda':
    print(torch.cuda.get_device_name(GPU_NUM))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(GPU_NUM)/1024**3,1), 'GB')

#Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 2000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2000,2000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2000,2000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2000,2000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2000,10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# optimizing model parameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9)
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size= 10, gamma= 0.1)
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=1)

def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    loss_train, correct = 0,0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        loss_train += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    loss_train /= num_batches
    correct /= size
    writer.add_scalar("Accuracy/train", correct, epoch)
    writer.add_scalar("Loss/train", loss_train, epoch)


def valid(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    valid_loss, correct = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            valid_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    valid_loss /= num_batches
    correct /= size
    print(f"Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {valid_loss:>8f} \n")
    writer.add_scalar("Accuracy/Valid", correct, epoch)
    writer.add_scalar("Loss/Valid", valid_loss, epoch)
    exp_lr_scheduler.step(valid_loss)


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    writer.add_scalar("Accuracy/test", correct)
    writer.add_scalar("Loss/test", test_loss)


if __name__ == "__main__":
    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, t)
        valid(valid_dataloader, model, loss_fn, t)
    test(test_dataloader, model, loss_fn)
    print("Done!")

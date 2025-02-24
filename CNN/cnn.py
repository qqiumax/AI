import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

# Mnist images to tensor (# of images, height, width, color_channels)
transform = transforms.ToTensor()

# Data
train_data = datasets.MNIST(root='CNN/mnist_train_data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='CNN/mnist_test_data', train=False, download=True, transform=transform)

# Batch size
train_loader = DataLoader(train_data, batch_size=3, shuffle=True)
test_loader = DataLoader(test_data, batch_size=3, shuffle=False)

# Define CNN
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.conv3 = nn.Conv2d(16, 23, 3, 1)
        
        # Calculate the correct input size for the first fully connected layer
        self._to_linear = None
        self.convs(torch.randn(1, 1, 28, 28))  # Pass a dummy input to calculate the size
        
        self.fc1 = nn.Linear(self._to_linear, 201)
        self.fc2 = nn.Linear(201, 157)
        self.fc3 = nn.Linear(157, 87)
        self.fc4 = nn.Linear(87, 37)
        self.fc5 = nn.Linear(37, 10)

    def convs(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        
        if self._to_linear is None:
            self._to_linear = x.view(-1).shape[0]
        return x

    def forward(self, X):
        X = self.convs(X)
        X = X.view(-1, self._to_linear)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = F.relu(self.fc4(X))
        X = self.fc5(X)
        return F.log_softmax(X, dim=1)

torch.manual_seed(37)
model = ConvolutionalNetwork()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

start_time = time.time()

# Training
epochs = 10
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b%500 == 0:
            print(f'Epoch {i} LOSS: {loss.item()} ACCURACY: {trn_corr.item()*100/(3*b):7.3f}%')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)


current_time = start_time
total_time = current_time - start_time
print(f'Total time: {total_time/60} minutes')

torch.save(model.state_dict(), 'CNN/handwritten_number_pred.pth')

test_load = DataLoader(test_data, batch_size=10000, shuffle=False)
with torch.no_grad():
    correct = 0
    for X_test, y_test in test_loader:
        y_val = model(X_test)
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum()
print(f'Test accuracy: {correct.item()*100/10000:.3f}%')

print(test_data[3157])
test_data[3157][0].reshape(28,28)

plt.imshow(test_data[3157][0].reshape(28,28))
plt.show()

model.eval()
with torch.no_grad():
    new_pred = model(test_data[3157][0].view(1,1,28,28)).argmax()

print(f'Predicted value: {new_pred.item()}')

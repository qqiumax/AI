import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time

# Mnist images to tensor (# of images, height, width, color_channels)
transform = transforms.ToTensor()

# Data
train_data = datasets.MNIST(root='GRU/mnist_train_data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='GRU/mnist_test_data', train=False, download=True, transform=transform)

# Batch size
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define GRU
class GRUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return F.log_softmax(out, dim=1)

input_dim = 28  # Each row of the image
hidden_dim = 128
output_dim = 10
num_layers = 6

torch.manual_seed(37)
model = GRUNetwork(input_dim, hidden_dim, output_dim, num_layers)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()

# Training
epochs = 7
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for epoch in range(epochs):
    trn_corr = 0
    tst_corr = 0

    for batch_idx, (X_train, y_train) in enumerate(train_loader):
        batch_idx += 1

        X_train = X_train.squeeze(1)  # Remove the channel dimension
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print(f'Epoch {epoch} LOSS: {loss.item()} ACCURACY: {trn_corr.item()*100/(32*batch_idx):7.3f}%')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    with torch.no_grad():
        for batch_idx, (X_test, y_test) in enumerate(test_loader):
            X_test = X_test.squeeze(1)  # Remove the channel dimension
            y_val = model(X_test)
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

current_time = time.time()
total_time = current_time - start_time
print(f'Total time: {total_time/60} minutes')

torch.save(model.state_dict(), 'GRU/gru_handwritten_number_pred.pth')

test_loader = DataLoader(test_data, batch_size=10000, shuffle=False)
with torch.no_grad():
    correct = 0
    for X_test, y_test in test_loader:
        X_test = X_test.squeeze(1)  # Remove the channel dimension
        y_val = model(X_test)
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum()
print(f'Test accuracy: {correct.item()*100/10000:.3f}%')

print(test_data[3157])
test_data[3157][0].reshape(28, 28)

plt.imshow(test_data[3157][0].reshape(28, 28))
plt.show()

model.eval()
with torch.no_grad():
    new_pred = model(test_data[3157][0].view(1, 28, 28)).argmax()

print(f'Predicted value: {new_pred.item()}')
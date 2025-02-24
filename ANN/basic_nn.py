import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Create a neural network class that inherits from nn.Module
class Basic(nn.Module):
    #input layer (4 features of the flower) --> hidden layers --> output layer (3 classes)
    def __init__(self,in_features=4, h1=25, h2=17, h3=13, h4=9, out_features=3):
        super().__init__() #instantiate nn.Module
        self.fc1 = nn.Linear(in_features,h1) #input layer
        self.fc2 = nn.Linear(h1,h2) #hidden layer
        self.fc3 = nn.Linear(h2,h3) #hidden layer
        self.fc4 = nn.Linear(h3,h4) #hidden layer
        self.out = nn.Linear(h4,out_features) #output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.out(x)  
        return x
    
#Instantiate the model
torch.manual_seed(37)
model = Basic()

#Get the data
url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
df = pd.read_csv(url)
df['variety'] = df['variety'].map({'Setosa':0.0,'Versicolor':1.0,'Virginica':2.0})

#Train split
X = df.drop('variety',axis=1).values
y = df['variety'].values

#Training
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=37)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10000
losses = []
for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred,y_train)
    losses.append(loss.item())  # Append the scalar value of the loss
    if i%100 == 0:
        print(f'Epoch {i} and loss is: {loss.item()}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Feed forward to get the test loss
with torch.no_grad():
    y_val = model.forward(X_test)
    loss = criterion(y_val,y_test)
print(f'Loss: {loss}')

#Test loss accuracy
correct = 0
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_val = model.forward(data)
        print(f'{i+1}.) {str(y_val)} {y_test[i]}')
        if y_val.argmax().item() == y_test[i]:
            correct += 1
print(f'We got {correct} correct!')

# Example of a new iris
new_iris = torch.tensor([5.9,3.0,5.1,1.8])
with torch.no_grad():
    print(model(new_iris))
    print(model(new_iris).argmax())

#Save the model
torch.save(model.state_dict(), 'IrisClassifier.pth')

#Loading the model
new_model = Basic()
new_model.load_state_dict(torch.load('IrisClassifier.pth'))
new_model.eval()
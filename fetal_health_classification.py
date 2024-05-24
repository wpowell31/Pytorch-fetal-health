import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

# device config
device = torch.device('mps')

# hyperparameters
n_features = 21 
hidden_size1 = 100
hidden_size2 = 200
num_classes = 3
num_epochs = 2000
learning_rate = 0.001

# load data
data = np.loadtxt('fetal_health.csv', delimiter=',', skiprows=1)

X = data[:, :-1]
y = data[:, -1]

# getting classes to be 0, 1, 2
y = y-1

print(data.shape)

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=31)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

print(X_train.shape, y_train.shape)

# Neural Net Model

class NeuralNet(nn.Module):

    def __init__(self, n_features, n_hidden1, n_hidden2, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(n_features, n_hidden1)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(n_hidden1, n_hidden2)
        self.l3 = nn.Linear(n_hidden2, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


model = NeuralNet(
    n_features = n_features,
    n_hidden1 = hidden_size1,
    n_hidden2 = hidden_size2,
    num_classes = num_classes
)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):
    # forward pass and loss
    outputs = model(X_train)
    y_train = y_train.squeeze().long()
    loss = criterion(outputs, y_train)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, loss = {loss.item():.3f}')

# test
with torch.no_grad():
    n_samples = y_test.shape[0]
    outputs = model(X_test)
    _, preds = torch.max(outputs, dim=1)
    n_correct = (preds == y_test.squeeze()).sum().item()
    print(f'Accuracy: {n_correct / n_samples}')

    # Accuracy is 92.2%

    


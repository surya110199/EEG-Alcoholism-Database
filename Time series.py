import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
file = sio.loadmat(r'uci_eeg.mat')

input_arr = file['X']
alcoholic = file['y_alcoholic']


def normalize(arr):
    for i in range(arr.shape[0]):
        arr[i] = (arr[i] - np.min(arr[i])) / (np.max(arr[i]) - np.min(arr[i]))
    return arr


def load_within_subject():

    X_train = np.load(r'Train_data_within.npy')
    X_train = normalize(X_train)
    X_test = np.load(r'Test_data_within.npy')
    X_test = normalize(X_test)
    y_train = np.load(r'Train_labels_within.npy')
    y_test = np.load(r'Test_labels_within.npy')
    return X_train, X_test, y_train, y_test


def load_cross_subject():
    X_train = np.load(r'Train_data_cross.npy')
    X_train = normalize(X_train)
    X_test = np.load(r'Test_data_cross.npy')
    X_test = normalize(X_test)
    y_train = np.load(r'Train_labels_cross.npy')
    y_test = np.load(r'Test_labels_cross.npy')
    return X_train, X_test, y_train, y_test


X = normalize(input_arr)
y = alcoholic.T

# Split your data into training and testing sets
# batch_size = 8
# num_batches = len(input_arr) // batch_size
# train_size = (int(0.7 * num_batches)) * batch_size  # 70% split
# test_size = (len(input_arr) - train_size - 1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size,
#                                                     random_state=42)
X_train, X_test, y_train, y_test = load_cross_subject()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Convert data to PyTorch tensors
X_train = torch.Tensor(X_train).to(device)
y_train = torch.Tensor(y_train).to(device)
X_test = torch.Tensor(X_test).to(device)
y_test = torch.Tensor(y_test).to(device)

# Create DataLoader objects for efficient data loading

batch_size = 64

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(256 * 64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.out(x)
        return x


class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.fc1 = nn.Linear(256 * 64, 64)
        self.lstm1 = nn.LSTM(64, 100, 1)
        self.drop = nn.Dropout(0.4)
        # self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        # x.shape: (batch_size, seq_len, input_size)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        out, _ = self.lstm1(x)
        out = torch.relu(out)
        out = self.drop(out)
        # out, _ = self.lstm2(out)
        out = self.fc(out)
        return out


class CONV_Network(nn.Module):
    def __init__(self, batch):
        super(CONV_Network, self).__init__()
        self.conv1d = nn.Conv1d(64, 100, kernel_size=3)
        self.pool = torch.nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(batch * 100 * 127, 100)
        self.out = nn.Linear(100, batch)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.pool(x)
        x = x.view(-1, (x.shape[0] * x.shape[1] * x.shape[2]))
        x = torch.relu(self.fc(x))
        x = torch.sigmoid(self.out(x))
        x = x.permute(1, 0)
        return x


model = LSTMClassifier().to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.1)

num_epochs = 200
best_test_accuracy = 0.0
best_F1 = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    correct_test = 0
    total = 0
    total_test = 0
    F1 = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        # outputs = torch.tensor(outputs, dtype=torch.float32, requires_grad=True)
        optimizer.zero_grad()
        # predicted = torch.tensor((outputs > 0.5), requires_grad=True, dtype=torch.float32)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        F1 = f1_score(labels.detach().cpu().numpy(), predicted.detach().cpu().numpy())
        # Print batch loss and accuracy
        # if (batch_idx + 1) % 10 == 0:
        #     batch_loss = total_loss / (batch_idx + 1)
        #     batch_accuracy = 100.0 * correct / total
        #     print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}], Batch Loss: {batch_loss:.4f},"
        #           f" Batch Accuracy: {batch_accuracy:.2f}%")
    scheduler.step()
    train_loss = total_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%", f"F1 Score:{F1}")
        train_loss = 0
        train_acc = 0
        F1 = 0
    model.eval()
    with torch.no_grad():
        for X_test_batch, y_test_batch in test_loader:
            outputs_test = model(X_test_batch)
            predicted_test = (outputs_test > 0.5).float()
            f1_test = f1_score(y_test_batch.detach().cpu().numpy(), predicted_test.detach().cpu().numpy())
            correct_test += (predicted_test == y_test_batch).sum().item()
            total_test += y_test_batch.size(0)
        test_acc = 100.0 * correct_test / total_test
        if f1_test > best_F1:
            best_F1 = f1_test
            print(f"Best F1, {best_F1}", f"Test Accuracy,{test_acc:.2f}%")

        # if test_acc > best_test_accuracy:
        #     best_test_accuracy = test_acc
        #     print(f"Test Accuracy,{best_test_accuracy:.2f}%")

print("Training finished.")

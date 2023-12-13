# Imports
import numpy as np
import scipy.io as sio
import os

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
# Torch Imports
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Visualization Imports
from matplotlib import pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def within_subject_split(mat_file):
    """
       Takes a .mat file as input

       Parameters
       ----------
       mat_file: .mat dataset file

       Returns
       -------
       train_data: (80% of N) x T x C for time-series,(80% of N) x 32 x 32 x 3 for image based
       train_labels: (80% of N) x 1 for time-series and image based
       test_data: (20% of N) x T x C for time-series,(20% of N) x 32 x 32 x 3 for image based
       test_labels: (20% of N) x 1 for time-series and image based
   """
    file = sio.loadmat(mat_file)
    X = file['data']
    y = file['y_alcoholic'].T

    # Extract subject IDs
    subject_ids = file['subjectid'].astype('int')

    subject_ids = subject_ids.reshape(np.shape(X)[0])

    # Determine the number of unique subjects
    num_subjects = len(np.unique(subject_ids))
    # Define the train-test split ratio
    test_size = 0.2  # You can adjust this ratio as needed

    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    for i in range(1, num_subjects + 1):
        idx_i = np.where(i == subject_ids)
        subject_data = X[idx_i]
        subject_labels = y[idx_i]
        X_train, X_test, y_train, y_test = train_test_split(
            subject_data, subject_labels, test_size=test_size, random_state=42)
        train_data.append(X_train)
        test_data.append(X_test)
        train_labels.append(y_train)
        test_labels.append(y_test)

    # Now, train_data, test_data, train_labels, and test_labels are available separately

    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    test_data = np.concatenate(test_data)
    test_labels = np.concatenate(test_labels)

    return train_data, test_data, train_labels, test_labels


def cross_subject_split(mat_file):
    """
           Takes a .mat file as input

           Parameters
           ----------
           mat_file: .mat dataset file

           Returns
           -------
           train_data: (80% of N) x T x C for time-series,(80% of N) x 32 x 32 x 3 for image based
           train_labels: (80% of N) x 1 for time-series and image based
           test_data: (20% of N) x T x C for time-series,(20% of N) x 32 x 32 x 3 for image based
           test_labels: (20% of N) x 1 for time-series and image based
       """
    mat = sio.loadmat(mat_file)
    data = mat['data'].astype('float32')

    # Extract labels for alcoholism, stimulus, and subject ID
    label_alcoholism = mat['y_alcoholic'].astype('int').reshape(data.shape[0])
    label_id = (mat['subjectid'].astype('int') - 1).reshape(data.shape[0])

    # Define variables for data splitting
    num_subject = 122
    num_datapoint = data.shape[0]
    mask = np.zeros(num_subject)

    # Split data into 80% training and 20% testing
    for i in range(num_subject):
        r = np.random.rand()
        if r < 0.8:
            mask[i] = 0
        else:
            mask[i] = 1

    # Separate data and labels for training and testing sets
    train_data = [data[i] for i in range(num_datapoint) if mask[label_id[i]] == 0]
    train_label_alcoholism = [label_alcoholism[i] for i in range(num_datapoint) if mask[label_id[i]] == 0]

    # Use the remaining 20% of subjects for testing
    test_data = [data[i] for i in range(num_datapoint) if mask[label_id[i]] == 1]
    test_label_alcoholism = [label_alcoholism[i] for i in range(num_datapoint) if mask[label_id[i]] == 1]

    # Reshape the data for compatibility
    train_data = np.concatenate(train_data)
    train_data = train_data.reshape((train_data.shape[0] // 32, 32, 32, 3))
    test_data = np.concatenate(test_data)
    test_data = test_data.reshape((test_data.shape[0] // 32, 32, 32, 3))

    # Prepare labels for training and testing
    train_label = np.array(train_label_alcoholism).reshape(-1, 1)
    test_label = np.array(test_label_alcoholism).reshape(-1, 1)

    return train_data, test_data, train_label, test_label


batch_size = 64
file = r'COMP4660&8420-UCI eeg dataset ass 2/uci_eeg_images.mat'
X_train, X_test, y_train, y_test = cross_subject_split(file)  # please use the function


# within_subject_split() for within-subject split.


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        # n_image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalizing the image
        labels = self.labels[index]
        n_image = torch.tensor(image, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        n_image = n_image.to(device)
        labels = labels.to(device)
        return n_image, labels


train_data = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

test_data = CustomDataset(X_test, y_test)

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


class CNN_Classifier(torch.nn.Module):
    def __init__(self):
        super(CNN_Classifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(32 * 15 * 15, 64)  # Adjusted the input size here
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # Flatten the tensor correctly
        x = self.fc1(x)
        x = self.relu(x)
        x = self.out(x)
        return x


class ResNet18(torch.nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.fc1 = torch.nn.Linear(1000, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.out = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.resnet(x)
        x = self.fc1(x)
        x = self.fc2(torch.relu(x))
        x = self.fc3(torch.relu(x))
        x = self.out(torch.relu(x))
        return x


model = CNN_Classifier().to(device)
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

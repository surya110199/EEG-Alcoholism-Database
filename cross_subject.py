# Import necessary libraries
from sklearn.model_selection import train_test_split
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load EEG data from a MATLAB file
mat = sio.loadmat(r'uci_eeg.mat')
data = mat['X'].astype('float32')

# Extract labels for alcoholism, stimulus, and subject ID
label_alcoholism = mat['y_alcoholic'].astype('int').reshape(data.shape[0])
label_stimulus = (mat['y_stimulus'].astype('int') - 1).reshape(data.shape[0])
label_id = (mat['subjectid'].astype('int') - 1).reshape(data.shape[0])

# Define variables for data splitting
num_subject = 122
num_datapoint = data.shape[0]
N, T, C = data.shape
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
train_label_stimulus = [label_stimulus[i] for i in range(num_datapoint) if mask[label_id[i]] == 0]
train_label_id = [label_id[i] for i in range(num_datapoint) if mask[label_id[i]] == 0]

# Use the remaining 20% of subjects for testing
test_data = [data[i] for i in range(num_datapoint) if mask[label_id[i]] == 1]
test_label_alcoholism = [label_alcoholism[i] for i in range(num_datapoint) if mask[label_id[i]] == 1]
test_label_stimulus = [label_stimulus[i] for i in range(num_datapoint) if mask[label_id[i]] == 1]
test_label_id = [label_id[i] for i in range(num_datapoint) if mask[label_id[i]] == 1]

# Reshape the data for compatibility
train_data = np.concatenate(train_data)
train_data = train_data.reshape((train_data.shape[0] // 256, 256, C))
test_data = np.concatenate(test_data)
test_data = test_data.reshape((test_data.shape[0] // 256, 256, C))

# Prepare labels for training and testing
train_label = np.array(train_label_alcoholism).reshape(-1, 1)
test_label = np.array(test_label_alcoholism).reshape(-1, 1)


np.save('Train_data_cross.npy', train_data)
np.save('Train_labels_cross.npy', train_label)
np.save('Test_data_cross.npy', test_data)
np.save('Test_labels_cross.npy', test_label)
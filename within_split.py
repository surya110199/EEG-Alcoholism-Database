from sklearn.model_selection import train_test_split
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler()
file = sio.loadmat(r'uci_eeg.mat')

# print(file.keys())
input_arr = file['X']
alcoholic = file['y_alcoholic']

X = input_arr
y = alcoholic.T

# Extract subject IDs
subject_ids = file['subjectid'].astype('int')

subject_ids = subject_ids.reshape(np.shape(input_arr)[0])

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

# Uncomment this file to save the files
# np.save('Train_data_within.npy', train_data)
# np.save('Train_labels_within.npy', train_labels)
# np.save('Test_data_within.npy', test_data)
# np.save('Test_labels_within.npy', test_labels)

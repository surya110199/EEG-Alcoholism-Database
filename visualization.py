import scipy.io as sio
import numpy as np
import random
import matplotlib.pyplot as plt

# Load the EEG data from the UCI EEG Alcoholism Database
mat = sio.loadmat('uci_eeg.mat')
eeg_data = mat['X']
labels_alcoholism = mat['y_alcoholic'].flatten()

# Find indices of alcoholic and normal subjects
alcoholic_indices = np.where(labels_alcoholism == 1)[0]
print(len(alcoholic_indices))
normal_indices = np.where(labels_alcoholism == 0)[0]

# Randomly select one alcoholic and one normal subject
alcoholic_subject_index = random.choice(alcoholic_indices)
normal_subject_index = random.choice(normal_indices)

# Select EEG data for the chosen subjects
alcoholic_eeg = eeg_data[alcoholic_subject_index][:, :10]
normal_eeg = eeg_data[normal_subject_index][:, :10]


print(alcoholic_eeg.shape, normal_eeg.shape)
# Plot EEG signals for the alcoholic and normal subjects
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(alcoholic_eeg)
plt.title('EEG Signal for an Alcoholic Subject')

plt.subplot(2, 1, 2)
plt.plot(normal_eeg)
plt.title('EEG Signal for a Normal (Non-Alcoholic) Subject')

plt.tight_layout()
plt.show()

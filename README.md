# EEG-Alcoholism-Database

## Getting started

This project is about using Deep Neural networks on the EEG Alcoholism Database to determine whether a subject is alcoholic or not. The original dataset is from the UCI Alcoholism database, accessed by the following link (https://archive.ics.uci.edu/dataset/121/eeg+database). My assignment uses the pre-processed version of the dataset using the code (https://github.com/ShiyaLiu/EEG-feature-filter-and-disguising/blob/master/DataPreprocessing/eegtoimg.py).
## Data Description

The dataset has two different versions. 

1) EEG as Time series data:- Each signal is measured using 64 electordes and sampled at 256Hz so the resulting tensor is of size 11057 x 256 x 64. (11057 is the number of trials performed.)
2) EEG as Image-based dataset:- In this method, each signal of 64 x 256 dimension is converted to an image of 32x32x3 resulting data size of 11057 x 32 x 32 x 3.

## Code Flow
1) Run within_split.py and cross_subject.py by uncommenting the last lines to save the .npy files
2) Run Image-Based and Time series datasets with different architectures implemented. Change the architecture by renaming the architecure name and check the accuracies for within and cross-subject split.
3) Play around with different hyper-parameter settings
4) You can visualize EEG signals under the visualization.py file.


## References:-

1) (https://archive.ics.uci.edu/dataset/121/eeg+database)


**Citation:**

```bibtex
@inproceedings{yao2018deep,
  title={Deep feature learning and visualization for EEG recording using autoencoders},
  author={Yao, Yue and Plested, Jo and Gedeon, Tom},
  booktitle={International Conference on Neural Information Processing},
  pages={554--566},
  year={2018},
  organization={Springer}
}

@article{yao2020information,
  title={Information-preserving feature filter for short-term EEG signals},
  author={Yao, Yue and Plested, Josephine and Gedeon, Tom},
  journal={Neurocomputing},
  volume={408},
  pages={91--99},
  year={2020},
  publisher={Elsevier}
}

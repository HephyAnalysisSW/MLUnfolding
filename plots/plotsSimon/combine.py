import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
import sys
import gc
import psutil
from datetime import datetime
import os
#from __future__ import print_function
import transformations as trf                             
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ExponentialLR
from MLUnfolding.Tools.user  import plot_directory


# Define file paths
train_files = [
    "/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut5GeV/TTbar_older/ML_Data_train.npy",
    "/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut5GeV/TTbar_171p5/ML_Data_train.npy",
    "/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut5GeV/TTbar_173p5/ML_Data_train.npy",
]

val_files = [
    "/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut5GeV/TTbar_older/ML_Data_validate.npy",
    "/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut5GeV/TTbar_171p5/ML_Data_validate.npy",
    "/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut5GeV/TTbar_173p5/ML_Data_validate.npy",
]

# Define the proportions for each dataset (first one gets 0.7, others get 1.0)
train_shares = [0.7, 1.0, 1.0]
val_shares = [0.7, 1.0, 1.0]  # Adjust if validation should have different proportions

def load_and_sample_data(file_paths, shares):
    sampled_data = []
    for file, share in zip(file_paths, shares):
        data = np.load(file)
        num_samples = int(len(data) * share)
        indices = np.random.choice(len(data), num_samples, replace=False)
        sampled_data.append(data[indices])
    return np.vstack(sampled_data)

# Load, sample, and concatenate train data
train_concatenated = load_and_sample_data(train_files, train_shares)
np.random.shuffle(train_concatenated)
np.save("/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut5GeV/ML_Data_train_combined_v2.npy", train_concatenated)

# Load, sample, and concatenate validation data
val_concatenated = load_and_sample_data(val_files, val_shares)
np.random.shuffle(val_concatenated)
np.save("/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut5GeV/ML_Data_validate_combined_v2.npy", val_concatenated)

print("Processing complete. Train and validation datasets saved.")

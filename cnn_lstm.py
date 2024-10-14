import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import re
import matplotlib.pyplot as plt
from scipy import signal
import os
import time


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

OVERLAP = 0.25
WINDOW_LENGTH = 1

def labelProcess(label):
  idx_to_label = {
    'False Alarm': 0,
    'nda': 1,
    'Seizure': 2,
    'Fall': 3
    }

  return idx_to_label[label]

def PPGWindow(ppg, overLap = OVERLAP, samplingRate = 25, windowLength = WINDOW_LENGTH, spect = False):
  i = 0
  listWindow = []
  while(i + int(windowLength*samplingRate)<len(ppg)):
    sigRaw = ppg[i:i + int(windowLength*samplingRate)]
    if spect:
      _, _, sigRaw = signal.spectrogram(sigRaw, samplingRate)
      sigRaw = np.squeeze(sigRaw)
    listWindow.append(sigRaw)
    i = i + int(overLap*samplingRate)

  return np.array(listWindow)

class OpenSeizure(Dataset):
    def __init__(self, dataList, labelList):
        self.data = dataList
        self.labelList = labelList
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data = PPGWindow(self.data[idx])
        data = torch.tensor(np.array(data))
        data = torch.permute(data, (1, 0)).type(torch.FloatTensor) # Change this if spectrogram
        label = torch.tensor(labelProcess(self.labelList[idx])) # map labels to 0,1,2

        return data, label

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size, num_filters, lstm_layers, dropout):
        super(CNN1D_LSTM, self).__init__()

        # 1D CNN layer
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, padding=1)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=hidden_dim, num_layers=lstm_layers,
                            batch_first=True, bidirectional=False)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape

        # Permute to (batch_size, input_dim, seq_len) for CNN1D
        x = x.permute(0, 2, 1)

        # Pass through CNN1D -> (batch_size, num_filters, seq_len)
        x = F.relu(self.conv1d(x))

        # Permute back to (batch_size, seq_len, num_filters) for LSTM
        x = x.permute(0, 2, 1)

        # Pass through LSTM -> (batch_size, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)

        # Take the last time step (seq_len)
        last_hidden_state = lstm_out[:, -1, :]

        # Pass through dropout and fully connected layer
        out = self.dropout(last_hidden_state)
        out = self.fc(out)

        return out

# Example usage
batch_size = 1
seq_len = 25
input_dim = 17
hidden_dim = 128
output_dim = 4
kernel_size = 3
num_filters = 64
lstm_layers = 2
dropout = 0.5

model = CNN1D_LSTM(input_dim, hidden_dim, output_dim, kernel_size, num_filters, lstm_layers, dropout)


# Random input with shape (batch_size, seq_len, input_dim)
x = torch.randn(batch_size, seq_len, input_dim)

# Forward pass
timnow = time.time()
output = model(x)
print(output.shape)  # Output shape should be (batch_size, output_dim)
print('time: ', '', time.time() - timnow)



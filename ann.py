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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_features = 25
n_classes = 4

class ConvNet1D(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer2 = nn.Flatten()
        self.layer3 = nn.Sequential(
            nn.Linear(425,1000),
            nn.ReLU(),
            nn.Linear(1000,960),
            nn.ReLU(),
            nn.Linear(960,100),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(100,n_classes),
            nn.Softmax())

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.layer2(x)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.criterion(output, y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        _, predicted = torch.max(output, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / len(y)
        self.log('val_acc', accuracy)
        return accuracy

model = ConvNet1D()

timnow = time.time()
xLow = torch.tensor(torch.rand(1, 25, 17))
tPLow = model(xLow)
print(tPLow)
print('time: ', '', time.time() - timnow)

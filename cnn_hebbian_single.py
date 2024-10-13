import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import re
import matplotlib.pyplot as plt
import scipy
import os
from datetime import datetime, timedelta
from scipy.signal import butter, lfilter
import time


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def labelProcess(label):

  if label=='Fall':
    label = 'Seizure'

  idx_to_label = {
    'False Alarm': 0,
    'nda': 1,
    'Seizure': 2,
    }
  return idx_to_label[label]

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b1, a1 = butter(order, normal_cutoff, btype='low', analog=False)
    b2, a2 = butter(order, normal_cutoff, btype='high', analog =False)
    return b1, a1, b2, a2

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b1, a1, b2, a2 = butter_lowpass(cutoff, fs, order=order)
    y1 = lfilter(b1, a1, data)
    y2 = lfilter(b2, a2, data)
    return y1, y2

def PPGWindow(ppg, overLap = 0.25, samplingRate = 25, windowLength = 1, spect = False):
  i = 0
  listWindowLow = []
  listWindowHigh = []
  while(i + int(windowLength*samplingRate)<len(ppg)):
    sigRaw = ppg[i:i + int(windowLength*samplingRate)]

    sigFilteredLow, sigFilteredHigh = butter_lowpass_filter(sigRaw, 12, 25, order=5)

    listWindowLow.append(sigFilteredLow)
    listWindowHigh.append(sigFilteredHigh)
    i = i + int(overLap*samplingRate)

  return np.array(listWindowLow), np.array(listWindowHigh)

class ContinuousOpenSeizure2Frame(Dataset):
  def __init__(self, dataList, labelList):
    self.data = dataList
    self.labelList = labelList
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    dataLow, dataHigh = PPGWindow(self.data[idx][0])

    dataLow = torch.tensor(np.array(dataLow))
    dataLow = torch.permute(dataLow, (1, 0)).type(torch.FloatTensor) # Change this if spectrogram

    dataHigh = torch.tensor(np.array(dataHigh))
    dataHigh = torch.permute(dataHigh, (1, 0)).type(torch.FloatTensor) # Change this if spectrogram

    labelNext = torch.tensor(labelProcess(self.labelList[idx][1])) # map labels to 0,1,2
    labelThis = torch.tensor(labelProcess(self.labelList[idx][0])) # map labels to 0,1,2

    return dataLow, dataHigh, labelNext, labelThis

class HebbianLayer(nn.Module):
    def __init__(self, input_features, output_features, lr=0.001):
        super(HebbianLayer, self).__init__()
        # Initialize weights
        self.weights = nn.Parameter(torch.randn(output_features, input_features) * 0.01)
        # Learning rate
        self.lr = lr

    def forward(self, x):
        # Compute output
        output = torch.mm(self.weights, x.t())  # assuming x is of shape (batch_size, input_features)
        return output.t()  # transpose to keep same shape as input batch

    def update_weights(self, x, y):
        # Hebbian learning rule: Δw = η * (y * x)
        # x: input, y: output
        delta_w = self.lr * torch.mm(y.t(), x)  # assuming y is the output from the forward pass
        self.weights.data += delta_w  # update weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_features = 25
n_classes = 3

class ConvNet1D(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Flatten()
        self.hebbianLayer = HebbianLayer(960, n_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.Linear1_1 = nn.Linear(1, n_classes)

    def forward(self, x, label = 'No'):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.hebbianLayer(out)

        if label!='No':
          labelExpanded = torch.tensor(label).float().unsqueeze(-1)
          outLabel = self.Linear1_1(labelExpanded)
        else:
          outLabel = _

        return out, outLabel

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        xLow, xHigh, yNext, yThis = batch

        outputNextLow, outLabel = self.forward(xLow, yThis)
        outputThisLow, _ = self.forward(xLow)

        lossNextLow = self.criterion(outputNextLow, yNext)
        lossThisLow  = self.criterion(outputNextLow, yThis)

        lossLabel = self.criterion(outLabel, yNext)

        loss = lossNextLow + lossThisLow + lossLabel

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        xLow, xHigh, yNext, yThis = batch

        outputNextLow, outLabel = self.forward(xLow, yThis)
        outputThisLow, _ = self.forward(xLow)


        _, predictedThisLow = torch.max(outputThisLow, 1)
        _, predictedNextLow = torch.max(outputNextLow, 1)

        _, predictedLabel = torch.max(outLabel, 1)


        correctLow = (predictedNextLow == yNext).sum().item()
        accuracyLow = correctLow /len(yNext)

        correctLabel = (predictedLabel == yNext).sum().item()
        accuracyLabel = correctLabel/ len(yNext)

        self.log('val_acc_Low', accuracyLow, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_label', accuracyLabel, on_step = True, on_epoch = True, prog_bar = True, logger = True)

        return accuracyLow, accuracyLabel

# Initialize a ModelCheckpoint callback to save the best model
checkpoint_callback = ModelCheckpoint(monitor='train_loss', mode='min', verbose = True)
model = ConvNet1D()

sum(p.numel() for p in model.parameters())

timnow = time.time()
xLow = torch.tensor(torch.rand(1, 25, 17))
yThis = 0
tPLow, tPLabel = model(xLow, yThis)
print(tPLow)
print('time: ', '', time.time() - timnow)

"""This loads the signal data from the raw simulation outputs from Richers et al (20XX) ."""
import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch

# from ..defaults import BATCH_SIZE
from ..logger import logger
BATCH_SIZE = 32

# _ROOT_URL = "https://raw.githubusercontent.com/starccato/data/main/training"
SIGNALS_CSV = f"../data/training/richers_1764.csv"
PARAMETERS_CSV = f"../data/training/richers_1764_parameters.csv"
TIME_CSV = f"../data/training/richers_1764_times.csv"


class Data(Dataset):
    def __init__(self, batch_size=BATCH_SIZE, frac=1, train=True , indices=None):
        ### read data from csv files
        self.parameters = pd.read_csv(PARAMETERS_CSV)
        self.signals = pd.read_csv(SIGNALS_CSV).astype("float32").T
        self.signals.index = [i for i in range(len(self.signals.index))]

        assert (
            self.signals.shape[0] == self.parameters.shape[0],
            "Signals and parameters must have the same number of rows (the number of signals)",
        )

        if frac < 1:
            init_shape = self.signals.shape
            n_signals = int(frac * self.signals.shape[0])
            # keep n_signals random signals columns
            self.signals = self.signals.sample(n=n_signals, axis=0)
            self.parameters = self.parameters.iloc[self.signals.index, :]
            logger.info(
                f"Frac of TrainingData being used {init_shape} -> {self.signals.shape}"
            )

        # remove unusual parameters and corresponding signals
        keep_idx = self.parameters["beta1_IC_b"] > 0
        self.parameters = self.parameters[keep_idx]
        self.parameters = self.parameters[["beta1_IC_b", "A(km)", "EOS"]]

        # one hot encode EOS
        eos = pd.get_dummies(self.parameters["EOS"], prefix="EOS")
        self.parameters = pd.concat([self.parameters.drop(columns=["EOS"]), eos], axis=1)

        self.signals = self.signals[keep_idx]
        self.signals = self.signals.values.T

        ### flatten signals and take last 256 timestamps
        temp_data = np.empty(shape=(256, 0)).astype("float32")

        for i in range(0, self.signals.shape[1]):
            signal = self.signals[:, i]
            signal = signal.reshape(1, -1)

            cut_signal = signal[:, int(len(signal[0]) - 256) : len(signal[0])]
            temp_data = np.insert(
                temp_data, temp_data.shape[1], cut_signal, axis=1
            )

        self.signals = temp_data

        if indices is not None:
            if train:
                self.signals = self.signals[:, indices]
                self.parameters = self.parameters.iloc[indices]
                self.indices = indices
            else:
                self.signals = self.signals[:, indices]
                self.parameters = self.parameters.iloc[indices]
                self.indices = indices

        self.batch_size = batch_size
        self.mean = self.signals.mean()
        self.std = np.std(self.signals, axis=None)
        self.scaling_factor = 5
        self.max_value = abs(self.signals).max()
        self.ylim_signal = (self.signals[:, :].min(), self.signals[:, :].max())

    def __str__(self):
        return f"TrainingData: {self.signals.shape}"

    def __repr__(self):
        return self.__str__()

    @property
    def raw_signals(self):
        return pd.read_csv(SIGNALS_CSV).astype("float32").T.values

    def summary(self):
        """Display summary stats about the data"""
        str = f"Signal Dataset mean: {self.mean:.3f} +/- {self.std:.3f}\n"
        str += f"Signal Dataset scaling factor (to match noise in generator): {self.scaling_factor}\n"
        str += f"Signal Dataset max value: {self.max_value}\n"
        # str += f"Signal Dataset max parameter value: {self.max_parameter_value}\n"
        str += f"Signal Dataset shape: {self.signals.shape}\n"
        str += f"Parameter Dataset shape: {self.parameters.shape}\n"
        logger.info(str)

    def standardize(self, signal):
        standardized_signal = (signal - self.mean) / self.std
        standardized_signal = standardized_signal / self.scaling_factor
        return standardized_signal

    def normalise_signals(self, signal):
        normalised_signal = signal / self.max_value
        return normalised_signal
    
    def normalise_parameters(self, parameters):
        normalised_parameters = parameters / self.max_parameter_value
        return normalised_parameters

    ### overloads ###
    def __len__(self):
        return self.signals.shape[1]

    @property
    def shape(self):
        return self.signals.shape
    
    def get_indices(self):
        return self.indices

    def __getitem__(self, idx):
        signal = self.signals[:, idx]
        signal = signal.reshape(1, -1)
        parameters = self.parameters.iloc[idx]
        # parameters = parameters.reshape(1, -1)

        normalised_signal = self.normalise_signals(signal)

        # return normalised_signal, parameters
        return torch.tensor(normalised_signal, dtype=torch.float32), torch.tensor(parameters, dtype=torch.float32)

    def get_loader(self) -> DataLoader:
        return DataLoader(
            self, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def get_signals_iterator(self):
        return next(iter(self.get_loader()))
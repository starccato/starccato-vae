import os
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm.auto import tqdm, trange

# from ..defaults import BATCH_SIZE, DEVICE, NC, NDF, NGF, NZ
from ..logger import logger
from ..nn import Encoder, Decoder, Model
from ..plotting import (
    plot_reconstruction,
    plot_waveform_grid,
    plot_latent_morphs
)
from .data import Data

class Trainer:
    def __init__(
        self
    ):
        pass
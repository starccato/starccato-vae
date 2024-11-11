import torch
from torch import nn

from src.starcattovae.nn.encoder import Encoder
from src.starcattovae.nn.decoder import Decoder

class Model(nn.Module):
    def __init__(self, Encoder, Decoder, DEVICE):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.DEVICE = DEVICE
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var
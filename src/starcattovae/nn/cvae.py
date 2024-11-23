import torch
from torch import nn

class CVAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, input_dim, condition_dim, DEVICE):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, condition_dim=condition_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim, condition_dim=condition_dim)
        self.DEVICE = DEVICE
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.DEVICE)  # sampling epsilon        
        z = mean + var * epsilon  # reparameterization trick
        return z
    
    def forward(self, x, condition):
        mean, log_var = self.encoder(x, condition)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.decoder(z, condition)
        return x_hat, mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, condition_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, z, condition):
        z = torch.cat([z, condition], dim=1)
        h = self.LeakyReLU(self.FC_hidden(z))
        h = self.LeakyReLU(self.FC_hidden2(h))
        x_hat = self.FC_output(h)
        return x_hat

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance 
        return mean, log_var
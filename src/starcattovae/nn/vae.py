import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, input_dim, param_dim, input_channels, DEVICE):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, param_dim=param_dim, output_dim=input_dim)
        self.DEVICE = DEVICE
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.DEVICE)  # sampling epsilon        
        z = mean + var * epsilon  # reparameterization trick
        return z
    
    def forward(self, y, x):
        mean, log_var = self.encoder(y)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        y_hat = self.decoder(z, x)
        return y_hat, mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, param_dim,output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim + param_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, z, x):
        h = torch.cat((z, x), dim=1)  # concatenate z and x
        h = self.LeakyReLU(self.FC_hidden(h))
        h = self.LeakyReLU(self.FC_hidden2(h))
        y_hat = self.FC_output(h)
        return y_hat

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, y):
        print(y.shape)
        h_ = self.LeakyReLU(self.FC_input(y))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance 
        return mean, log_var
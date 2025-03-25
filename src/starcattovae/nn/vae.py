# import torch
# from torch import nn

# class VAE(nn.Module):
#     def __init__(self, latent_dim, hidden_dim, input_dim, input_channels, DEVICE):
#         super(VAE, self).__init__()
#         self.encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
#         self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)
#         self.DEVICE = DEVICE
        
#     def reparameterization(self, mean, var):
#         epsilon = torch.randn_like(var).to(self.DEVICE)  # sampling epsilon        
#         z = mean + var * epsilon  # reparameterization trick
#         return z
    
#     def forward(self, x):
#         mean, log_var = self.encoder(x)
#         z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
#         x_hat = self.decoder(z)
#         return x_hat, mean, log_var

# class Decoder(nn.Module):
#     def __init__(self, latent_dim, hidden_dim, output_dim):
#         super(Decoder, self).__init__()
#         self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
#         self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
#         self.FC_output = nn.Linear(hidden_dim, output_dim)
#         self.LeakyReLU = nn.LeakyReLU(0.2)
        
#     def forward(self, z):
#         h = self.LeakyReLU(self.FC_hidden(z))
#         h = self.LeakyReLU(self.FC_hidden2(h))
#         x_hat = self.FC_output(h)
#         print(x_hat.shape)
#         return x_hat

# class Encoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim):
#         super(Encoder, self).__init__()
#         self.FC_input = nn.Linear(input_dim, hidden_dim)
#         self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
#         self.FC_mean = nn.Linear(hidden_dim, latent_dim)
#         self.FC_var = nn.Linear(hidden_dim, latent_dim)
#         self.LeakyReLU = nn.LeakyReLU(0.2)
        
#     def forward(self, x):
#         h_ = self.LeakyReLU(self.FC_input(x))
#         h_ = self.LeakyReLU(self.FC_input2(h_))
#         mean = self.FC_mean(h_)
#         log_var = self.FC_var(h_)  # encoder produces mean and log of variance 
#         return mean, log_var

import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, input_channels, input_dim, DEVICE):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, hidden_dim, latent_dim, input_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_channels, input_dim)
        self.DEVICE = DEVICE

    def reparameterization(self, mean, log_var):
        std = torch.exp(0.5 * torch.clamp(log_var, min=-10, max=10))  # Clamping log_var to prevent extreme values
        epsilon = torch.randn_like(std).to(self.DEVICE)
        return mean + std * epsilon * 0.05  # Smaller scaling to avoid large latent jumps

    def forward(self, x):
        mean, log_var = self.encoder(x.unsqueeze(1))  # Add channel dim
        z = self.reparameterization(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat.view(x_hat.size(0), -1), mean, log_var  # Flatten output


class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, latent_dim, input_dim):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2)
        )

        conv_output_dim = input_dim // 8
        self.flatten_dim = hidden_dim * 4 * conv_output_dim

        self.fc_mean = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)
        return self.fc_mean(h), self.fc_log_var(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_channels, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim * 4 * (output_dim // 8))

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([hidden_dim * 2, output_dim // 4]),  # Normalize before activation
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([hidden_dim, output_dim // 2]),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(hidden_dim, output_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), -1, h.size(1) // (self.deconv_layers[0].in_channels))
        x_hat = self.deconv_layers(h)
        # x_hat = torch.tanh(x_hat)  # Apply Tanh to limit range to (-1, 1)
        return x_hat
import torch
from torch import nn

class VAECNN(nn.Module):
    def __init__(self, input_length, hidden_dim, latent_dim, DEVICE):
        """
        CNN-based VAE for 1D signals.
        
        Args:
            latent_dim (int): Size of the latent space.
            input_length (int): Length of the input signal.
            num_channels (int): Number of input channels (e.g., 1 for single-channel signals).
            DEVICE (str): Device to use ('cpu' or 'cuda').
        """
        super(VAECNN, self).__init__()
        self.encoder = Encoder(input_length=input_length, latent_dim=latent_dim)
        self.decoder = Decoder(input_length=input_length, latent_dim=latent_dim)
        self.DEVICE = DEVICE

    def reparameterization(self, mean, log_var):
        epsilon = torch.randn_like(log_var).to(self.DEVICE)  # Sampling epsilon
        z = mean + torch.exp(0.5 * log_var) * epsilon  # Reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var

class Encoder(nn.Module):
    def __init__(self, input_length, latent_dim):
        super(Encoder, self).__init__()
        """
        CNN Encoder for 1D signal.
        
        Args:
            input_length (int): Length of the input signal.
            latent_dim (int): Size of the latent space.
        """
        self.main = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        self.feature_map_length = input_length // (2 ** 3)  # Adjust based on the number of Conv layers (3)
        self.fc_mean = nn.Linear(128 * self.feature_map_length, latent_dim)
        self.fc_log_var = nn.Linear(128 * self.feature_map_length, latent_dim)

    def forward(self, x):
        h_ = self.main(x)
        mean = self.fc_mean(h_)
        log_var = self.fc_log_var(h_)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))
        x_hat = self.FC_output(h)
        return x_hat
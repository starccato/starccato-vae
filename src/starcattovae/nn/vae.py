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
#         # print(x_hat.shape)
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
        """
        Variational Autoencoder with CNN-based Encoder and Decoder.
        Args:
            latent_dim: Dimensionality of the latent space.
            hidden_dim: Number of filters in the convolutional layers.
            input_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            input_dim: Length of the input signal (e.g., time-series length or image width).
            DEVICE: Device to run the model on (e.g., 'cuda' or 'cpu').
        """
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels=input_channels, hidden_dim=hidden_dim, latent_dim=latent_dim, input_dim=input_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_channels=input_channels, output_dim=input_dim)
        self.DEVICE = DEVICE
        
    def reparameterization(self, mean, var):
        """
        Reparameterization trick to sample from N(mean, var) using N(0, 1).
        """
        epsilon = torch.randn_like(var).to(self.DEVICE)  # Sampling epsilon
        z = mean + var * epsilon  # Reparameterization trick
        return z
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        Args:
            x: Input tensor (batch_size, input_channels, input_dim).
        Returns:
            x_hat: Reconstructed input.
            mean: Mean of the latent distribution.
            log_var: Log variance of the latent distribution.
        """
        print(x.shape)
        x = x.unsqueeze(1)
        print(x.shape)
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # Convert log_var to var
        print(z.shape)
        x_hat = self.decoder(z)
        # flatten then second dim
        x_hat = x_hat.view(x_hat.size(0), -1)
        print(x_hat.shape)
        return x_hat, mean, log_var


class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, latent_dim, input_dim):
        """
        CNN-based Encoder for the VAE.
        Args:
            input_channels: Number of input channels.
            hidden_dim: Number of filters in the convolutional layers.
            latent_dim: Dimensionality of the latent space.
            input_dim: Length of the input signal.
        """
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=hidden_dim * 2, out_channels=hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2)
        )

        # Compute the size of the flattened feature map
        conv_output_dim = input_dim // 8  # Downsampled by a factor of 2^3 (3 conv layers with stride=2)
        self.flatten_dim = hidden_dim * 4 * conv_output_dim

        # Fully connected layers for mean and log variance
        self.fc_mean = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        """
        Forward pass through the encoder.
        Args:
            x: Input tensor (batch_size, input_channels, input_dim).
        Returns:
            mean: Mean of the latent distribution.
            log_var: Log variance of the latent distribution.
        """
        print(x.shape)
        h = self.conv_layers(x)  # Pass through convolutional layers
        print(h.shape)
        h = h.view(h.size(0), -1)  # Flatten the feature map
        mean = self.fc_mean(h)  # Compute mean
        log_var = self.fc_log_var(h)  # Compute log variance
        print(mean.shape)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_channels, output_dim):
        """
        CNN-based Decoder for the VAE.
        Args:
            latent_dim: Dimensionality of the latent space.
            hidden_dim: Number of filters in the convolutional layers.
            output_channels: Number of output channels.
            output_dim: Length of the output signal.
        """
        super(Decoder, self).__init__()
        # Fully connected layer to expand latent space to feature map
        self.fc = nn.Linear(latent_dim, hidden_dim * 4 * (output_dim // 8))

        # Transposed convolutional layers for upsampling
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(in_channels=hidden_dim, out_channels=output_channels, kernel_size=4, stride=2, padding=1),
            # nn.Sigmoid()  # Output values between 0 and 1
        )

    def forward(self, z):
        """
        Forward pass through the decoder.
        Args:
            z: Latent variable (batch_size, latent_dim).
        Returns:
            x_hat: Reconstructed input (batch_size, output_channels, output_dim).
        """
        # Expand latent space to feature map
        h = self.fc(z)
        # Reshape to match the input dimensions of the first transposed convolutional layer
        h = h.view(h.size(0), -1, (h.size(1) // (self.deconv_layers[0].in_channels)))
        # Pass through transposed convolutional layers
        x_hat = self.deconv_layers(h)
        print(x_hat.shape)
        return x_hat
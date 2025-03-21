import torch
from torch import nn

# note:
# y = signal - our observation
# x = parameter
# z = latent variable
# h = hidden layer

class CVAE(nn.Module):
    def __init__(self, batch_size, latent_dim, hidden_dim, param_dim, signal_dim, num_components, DEVICE):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_components = num_components
        self.batch_size = batch_size

        self.r1 = Encoder(signal_dim=signal_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_components=num_components)
        self.r2 = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim, condition_dim=condition_dim)
        self.q = Q(batch_size=batch_size, signal_dim=signal_dim, param_dim=param_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        # declare mixture gaussian here
        self.bimix_gauss 

        self.DEVICE = DEVICE
    
    # all the new implementation here
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.DEVICE)  # sampling epsilon        
        z = mean + var * epsilon  # reparameterization trick
        return z
    
    def forward(self, x, y):
        r1_mean, r1_log_var, r1_weights = self.r1(y)
        r1_zy_samp = 

        # output from q network
        q_zxy_mean, q_zxy_log_var = self.q(x, y)
        q_zxy_z_samp = self.q.sample_from_gaussian_distribution(self.batch_size, self.latent_dim, q_zxy_mean, q_zxy_log_var)



        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.decoder(z, condition)
        return x_hat, mean, log_var

# sampling from latent conditioned on signal
# todo fix this
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, param_dim, signal_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim + signal_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, param_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, z, signal):
        z = torch.cat([z, signal], dim=1)
        h = self.LeakyReLU(self.FC_hidden(z))
        h = self.LeakyReLU(self.FC_hidden2(h))
        x_hat = self.FC_output(h)
        return x_hat
    
# only use signals, no params
class Encoder(nn.Module):
    def __init__(self, signal_dim, hidden_dim, latent_dim, num_channels=1, num_components=32, kernel_size=3, stride=1, padding=1):
        super(Encoder, self).__init__()
        self.num_components = num_components
        self.latent_dim = latent_dim

        # Convolutional layers for multichannel time-series data
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )

        # Outputs for GMM parameters
        self.FC_means = nn.Linear(hidden_dim, latent_dim * num_components)  # Means for each component
        self.FC_log_vars = nn.Linear(hidden_dim, latent_dim * num_components)  # Log variances for each component
        self.FC_weights = nn.Linear(hidden_dim, num_components)  # Mixture weights

    def forward(self, y):
        # Pass through convolutional layers
        h = self.conv_layers(y)
        h = self.fc_layers(h)
        # Compute GMM parameters
        means = self.FC_means(h).view(-1, self.num_components, self.latent_dim)
        log_vars = self.FC_log_vars(h).view(-1, self.num_components, self.latent_dim)
        weights = torch.softmax(self.FC_weights(h), dim=1)

        return means, log_vars, weights

# conditioned parameters on signals
class Q(nn.Module):
    def __init__(self, signal_dim, param_dim, hidden_dim, latent_dim, num_channels=1, kernel_size=3, stride=1, padding=1):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )

        # Outputs for GMM parameters
        self.FC_means = nn.Linear(hidden_dim, latent_dim)
        self.FC_log_vars = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)

        h_ = self.conv_layers(xy)
        h_ = self.fc_layers(h_)

        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)

        return mean, log_var
    
    def sample_from_gaussian(self, batch_size, latent_dim, mean, log_var):
        epsilon = torch.randn((batch_size, latent_dim), dtype=torch.float32)
        z = mean + torch.exp(0.5 * log_var) * epsilon # why the 0.5?
        return z
import torch
from torch import nn
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily

# note:
# y = signal - our observation
# x = parameter
# z = latent variable
# h = hidden layer

class CVAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, param_dim, signal_dim, num_components, DEVICE):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_components = num_components

        self.r1 = Encoder(signal_dim=signal_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_components=num_components)
        self.r2 = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, param_dim=param_dim, signal_dim=signal_dim)
        self.q = Q(signal_dim=signal_dim, param_dim=param_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        # declare mixture gaussian here

        self.DEVICE = DEVICE
    
    # all the new implementation here
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.DEVICE)  # sampling epsilon        
        z = mean + var * epsilon  # reparameterization trick
        return z
    
    def forward(self, x, y):
        # r1 network
        r1_mean, r1_log_var, r1_weights = self.r1(y)
        r1_zy_z_samp, bimix_gauss = self.r1.define_and_sample_gmm(r1_weights, r1_mean, r1_log_var) # use this to sample from the mixture model, allows for multimodal distributions. Can change later but this is what Gabbard et al. used

        # output from q network
        q_zxy_mean, q_zxy_log_var = self.q(x, y)
        q_zxy_z_samp = self.q.reparameterization(q_zxy_mean, q_zxy_log_var)

        # r2 network
        r2_x_mean, r2_x_log_var = self.r2(q_zxy_z_samp, y)

        # calculate losses
        loss, recontruction_loss_x, KL = self.loss(x, y, r2_x_mean, r2_x_log_var, q_zxy_z_samp, q_zxy_mean, q_zxy_log_var, r1_mean, r1_log_var, r1_weights, bimix_gauss)

        return r2_x_mean, loss, recontruction_loss_x, KL
    
    ### this function needs a lot more work
    def loss(self, x, y, r2_x_mean, r2_x_log_var, q_zxy_z_samp, q_zxy_mean, q_zxy_log_var, r1_mean, r1_log_var, r1_weights, bimix_gauss):
        ### recontruction loss of x, using Gabbard et al's implementation for now
        # reconstruction_loss_x = nn.functional.mse_loss(r2_x_mean, x, reduction='sum')
        # reconstruction_loss_x *= 1 * x.shape[1]
        # Compute the normalizing factor: -0.5 * log(sig^2) - 0.5 * log(2 * pi)
        small_constant = 1e-8
        normalising_factor_x = -0.5 * (r2_x_log_var + torch.log(2.0 * torch.pi + small_constant))

        # Compute the squared difference: (mu - x)^2
        square_diff_between_mu_and_x = (r2_x_mean - x) ** 2

        # Compute the exponent term: -0.5 * (mu - x)^2 / sig^2
        inside_exp_x = -0.5 * square_diff_between_mu_and_x / (torch.exp(r2_x_log_var) + small_constant)

        # Sum over the parameter dimension: sum_dim(-0.5*log(sig^2) - 0.5*log(2*pi) - 0.5*(mu - x)^2 / sig^2)
        reconstruction_loss_x = torch.sum(normalising_factor_x + inside_exp_x, dim=1, keepdim=True)

        ### NOTE: they seem to multiply by -1.0 in Gabbard et al's implementation
 
        ### compute analytic self-entropy of q(z|x,y), using Gabbard et al's implementation for now
        # Compute the normalizing factor: -0.5 * log(sig^2) - 0.5 * log(2 * pi)
        normalising_factor_kl = -0.5 * (q_zxy_log_var + torch.log(2.0 * torch.pi + small_constant))

        # Compute the squared difference: (mu - x)^2
        square_diff_between_qz_and_q = (q_zxy_mean - q_zxy_z_samp) ** 2

        # Compute the exponent term: -0.5 * (mu - x)^2 / sig^2
        inside_exp_q = -0.5 * square_diff_between_qz_and_q / (torch.exp(q_zxy_log_var) + small_constant)

        # Sum over the latent dimension: sum_dim(-0.5*log(sig^2) - 0.5*log(2*pi) - 0.5*(mu - x)^2 / sig^2)
        log_q_q = torch.sum(normalising_factor_kl + inside_exp_q, dim=1, keepdim=True)

        ### compute KL between q(z|x,y) and r1(z|y), using Gabbard et al's implementation for now
        log_r1_q = bimix_gauss.log_prob(q_zxy_z_samp)  # Shape: (batch_size,)
        KL = torch.mean(log_q_q - log_r1_q)

        loss = reconstruction_loss_x + KL

        return loss, reconstruction_loss_x, KL

# sampling from latent conditioned on signal
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, param_dim, signal_dim, num_channels=1, kernel_size=3, stride=1, padding=1):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.signal_dim = signal_dim

        # Fully connected layers to combine z and y
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim + signal_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )

        # Convolutional layers for decoding
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()  # Ensure non-negative output
        )

        # Output layers for loc (x_hat) and log_var
        self.fc_loc = nn.Linear(hidden_dim, param_dim)
        self.fc_log_var = nn.Linear(hidden_dim, param_dim)

    def forward(self, z, y):
        """
        Forward pass for the decoder.
        Args:
            z: Latent variable (batch_size, latent_dim)
            y: Signal (batch_size, signal_dim)
        Returns:
            x_hat: Reconstructed parameters (batch_size, param_dim)
            log_var: Log variance of the reconstructed parameters (batch_size, param_dim)
        """
        # Concatenate z and y
        combined = torch.cat([z, y], dim=1)  # Shape: (batch_size, latent_dim + signal_dim)

        # Pass through fully connected layers
        h = self.fc_layers(combined)  # Shape: (batch_size, hidden_dim)

        # Reshape for convolutional layers
        h = h.unsqueeze(2)  # Add a channel dimension for ConvTranspose1d (batch_size, hidden_dim, 1)

        # Pass through convolutional layers
        h = self.conv_layers(h)  # Shape: (batch_size, num_channels, signal_dim)

        # Reshape the output to match the parameter dimension
        h = h.squeeze(2)  # Remove the extra dimension (batch_size, hidden_dim)

        # Compute loc (x_hat) and log_var
        x_hat = self.fc_loc(h)  # Shape: (batch_size, param_dim)
        log_var = self.fc_log_var(h)  # Shape: (batch_size, param_dim)

        return x_hat, log_var
    
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
    
    def define_and_sample_gmm(r1_weight, r1_mean, r1_log_var):
        """
        Define the r1(z|y) mixture model and sample from it.

        Args:
            r1_weight: Tensor of shape (batch_size, num_components), logits for the mixture weights.
            r1_mean: Tensor of shape (batch_size, num_components, latent_dim), means of the Gaussian components.
            r1_log_var: Tensor of shape (batch_size, num_components, latent_dim), log variances of the Gaussian components.

        Returns:
            r1_zy_samp: Samples drawn from the r1(z|y) mixture model.
        """
        # Compute the standard deviations from the log variances
        r1_scale = torch.exp(0.5 * r1_log_var)

        # Define the mixture distribution
        mixture_distribution = Categorical(logits=r1_weight)

        # Define the Gaussian components
        components_distribution = MultivariateNormal(
            loc=r1_mean,
            scale_tril=torch.diag_embed(r1_scale)  # Convert scale_diag to scale_tril
        )

        # Define the MixtureSameFamily distribution
        bimix_gauss = MixtureSameFamily(mixture_distribution, components_distribution)

        # Sample from the mixture model
        r1_zy_samp = bimix_gauss.sample()

        return r1_zy_samp, bimix_gauss

# conditioned parameters on signals
class Q(nn.Module):
    def __init__(self, signal_dim, param_dim, hidden_dim, latent_dim, num_channels=1, kernel_size=3, stride=1, padding=1):
        super(Q, self).__init__()
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
    
    def reparameterization(self, mean, log_var):
        var = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(var).to(self.DEVICE)
        z = mean + var * epsilon  # reparameterization trick
        return z
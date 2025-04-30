import math
import torch
from torch import nn
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily

# note:
# y = signal - our observation
# x = parameter
# z = latent variable
# h = hidden layer

SMALL_CONSTANT = 1e-8

RAMP = False

def compute_ramp(idx, ramp_start=1, ramp_end=100):
    """Compute the ramp value based on log-scaling."""
    if not RAMP:
        return 1.0

    idx = max(float(idx), 1.0)  # Clamp idx to be at least 1.0 to avoid log(0)
    numerator = math.log(idx) - math.log(ramp_start)
    denominator = math.log(ramp_end) - math.log(ramp_start)
    ramp = numerator / denominator

    # Clamp ramp between 0.0 and 1.0
    ramp = max(0.0, min(1.0, ramp))
    
    return ramp

class CVAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, param_dim, signal_dim, num_components, num_epochs, DEVICE):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_components = num_components
        self.num_epochs = num_epochs

        self.r1 = Encoder(signal_dim=signal_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.r2 = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, param_dim=param_dim, signal_dim=signal_dim)
        self.q = Q(signal_dim=signal_dim, param_dim=param_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

        self.DEVICE = DEVICE
    
    def forward(self, x, y, epoch): # checked
        # r1 network
        r1_mean, r1_log_var = self.r1(y)
        r1_z_samp = self.r1.reparameterization(r1_mean, r1_log_var)

        # output from q network
        q_zxy_mean, q_zxy_log_var = self.q(x, y)
        q_zxy_z_samp = self.q.reparameterization(q_zxy_mean, q_zxy_log_var)

        # r2 network
        r2_x_mean, r2_x_log_var = self.r2(q_zxy_z_samp, y)

        # calculate losses
        loss, reconstruction_loss_x, KL = self.loss(
            x, r2_x_mean, r2_x_log_var,
            q_zxy_mean, q_zxy_log_var,
            r1_mean, r1_log_var
        )

        return r2_x_mean, loss, reconstruction_loss_x, KL
    
    def posterior_samples(self, y, num_samples):
        x_samples = []
        for i in range(num_samples):
            # r1 network
            r1_mean, r1_log_var = self.r1(y)
            r1_z_samp = self.r1.reparameterization(r1_mean, r1_log_var)

            # r2 network
            r2_x_mean, _ = self.r2(r1_z_samp, y)
            x_samples.append(r2_x_mean)

        # Stack the list into a single tensor
        return torch.cat(x_samples, dim=0)

    
    def loss(
        self, x, r2_x_mean, r2_x_log_var,
        q_zxy_mean, q_zxy_log_var,
        r1_mean, r1_log_var
    ): # unsure whether reconstruction loss is correct, I think KL is correct based on the paper
        small_constant = 1e-8

        # Reconstruction loss (ELBO component)
        normalising_factor_x = -0.5 * (
            r2_x_log_var + torch.log(torch.tensor(2.0 * torch.pi + small_constant, device=r2_x_log_var.device, dtype=r2_x_log_var.dtype))
        )
        square_diff_between_mu_and_x = (r2_x_mean - x) ** 2
        inside_exp_x = -0.5 * square_diff_between_mu_and_x / (torch.exp(r2_x_log_var) + small_constant)
        reconstruction_loss_x = torch.sum(normalising_factor_x + inside_exp_x, dim=1)
        reconstruction_loss_x = torch.mean(reconstruction_loss_x)  # Make scalar

        # KL divergence between q(z|x, y) and r(z|y)
        kl_divergence = 0.5 * torch.sum(
            torch.exp(q_zxy_log_var - r1_log_var)  # σ_q^2 / σ_r^2
            + ((r1_mean - q_zxy_mean) ** 2) / (torch.exp(r1_log_var) + small_constant)  # (μ_r - μ_q)^2 / σ_r^2
            - 1
            + r1_log_var - q_zxy_log_var,  # log(σ_r^2 / σ_q^2)
            dim=1
        )
        KL = torch.mean(kl_divergence)  # Make scalar

        # Total loss
        loss = -reconstruction_loss_x + KL
        return loss, reconstruction_loss_x, KL

    # def loss(
    #     self, x, r2_x_mean, r2_x_log_var,
    #     q_zxy_mean, q_zxy_log_var,
    #     r1_mean, r1_log_var
    # ):
    #     small_constant = 1e-8

    #     # Split x into numeric and one-hot-encoded parts
    #     x_numeric = x[:, 0]  # First column is the numeric variable
    #     x_one_hot = x[:, 1:]  # Remaining columns are one-hot-encoded variables

    #     # Split r2_x_mean and r2_x_log_var accordingly
    #     r2_x_mean_numeric = r2_x_mean[:, 0]
    #     r2_x_log_var_numeric = r2_x_log_var[:, 0]
    #     r2_x_mean_one_hot = r2_x_mean[:, 1:]
    #     r2_x_log_var_one_hot = r2_x_log_var[:, 1:]

    #     # Reconstruction loss for numeric variable (Gaussian log-likelihood)
    #     normalising_factor_numeric = -0.5 * (
    #         r2_x_log_var_numeric + torch.log(torch.tensor(2.0 * torch.pi + small_constant, device=r2_x_log_var_numeric.device, dtype=r2_x_log_var_numeric.dtype))
    #     )
    #     square_diff_numeric = (r2_x_mean_numeric - x_numeric) ** 2
    #     inside_exp_numeric = -0.5 * square_diff_numeric / (torch.exp(r2_x_log_var_numeric) + small_constant)
    #     reconstruction_loss_numeric = normalising_factor_numeric + inside_exp_numeric
    #     reconstruction_loss_numeric = torch.mean(reconstruction_loss_numeric)  # Make scalar

    #     # Reconstruction loss for one-hot-encoded variables (Binary Cross-Entropy)
    #     reconstruction_loss_one_hot = nn.BCEWithLogitsLoss()(r2_x_mean_one_hot, x_one_hot)

    #     # Total reconstruction loss
    #     reconstruction_loss_x = reconstruction_loss_numeric + reconstruction_loss_one_hot

    #     # KL divergence between q(z|x, y) and r(z|y)
    #     kl_divergence = 0.5 * torch.sum(
    #         torch.exp(q_zxy_log_var - r1_log_var)  # σ_q^2 / σ_r^2
    #         + ((r1_mean - q_zxy_mean) ** 2) / (torch.exp(r1_log_var) + small_constant)  # (μ_r - μ_q)^2 / σ_r^2
    #         - 1
    #         + r1_log_var - q_zxy_log_var,  # log(σ_r^2 / σ_q^2)
    #         dim=1
    #     )
    #     KL = torch.mean(kl_divergence)  # Make scalar

    #     # Total loss
    #     loss = -reconstruction_loss_x + KL
    #     return loss, reconstruction_loss_x, KL


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, param_dim, signal_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.signal_dim = signal_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.y_fc_layers = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(hidden_dim, param_dim)        
        )
        self.fc_log_var = nn.Sequential(
            nn.Linear(hidden_dim, param_dim)       
        )

    def forward(self, z, y):
        hy = self.y_fc_layers(y)
        zhy = torch.cat([z, hy], dim=1)
        hzy = self.fc_layers(zhy)
        x_hat = self.fc_loc(hzy)
        log_var = self.fc_log_var(hzy)
        return x_hat, log_var

class Encoder(nn.Module):
    def __init__(self, signal_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.FC_means = nn.Linear(hidden_dim, latent_dim)
        self.FC_log_vars = nn.Linear(hidden_dim, latent_dim)

    def forward(self, y):
        h = self.fc_layers(y)
        means = self.FC_means(h)
        log_vars = self.FC_log_vars(h)
        return means, log_vars

    def reparameterization(self, mean, log_var):
        var = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    # def define_and_sample_gmm(self, r1_weight, r1_mean, r1_log_var, ramp):
    #     # Ensure numerical stability by clamping log_var
    #     r1_scale = torch.exp(0.5 * torch.clamp(r1_log_var, min=-10, max=10))  # Clamp log_var to prevent extreme values
    #     r1_scale = torch.clamp(r1_scale, min=1e-6)  # Ensure that the scale is not too small

    #     # Debugging prints
    #     print("r1_mean:", r1_mean)
    #     print("r1_log_var:", r1_log_var)
    #     print("r1_scale:", r1_scale)

    #     # Apply KL annealing to weight
    #     r1_weight = torch.clamp(r1_weight, min=SMALL_CONSTANT)  # Ensure weights are positive
    #     r1_weight = r1_weight * ramp  # Apply ramp to weights

    #     # Debugging prints
    #     print("r1_weight:", r1_weight)

    #     # Define the mixture distribution
    #     mixture_distribution = Categorical(logits=r1_weight)  # Categorical distribution over components
    #     components_distribution = MultivariateNormal(
    #         loc=r1_mean, 
    #         scale_tril=torch.diag_embed(r1_scale)  # Diagonal covariance matrix
    #     )

    #     # Define the Mixture of Gaussians distribution
    #     bimix_gauss = MixtureSameFamily(mixture_distribution, components_distribution)

    #     # Sample from the distribution
    #     r1_zy_samp = bimix_gauss.sample()

    #     # Debugging prints
    #     print("r1_zy_samp:", r1_zy_samp)

    #     return r1_zy_samp, bimix_gauss


class Q(nn.Module):
    def __init__(self, signal_dim, param_dim, hidden_dim, latent_dim):
        super(Q, self).__init__()
        self.latent_dim = latent_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim + param_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.y_fc_layers = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, y):
        hy = self.y_fc_layers(y)
        xhy = torch.cat([x, hy], dim=1)
        hxy = self.fc_layers(xhy)
        mean = self.FC_mean(hxy)
        log_var = self.FC_var(hxy)
        return mean, log_var

    def reparameterization(self, mean, log_var):
        var = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z
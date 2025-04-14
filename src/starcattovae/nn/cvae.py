import torch
from torch import nn
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily

# note:
# y = signal - our observation
# x = parameter
# z = latent variable
# h = hidden layer

class CVAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, param_dim, signal_dim, num_components, num_epochs, DEVICE):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_components = num_components
        self.num_epochs = num_epochs

        self.r1 = Encoder(signal_dim=signal_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_components=num_components)
        self.r2 = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, param_dim=param_dim, signal_dim=signal_dim)
        self.q = Q(signal_dim=signal_dim, param_dim=param_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

        self.DEVICE = DEVICE
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.DEVICE)  # sampling epsilon        
        z = mean + var * epsilon  # reparameterization trick
        return z
    
    def forward(self, x, y, epoch):
        # r1 network
        r1_mean, r1_log_var, r1_weights = self.r1(y)
        r1_zy_z_samp, bimix_gauss = self.r1.define_and_sample_gmm(r1_weights, r1_mean, r1_log_var)

        # output from q network
        q_zxy_mean, q_zxy_log_var = self.q(x, y)
        q_zxy_z_samp = self.q.reparameterization(q_zxy_mean, q_zxy_log_var)

        # r2 network
        r2_x_mean, r2_x_log_var = self.r2(q_zxy_z_samp, y)

        # calculate losses
        loss, reconstruction_loss_x, KL = self.loss(x, y, r2_x_mean, r2_x_log_var, q_zxy_z_samp, q_zxy_mean, q_zxy_log_var, r1_mean, r1_log_var, r1_weights, bimix_gauss, epoch)

        return r2_x_mean, loss, reconstruction_loss_x, KL
    
    def annealing(self, epoch):
        """Anneal the KL divergence term in the loss function"""
        if epoch < self.num_epochs / 2:
            return 0.0
        else:
            return min(1.0, (epoch - self.num_epochs / 2) / (self.num_epochs / 2))
    
    def loss(self, x, y, r2_x_mean, r2_x_log_var, q_zxy_z_samp, q_zxy_mean, q_zxy_log_var, r1_mean, r1_log_var, r1_weights, bimix_gauss, epoch):
        small_constant = 1e-8
        normalising_factor_x = -0.5 * (r2_x_log_var + torch.log(2.0 * torch.pi + small_constant))
        square_diff_between_mu_and_x = (r2_x_mean - x) ** 2
        inside_exp_x = -0.5 * square_diff_between_mu_and_x / (torch.exp(r2_x_log_var) + small_constant)
        reconstruction_loss_x = torch.sum(normalising_factor_x + inside_exp_x, dim=1, keepdim=True)

        normalising_factor_kl = -0.5 * (q_zxy_log_var + torch.log(2.0 * torch.pi + small_constant))
        square_diff_between_qz_and_q = (q_zxy_mean - q_zxy_z_samp) ** 2
        inside_exp_q = -0.5 * square_diff_between_qz_and_q / (torch.exp(q_zxy_log_var) + small_constant)
        log_q_q = torch.sum(normalising_factor_kl + inside_exp_q, dim=1, keepdim=True)
        ramp = self.annealing(epoch)
        log_r1_q = bimix_gauss.log_prob(q_zxy_z_samp)
        KL = torch.mean(log_q_q - log_r1_q)
        KL = KL * ramp

        loss = reconstruction_loss_x + KL
        return loss, reconstruction_loss_x, KL


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

        self.fc_loc = nn.Linear(hidden_dim, param_dim)
        self.fc_log_var = nn.Linear(hidden_dim, param_dim)

    def forward(self, z, y):
        hy = self.y_fc_layers(y)
        zhy = torch.cat([z, hy], dim=1)
        hzy = self.fc_layers(zhy)
        x_hat = self.fc_loc(hzy)
        log_var = self.fc_log_var(hzy)
        return x_hat, log_var

class Encoder(nn.Module):
    def __init__(self, signal_dim, hidden_dim, latent_dim, num_components):
        super(Encoder, self).__init__()
        self.num_components = num_components
        self.latent_dim = latent_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.FC_means = nn.Linear(hidden_dim, latent_dim * num_components)
        self.FC_log_vars = nn.Linear(hidden_dim, latent_dim * num_components)
        self.FC_weights = nn.Linear(hidden_dim, num_components)

    def forward(self, y):
        h = self.fc_layers(y)
        means = self.FC_means(h).view(-1, self.num_components, self.latent_dim)
        log_vars = self.FC_log_vars(h).view(-1, self.num_components, self.latent_dim)
        weights = torch.softmax(self.FC_weights(h), dim=1)
        return means, log_vars, weights

    def define_and_sample_gmm(self, r1_weight, r1_mean, r1_log_var):
        r1_scale = torch.exp(0.5 * r1_log_var)
        mixture_distribution = Categorical(logits=r1_weight)
        components_distribution = MultivariateNormal(
            loc=r1_mean,
            scale_tril=torch.diag_embed(r1_scale)
        )
        bimix_gauss = MixtureSameFamily(mixture_distribution, components_distribution)
        r1_zy_samp = bimix_gauss.sample()
        return r1_zy_samp, bimix_gauss


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
        print(x.shape)
        print(y.shape)
        hy = self.y_fc_layers(y)
        print(hy.shape)
        xhy = torch.cat([x, hy], dim=1)
        print(xhy.shape)
        hxy = self.fc_layers(xhy)
        print(hxy.shape)
        mean = self.FC_mean(hxy)
        log_var = self.FC_var(hxy)
        return mean, log_var

    def reparameterization(self, mean, log_var):
        var = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z
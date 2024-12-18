import torch
import torch.nn as nn
import torch.nn.functional as F

from common.registry import registry

def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

@registry.register_method("conditional_ddpm")
class ConditionalDDPM:
    def __init__(
        self,
        num_train_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        betas=None,
        beta_schedule="linear",
    ):

        if betas is not None:
            self.betas = betas
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, steps=num_train_steps, dtype=torch.float32)

        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0) # a_t_bar
        self.alphas_bar_prev = F.pad(self.alphas_bar, [1, 0], value=1)[:num_train_steps] # a_(t-1)_bar

        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - self.alphas_bar)
        self.sqrt_recip_alphas_bar = torch.sqrt(1 / self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.sqrt(1. / self.alphas_bar - 1)

        self.posterior_var = self.betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)
        self.posterior_log_var_clipped = torch.log(self.posterior_var[1:])
        self.posterior_mean_coef1 = torch.sqrt(self.alphas_bar_prev) * self.betas / (1. - self.alphas_bar)
        self.posterior_mean_coef2 = torch.sqrt(self.alphas) * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)

        self.num_train_steps = num_train_steps

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)
        self.alphas_bar_prev = self.alphas_bar_prev.to(device)
        self.sqrt_alphas_bar = self.sqrt_alphas_bar.to(device)
        self.sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar.to(device)
        self.sqrt_recip_alphas_bar = self.sqrt_recip_alphas_bar.to(device)
        self.sqrt_recipm1_alphas_bar = self.sqrt_recipm1_alphas_bar.to(device)

        self.posterior_var = self.posterior_var.to(device)
        self.posterior_log_var_clipped = self.posterior_log_var_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)

        return self

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t, eps):
        model_log_var = torch.log(torch.cat([self.posterior_var[1:2], self.betas[1:]]))
        model_log_var = extract(model_log_var, t, x_t.shape)

        x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
        model_mean, _ = self.q_mean_variance(x_0, x_t, t)

        return model_mean, model_log_var

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample(self, x_0, t=None, return_noise=False):
        """
        sample conditional flow sample
        """

        if t is None:
            nt = torch.rand(x_0.shape[0]).type_as(x_0)
            t = (nt * self.num_train_steps).to(torch.int64)

        assert len(t) == x_0.shape[0]

        eps = self.sample_noise_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * eps

        return nt, x_t, eps

    @classmethod
    def from_config(cls, cfg):

        return cls()

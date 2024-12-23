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

        scale = 1000 / num_train_steps
        beta_start *= scale
        beta_end *= scale
        if betas is not None:
            self.betas = betas
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, steps=num_train_steps, dtype=torch.float32)

        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0) # a_t_bar
        self.alphas_bar_prev = F.pad(self.alphas_bar, [1, 0], value=1)[:num_train_steps] # a_(t-1)_bar

        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - self.alphas_bar)
        self.sqrt_recip_alphas_bar = torch.sqrt(1. / self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.sqrt(1. / self.alphas_bar - 1)

        self.posterior_var = self.betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)
        self.posterior_log_var_clipped = torch.log(self.posterior_var.clamp(min =1e-20))
        self.posterior_mean_coef_x0 = torch.sqrt(self.alphas_bar_prev) * self.betas / (1. - self.alphas_bar)
        self.posterior_mean_coef_xt = torch.sqrt(self.alphas) * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)

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
        self.posterior_mean_coef_x0 = self.posterior_mean_coef_x0.to(device)
        self.posterior_mean_coef_xt = self.posterior_mean_coef_xt.to(device)

        return self

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    # sample x_t ~ q(x_t | x_0)
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            eps = self.sample_noise_like(x_0)
        else:
            eps = noise

        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +  \
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * eps

        return x_t, eps

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef_x0, t, x_0.shape) * x_0 +
            extract(self.posterior_mean_coef_xt, t, x_t.shape) * x_t
        )
        posterior_var = extract(
            self.posterior_var, t, x_t.shape
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_var, posterior_log_var_clipped


    def predict_xstart_from_eps(self, x_t, t, eps):
        """
        predict x_0 from predicted noise and x_t refer to eq (4)
        """
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )


    def p_mean_variance(self, x_t, t, pred_eps, clip_denoised=True):
        """
        predict mean and variance of p(x_{t-1} | x_t)
        """
        x_0_pred = self.predict_xstart_from_eps(x_t, t, eps=pred_eps)
        if clip_denoised:
            x_0 = torch.clamp(x_0_pred, min=-1., max=1.)

        model_mean, model_var, model_log_var = self.q_mean_variance(x_0, x_t, t)

        return model_mean, model_var, model_log_var


    def p_sample(self, x_t, t, pred_eps, clip_denoised=True):
        """
        compute x_{t-1} using pred_epsilon
        """
        model_mean, _, model_log_var = self.p_mean_variance(x_t, t, pred_eps,
                                                    clip_denoised=clip_denoised)
        z_noise = self.sample_noise_like(x_t)

        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))

        pred_img = model_mean + nonzero_mask * (0.5 * model_log_var).exp() * z_noise

        return pred_img

    def sample(self, x_0, t=None, return_noise=False):
        """
        sample conditional flow sample
        """

        if t is None:
            nt = torch.rand(x_0.shape[0]).type_as(x_0)
            t = (nt * self.num_train_steps).to(torch.int64)

        x_t, eps = self.q_sample(x_0, t)

        if not return_noise:
            return t, x_t

        return t, x_t, eps

    @classmethod
    def from_config(cls, cfg):

        num_train_steps = cfg.get("num_train_steps", 1000)
        beta_start = cfg.get("beta_start", 1e-4)
        beta_end = cfg.get("beta_end", 0.02)
        betas = cfg.get("betas", None)
        beta_scheduler = cfg.get("beta_scheduler", "linear")

        return cls(num_train_steps, beta_start, beta_end, betas, beta_scheduler)

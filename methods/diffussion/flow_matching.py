import torch
import torch.nn as nn

from common.registry import registry

def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

@registry.register_method("conditional_flow_matching")
class ConditionalFlowMatching:
    def __init__(self, sigma):
        self.sigma = sigma

    def compute_mu_t(self, x0, x1, t):
        """
        refer to N(t * x_1 + (1 - t) * x_0, sigma)
        x0: source distribution
        x1: target distribution
        t: time
        """

        if t.dim() != x0.dim():
            t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0


    def compute_sigma_t(self, t):
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x1)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        x0: source distribution
        x1: target distribution
        t: time
        xt: sample drawn
        """
        del t, xt
        return x1 - x0

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample(self, x0, x1, t=None, return_noise=False):
        """
        sample conditional flow sample
        """

        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0]

        epsilon = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, epsilon)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, epsilon
        else:
            return t, xt, ut

    @classmethod
    def from_config(cls, cfg):
        sigma = cfg.sigma

        return cls(sigma)

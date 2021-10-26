import torch


def uniform_to_exp(log_mean, uniform=None, enable_grad=False):
    if uniform is not None:
        assert uniform.size() == log_mean.size()
    else:
        uniform = torch.distributions.utils.clamp_probs(torch.rand_like(log_mean))

    exp = torch.exp(log_mean + torch.log(-torch.log(uniform)))
    if enable_grad:
        exp.requires_grad_(True)

    return exp

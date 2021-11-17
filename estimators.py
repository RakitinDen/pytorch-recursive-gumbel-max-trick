import torch


def uniform_to_exp(logits, uniform=None, enable_grad=False):
    if uniform is not None:
        assert uniform.size() == logits.size()
    else:
        uniform = torch.distributions.utils.clamp_probs(torch.rand_like(logits))

    exp = torch.exp(logits + torch.log(-torch.log(uniform)))
    if enable_grad:
        exp.requires_grad_(True)

    return exp


def reattach_exp_to_new_logits(logits, exp):
    exp = torch.exp(torch.log(exp.detach()) + logits - logits.detach())
    return exp


def E_reinforce(loss_value, logits, exp, plus_samples=1, mask_unused_values=None, **kwargs):
    batch_size = logits.shape[0] // plus_samples

    loss_value = loss_value.detach()
    exp = exp.detach()

    log_prob = -logits - torch.exp(torch.log(exp) - logits)
    if mask_unused_values is not None:
        log_prob = mask_unused_values(log_prob, **kwargs)

    dims_except_batch = tuple(-i for i in range(1, logits.ndimension()))
    log_prob = log_prob.sum(dim=dims_except_batch)

    score = torch.autograd.grad([log_prob], [logits], grad_outputs=torch.ones_like(log_prob))[0]

    if plus_samples > 1:
        score_shape = (batch_size, plus_samples) + logits.shape[1:]
        score = score.view(score_shape)

        loss_value = loss_value.view(batch_size, plus_samples)
        loss_value = loss_value - loss_value.mean(dim=-1)[:, None]
        for i in range(logits.ndimension() - 1):
            loss_value = loss_value.unsqueeze(-1)

        grad = (loss_value * score).sum(dim=1) / (plus_samples - 1)
    else:
        for i in range(logits.ndimension() - 1):
            loss_value = loss_value.unsqueeze(-1)

        grad = loss_value * score

    return grad


def T_reinforce(loss_value, struct_var, logits, f_log_prob, plus_samples=1, **kwargs):
    batch_size = logits.shape[0] // plus_samples

    loss_value = loss_value.detach()
    struct_var = struct_var.detach()
    log_prob = f_log_prob(struct_var, logits, **kwargs)
    score = torch.autograd.grad([log_prob], [logits], grad_outputs=torch.ones_like(log_prob))[0]

    if plus_samples > 1:
        score_shape = (batch_size, plus_samples) + logits.shape[1:]
        score = score.view(score_shape)

        loss_value = loss_value.view(batch_size, plus_samples)
        loss_value = loss_value - loss_value.mean(dim=-1)[:, None]
        for i in range(logits.ndimension() - 1):
            loss_value = loss_value.unsqueeze(-1)

        grad = (loss_value * score).sum(dim=1) / (plus_samples - 1)
    else:
        for i in range(logits.ndimension() - 1):
            loss_value = loss_value.unsqueeze(-1)

        grad = loss_value * score

    return grad


def relax(loss_value, struct_var, logits, exp, critic, f_log_prob, f_cond, uniform=None, **kwargs):
    loss_value = loss_value.detach()
    struct_var = struct_var.detach()

    logits = logits.detach().requires_grad_(True)
    exp = reattach_exp_to_new_logits(logits, exp)

    exp_cond = f_cond(struct_var, logits, uniform, **kwargs)

    baseline_exp = critic(exp)
    baseline_cond = critic(exp_cond).squeeze()

    diff = loss_value - baseline_cond
    log_prob = f_log_prob(struct_var, logits, **kwargs)

    score, = torch.autograd.grad(
        [log_prob],
        [logits],
        grad_outputs = torch.ones_like(log_prob)
    )

    d_baseline_exp, = torch.autograd.grad(
        [baseline_exp],
        [logits],
        create_graph=True,
        retain_graph=True,
        grad_outputs=torch.ones_like(baseline_exp)
    )

    d_baseline_cond, = torch.autograd.grad(
        [baseline_cond],
        [logits],
        create_graph=True,
        retain_graph=True,
        grad_outputs=torch.ones_like(baseline_cond)
    )

    for i in range(logits.ndimension() - 1):
        diff = diff.unsqueeze(-1)

    grad = diff * score + d_baseline_exp - d_baseline_cond

    assert grad.size() == logits.size()

    return grad

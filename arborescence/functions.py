import torch
import sys
sys.path.append('../')

from estimators import uniform_to_exp
from edmonds import get_arborescence_batch
from arborescence.utils import Arborescence, expand_mask, calc_trace_log_prob

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# defines F_struct for arborescence
# applies the Chu-Liu Edmonds' algorithm to a perturbed weight matrix
def arb_struct(exp, lengths, root=0, **kwargs):
    assert exp.ndimension() == 3

    batch_size = exp.shape[0]

    arbs, min_xs, min_ys, masks = get_arborescence_batch(
        exp.detach().cpu(), root, lengths.cpu().int())

    arbs = arbs.to(device)
    trace = []

    for i in range(batch_size):
        trace_i = {}
        trace_i['min_x'] = min_xs[i].to(device)
        trace_i['min_y'] = min_ys[i].to(device)
        bool_mask = masks[i].to(device)
        bool_mask = ~expand_mask(bool_mask.bool())
        mask_i = (bool_mask).float().masked_fill_(bool_mask, float('inf'))
        trace_i['mask'] = mask_i
        trace.append(trace_i)

    return Arborescence(arbs, trace)


# defines F_log_prob for arborescence
# calculates the log probability log(p(T)) of the execution trace
def arb_log_prob(struct_var, logits, lengths, **kwargs):
    batch_size = logits.shape[0]
    dim = logits.shape[1]
    used_logits = logits.masked_fill(
        torch.eye(dim, dtype=torch.bool).unsqueeze(0).repeat(batch_size, 1, 1).to(device),
        float('inf')
    )

    trace = struct_var.trace
    res = torch.zeros(len(trace)).to(device)

    for i in range(batch_size):
        res[i] += calc_trace_log_prob(used_logits[i], trace[i], lengths[i])

    return res


# defines F_cond for arborescence
# samples from the conditional distribution p(E | T) of exponentials given the execution trace
def arb_cond(struct_var, logits, uniform, lengths, **kwargs):
    batch_size = logits.shape[0]
    cond_exp = torch.zeros_like(logits)

    for i in range(batch_size):
        trace_i = struct_var.trace[i]
        mask = trace_i['mask']
        min_x = trace_i['min_x']
        min_y = trace_i['min_y']

        masked_logits = mask + logits[i, :lengths[i], :lengths[i]]

        logits_of_minimums = -torch.logsumexp(-masked_logits, dim=(-1, -2))
        uniform_of_minimums = uniform[i][min_x, min_y]
        log_minimums = logits_of_minimums + torch.log(-torch.log(uniform_of_minimums))
        log_minimums = log_minimums[:, None, None] - mask
        cond_exp[i, :lengths[i], :lengths[i]] += torch.exp(log_minimums).sum(dim=0)

        non_minimums = torch.zeros_like(logits[i, :lengths[i], :lengths[i]])
        non_minimums[min_x, min_y] = float('-inf')

        cond_exp[i, :lengths[i], :lengths[i]] += uniform_to_exp(
            logits=logits[i, :lengths[i], :lengths[i]] + non_minimums,
            uniform=uniform[i, :lengths[i], :lengths[i]],
            enable_grad=False)


    return cond_exp

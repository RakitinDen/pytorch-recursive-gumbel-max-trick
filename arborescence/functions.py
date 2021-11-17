import torch

import sys
sys.path.append('../')
from estimators import uniform_to_exp
from edmonds import get_arborescence_batch
from arborescence.utils import Arborescence, expand_mask, calc_trace_log_prob

def arb_struct(exp, lengths, root=0, **kwargs):
    '''
    Defines F_struct for arborescence
    Applies the Chu-Liu Edmonds' algorithm to a weight matrix 'exp'
    Calculates the minimal spanning arborescence and the execution trace
    get_arborescence_batch is implemented in C++ and performs all the fundamental calculations

    Input
    --------------------
    exp         : torch.Tensor | batch_size x dim x dim |
                  Contains a batch of adjacency matrices

    lengths     : torch.Tensor | batch_size | 
                  Contains dimensions of graphs in the batch (lengths[i] <= dim)

    root        : int
                  Defines root of the arborescence for all the graphs in a batch

    **kwargs    : Needed to support usage of different F_struct in the estimators' implementation

    Output
    --------------------
    struct_var  : Arborescence (defined in arborescence.utils)
                  Contains the spanning arborescences with the corresponding execution traces
    '''
    batch_size = exp.shape[0]

    arborescence, min_xs, min_ys, masks = get_arborescence_batch(
        exp.detach().cpu(), root, lengths.cpu().int())

    trace = []
    for i in range(batch_size):
        trace_i = {}
        trace_i['min_x'] = min_xs[i]
        trace_i['min_y'] = min_ys[i]
        bool_mask = masks[i]
        bool_mask = ~expand_mask(bool_mask.bool())
        mask_i = (bool_mask).float().masked_fill_(bool_mask, float('inf'))
        trace_i['mask'] = mask_i
        trace.append(trace_i)

    struct_var = Arborescence(arborescence, trace)
    return struct_var

def arb_log_prob(struct_var, logits, lengths, **kwargs):
    '''
    Defines F_log_prob for arborescence
    Calculates the log probability log(p(T)) of the execution trace

    Input
    --------------------
    struct_var  : Arborescence (defined in arborescence.utils)
                  Contains a batch of arborescences with the corresponding execution traces

    logits      : torch.Tensor | batch_size x dim x dim |
                  Contains parameters (log(mean)) of the exponential distributions of edge weights

    lengths     : torch.Tensor | batch_size | 
                  Contains dimensions of graphs in the batch (lengths[i] <= dim)

    **kwargs    : Needed to support usage of different F_log_prob in the estimators' implementation

    Output
    --------------------
    log_prob    : torch.Tensor | batch_size |
                  Contains log probabilities of the execution traces for graphs in the batch
    '''
    batch_size = logits.shape[0]
    dim = logits.shape[1]

    used_logits = logits.masked_fill(
        torch.eye(dim, dtype=torch.bool).unsqueeze(0).repeat(batch_size, 1, 1),
        float('inf')
    )

    trace = struct_var.trace
    log_prob = torch.zeros(batch_size)

    for i in range(batch_size):
        log_prob[i] += calc_trace_log_prob(used_logits[i], trace[i], lengths[i])

    return log_prob

def arb_cond(struct_var, logits, uniform, lengths, **kwargs):
    '''
    Defines F_cond for arborescence
    Samples for the conditional distribution p(E | T) of exponentials given the execution trace

    Input
    --------------------
    struct_var  : Arborescence (defined in arborescence.utils)
                  Contains a batch of arborescences with the corresponding execution traces

    logits      : torch.Tensor | batch_size x dim x dim |
                  Contains parameters (log(mean)) of the exponential distributions of edge weights

    uniform     : torch.Tensor | batch_size x dim x dim |
                  Contains realizations of the independent uniform variables, that will be transformed to conditional samples

    lengths     : torch.Tensor | batch_size | 
                  Contains dimensions of graphs in the batch (lengths[i] <= dim)

    **kwargs    : Needed to support usage of different F_cond in the estimators' implementation

    Output
    --------------------
    cond_exp    : torch.Tensor | batch_size x dim x dim |
                  Contains conditional samples from p(E | T)
    '''
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

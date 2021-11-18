import torch

import sys
sys.path.append('../')
from estimators import uniform_to_exp
from matching.utils import Matching

def matching_struct(exp, **kwargs):
    '''
    Defines F_struct for perfect matching
    Applies the "crossing" algorithm to a weight matrix 'exp'
    It recursively finds argminimum and excludes the corresponding row and column

    Input
    --------------------
    exp         : torch.Tensor | batch_size x dim x dim |
                  Contains a batch of weight matrices

    **kwargs    : Needed to support usage of different F_struct in the estimators' implementation

    Output
    --------------------
    struct_var  : Matching (defined in matching.utils)
                  Contains perfect matchings represented as binary masks
                  with the corresponding execution traces
    '''
    batch_size = exp.shape[0]
    dim = exp.shape[1]

    matching = torch.zeros(batch_size, dim, dim)
    mask = torch.zeros(batch_size, dim, dim, dim)

    cur_exp = exp.clone()
    min_x = torch.zeros(batch_size, dim, dtype=torch.long)
    min_y = torch.zeros(batch_size, dim, dtype=torch.long)

    for i in range(dim):
        min_values, argmin = torch.min(cur_exp, dim=-1)
        min_x_i = torch.argmin(min_values, dim=-1)
        min_y_i = argmin[torch.arange(batch_size), min_x_i]
        min_x[:, i] = min_x_i
        min_y[:, i] = min_y_i

        matching[torch.arange(batch_size), min_x_i, min_y_i] = 1

        mask[:, i, :, :] = cur_exp
        cur_exp[torch.arange(batch_size), min_x_i, :] = float('inf')
        cur_exp[torch.arange(batch_size), :, min_y_i] = float('inf')

    mask[mask != float('inf')] = 0

    trace = {
        'mask'  : mask,
        'min_x' : min_x,
        'min_y' : min_y
    }

    struct_var = Matching(matching, trace)
    return struct_var

def matching_log_prob(struct_var, logits, **kwargs):
    '''
    Defines F_log_prob for perfect matching
    Calculates the log probability log(p(T)) of the execution trace

    Input
    --------------------
    struct_var  : Matching (defined in matching.utils)
                  Contains perfect matchings represented as binary masks
                  with the corresponding execution traces

    logits      : torch.Tensor | batch_size x dim x dim |
                  Contains parameters (log(mean)) of the exponential distributions of edge weights

    **kwargs    : Needed to support usage of different F_log_prob in the estimators' implementation

    Output
    --------------------
    log_prob    : torch.Tensor | batch_size |
                  Contains log probabilities of the execution traces for graphs in the batch
    '''
    mask = struct_var.trace['mask']
    min_logits_sum = -(logits * struct_var.matching).sum(dim=(-1, -2))
    other_logits_sum = -torch.sum(torch.logsumexp(-(mask + logits[:, None, :, :]), dim=(-1, -2)), dim=-1)
    log_prob = min_logits_sum + other_logits_sum
    return log_prob

def matching_cond(struct_var, logits, uniform, **kwargs):
    '''
    Defines F_cond for perfect matching
    Samples from the conditional distribution p(E | T) of exponentials given the execution trace

    Input
    --------------------
    struct_var  : Matching (defined in matching.utils)
                  Contains perfect matchings represented as binary masks
                  with the corresponding execution traces

    logits      : torch.Tensor | batch_size x dim x dim |
                  Contains parameters (log(mean)) of the exponential distributions of edge weights

    uniform     : torch.Tensor | batch_size x dim x dim |
                  Contains realizations of the independent uniform variables, that will be transformed to conditional samples

    **kwargs    : Needed to support usage of different F_cond in the estimators' implementation

    Output
    --------------------
    cond_exp    : torch.Tensor | batch_size x dim x dim |
                  Contains conditional samples from p(E | T)
    '''
    matching = struct_var.matching
    mask = struct_var.trace['mask']
    min_x = struct_var.trace['min_x']
    min_y = struct_var.trace['min_y']

    batch_size = struct_var.matching.shape[0]
    dim = struct_var.matching.shape[1]

    exp = uniform_to_exp(logits, uniform)
    cond_exp = torch.zeros_like(logits)

    min_parameters = -torch.logsumexp(-(mask + logits[:, None, :, :]), dim=(-1, -2))

    uniform_for_min = torch.zeros(batch_size, dim)

    for i in range(batch_size):
        uniform_for_min[i] = uniform[i][min_x[i], min_y[i]]

    exp_min = uniform_to_exp(min_parameters, uniform_for_min)

    bin_mask = mask.clone()
    bin_mask[bin_mask == 0] = 1
    bin_mask[bin_mask == float('inf')] = 0

    cond_exp += (bin_mask * exp_min[:, :, None, None]).sum(dim=1)

    non_minimums = 1 - matching
    cond_exp += non_minimums * exp

    return cond_exp

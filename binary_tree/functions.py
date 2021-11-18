import torch

import sys
sys.path.append('../')
from binary_tree.utils import BinaryTree, build_tree
from estimators import uniform_to_exp

def bin_tree_struct(exp, lengths=None, **kwargs):
    '''
    Defines F_struct for binary tree
    Applies the divide and conquer algorithm from the paper

    Input
    --------------------
    exp         : torch.Tensor | batch_size x dim |
                  Contains a batch of arrays

    lengths     : torch.Tensor | batch_size | 
                  Contains lengths of arrays in the batch (lengths[i] <= dim)

    **kwargs    : Needed to support usage of different F_struct in the estimators' implementation

    Output
    --------------------
    struct_var  : BinaryTree (defined in binary_tree.utils)
    '''
    batch_size = exp.shape[0]
    dim = exp.shape[1]

    masks = -torch.log(torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1))
    trees = []
    heights = torch.zeros(batch_size)

    for batch_idx in range(batch_size):
        if lengths is None:
            right = dim
        else:
            right = lengths[batch_idx].item()

        left = 0
        level = 0

        tree = build_tree(batch_idx, exp, left, right, level, masks, heights)
        trees.append(tree)

    struct_var = BinaryTree(masks, trees, heights)
    return struct_var

def bin_tree_log_prob(struct_var, logits, **kwargs):
    '''
    Defines F_log_prob for binary tree
    Calculates the log probability log(p(X)) of the binary tree
    Note: here the execution trace is in one-to-one correspondance with the binary tree itself

    Input
    --------------------
    struct_var  : BinaryTree (defined in binary_tree.utils)

    logits      : torch.Tensor | batch_size x dim |
                  Contains parameters (log(mean)) of the exponential distributions of elements in arrays

    **kwargs    : Needed to support usage of different F_log_prob in the estimators' implementation

    Output
    --------------------
    log_prob    : torch.Tensor | batch_size |
                  Contains log probabilities of the binary trees
    '''
    batch_size = logits.shape[0]
    dim = logits.shape[1]

    logits_expanded = logits.unsqueeze(1).repeat((1, dim, 1))
    masked_logits = struct_var.masks + logits_expanded
    log_probs = -logits - torch.logsumexp(-masked_logits, dim=-1)

    return log_probs.sum(dim=-1)

def bin_tree_cond(struct_var, logits, uniform, **kwargs):
    '''
    Defines F_cond for arborescence
    Samples from the conditional distribution p(E | T) of exponentials given the execution trace

    Input
    --------------------
    struct_var  : BinaryTree (defined in binary_tree.utils)

    logits      : torch.Tensor | batch_size x dim |
                  Contains parameters (log(mean)) of the exponential distributions of elements in arrays

    uniform     : torch.Tensor | batch_size x dim |
                  Contains realizations of the independent uniform variables, that will be transformed to conditional samples

    **kwargs    : Needed to support usage of different F_cond in the estimators' implementation

    Output
    --------------------
    cond_exp    : torch.Tensor | batch_size x dim |
                  Contains conditional samples from p(E | X) = p(E | T)
    '''
    batch_size = logits.shape[0]
    dim = logits.shape[1]

    logits_expanded = logits.unsqueeze(1).repeat((1, dim, 1))
    masked_logits = struct_var.masks + logits_expanded
    min_logits = -torch.logsumexp(-masked_logits, dim=-1)

    minimums = uniform_to_exp(logits=min_logits, uniform=uniform)
    bin_mask = torch.exp(-struct_var.masks)
    cond_exp = (minimums.unsqueeze(-1).repeat((1, 1, dim)) * bin_mask).sum(dim=1)

    return cond_exp

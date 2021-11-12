import torch
from torch.nn.functional import log_softmax
import math

import sys
sys.path.append('../')

from binary_tree.utils import BinaryTree, build_tree
from estimators import uniform_to_exp

def bin_tree_struct(exp, lengths=None, **kwargs):
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

    return BinaryTree(masks, trees, heights)

def bin_tree_log_prob(struct_var, logits, **kwargs):
    batch_size = logits.shape[0]
    dim = logits.shape[1]

    logits_expanded = logits.unsqueeze(1).repeat((1, dim, 1))
    masked_logits = struct_var.masks + logits_expanded
    log_probs = -logits - torch.logsumexp(-masked_logits, dim=-1)

    return log_probs.sum(dim=-1)

def bin_tree_cond(struct_var, logits, uniform, **kwargs):
    batch_size = logits.shape[0]
    dim = logits.shape[1]

    logits_expanded = logits.unsqueeze(1).repeat((1, dim, 1))
    masked_logits = struct_var.masks + logits_expanded
    min_logits = -torch.logsumexp(-masked_logits, dim=-1)

    minimums = uniform_to_exp(logits=min_logits, uniform=uniform)
    bin_mask = torch.exp(-struct_var.masks)
    exp_cond = (minimums.unsqueeze(-1).repeat((1, 1, dim)) * bin_mask).sum(dim=1)

    return exp_cond

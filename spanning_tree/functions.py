import torch

import sys
sys.path.append('../')
from estimators import uniform_to_exp
from spanning_tree.utils import BatchDisjointSets, span_mask_unused_values, get_lightest_edge

def span_struct(exp, **kwargs):
    '''
    Defines F_struct for spanning tree
    Applies the Kruskals' algorithm to a weight matrix 'exp'
    Calculates the minimal spanning tree and the execution trace
    They are merged in one output, represented as the sequence of edges

    Input
    --------------------
    exp      : torch.Tensor | batch_size x dim x dim |
               Contains a batch of adjacency matrices

    **kwargs : Needed to support usage of different F_struct in the estimators' implementation

    Output
    --------------------
    edges    : torch.Tensor | batch_size x (dim - 1) x 2
               Contains the sequence of edges in the spanning trees
               Order is defined by the execution of the Kruskal's algorithm
    '''
    batch_size = exp.shape[0]
    dim = exp.shape[1]

    disjoint_sets = BatchDisjointSets(batch_size, dim)
    edges = torch.empty(batch_size, dim - 1, 2, dtype=torch.long)

    exp = span_mask_unused_values(exp)

    for i in range(dim - 1):
        edges_no_loop = exp.masked_fill(disjoint_sets.get_mask(), float('inf'))
        u, v = get_lightest_edge(edges_no_loop)
        assert torch.all(u < v)
        set_u = disjoint_sets.find(u)
        set_v = disjoint_sets.find(v)
        disjoint_sets.union(set_u, set_v)
        edges[:, i, 0] = u
        edges[:, i, 1] = v

    return edges

def span_log_prob(struct_var, logits, **kwargs):
    '''
    Defines F_log_prob for spanning tree
    Calculates the log probability log(p(T)) of the execution trace

    Input
    --------------------
    struct_var  : torch.Tensor | batch_size x (dim - 1) x 2
                  Contains the sequence of edges in the spanning trees
                  Order is defined by the execution of the Kruskal's algorithm

    logits      : torch.Tensor | batch_size x dim x dim |
                  Contains parameters (log(mean)) of the exponential distributions of edge weights

    **kwargs    : Needed to support usage of different F_log_prob in the estimators' implementation

    Output
    --------------------
    log_prob    : torch.Tensor | batch_size |
                  Contains log probabilities of the execution traces for graphs in the batch
    '''
    batch_size = logits.shape[0]
    dim = logits.shape[1]
    logits = span_mask_unused_values(logits, float('inf'))

    disjoint_sets = BatchDisjointSets(batch_size, dim)
    log_prob = torch.zeros(batch_size)
    for i in range(dim - 1):
        u, v = struct_var[:, i, 0], struct_var[:, i, 1]
        masked_logits = logits.masked_fill(disjoint_sets.get_mask(),
                                                float('inf'))

        log_prob += (-logits[range(batch_size), u, v]) - (-masked_logits).logsumexp((1, 2))
        set_u = disjoint_sets.find(u)
        set_v = disjoint_sets.find(v)
        disjoint_sets.union(set_u, set_v)
    return log_prob

# defines F_cond for spanning tree
# samples from the conditional distribution p(E | T) of exponentials given the execution trace
def span_cond(struct_var, logits, uniform, **kwargs):
    '''
    Defines F_cond for arborescence
    Samples for the conditional distribution p(E | T) of exponentials given the execution trace

    Input
    --------------------
    struct_var  : torch.Tensor | batch_size x (dim - 1) x 2
                  Contains the sequence of edges in the spanning trees
                  Order is defined by the execution of the Kruskal's algorithm

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
    batch_size = logits.shape[0]
    dim = logits.shape[1]

    disjoint_sets = BatchDisjointSets(batch_size, dim)
    visited = torch.zeros_like(logits).bool()

    exp = uniform_to_exp(logits, uniform)
    cond_exp = torch.zeros_like(exp)

    max_value = 0
    for i in range(dim - 1):
        v_1, v_2 = struct_var[:, i, 0], struct_var[:, i, 1]

        masked_logits = logits.masked_fill(disjoint_sets.get_mask(), float('inf'))
        min_logits = (-masked_logits).logsumexp((1, 2))
        min_exp = torch.exp(min_logits + torch.log(-torch.log(uniform[range(batch_size), v_1, v_2])))
        cond_exp[range(batch_size), v_1, v_2] = max_value + min_exp

        max_value += min_exp

        visited[range(batch_size), v_1, v_2] = True

        set_v_1 = disjoint_sets.find(v_1)
        set_v_2 = disjoint_sets.find(v_2)
        disjoint_sets.union(set_v_1, set_v_2)

        impossible = (disjoint_sets.get_mask() * (~visited))

        cond_exp += impossible * (max_value[:, None, None] + exp)
        visited[impossible] = True 

    return span_mask_unused_values(cond_exp, value=0.0)

import torch
import sys
sys.path.append('../')

from spanning_tree.utils import BatchDisjointSets, span_mask_unused_values, get_lightest_edge
from structured_variable import uniform_to_exp


# defines F_struct for spanning tree
# applies Kruskals' for perturbed weight matrix
def span_struct(exp, **kwargs):
    assert(exp.ndimension() == 3)
    batch_size = exp.shape[0]
    n_vertices = exp.shape[1]

    disjoint_sets = BatchDisjointSets(batch_size, n_vertices)
    edges = torch.empty(batch_size, n_vertices - 1, 2, dtype=torch.long)

    exp = span_mask_unused_values(exp)

    for i in range(n_vertices - 1):
        edges_no_loop = exp.masked_fill(disjoint_sets.get_mask(), float('inf'))
        u, v = get_lightest_edge(edges_no_loop)
        assert torch.all(u < v)
        set_u = disjoint_sets.find(u)
        set_v = disjoint_sets.find(v)
        disjoint_sets.union(set_u, set_v)
        edges[:, i, 0] = u
        edges[:, i, 1] = v

    return edges


# defines F_log_prob for spanning tree
# calculates log probability log(p(T)) of the execution trace
def span_log_prob(struct_var, logits, **kwargs):
    logits = span_mask_unused_values(logits, float('inf'))
    assert (struct_var.ndimension() == logits.ndimension())

    batch_size = logits.shape[0]
    n_vertices = logits.shape[1]

    disjoint_sets = BatchDisjointSets(batch_size, n_vertices)
    log_p = torch.zeros(batch_size)
    for i in range(n_vertices - 1):
        u, v = struct_var[:, i, 0], struct_var[:, i, 1]
        masked_logits = logits.masked_fill(disjoint_sets.get_mask(),
                                                float('inf'))

        log_p += (-logits[range(batch_size), u, v]) - (-masked_logits).logsumexp((1, 2))
        set_u = disjoint_sets.find(u)
        set_v = disjoint_sets.find(v)
        disjoint_sets.union(set_u, set_v)
    return log_p

 
# defines F_cond for spanning tree
# samples from conditional distribution p(E | T) of exponentials given execution trace
def span_cond(struct_var, logits, uniform, **kwargs):
    batch_size = logits.shape[0]
    n_vertices = logits.shape[1]

    disjoint_sets = BatchDisjointSets(batch_size, n_vertices)
    visited = torch.zeros_like(logits).bool()

    exp = uniform_to_exp(logits, uniform)
    exp_cond = torch.zeros_like(exp)

    max_value = 0
    for i in range(n_vertices - 1):
        v_1, v_2 = b[:, i, 0], b[:, i, 1]

        masked_logits = logits.masked_fill(disjoint_sets.get_mask(), float('inf'))
        min_logits = (-masked_logits).logsumexp((1, 2))
        min_exp = torch.exp(min_logits + torch.log(-torch.log(v[range(batch_size), v_1, v_2])))
        exp_cond[range(batch_size), v_1, v_2] = max_value + min_exp

        max_value += min_exp

        visited[range(batch_size), v_1, v_2] = True

        set_v_1 = disjoint_sets.find(v_1)
        set_v_2 = disjoint_sets.find(v_2)
        disjoint_sets.union(set_v_1, set_v_2)

        impossible = (disjoint_sets.get_mask() * (~visited))

        exp_cond += impossible * (max_value[:, None, None] + exp)
        visited[impossible] = True 

    return span_mask_unused_values(exp_cond, value=0.0)

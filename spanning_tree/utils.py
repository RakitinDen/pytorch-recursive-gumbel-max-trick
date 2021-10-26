import torch


class BatchDisjointSets():
    
    def __init__(self, bsz, n_vertices):
        self.bsz = bsz
        self.n_vertices = n_vertices
        self.sets = torch.eye(self.n_vertices, dtype=torch.long)
        self.sets = self.sets.unsqueeze(0).repeat(self.bsz, 1, 1)

    def find(self, v):
        has_v = self.sets[range(self.bsz), :, v] == 1
        return torch.argmax(has_v.long(), dim=1)

    def union(self, set_u, set_v):
        assert(torch.all(set_u != set_v))
        r_bsz = range(self.bsz)
        self.sets[r_bsz, set_u] = self.sets[r_bsz, set_u] + self.sets[r_bsz, set_v]
        self.sets[r_bsz, set_v] = 0
        assert(torch.all(self.sets <= 1))

    def get_mask(self):
        return torch.einsum('bij,bik->bjk', (self.sets.float(), self.sets.float())).bool()


def span_mask_unused_values(w, value=float('inf'), **kwargs):
    batch_size, n_vertices, _ = w.size()
    mask = torch.tril(torch.ones(1, n_vertices, n_vertices, dtype=torch.bool))
    masked_w = w.masked_fill(mask, value)
    return masked_w


def get_lightest_edge(w):
    batch_size, n_vertices, _ = w.size()
    flat_w = w.view(batch_size, n_vertices * n_vertices)
    flat_edge = torch.argmin(flat_w, dim=1)
    return flat_edge // n_vertices, flat_edge % n_vertices

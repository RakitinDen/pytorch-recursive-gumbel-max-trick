import torch

class Arborescence:
    def __init__(self, arborescence, trace):
        self.arborescence = arborescence
        self.trace = trace


    def detach(self):
        arborescence = self.arborescence.detach()
        trace = self.trace

        return Arborescence(arborescence, trace)


def expand_mask(mask):
    out = ~mask
    return mask[:, None, :] * out[:, :, None]


def calc_trace_log_prob(logits, trace, length):
    mask = trace['mask']
    min_x = trace['min_x']
    min_y = trace['min_y']

    logits_cut = logits[:length, :length]

    masked_logits = mask + logits_cut

    log_prob = -logits_cut[min_x, min_y] - torch.logsumexp(-masked_logits, dim=(-1, -2))
    return log_prob.sum()


# used in E_reinforce for correct calculation of score
# masks diagonal elements
# masks elements that are not included in the graph (needed in case of different lengths in a batch)
def arb_mask_unused_values(values, root=0, lengths=None, **kwargs):
    batch_size = values.shape[0]
    dim = values.shape[1]
    mask = 1 - torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1)
    mask[:, :, 0] = 0

    if lengths is not None:
        mask_lengths = lengths[:, None] > torch.arange(dim)[None, :]
        mask_lengths = mask_lengths[:, None, :] * mask_lengths[:, :, None]
        mask *= mask_lengths

    return values * mask

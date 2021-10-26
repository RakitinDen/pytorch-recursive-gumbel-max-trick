import torch

class Arborescence:
    def __init__(self, arborescence, trace):
        self.arborescence = arborescence
        self.trace = trace


    def detach(self):
        arborescence = self.arborescence.detach()

        trace = []
        for trace_i in self.trace:
            trace.append(trace_i)
            trace[-1]['mask'] = trace[-1]['mask'].detach()
            trace[-1]['min_x'] = trace[-1]['min_x'].detach()
            trace[-1]['min_y'] = trace[-1]['min_y'].detach()

        return Arborescence(arborescence, trace)


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
def arb_mask_unused_values(values, root=0, **kwargs):
    batch_size = values.shape[0]
    dim = values.shape[1]
    mask = 1 - torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    mask[:, :, 0] = 0
    return values * mask

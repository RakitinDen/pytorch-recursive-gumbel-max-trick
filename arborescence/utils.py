import torch

class Arborescence:
    def __init__(self, arborescence, trace):
        self.arborescence = arborescence
        self.trace = trace

    def detach(self):
        arborescence = self.arborescence.detach()
        trace = self.trace

        return Arborescence(arborescence, trace)

    def to(self, device):
        arborescence = self.arborescence.to(device)
        trace = []
        for i in range(len(self.trace)):
            trace_i = {}
            trace_i['min_x'] = self.trace[i]['min_x'].to(device)
            trace_i['min_y'] = self.trace[i]['min_y'].to(device)
            trace_i['mask'] = self.trace[i]['mask'].to(device)
            trace.append(trace_i)

        return Arborescence(arborescence, trace)

def expand_mask(mask):
    out = ~mask
    return mask[:, None, :] * out[:, :, None]

def calc_trace_log_prob(logits, trace, length):
    '''
    Calculates log probability of the execution trace for one object
    Used in arb_log_prob (see functions.py)

    Input
    --------------------
    logits      : torch.Tensor | dim x dim |
                  Contains parameters (log(mean)) of the exponential distributions of edge weights
                  Unused weights (diagonal and edges going to the root) are set to be +inf

    trace       : dict
         mask   : torch.Tensor | n_steps x dim x dim |
         min_x  : torch.Tensor | n_steps |
         min_y  : torch.Tensor | n_steps | 

                  Contains the execution trace
                  Each of 'n_steps' minimum choices in the algorithm is encoded as the triple (mask, min_x, min_y)
                  Mask defines the elements among which the minimum is taken, (min_x, min_y) defines the argminimum

    length      : int
                  Defines dimension size of the graph (length <= dim)

    Output
    --------------------
    log_prob    : torch.Tensor | size = (1,) |
                  Contains log probabily of the execution trace
    '''
    mask = trace['mask']
    min_x = trace['min_x']
    min_y = trace['min_y']

    logits_cut = logits[:length, :length]
    masked_logits = mask + logits_cut

    log_prob = -logits_cut[min_x, min_y] - torch.logsumexp(-masked_logits, dim=(-1, -2))
    return log_prob.sum()

def arb_mask_unused_values(values, root=0, lengths=None, **kwargs):
    '''
    Used in E-REINFORCE for correct calculation of the exponential score function
    Masks diagonal elements
    Masks weights corresponding to the edges that go into the root
    Masks elements that are not included in the graph (needed in case of different lengths in a batch)

    Input
    --------------------
    values        : torch.Tensor | batch_size x dim x dim |
                    Contains a tensor of elements to be masked

    root          : int
                    Defines root of the arborescence for all the graphs in a batch

    lengths       : torch.Tensor | batch_size | 
                    Contains dimensions of graphs in the batch (lengths[i] <= dim)

    **kwargs      : Needed to support usage of different mask functions in the estimators' implementation

    Output
    --------------------
    masked_values : torch.Tensor | batch_size x dim x dim |
                    Contains a tensor of masked elements
    '''
    batch_size = values.shape[0]
    dim = values.shape[1]
    mask = 1 - torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1)
    mask[:, :, root] = 0

    if lengths is not None:
        mask_lengths = lengths[:, None] > torch.arange(dim)[None, :]
        mask_lengths = mask_lengths[:, None, :] * mask_lengths[:, :, None]
        mask *= mask_lengths

    masked_values = values * mask
    return masked_values

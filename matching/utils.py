import torch

class Matching:
    '''
    Defines the class wrapper for perfect matching

    matching     : torch.Tensor | batch_size x dim x dim |
                   Binary matrix that defines the perfect matching (permutation matrix)

    trace        : dict
    trace.mask   : torch.Tensor | batch_size x dim x dim x dim |
    trace.min_x  : torch.Tensor | batch_size x dim |
    trace.min_y  : torch.Tensor | batch_size x dim |
                   Contains the execution trace
                   mask[b, i] defines a set of elements among which the argminimum is chosen at step 'i' in the batch 'b'
                   (min_x, min_y)[b, i] is the argminimum chosen at step 'i' in the batch 'b'
    '''
    def __init__(self, matching, trace):
        self.matching = matching
        self.trace = trace

    def detach(self):
        matching = self.matching.detach()
        trace = self.trace
        return Matching(matching, trace)

    def to(self, device):
        matching = self.matching.to(device)
        trace = {}
        trace['mask'] = self.trace['mask'].to(device)
        trace['min_x'] = self.trace['min_x'].to(device)
        trace['min_y'] = self.trace['min_y'].to(device)
        return Matching(matching, trace)

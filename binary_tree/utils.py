import torch

class Node:
    def __init__(self, value=None):
        self.value = value
        self.left = None
        self.right = None
        self.depth = None

class BinaryTree():
    '''
    Defines the class wrapper for binary tree

    masks    : torch.Tensor | batch_size x dim x dim |
               Vector masks[b, i] contains zeros in positions among which element number i was minimal
               during the algotihm execution
               The other elements are equal to -inf

    trees    : list[Node_(1), ..., Node_(batch_size)]
               Contains representations of the trees using the class Node

    heights  : torch.Tensor | batch_size |
               Contains heights of the trees
    '''
    def __init__(self, masks, trees, heights):
        self.masks = masks
        self.trees = trees
        self.heights = heights

    def detach(self):
        masks = self.masks.detach()
        return BinaryTree(masks, self.trees, self.heights)

def build_tree(batch_idx, exp, left, right, level, masks, heights):
    '''
    Applies the binary algorithm for one object in the batch

    Input
    --------------------
    batch_idx  : int

    exp        : torch.Tensor | batch_size x dim |
                 Contains a batch of arrays (inputs of the algorithm)

    left       : int
    right      : int 
                 Define limits among which the minimum is taken at current recursion level

    level      : int
                 Current level of recursion

    masks      : torch.Tensor | batch_size x dim x dim |
                 The same thing as in the definition of the class BinaryTree; updated during the recursion

    heights    : torch.Tensor | batch_size |
                 The same thing as in the definition of the class BinaryTree; updated during the recursion

    Output
    --------------------
    node       : Node
                 The subtree at current recursion level
    '''
    if left >= right:
        return None

    heights[batch_idx] = max(heights[batch_idx], level + 1)
    min_index = exp[batch_idx][left:right].argmin() + left
    masks[batch_idx, min_index, left:right] = 0

    node = Node(min_index)

    node.left = build_tree(batch_idx, exp, left, min_index, level + 1, masks, heights)
    node.right = build_tree(batch_idx, exp, min_index + 1, right, level + 1, masks, heights)

    return node

def bin_tree_mask_unused_values(values, lengths=None, **kwargs):
    '''
    Used in E-REINFORCE for correct calculation of the exponential score function
    Masks elements that are not included in the array (needed in case of different lengths in a batch)

    Input
    --------------------
    values        : torch.Tensor | batch_size x dim |
                    Contains a tensor of elements to be masked

    lengths       : torch.Tensor | batch_size | 
                    Contains lengths of arrays in the batch (lengths[i] <= dim)

    **kwargs      : Needed to support usage of different mask functions in the estimators' implementation

    Output
    --------------------
    masked_values : torch.Tensor | batch_size x dim x dim |
                    Contains the masked tensor
    '''
    if lengths is None:
        return values

    batch_size = values.shape[0]
    dim = values.shape[1]
    mask = torch.arange(dim)[None, :] < lengths[:, None]
    masked_values = values * mask
    return masked_values

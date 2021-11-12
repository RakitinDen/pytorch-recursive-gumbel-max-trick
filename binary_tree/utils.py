import torch

class Node:
    def __init__(self, value=None):
        self.value = value
        self.left = None
        self.right = None
        self.depth = None

class BinaryTree():
    def __init__(self, masks, trees, heights):
        self.masks = masks
        self.trees = trees
        self.heights = heights

    def detach(self):
        masks = self.masks.detach()
        return BinaryTree(masks, self.trees, self.heights)

def build_tree(batch_idx, exp, left, right, level, masks, heights):
    if left >= right:
        return None

    heights[batch_idx] = max(heights[batch_idx], level)
    min_index = exp[batch_idx][left:right].argmin() + left
    masks[batch_idx, min_index, left:right] = 0

    node = Node(min_index)

    node.left = build_tree(batch_idx, exp, left, min_index, level + 1, masks, heights)
    node.right = build_tree(batch_idx, exp, min_index + 1, right, level + 1, masks, heights)

    return node

# used in E_reinforce for correct calculation of score
# masks vectors in a batch according to the corresponding length
def bin_tree_mask_unused_values(values, lengths=None, **kwargs):
    if lengths is None:
        return values

    batch_size = values.shape[0]
    dim = values.shape[1]
    mask = torch.arange(dim)[None, :] < lengths[:, None]

    return values * mask

# Binary tree

This folder contains the implementation of `f_struct`, `f_log_prob` and `f_cond` for binary tree. `f_struct` applies the divide-and-conquer algorithm from the paper that finds argminimum in the array and makes the recursive call to the subarrays at the left and at the right of it. `f_log_prob` calculates the log probability of the binary tree and `f_cond` returns a sample from the conditional distribution of exponential variables given the binary tree. 
Note: here the execution trace `T` and the binary tree `X` are in one-to-one correspondance.

<p align="middle">
	<img width="450" src="../figures/binary_tree.png">
</p>

Toy experiment consists in optimizing a vector of parameters of exponentials with respect to the loss function, that is equal to the expected negative height of the binary tree. The optimal configuration of the binary tree is the list (left-to-right or right-to-left).
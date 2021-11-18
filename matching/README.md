# Perfect matching

This folder contains the implementation of `f_struct`, `f_log_prob` and `f_cond` for perfect matching. `f_struct` applies the 'crossing' algorithm that recursively finds argminimum in the matrix and excludes the corresponding row and column. It returns the binary matrix corresponding to the matching along with the execution trace. `f_log_prob` calculates the log probability of the execution trace and `f_cond` returns a sample from the conditional distribution of exponential variables given the execution trace.

<p align="middle">
	<img width="300" src="../figures/matching.png">
</p>

Toy experiment consists in optimizing a vector of parameters of exponentials with respect to the loss function, that is equal to the expected negative number of matching elements on the main diagonal. The optimal configuration of the perfect matching corresponds to the identity matrix.
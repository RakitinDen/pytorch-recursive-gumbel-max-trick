# Spanning tree

This folder contains the implementation of `f_struct`, `f_log_prob` and `f_cond` for spanning tree. `f_struct` applies the Kruskal's algorithm that returns the minimal spanning tree of the weighted undirected graph along with its execution trace. `f_log_prob` calculates the log probability of the execution trace and `f_cond` returns a sample from the conditional distribution of exponential variables given the execution trace.

Toy experiment consists in optimizing a vector of parameters of exponentials with respect to the loss function, that is equal to the expected number of edges that are not connected to the vertex with index 0. The optimal configuration of the spanning tree is the "star" in which all the edges are connected to the vertex with index 0.

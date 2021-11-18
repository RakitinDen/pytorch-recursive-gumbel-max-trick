# Arborescence

This folder contains the implementation of `f_struct`, `f_log_prob` and `f_cond` for arborescence. `f_struct` applies the Chu-Liu Edmonds' algorithm that returns the minimal spanning arborescence of the weighted directed graph along with its execution trace. `f_log_prob` calculates the log probability of the execution trace and `f_cond` returns a sample from the conditional distribution of exponential variables given the execution trace.

Chu-Liu Edmonds' algorithm is implemented in C++. To compile and install it, go to the folder `edmonds` and run:
```
python setup.py install
```
Toy experiment consists in optimizing a vector of parameters of exponentials with respect to the loss function, that is equal to the expected negative maximal number of outgoing edges amond all the vertices in the arborescence. The optimal configuration of the arborescence is the graph in which all the edges come from the root.
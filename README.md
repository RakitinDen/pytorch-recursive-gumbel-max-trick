# Leveraging Recursive Gumbel-Max Trick for Approximate Inference in Combinatorial Spaces

This repository contains the PyTorch impementation of main algorithms and how-to-use examples from our paper [Leveraging Recursive Gumbel-Max Trick for Approximate Inference in Combinatorial Spaces](https://arxiv.org/abs/2110.15072).

The repository contains the code for 4 structured variables:
* Arborescence (Edmonds' algorithm)
* Binary tree (divide-and-conquer algorithm)
* Perfect matching ('crossing' algorithm)
* Spanning tree (Kruskal's algorithm).

For each of the variables, the implementation contains:
* Sampling structured variable and the execution trace
* Calculating log probability of the execution trace
* Sampling from the conditional distribution of the exponentials given the execution trace
* A toy optimization experiment.

In addition, the repository contains the implementation of different gradient estimators.

### Abstract

Structured latent variables allow incorporating meaningful prior knowledge into deep learning models. However, learning with such variables remains challenging because of their discrete nature. Nowadays, the standard learning approach is to define a latent variable as a perturbed algorithm output and to use a differentiable surrogate for training. In general, the surrogate puts additional constraints on the model and inevitably leads to biased gradients. To alleviate these shortcomings, we extend the Gumbel-Max trick to define distributions over structured domains. We avoid the differentiable surrogates by leveraging the score function estimators for optimization. In particular, we highlight a family of recursive algorithms with a common feature we call stochastic invariant. The feature allows us to construct reliable gradient estimates and control variates without additional constraints on the model. In our experiments, we consider various structured latent variable models and achieve results competitive with relaxation-based counterparts.

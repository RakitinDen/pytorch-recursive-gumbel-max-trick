import argparse
import os
import pickle

import numpy as np
import torch

from itertools import chain
from tqdm import tqdm

import sys
sys.path.append('../../')
from spanning_tree.critics import RELAXCritic
from spanning_tree.functions import span_struct, span_log_prob, span_cond
from spanning_tree.utils import span_mask_unused_values
from estimators import uniform_to_exp, E_reinforce, T_reinforce, relax

def _parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dim', type=int, default=7, help='Number of graph vertices')
    parser.add_argument('--estimator', choices=['E_reinforce', 'T_reinforce', 'relax'], default='T_reinforce')
    parser.add_argument('--num_mc', type=int, default=10, help='Number of MC samples at eval')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of independent stochastic gradient samples at training')
    parser.add_argument('--plus_samples', type=int, default=1, help='Number of samples used in inner averaging in REINFORCE+; 1 corresponds to REINFORCE')

    parser.add_argument('--hidden', type=int, default=32, help='Dimension of hidden layer for RELAX critic')
    parser.add_argument('--iters', type=int, default=50000, help='Number of iterations to train')
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")

    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--exp_path', type=str, default="./exp_log/", help='Where to save all stuff')
    parser.add_argument('--precision', choices=["float", "double"], default="float", help="Default tensor type")
    parser.add_argument('--seed', type=int, default=23, help='Random seed')

    return parser.parse_args(args)


def span_loss(edges):
    '''
    Loss is the number of edges that are not connected to the vertex with index 0
    The optimal configuration of the spanning tree is the "star"
    In this configuration all edges are connected to the vertex with index 0

    Input
    --------------------
    edges    : torch.Tensor | batch_size x (dim - 1) x 2
               Contains the sequence of edges in the spanning trees
               Order is defined by the execution of the Kruskal's algorithm

    Output
    --------------------
    loss        : torch.Tensor | batch_size |
    '''
    edges_not_from_zero = edges[:, 0, :] != 0
    edges_not_to_zero = edges[:, 1, :] != 0
    loss = (edges_not_from_zero & edges_not_to_zero).sum(dim=-1)
    return loss.float()

def run_toy_example(args=None):
    args = _parse_args(args)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = "cuda"
        if args.precision == "float":
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.cuda.DoubleTensor")
    else:
        device = "cpu"
        if args.precision == "float":
            torch.set_default_tensor_type("torch.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.DoubleTensor")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)

    # logits to optimize in the experiment
    logits = torch.zeros(size=(args.dim, args.dim), requires_grad=True)

    if args.estimator == 'E_reinforce':
        estimator = E_reinforce
        critic = None
        tunable = []
    elif args.estimator == 'T_reinforce':
        estimator = T_reinforce
        critic = None
        tunable = []
    elif args.estimator == 'relax':
        if args.plus_samples > 1:
            raise ValueError("RELAX does not support LOO baseline")

        estimator = relax
        critic = RELAXCritic(args.dim, args.hidden)
        tunable = critic.parameters()
    else:
        raise ValueError("only E_reinforce, T_reinforce or relax")

    if args.plus_samples > 1:
        args.estimator += '+'

    history = {
        "grad_std": [],
        "mean_objective": [],
    }

    optim = torch.optim.Adam(chain([logits], tunable), args.lr)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # start of the cycle
    for i in tqdm(range(args.iters)):
        if i % args.eval_every == 0:
            u_mc = torch.distributions.utils.clamp_probs(torch.rand(args.num_mc * args.plus_samples, args.dim, args.dim))
            v_mc = torch.distributions.utils.clamp_probs(torch.rand(args.num_mc * args.plus_samples, args.dim, args.dim))
            logits_mc = logits.unsqueeze(0).expand(args.num_mc * args.plus_samples, args.dim, args.dim)

            exp_mc = uniform_to_exp(logits_mc, u_mc, enable_grad=True)
            struct_var_mc = span_struct(exp_mc)
            loss_value_mc = span_loss(struct_var_mc)

            d_logits = estimator(
                loss_value=loss_value_mc, struct_var=struct_var_mc,
                logits=logits_mc,
                f_log_prob=span_log_prob, f_cond=span_cond,
                exp=exp_mc, uniform=v_mc, critic=critic,
                mask_unused_values=span_mask_unused_values,
                plus_samples=args.plus_samples
            ).detach().cpu().numpy()

            grad_std = np.std(d_logits, 0)
            mean_objective = loss_value_mc.mean().item()
            history["grad_std"].append(grad_std)
            history["mean_objective"].append(mean_objective)

        optim.zero_grad()

        u = torch.distributions.utils.clamp_probs(torch.rand(args.num_samples * args.plus_samples, args.dim, args.dim))
        v = torch.distributions.utils.clamp_probs(torch.rand(args.num_samples * args.plus_samples, args.dim, args.dim))
        logits_n = logits.unsqueeze(0).expand(args.num_samples * args.plus_samples, args.dim, args.dim)

        exp = uniform_to_exp(logits_n, u, enable_grad=True)
        struct_var = span_struct(exp)
        loss_value = span_loss(struct_var)

        d_logits = estimator(
            loss_value=loss_value, struct_var=struct_var,
            logits=logits_n,
            f_log_prob=span_log_prob, f_cond=span_cond,
            exp=exp, uniform=v, critic=critic,
            mask_unused_values=span_mask_unused_values,
            plus_samples=args.plus_samples
        )

        if tunable:
            (d_logits ** 2).sum(dim=(1, 2)).mean().backward()

        d_logits = d_logits.mean(0)
        logits.backward(d_logits)
        optim.step()
    # end of the cycle


    for key in history.keys():
        history[key] = np.array(history[key])

    with open(args.exp_path + "history_{0}.pkl".format(args.estimator), "wb") as file:
        pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)

    if tunable:
        torch.save(critic.state_dict(), args.exp_path + "critic_{0}.pt".format(args.estimator))

    torch.save(logits.detach().cpu(), args.exp_path + "logits_{0}.pt".format(args.estimator))


if __name__ == '__main__':
    run_toy_example()

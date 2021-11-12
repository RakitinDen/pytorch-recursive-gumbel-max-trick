import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('../../')

from binary_tree.critics import RELAXCritic
from binary_tree.functions import bin_tree_struct, bin_tree_log_prob, bin_tree_cond
from binary_tree.utils import bin_tree_mask_unused_values
from estimators import uniform_to_exp, E_reinforce, T_reinforce, relax

import os
from itertools import chain
import pickle
from tqdm import tqdm


def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=7, help='Dimension of vector to build a tree')
    parser.add_argument('--estimator', choices=['E_reinforce', 'T_reinforce', 'relax'], default='T_reinforce')
    parser.add_argument('--hidden', type=int, default=32, help='Dimension of hidden layer for RELAX critic')
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--iters', type=int, default=50000, help='Number of iterations to train')
    parser.add_argument('--precision', choices=["float", "double"], default="double", help="Default tensor type")
    parser.add_argument('--exp_path', type=str, default="./exp_log/", help='Where to save all stuff')
    parser.add_argument('--seed', type=int, default=23, help='Random seed')
    parser.add_argument('--num_mc', type=int, default=10, help='Number of MC samples at eval')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of MC samples at training')

    return parser.parse_args(args)


def loss(struct_var):
    return -struct_var.heights - 1

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

    # this random seed setup to control critics initialization
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)

    logits = torch.zeros(size=(args.d,), requires_grad=True)

    if args.estimator == 'E_reinforce':
        estimator = E_reinforce
        critic = None
        tunable = []
    elif args.estimator == 'T_reinforce':
        estimator = T_reinforce
        critic = None
        tunable = []
    elif args.estimator == "relax":
        estimator = relax
        critic = RELAXCritic(args.d, args.hidden)
        tunable = critic.parameters()
    else:
        raise ValueError("only E_reinforce, T_reinforce or relax")

    num_samples = args.num_samples
    num_mc_samples = 10
    history = {
        "grad_mean": [],
        "grad_std": [],
        "critic_loss": [],
        "mean_objective": [],
    }

    optim = torch.optim.Adam(chain([logits], tunable), args.lr)
    # this random seed setup to get first point of mean objective equal across estimators
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for i in tqdm(range(args.iters)):
        if i % 500 == 499:
            u_mc = torch.distributions.utils.clamp_probs(torch.rand(args.num_mc, args.d))
            v_mc = torch.distributions.utils.clamp_probs(torch.rand(args.num_mc, args.d))
            logits_mc = logits.unsqueeze(0).expand(args.num_mc, args.d)

            exp_mc = uniform_to_exp(logits_mc, u_mc, enable_grad=True)
            struct_var_mc = bin_tree_struct(exp_mc)
            loss_value_mc = loss(struct_var_mc)
            mean_objective = loss_value_mc.mean().item()

            d_logits = estimator(
                loss_value=loss_value_mc, struct_var=struct_var_mc,
                logits=logits_mc,
                f_log_prob=bin_tree_log_prob, f_cond=bin_tree_cond,
                exp=exp_mc, uniform=v_mc, critic=critic,
                mask_unused_values=bin_tree_mask_unused_values
            ).detach().cpu().numpy()

            if args.estimator != "relax":
                critic_loss = 0.0
            else:
                critic_loss = (d_logits ** 2).sum(axis=-1).mean()

            grad_mean = np.mean(d_logits, 0)
            grad_std = np.std(d_logits, 0)
            history["grad_mean"].append(grad_mean)
            history["grad_std"].append(grad_std)
            history["mean_objective"].append(mean_objective)
            history["critic_loss"].append(critic_loss)

        optim.zero_grad()

        u = torch.distributions.utils.clamp_probs(torch.rand(args.num_samples, args.d))
        v = torch.distributions.utils.clamp_probs(torch.rand(args.num_samples, args.d))
        logits_n = logits.unsqueeze(0).expand(args.num_samples, args.d)

        exp = uniform_to_exp(logits_n, u, enable_grad=True)
        struct_var = bin_tree_struct(exp)
        loss_value = loss(struct_var)

        d_logits = estimator(
            loss_value=loss_value, struct_var=struct_var,
            logits=logits_n,
            f_log_prob=bin_tree_log_prob, f_cond=bin_tree_cond,
            exp=exp, uniform=v, critic=critic
        ).mean(dim=0)

        if tunable:
            (d_logits ** 2).sum().backward()

        logits.backward(d_logits.squeeze())
        optim.step()

    for key in history.keys():
        history[key] = np.array(history[key])

    with open(args.exp_path + "history_{0}.pkl".format(args.estimator), "wb") as file:
        pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)

    if tunable:
        torch.save(critic.state_dict(), args.exp_path + "critic_{0}.pt".format(args.estimator))

    torch.save(logits.detach().cpu(), args.exp_path + "logits_{0}.pt".format(args.estimator))


if __name__ == '__main__':
    run_toy_example()

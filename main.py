# -*- coding: utf-8 -*-
import argparse
import os
import torch
from torch import multiprocessing as mp

from jaco_arm import JacoEnv
from model import ActorCritic
from optim import SharedRMSprop
from train import train
from test import test
from utils import Counter


parser = argparse.ArgumentParser(description='a3c-mujoco')
parser.add_argument('--width', type=int, default=64, help='RGB width')
parser.add_argument('--height', type=int, default=64, help='RGB height')
parser.add_argument('--random', action='store_true', help='Rendering random agent')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--num-processes', type=int, default=6, metavar='N',
                    help='Number of training async agents (does not include single validation agent)')
parser.add_argument('--T-max', type=int, default=5e7, metavar='STEPS', help='Number of training steps')
parser.add_argument('--t-max', type=int, default=100, metavar='STEPS', help='Max number of forward steps for A3C before update')
parser.add_argument('--max-episode-length', type=int, default=100, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--hidden-size', type=int, default=128, metavar='SIZE', help='Hidden size of LSTM cell')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--trace-decay', type=float, default=1, metavar='λ', help='Eligibility trace decay factor')
parser.add_argument('--lr', type=float, default=1e-4, metavar='η', help='Learning rate')
parser.add_argument('--lr-decay', action='store_true', help='Linearly decay learning rate to 0')
parser.add_argument('--rmsprop-decay', type=float, default=0.99, metavar='α', help='RMSprop decay factor')
parser.add_argument('--entropy-weight', type=float, default=0.01, metavar='β', help='Entropy regularisation weight')
parser.add_argument('--no-time-normalisation', action='store_true', help='Do not normalise loss by number of time steps')
parser.add_argument('--max-gradient-norm', type=float, default=40, metavar='VALUE',
                    help='Max value of gradient L2 norm for gradient clipping')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=5e4, metavar='STEPS',
                    help='Number of training steps between evaluations (roughly)')
parser.add_argument('--evaluation-episodes', type=int, default=30, metavar='N',
                    help='Number of evaluation episodes to average over')
parser.add_argument('--render', action='store_true', help='Render evaluation agent')
parser.add_argument('--overwrite', action='store_true', help='Overwrite results')
parser.add_argument('--frame_skip', type=int, default=100, help="Frame skipping in environment. Repeats last agent action.")
parser.add_argument('--rewarding_distance', type=float, default=0.1, help='Distance from target at which reward is provided.')
parser.add_argument('--control_magnitude', type=float, default=0.8, help='Fraction of actuator range used as control inputs.')
parser.add_argument('--reward_continuous', action='store_true', help='if True, provides rewards at every timestep')

if __name__ == '__main__':
    # BLAS setup
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # Setup
    args = parser.parse_args()
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))
    args.env = 'jaco'
    args.non_rgb_state_size = 18  # 9 joints qpos and qvel TODO: don't hardcode!

    mp.set_start_method('spawn')
    torch.manual_seed(args.seed)
    T = Counter()  # Global shared counter

    # Results dir
    if not os.path.exists('results'):
        os.makedirs('results')
    elif not args.overwrite:
        raise OSError('results dir exists and overwrite flag not passed')

    # Create shared network
    env = JacoEnv(args.width,
                  args.height,
                  args.frame_skip,
                  args.rewarding_distance,
                  args.control_magnitude,
                  args.reward_continuous)

    shared_model = ActorCritic(None, args.non_rgb_state_size, None,
                               args.hidden_size)
    shared_model.share_memory()
    if args.model and os.path.isfile(args.model):
        # Load pretrained weights
        shared_model.load_state_dict(torch.load(args.model))
    # Create optimiser for shared network parameters with shared statistics
    optimiser = SharedRMSprop(
        shared_model.parameters(), lr=args.lr, alpha=args.rmsprop_decay)
    optimiser.share_memory()

    # Start validation agent
    processes = []
    p = mp.Process(target=test, args=(0, args, T, shared_model))
    p.start()
    processes.append(p)

    if not args.evaluate:
        # Start training agents
        for rank in range(1, args.num_processes + 1):
            p = mp.Process(
                target=train, args=(rank, args, T, shared_model, optimiser))
            p.start()
            processes.append(p)

    # Clean up
    for p in processes:
        p.join()

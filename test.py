# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
import torch
from torch.autograd import Variable

from jaco_arm import JacoEnv
from model import ActorCritic
from utils import state_to_tensor, plot_line

import matplotlib.pyplot as plt


def test(rank, args, T, shared_model):
    torch.manual_seed(args.seed + rank)

    env = JacoEnv(args.width,
                  args.height,
                  args.frame_skip,
                  args.rewarding_distance,
                  args.control_magnitude,
                  args.reward_continuous)
    env.seed(args.seed + rank)
    if args.render:
        (_, _, obs_rgb_view2) = env.reset()
        plt.ion()
        f, ax = plt.subplots()
        im = ax.imshow(obs_rgb_view2)

    model = ActorCritic(None, args.non_rgb_state_size, None, args.hidden_size)
    model.eval()
    can_test = True  # Test flag
    t_start = 1  # Test step counter to check against global counter
    rewards, steps = [], []  # Rewards and steps for plotting
    n_digits = str(
        len(str(args.T_max)))  # Max num. of digits for logging steps
    done = True  # Start new episode

    while T.value() <= args.T_max:
        if can_test:
            t_start = T.value()  # Reset counter

            # Evaluate over several episodes and average results
            avg_rewards, avg_episode_lengths = [], []
            for _ in range(args.evaluation_episodes):
                while True:
                    # Reset or pass on hidden state
                    if done:
                        # Sync with shared model every episode
                        model.load_state_dict(shared_model.state_dict())
                        hx = Variable(
                            torch.zeros(1, args.hidden_size), volatile=True)
                        cx = Variable(
                            torch.zeros(1, args.hidden_size), volatile=True)
                        # Reset environment and done flag
                        state = state_to_tensor(env.reset())
                        action, reward, done, episode_length = (0, 0, 0, 0, 0,
                                                                0), 0, False, 0
                        reward_sum = 0

                    # Calculate policy
                    policy, _, (hx, cx) = model(
                        Variable(
                            state[0], volatile=True),
                        Variable(
                            state[1], volatile=True),
                        (hx.detach(),
                         cx.detach()))  # Break graph for memory efficiency

                    # Choose action greedily
                    action = [p.max(1)[1].data[0, 0] for p in policy]

                    # Step
                    state, reward, done = env.step(action)
                    obs_rgb_view1 = state[1]
                    obs_rgb_view2 = state[2]
                    state = state_to_tensor(state)
                    reward_sum += reward
                    done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
                    episode_length += 1  # Increase episode counter

                    # Optionally render validation states
                    if args.render:
                        # rendering the first camera view
                        im.set_data(obs_rgb_view1)
                        plt.draw()
                        plt.pause(0.05)

                        # rendering mujoco simulation
                        # viewer = mujoco_py.MjViewer(env.sim)
                        # viewer.render()

                    # Log and reset statistics at the end of every episode
                    if done:
                        avg_rewards.append(reward_sum)
                        avg_episode_lengths.append(episode_length)
                        break

            print(('[{}] Step: {:<' + n_digits +
                   '} Avg. Reward: {:<8} Avg. Episode Length: {:<8}').format(
                       datetime.utcnow().strftime(
                           '%Y-%m-%d %H:%M:%S,%f')[:-3], t_start,
                       sum(avg_rewards) / args.evaluation_episodes,
                       sum(avg_episode_lengths) / args.evaluation_episodes))

            rewards.append(avg_rewards)  # Keep all evaluations
            steps.append(t_start)
            plot_line(steps, rewards)  # Plot rewards
            torch.save(model.state_dict(),
                       os.path.join('results', str(t_start) +
                                    '_model.pth'))  # Checkpoint model params
            can_test = False  # Finish testing
            if args.evaluate:
                return
        else:
            if T.value() - t_start >= args.evaluation_interval:
                can_test = True

        time.sleep(0.001)  # Check if available to test every millisecond

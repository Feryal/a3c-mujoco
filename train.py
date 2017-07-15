# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable

from jaco_arm import JacoEnv
from model import ActorCritic
from utils import state_to_tensor


# Transfers gradients from thread-specific model to shared model
def _transfer_grads_to_shared_model(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


# Adjusts learning rate
def _adjust_learning_rate(optimiser, lr):
    for param_group in optimiser.param_groups:
        param_group['lr'] = lr


# Updates networks
def _update_networks(args, T, model, shared_model, loss, optimiser):
    # Zero shared and local grads
    optimiser.zero_grad()
    # Calculate gradients (not losses defined as negatives of normal update rules for gradient descent)
    loss.backward()
    # Gradient L2 norm clipping
    nn.utils.clip_grad_norm(model.parameters(), args.max_gradient_norm, 2)

    # Transfer gradients to shared model and update
    _transfer_grads_to_shared_model(model, shared_model)
    optimiser.step()
    if args.lr_decay:
        # Linearly decay learning rate
        _adjust_learning_rate(optimiser,
                              max(args.lr * (args.T_max - T.value()) /
                                  args.T_max, 1e-32))


# Trains model
def _train(args, T, model, shared_model, optimiser, policies, Vs, actions,
           rewards, R):
    policy_loss, value_loss = 0, 0
    A_GAE = torch.zeros(1, 1)  # Generalised advantage estimator Ψ
    # Calculate n-step returns in forward view, stepping backwards from the last state
    t = len(rewards)
    for i in reversed(range(t)):
        # R ← r_i + γR
        R = rewards[i] + args.discount * R
        # Advantage A ← R - V(s_i; θ)
        A = R - Vs[i]
        # dθ ← dθ - ∂A^2/∂θ
        value_loss += 0.5 * A**2  # Least squares error

        # TD residual δ = r + γV(s_i+1; θ) - V(s_i; θ)
        td_error = rewards[i] + args.discount * Vs[i + 1].data - Vs[i].data
        # Generalised advantage estimator Ψ (roughly of form ∑(γλ)^t∙δ)
        A_GAE = A_GAE * args.discount * args.trace_decay + td_error
        # dθ ← dθ - ∇θ∙log(π(a_i|s_i; θ))∙Ψ - β∙∇θH(π(s_i; θ))
        for j, p in enumerate(policies[i]):
            policy_loss -= p.gather(
                1, actions[i][j].detach().unsqueeze(0)).log() * Variable(A_GAE)
            # policy_loss -= args.entropy_weight * -(p.log() * p).sum(1).mean(0)
            policy_loss -= args.entropy_weight * -(p.log() * p).sum(1)

    # Optionally normalise loss by number of time steps
    if not args.no_time_normalisation:
        policy_loss /= t
        value_loss /= t
    # Update networks
    _update_networks(args, T, model, shared_model, policy_loss + value_loss,
                     optimiser)


# Acts and trains model
def train(rank, args, T, shared_model, optimiser):
    torch.manual_seed(args.seed + rank)

    env = JacoEnv(args.width,
                  args.height,
                  args.frame_skip,
                  args.rewarding_distance,
                  args.control_magnitude,
                  args.reward_continuous)
    env.seed(args.seed + rank)

    # TODO: pass in the observation and action space
    model = ActorCritic(None, args.non_rgb_state_size, None, args.hidden_size)
    model.train()

    t = 1  # Thread step counter
    done = True  # Start new episode

    while T.value() <= args.T_max:
        # Sync with shared model at least every t_max steps
        model.load_state_dict(shared_model.state_dict())
        # Get starting timestep
        t_start = t

        # Reset or pass on hidden state
        if done:
            hx = Variable(torch.zeros(1, args.hidden_size))
            cx = Variable(torch.zeros(1, args.hidden_size))
            # Reset environment and done flag
            state = state_to_tensor(env.reset())
            action, reward, done, episode_length = (0, 0, 0, 0, 0,
                                                    0), 0, False, 0

        else:
            # Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
            hx = hx.detach()
            cx = cx.detach()

        # Lists of outputs for training
        policies, Vs, actions, rewards = [], [], [], []

        while not done and t - t_start < args.t_max:
            # Calculate policy and value
            policy, V, (hx, cx) = model(
                Variable(state[0]), Variable(state[1]), (hx, cx))

            # Sample action
            action = [
                p.multinomial().data[0, 0] for p in policy
            ]  # Graph broken as loss for stochastic action calculated manually

            # Step
            state, reward, done = env.step(action)
            state = state_to_tensor(state)
            done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
            episode_length += 1  # Increase episode counter

            # Save outputs for online training
            [
                arr.append(el)
                for arr, el in zip((policies, Vs, actions, rewards), (
                    policy, V, Variable(torch.LongTensor(action)), reward))
            ]

            # Increment counters
            t += 1
            T.increment()

        # Break graph for last values calculated (used for targets, not directly as model outputs)
        if done:
            # R = 0 for terminal s
            R = Variable(torch.zeros(1, 1))

        else:
            # R = V(s_i; θ) for non-terminal s
            _, R, _ = model(Variable(state[0]), Variable(state[1]), (hx, cx))
            R = R.detach()
        Vs.append(R)

        # Train the network
        _train(args, T, model, shared_model, optimiser, policies, Vs, actions,
               rewards, R)

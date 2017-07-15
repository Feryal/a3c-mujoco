# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import init


class ActorCritic(nn.Module):
    def __init__(self, observation_space, non_rgb_rgb_state_size, action_space,
                 hidden_size):
        super(ActorCritic, self).__init__()
        self.rgb_state_size = (6, 128, 128)
        self.action_size = 5
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax()

        # the archtecture is adapted from Sim2Real (Rusu et. al., 2016)
        self.conv1 = nn.Conv2d(
            self.rgb_state_size[0], 16, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.fc1 = nn.Linear(1152 + non_rgb_rgb_state_size, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.fc_actor1 = nn.Linear(hidden_size, self.action_size)
        self.fc_actor2 = nn.Linear(hidden_size, self.action_size)
        self.fc_actor3 = nn.Linear(hidden_size, self.action_size)
        self.fc_actor4 = nn.Linear(hidden_size, self.action_size)
        self.fc_actor5 = nn.Linear(hidden_size, self.action_size)
        self.fc_actor6 = nn.Linear(hidden_size, self.action_size)
        self.fc_critic = nn.Linear(hidden_size, 1)

        # Orthogonal weight initialisation
        for name, p in self.named_parameters():
            if 'weight' in name:
                init.orthogonal(p)
            elif 'bias' in name:
                init.constant(p, 0)

    def forward(self, non_rgb_state, rgb_state, h):
        x = self.relu(self.conv1(rgb_state))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(torch.cat((x, non_rgb_state), 1))
        h = self.lstm(x, h)  # h is (hidden state, cell state)
        x = h[0]
        policy1 = self.softmax(self.fc_actor1(x)).clamp(
            max=1 - 1e-20)  # Prevent 1s and hence NaNs
        policy2 = self.softmax(self.fc_actor2(x)).clamp(max=1 - 1e-20)
        policy3 = self.softmax(self.fc_actor3(x)).clamp(max=1 - 1e-20)
        policy4 = self.softmax(self.fc_actor4(x)).clamp(max=1 - 1e-20)
        policy5 = self.softmax(self.fc_actor5(x)).clamp(max=1 - 1e-20)
        policy6 = self.softmax(self.fc_actor6(x)).clamp(max=1 - 1e-20)
        V = self.fc_critic(x)
        return (policy1, policy2, policy3, policy4, policy5, policy6), V, h

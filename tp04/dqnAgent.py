import argparse
import sys
import matplotlib
import gym
import gridworld
import torch
from utils import *
from base import BaseAgent
from torch.utils.tensorboard import SummaryWriter
import copy
from memory import Memory
import numpy as np
import datetime

# matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")


class DQNAgent(BaseAgent):
    def __init__(self, env, opt, writer=None):
        self.name = "DQN"
        self.opt = opt
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)

        self.gamma = opt.gamma

        self.eps = opt.eps_0
        self.eps_0 = opt.eps_0
        self.eta = opt.eta

        self.test = False
        self.insize = self.featureExtractor.outSize
        self.outsize = self.action_space.n
        self.layers = opt.layers

        self.q = NN(self.insize, self.outsize, self.layers)
        self.q_target = copy.deepcopy(self.q)

        self.LR = opt.lr

        self.optim = torch.optim.SGD(params=self.q.parameters(), lr=self.LR)
        self.train_loss = torch.nn.SmoothL1Loss(
            reduction="none"
        )

        self.buffer_size = opt["buffer_size"]
        self.prioritized = opt.prioritized
        self.memory = Memory(self.buffer_size, prior=self.prioritized)

        self.batch_size = opt.batch_size
        self.global_id = 0

    def act(self, observation, reward, done):
        self.eps = self.eps_0 / (1 + self.eta * self.episode)
        observation = self.get_features(observation)
        assert isinstance(observation, torch.Tensor)
        action_values = self.q(observation).squeeze()

        if np.random.rand() < self.eps and not self.test:  # explore
            return self.action_space.sample()
        else:
            assert isinstance(observation, torch.Tensor)
            return torch.argmax(action_values.detach()).item()

    def store(self, s, a, s_p, r, done):
        # Store feature-extracted states
        self.memory.store((s, a, s_p, r, done))

    def learn(self):
        if self.prioritized:
            idx, w, batch = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)

        # prepare batches
        old_ob_t = torch.Tensor([transition[0] for transition in batch])
        action_t = torch.Tensor([transition[1] for transition in batch]).long()
        ob_t = torch.Tensor([transition[2] for transition in batch])
        reward_t = torch.Tensor([transition[3] for transition in batch]).float()
        done_t = torch.logical_not(
            torch.Tensor(([transition[4] for transition in batch]))
        ).int()
        # 1 if not done, 0 if done.

        with torch.no_grad():
            q_values_t = self.q_target(ob_t)
            max_q, _ = torch.max(q_values_t, dim=1)
            target = reward_t + self.gamma * max_q * done_t
        est = self.q.forward(old_ob_t).gather(1, action_t.unsqueeze(-1)).squeeze(1)
        loss = self.train_loss(est, target)  # no reduction
        if self.prioritized:
            loss *= torch.Tensor(w).squeeze(-1)  # IS debiasing.
            loss /= w.max()  # weight normalization
        loss = loss.mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()

        # Update memory for Prioritized Experience Replay
        if self.prioritized:
            with torch.no_grad():
                errors = torch.abs(est - target)
                self.memory.update(idx, errors)

        self.writer.add_scalar("TargetLoss/train", loss.item(), self.global_id)


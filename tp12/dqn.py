from copy import deepcopy

import numpy as np
import torch
from torch import autograd
from torch.optim import SGD, Adam
import torch.nn as nn

from base import BaseAgent, ApproxFunction
from buffer import Memory
from policies import GreedyPolicy

class DoubleDQNAgent(BaseAgent):
    def __init__(self, env, opt):
        super().__init__(env, opt)
        self.name = "DoubleDQN"
        self.qfunction = ApproxFunction(
            self.featureExtractor.outSize*2,
            self.action_space.n,
            opt.hidden_dim
        )

        if opt.target:
            self.targetq = deepcopy(self.qfunction)
        if opt.replay:
            self.memory = Memory(mem_size=opt.mem_size, items=5)

        self.mse_loss = nn.MSELoss()
        self.optim = Adam(params=self.qfunction.parameters(), lr=opt.lr)
        self.env_policy = GreedyPolicy(self.action_space, opt.epsilon, (1 - opt.eta))

        self.last_ob = None
        self.action = None
        self.t = 0

   
    def learn(self):
        print("Learning")
        last_ob, actions, rewards, ob, mask = self.memory.sample(self.opt.batch_size)
        action_values = self.targetq(ob).detach()
        argmax_actions= torch.argmax(self.qfunction(ob).detach(), 1)
        max_action_values = torch.gather(action_values, 1, argmax_actions.unsqueeze(1)).squeeze()
        target = torch.where(mask, rewards, rewards + self.opt.gamma * max_action_values)

        loss = self.mse_loss(
            torch.gather(self.qfunction(last_ob), 1, actions.unsqueeze(1)).squeeze(),
            target
        )

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def act(self, ob, reward, done, truncated, goal):
        ob = torch.cat([self.get_features(ob), self.get_features(goal)], 0)
        action_values = self.qfunction(ob).detach().squeeze()

        if self.action is not None:
            self.memory.add(
                (self.last_ob, 
                torch.tensor(self.action), 
                torch.tensor(reward), 
                ob, 
                torch.tensor(done and (not truncated)))
            )
            if (not self.test) and (self.memory.size > self.opt.learning_offset):
                self.learn()

        self.last_ob = ob
        if (self.opt.target) and (self.t % self.opt.C == 0):
            self.targetq = deepcopy(self.qfunction)

        if done:
            self.action = None   
        elif self.test:
            self.action = int(np.argmax(action_values)) #self.env_policy.chose(0, action_values)
        else:
            self.action = self.env_policy.chose(action_values)
        self.t += 1
        return self.action

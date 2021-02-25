from copy import deepcopy
import pickle

import numpy as np
import torch
from torch import autograd
from torch.distributions.categorical import Categorical
from torch.optim import SGD, Adam
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from base import BaseImitation, ApproxFunction
from buffer import Memory
from policies import GreedyPolicy

class BehaviorClone(BaseImitation):
    def __init__(self, env, opt, writer=None):
        super().__init__(env, opt)
        self.name = "BehaviorClone"
        self.writer = writer

        self.actor = ApproxFunction(
            self.featureExtractor.outSize,
            self.action_space.n,
            opt.hidden_dim,
            hidden_activation = torch.tanh,
            output_activation = None
        )

        data = TensorDataset(self.expert_states, self.toIndexAction(self.expert_actions))
        self.loader = DataLoader(data, batch_size=opt.batch_size)
        self.loss = nn.CrossEntropyLoss()
        self.optim = Adam(params=self.actor.parameters(), lr=opt.lr)

        self.t = 0
        self.epi = 0
        

    def learn(self):
        print("Learning ! ")
        for states, expert_actions in self.loader:
            actor_actions = self.actor(states)
            loss = self.loss(actor_actions, expert_actions)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.t += 1

            self.writer.add_scalar("Loss", loss.item(), self.t)
            accuracy = (actor_actions.argmax(-1) == expert_actions).to(torch.float).mean()
            self.writer.add_scalar("Accuracy", accuracy.item(), self.t)
    
    def act(self, ob, reward, done, truncated):
        if done:
            if self.epi % self.opt.C == 0:
                for i in range(self.opt.C):
                    self.learn()
            self.epi += 1
        
        ob = self.get_features(ob)
        action_values = self.actor(ob).squeeze()
        # if self.test:
        #     return int(np.argmax(action_values.detach()))
        dist = Categorical(torch.softmax(action_values,0))
        self.action = dist.sample()
        return int(self.action)
        
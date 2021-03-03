from copy import deepcopy

import numpy as np
import torch
from torch import autograd
from torch.distributions.normal import Normal
from torch.optim import SGD, Adam
import torch.nn as nn

from base import BaseAgent, ApproxFunction
from buffer import Memory


class DDPG(BaseAgent):
    def __init__(self, env, opt):
        super().__init__(env, opt)
        self.name = "DDPG"
        self.critic = ApproxFunction(
            self.featureExtractor.outSize+self.action_space.shape[0],
            1,
            opt.hidden_dim
        )

        self.actor =  ApproxFunction(
            self.featureExtractor.outSize,
            self.action_space.shape[0],
            opt.hidden_dim,
            torch.tanh
        )

        self.targetcritic = deepcopy(self.critic)
        self.targetactor = deepcopy(self.actor)
        self.memory = Memory(mem_size=opt.mem_size, items=5)

        self.huber_loss = nn.SmoothL1Loss()
        self.optim_critic = Adam(params=self.critic.parameters(), lr=opt.critic_lr)
        self.optim_actor = Adam(params=self.actor.parameters(), lr=opt.actor_lr)

        self.last_ob = None
        self.action = None
        self.sigma = torch.diag(torch.tensor(self.opt.sigma, dtype=torch.float))
        self.t = 0
        self.optim_t = 0

    def update_target(self, target, learner):
        with torch.no_grad():
            for ptarget, plearner in zip(target.parameters(), learner.parameters()):
                new_val = self.opt.rho * ptarget.data + (1 - self.opt.rho) * plearner.data
                ptarget.copy_(new_val)

    def learn(self):
        last_ob, actions, rewards, ob, mask = self.memory.sample(self.opt.batch_size)
        
        next_actions = self.targetactor(ob).detach()
        next_action_values = self.targetcritic(torch.cat([ob, next_actions], 1)).detach().squeeze()
        target = torch.where(mask, rewards, rewards + self.opt.gamma * next_action_values)

        loss = self.huber_loss(
            self.critic(torch.cat([last_ob, actions], 1)).squeeze(),
            target
        )

        self.optim_critic.zero_grad()
        loss.backward()
        self.optim_critic.step()

        self.writer.add_scalar(
            "Critic/Loss", loss.item(), self.optim_t
            )
        
        actions = self.actor(last_ob) # TODO: directly keep the grad in the sampled actions
        action_values = self.critic(torch.cat([last_ob, actions], 1))
        loss = - torch.mean(action_values)
        self.optim_actor.zero_grad()
        loss.backward()
        self.optim_actor.step()

        self.update_target(self.targetcritic, self.critic)
        self.update_target(self.targetactor, self.actor)

        self.writer.add_scalar(
            "Actor/Loss", loss.item(), self.optim_t
        )

        self.optim_t += 1

    def act(self, ob, reward, done, truncated):
        ob = self.get_features(ob)
        action = self.actor(ob)
        # Scale to action space
        action = action * self.action_space.high[0]

        if self.test:
            return action.detach().numpy()

        if self.action is not None:
            self.memory.add(
                (self.last_ob, 
                self.action[0], 
                torch.tensor(reward, dtype=torch.float32), 
                ob, 
                torch.tensor(done and (not truncated)))
            )

        if self.memory.size > self.opt.batch_size:
            self.learn()

        self.last_ob = ob
        self.t += 1

        if done:
            self.action = None
            return self.action

        self.sigma *= self.opt.decrease_rate
        dist = Normal(action.squeeze(), self.sigma)
        self.action = torch.clip(dist.sample(),
                float(self.action_space.low[0]),
                float(self.action_space.high[0]))
        
        # Reshape to action space shape
        action = self.action.numpy().reshape(self.action_space.shape)
        return action

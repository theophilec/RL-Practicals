from copy import deepcopy

import numpy as np
import torch
from torch import autograd
from torch.distributions.categorical import Categorical
from torch.optim import SGD, Adam
import torch.nn as nn

from base import BaseAgent, ApproxFunction
from buffer import Memory

class ActorCritic(BaseAgent):
    def __init__(self, env, opt, writer=None):
        super().__init__(env, opt)
        self.name = "AC"
        self.writer = writer
        self.critic = ApproxFunction(
            self.featureExtractor.outSize,
            1,
            opt.hidden_dim
        )
        self.targetcritic = deepcopy(self.critic)
        self.actor =  ApproxFunction(
            self.featureExtractor.outSize,
            self.action_space.n,
            opt.hidden_dim,
            nn.Softmax(dim=-1)
        )
        self.huber_loss = nn.SmoothL1Loss()
        self.optim_critic = Adam(params=self.critic.parameters(), lr=opt.critic_lr)
        self.optim_actor = Adam(params=self.actor.parameters(), lr=opt.actor_lr)
        self.memory = Memory(mem_size=opt.mem_size, items=6, replace=False)

        self.last_ob = None
        self.action = None
        self.logprob = None
        self.traj_start = 0
        self.t = 0

    def learn(self):
        print("Learning ! ")
        last_ob, actions, logprobs, returns, ob, mask = self.memory.sample(self.memory.size)
        if self.opt.returns == "td":
            state_values = self.targetcritic(ob).detach().squeeze()
            target = torch.where(mask, returns, returns + self.opt.gamma * state_values)
        # elif self.opt.returns == "gae":
        #     state_values = self.targetcritic(ob).detach().squeeze()
        #     target = torch.where(mask, returns, returns + self.opt.gamma * state_values)
        elif self.opt.returns == "rollout" or (self.opt.returns == "gae"):
            target = returns
        last_state_values = self.critic(last_ob).squeeze()

        loss = self.huber_loss(
            last_state_values,
            target
        )

        self.optim_critic.zero_grad()
        loss.backward()
        self.writer.add_scalar(
            "Loss/Critic", loss.item(), self.t
        )
        self.writer.add_scalar(
            "Norm/Critic", torch.norm(self.critic.fcinput.weight.data), self.t
        )
        self.optim_critic.step()

        advantages = target - self.targetcritic(last_ob).squeeze()
        loss = - torch.mean(logprobs * advantages.detach())
        
        self.optim_actor.zero_grad()
        loss.backward()
        self.writer.add_scalar(
            "Loss/Actor", loss.item(), self.t
        )
        self.writer.add_scalar(
            "Norm/Actor", torch.norm(self.actor.fcinput.weight.data), self.t
        )
        self.optim_actor.step()

    def compute_returns(self, pnt):
        returns = 0
        for t in range(self.memory.size - 1, pnt - 1, -1):
            reward = float(self.memory.memory[3][t])
            self.memory.memory[3][t] += returns
            returns = self.opt.gamma * (reward + returns)
    
    def compute_gae(self, pnt):
        returns = 0
        for t in range(self.memory.size - 1, pnt - 1, -1):
            reward = float(self.memory.memory[3][t])
            next_ob = self.memory.memory[4][t]
            last_ob = self.memory.memory[0][t]
            done = bool(self.memory.memory[-1][t])
            td_target = reward + (1-done) * self.opt.gamma * self.targetcritic(next_ob).detach().squeeze()
            discount_returns = self.opt.lambd * self.opt.gamma * returns
            self.memory.memory[3][t] = td_target + discount_returns
            returns = discount_returns + td_target - self.targetcritic(last_ob).detach().squeeze() #td error

    def act(self, ob, reward, done, truncated):
        ob = self.get_features(ob)
        action_values = self.actor(ob).squeeze()
        if self.test:
            return int(np.argmax(action_values.detach()))
        
        self.t += 1
        if self.action is not None:
            self.memory.add(
                (self.last_ob,
                self.action,
                self.logprob,
                torch.tensor(reward), 
                ob, 
                torch.tensor(done and (not truncated)))
            )

        self.last_ob = ob
        if (self.t % self.opt.C == 0):
            self.targetcritic = deepcopy(self.critic)

        if done:
            if self.opt.returns == "rollout":
                self.compute_returns(self.traj_start)
                self.traj_start = self.memory.size
            if self.opt.returns == "gae":
                self.compute_gae(self.traj_start)
                self.traj_start = self.memory.size
            
            self.action = None
            if self.memory.size > self.opt.learning_minimum:
                self.learn()
                self.memory.empty() # on policy learning
                self.traj_start = 0
            return self.action
        else:
            dist = Categorical(action_values)
            self.action = dist.sample()
            self.logprob = dist.log_prob(self.action)
            return int(self.action)
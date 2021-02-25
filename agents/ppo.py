from copy import deepcopy

import numpy as np
import torch
from torch import autograd
from torch.distributions.categorical import Categorical
from torch.optim import SGD, Adam
import torch.nn as nn
from torch.utils.data import DataLoader

from base import BaseAgent, ApproxFunction
from buffer import Memory
from policies import GreedyPolicy

class PPO(BaseAgent):
    def __init__(self, env, opt, writer=None):
        super().__init__(env, opt)
        self.name = "PPO"
        if opt.loss not in ["kl", "clip"]:
            raise ValueError(f"'loss' parameter accepts two values: 'kl' or 'clip'. '{opt.loss}' is not valid.")
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
        self.kldiv = nn.KLDivLoss(reduction="batchmean")
        self.optim_critic = Adam(params=self.critic.parameters(), lr=opt.critic_lr)
        self.optim_actor = Adam(params=self.actor.parameters(), lr=opt.actor_lr)
        self.memory = Memory(mem_size=opt.mem_size, items=6, replace=False)
        self.beta = self.opt.beta # init beta

        self.last_ob = None
        self.action = None
        self.logprob = None
        self.traj_start = 0
        self.t = 0
        self.optim_step = 0

    def learn(self):
        print("Learning ! ")
        data = self.memory.as_dataset()
        loader = DataLoader(data, batch_size=self.opt.batch_size)
        
        for i in range(self.opt.num_epochs):
            for last_ob, actions, old_logprobs, returns, ob, mask in loader:
                self.optim_step += 1
                if self.opt.returns == "td":
                    state_values = self.targetcritic(ob).detach().squeeze()
                    target = torch.where(mask, returns, returns + self.opt.gamma * state_values)
                elif self.opt.returns == "rollout":
                    target = returns
                last_state_values = self.critic(last_ob).squeeze()

                loss = self.huber_loss(
                    last_state_values,
                    target
                )

                self.optim_critic.zero_grad()
                loss.backward()
                self.writer.add_scalar(
                    "Loss/Critic", loss.item(), self.optim_step
                )
                self.writer.add_scalar(
                    "Norm/Critic", torch.norm(self.critic.fcinput.weight.data), self.optim_step
                )
                self.optim_critic.step()

                advantages = target - self.targetcritic(last_ob).squeeze()
                advantages = advantages.detach()
                new_probs = torch.gather(self.actor(last_ob), 1, actions.unsqueeze(1)).squeeze()
                ratios = new_probs /  torch.exp(old_logprobs).detach()

                if self.opt.loss == "kl":
                    loss_obj = - torch.mean(ratios * advantages)
                    loss_kl = self.kldiv(old_logprobs.detach().squeeze(), new_probs)
                    loss = loss_obj + self.beta * loss_kl
                elif self.opt.loss == "clip":
                    clipped_ratios = torch.clip(ratios, 1 - self.opt.epsilon, 1 + self.opt.epsilon)
                    loss_i = torch.minimum(ratios * advantages,
                                           clipped_ratios * advantages)
                    loss = torch.mean(loss_i)
                
                self.optim_actor.zero_grad()
                loss.backward()
                self.writer.add_scalar(
                    "Loss/Actor", loss.item(), self.optim_step
                )
                self.writer.add_scalar(
                    "Norm/Actor", torch.norm(self.actor.fcinput.weight.data), self.optim_step
                )
                self.optim_actor.step()
        
        last_ob, actions, old_logprobs, _, _, _ = self.memory.sample(self.memory.size)
        new_probs = torch.gather(self.actor(last_ob), 1, actions.unsqueeze(1)).squeeze()
        d = self.kldiv(old_logprobs, new_probs)
        if d < self.opt.dtarg / 1.5:
            self.beta /= 2
        elif d > self.opt.dtarg * 1.5:
            self.beta *= 2

    def compute_returns(self, pnt):
        returns = 0
        for t in range(self.memory.size - 1, pnt - 1, -1):
            reward = float(self.memory.memory[3][t])
            self.memory.memory[3][t] += returns
            returns = self.opt.gamma * (reward + returns)

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
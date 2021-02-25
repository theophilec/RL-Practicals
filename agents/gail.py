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
from gans import Discriminator
from policies import GreedyPolicy

class GAILAgent(BaseImitation):
    def __init__(self, env, opt, writer=None):
        super().__init__(env, opt)
        self.name = "GAIL"
        self.writer = writer

        self.discriminator = Discriminator(
            input_size = self.featureExtractor.outSize+self.action_space.n,
            hidden_size=opt.hidden_dim[0]
        )
        # self.critic = ApproxFunction(
        #     self.featureExtractor.outSize,
        #     1,
        #     opt.hidden_dim
        # )
        self.actor = ApproxFunction(
            self.featureExtractor.outSize,
            self.action_space.n,
            opt.hidden_dim,
            hidden_activation = torch.tanh,
            output_activation = nn.Softmax(-1)
        )

        data = TensorDataset(self.expert_states, self.toIndexAction(self.expert_actions))
        self.expert_loader = DataLoader(data, batch_size=opt.batch_size)
        self.loss = nn.BCELoss()
        # self.optim_critic = Adam(params=self.critic.parameters(), lr=opt.lr)
        self.optim_actor = Adam(params=self.actor.parameters(), lr=opt.lr)
        self.optim_discrim = Adam(params=self.discriminator.parameters(), lr=opt.lr)
        self.memory = Memory(mem_size=opt.mem_size, items=6, replace=False)

        self.t = 0
        self.optim_step = 0
        self.epi = 0
        self.device = None
        self.traj_start = 0
        self.logprob = None
        self.action = None

        if self.opt.device:
            self.device = torch.device(self.opt.device)
            self.discriminator.to(self.device)
            self.actor.to(self.device)

    def discriminator_step(self, expert_batch, agent_batch):
        one_labels = torch.ones(expert_batch.size(0), device=self.device)
        zero_labels = torch.zeros(agent_batch.size(0), device=self.device)

        expert_probs = self.discriminator(expert_batch).flatten()
        agent_probs = self.discriminator(agent_batch).flatten()
        loss_expert = self.loss(expert_probs, one_labels)
        loss_agent = self.loss(agent_probs, zero_labels)
        loss_D = loss_expert + loss_agent
        
        #Calculate Gradients and Update weights
        self.optim_discrim.zero_grad()
        loss_D.backward()
        self.optim_discrim.step()

        self.writer.add_scalar("Discriminator/Expert Score", 
                                expert_probs.mean().item(), self.optim_step)
        self.writer.add_scalar("Discriminator/Agent Score",
                                agent_probs.mean().item(), self.optim_step)
        

    def learn(self):
        print("Learning ! ")
        data = self.memory.as_dataset()
        agent_loader = DataLoader(data, batch_size=self.opt.batch_size)
        expert_loader = iter(self.expert_loader)
        for last_ob, actions, old_logprobs, returns, ob, mask in agent_loader:
            try:
                expert_ob, expert_actions = next(expert_loader)
            except StopIteration:
                expert_loader = iter(self.expert_loader)
                expert_ob, expert_actions = next(expert_loader)
            expert_pairs = torch.cat([expert_ob, self.toOneHot(expert_actions)], 1)
            agent_pairs = torch.cat([last_ob, self.toOneHot(actions)], 1)
            self.optim_step += 1
            self.discriminator_step(expert_pairs, agent_pairs)

            advantages = returns # no baseline for now
            probs = self.actor(last_ob)
            selected_probs = torch.gather(probs, 1, actions.unsqueeze(1)).squeeze()
            ratios = selected_probs /  torch.exp(old_logprobs).detach()
            clipped_ratios = torch.clip(ratios, 1 - self.opt.epsilon, 1 + self.opt.epsilon)
            loss_i = torch.minimum(ratios * advantages,
                                    clipped_ratios * advantages)
            loss_obj = - torch.mean(loss_i)
            entropy = (torch.log(probs + 1e-4) * probs).sum(-1)
            loss = loss_obj + self.opt.beta * entropy.mean()
            
            self.optim_actor.zero_grad()
            loss.backward()
            self.writer.add_scalar(
                "Actor/Loss", loss.item(), self.optim_step
            )
            self.writer.add_scalar(
                "Actor/Norm", torch.norm(self.actor.fcinput.weight.data), self.optim_step
            )
            self.optim_actor.step()

    def compute_avg_returns(self, pnt):
        returns = 0
        for t in range(self.memory.size - 1, pnt - 1, -1):
            last_ob = self.memory.memory[0][t].unsqueeze(0)
            action = self.memory.memory[1][t].unsqueeze(0)
            pair = torch.cat([last_ob, self.toOneHot(action)], 1)
            rt = torch.log(self.discriminator(pair))
            # Clip
            rt = torch.clip(rt, self.opt.reward_clip[0], self.opt.reward_clip[1]).item()
            returns += rt
            self.memory.memory[3][t] += returns / (self.memory.size - t) #mean rewards
    
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
                torch.tensor(0.0), #init avg reward as 0
                ob, 
                torch.tensor(done and (not truncated)))
            )

        self.last_ob = ob
        # if (self.t % self.opt.C == 0):
        #     self.targetcritic = deepcopy(self.critic)

        if done:
            self.compute_avg_returns(self.traj_start)
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
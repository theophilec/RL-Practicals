from copy import deepcopy

import numpy as np
import torch
from torch import autograd
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim import SGD, Adam
import torch.nn as nn

from base import BaseAgent, ApproxFunction
from buffer import MultiAgentMemory
from utils import *


class MADDPG():
    def __init__(self, env, opt):
        self.name = "MADDPG"
        self.test = False
        assert env.n == len(opt.observation_shapes)
        self.num_agents = env.n

        self.opt = opt
        self.action_space = env.action_space

        self.critics = [ApproxFunction(
            sum(opt.observation_shapes) + (env.world.dim_p + env.world.dim_c) * env.n,
            1,
            opt.hidden_dim
        ).to(opt.device) for i in range(env.n)]

        self.actors =  [ApproxFunction(
            opt.observation_shapes[i],
            env.world.dim_p + env.world.dim_c,
            opt.hidden_dim,
            torch.tanh
        ).to(opt.device) for i in range(env.n)]

        self.targetcritics = [deepcopy(critic) for critic in self.critics]
        self.targetactors = [deepcopy(actor) for actor in self.actors]
        self.memory = MultiAgentMemory(mem_size=opt.mem_size, n=env.n, items=5)

        self.huber_loss = nn.SmoothL1Loss()
        self.optim_critics = [Adam(params=critic.parameters(), lr=opt.critic_lr) for critic in self.critics]
        self.optim_actors = [Adam(params=actor.parameters(), lr=opt.actor_lr) for actor in self.actors]

        self.noise = Orn_Uhlen(env.world.dim_p + env.world.dim_c, sigma=opt.sigma)
        self.last_ob = [None]
        self.action = [None]
        self.t = 0
        self.optim_t = 0

    def update_target(self, target, learner):
        with torch.no_grad():
            for ptarget, plearner in zip(target.parameters(), learner.parameters()):
                new_val = self.opt.rho * ptarget.data + (1 - self.opt.rho) * plearner.data
                ptarget.copy_(new_val)

    def learn(self):
        last_ob, actions, rewards, ob, mask = self.memory.sample(self.opt.batch_size, device=self.opt.device)

        for i in range(self.num_agents):
            next_actions_target = [self.targetactors[n](ob[n]).detach() for n in range(self.num_agents)]
            next_sa_pair = torch.cat(list(ob) + next_actions_target, -1)
            next_action_values = self.targetcritics[i](next_sa_pair).detach().squeeze()
            target = torch.where(mask[i], rewards[i], rewards[i] + self.opt.gamma * next_action_values)
            advantages = self.critics[i](torch.cat(list(last_ob) + list(actions), -1)).squeeze()

            loss = self.huber_loss(
                advantages,
                target
            )

            self.writer.add_scalar(
                f"Agent_{i+1}/Critic", loss.item(), self.optim_t
            )

            self.optim_critics[i].zero_grad()
            loss.backward()
            self.optim_critics[i].step()
        
            actions_list = list(deepcopy(actions))
            actions_list[i] = self.actors[i](last_ob[i])
            action_values = self.critics[i](torch.cat(list(last_ob) + actions_list, -1)).squeeze()
            loss = - torch.mean(action_values)

            self.writer.add_scalar(
                f"Agent_{i+1}/Actor", loss.item(), self.optim_t
            )

            self.optim_actors[i].zero_grad()
            loss.backward(retain_graph=True)
            self.optim_actors[i].step()

            self.update_target(self.targetcritics[i], self.critics[i])
            self.update_target(self.targetactors[i], self.actors[i])

        self.optim_t += 1

    def act(self, ob, reward, done, truncated):
        ob = [torch.tensor(o.flatten(), dtype=torch.float, device=self.opt.device) for o in ob]
        actions = [actor(o).squeeze().cpu() for actor, o in zip(self.actors, ob)]
        if self.test:
            return deepcopy([a.detach().numpy() for a in actions])

        if sum([int(a is not None) for a in self.action]) > 0:
            self.memory.add(
                [(o, 
                a, 
                torch.clip(torch.tensor(r, dtype=torch.float), self.opt.reward_clip[0], self.opt.reward_clip[1]), 
                o_next, 
                torch.tensor(d and (not truncated)))
                for o, a, r, o_next, d in zip(self.last_ob, self.action, reward, ob, done)]
            )

        if self.t % self.opt.optim_freq and (self.memory.size > self.opt.batch_size):
            self.learn()

        next_actions = []
        for a, d in zip(actions, done):
            if d:
                next_actions.append(None)
            else:
                next_actions.append(
                    torch.clip(
                        a.detach() + self.noise.sample(),
                        float(self.opt.action_space_low),
                        float(self.opt.action_space_high)
                    ))
        self.action = next_actions
        self.last_ob = ob
        self.t += 1
        return deepcopy([a.detach().cpu().numpy() for a in self.action])

    def save(self, path):
        save_dict = {}
        for i in range(self.num_agents):
            save_dict[f"actor{i}"] = self.actors[i].state_dict()
            save_dict[f"critic{i}"] = self.critics[i].state_dict()
            save_dict[f"optim_actor{i}"] = self.optim_actors[i].state_dict()
            save_dict[f"optim_critic{i}"] = self.optim_critics[i].state_dict()
        with open(path, "wb") as f:
            torch.save(save_dict, path)
            print(f"Model saved at {path}")
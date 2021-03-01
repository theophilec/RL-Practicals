from copy import deepcopy

import numpy as np
import torch
from torch import autograd
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim import SGD, Adam
import torch.nn as nn

from base import BaseAgent, ApproxFunction
from buffer import MultiAgentMemory


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
        ) for i in range(env.n)]

        self.actors =  [ApproxFunction(
            opt.observation_shapes[i],
            env.world.dim_p + env.world.dim_c,
            opt.hidden_dim,
            torch.tanh
        ) for i in range(env.n)]

        self.targetcritics = [deepcopy(critic) for critic in self.critics]
        self.targetactors = [deepcopy(actor) for actor in self.actors]
        self.memory = MultiAgentMemory(mem_size=opt.mem_size, n=env.n, items=5)

        self.huber_loss = nn.SmoothL1Loss()
        self.optim_critics = [Adam(params=critic.parameters(), lr=opt.critic_lr) for critic in self.critics]
        self.optim_actors = [Adam(params=actor.parameters(), lr=opt.actor_lr) for actor in self.actors]

        self.last_ob = [None]
        self.action = [None]
        self.sigma = torch.diag(torch.tensor(self.opt.sigma, dtype=torch.float))
        self.t = 0

    def update_target(self, target, learner):
        with torch.no_grad():
            for ptarget, plearner in zip(target.parameters(), learner.parameters()):
                new_val = self.opt.rho * ptarget.data + (1 - self.opt.rho) * plearner.data
                ptarget.copy_(new_val)

    def learn(self):
        print("Learning ...")
        last_ob, actions, rewards, ob, mask = self.memory.sample(self.opt.batch_size)
        next_actions_target = [self.targetactors[n](ob[:, n, :]).detach() for n in range(self.num_agents)]
        next_sa_pair = torch.cat([ob.reshape(ob.size(0), -1)] + next_actions_target, -1)

        for i in range(self.num_agents):
            next_action_values = self.targetcritics[i](next_sa_pair).detach().squeeze()
            target = torch.where(mask[:, i], rewards[:, i], rewards[:, i] + self.opt.gamma * next_action_values)
            advantages = self.critics[i](torch.cat([last_ob.reshape(ob.size(0), -1), 
                                                    actions.reshape(ob.size(0), -1)], -1)).squeeze()

            loss = self.huber_loss(
                advantages,
                target
            )

            self.optim_critics[i].zero_grad()
            loss.backward()
            self.optim_critics[i].step()
        
            actor_actions = [actor(o) for actor, o in zip(self.actors, last_ob.permute(1,0,2))]
            action_values = self.critics[i](torch.cat([last_ob.reshape(ob.size(0), -1)] + actor_actions, -1)).squeeze()
            loss = - torch.mean(action_values)
            self.optim_actors[i].zero_grad()
            loss.backward(retain_graph=True)
            self.optim_actors[i].step()

        self.update_target(self.targetcritics[i], self.critics[i])
        self.update_target(self.targetactors[i], self.actors[i])

    def act(self, ob, reward, done, truncated):
        ob = [torch.tensor(o.flatten(), dtype=torch.float) for o in ob]
        actions = [actor(o).squeeze() for actor, o in zip(self.actors, ob)]
        if self.test:
            return [a.detach().numpy() for a in actions]

        if sum([int(a is not None) for a in self.action]) > 0:
            self.memory.add(
                [(o, 
                a, 
                torch.clip(torch.tensor(r, dtype=torch.float), self.opt.reward_clip[0], self.opt.reward_clip[1]), 
                op, 
                torch.tensor(d and (not truncated)))
                for o, a, r, op, d in zip(self.last_ob, self.action, reward, ob, done)]
            )            

        if self.t > self.opt.learning_minimum:
            self.learn()

        next_actions = []
        for a, d in zip(actions, done):
            if d:
                next_actions.append(None)
            else:
                dist = MultivariateNormal(a.squeeze(), self.sigma)
                next_actions.append(
                    torch.clip(
                        dist.sample(),
                        float(self.opt.action_space_low),
                        float(self.opt.action_space_high)
                    ))
        self.action = next_actions
        self.last_ob = ob
        # self.sigma *= self.opt.decrease_rate    
        self.t += 1
        return [a.detach().numpy() for a in self.action]

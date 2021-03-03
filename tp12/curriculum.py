from copy import deepcopy

import numpy as np
import torch
from torch import autograd
from torch.optim import SGD, Adam
import torch.nn as nn

from base import BaseAgent, ApproxFunction
from buffer import Memory
from policies import GreedyPolicy

class Wrapper():
    def __init__(self, agent):
        self.agent = agent
        self.history = []
        self.last_ob = None
    
    def get_features(self, *args):
        return self.agent.get_features(*args)

    def add_writer(self, writer):
        self.agent.writer = writer 
    
    def store_hinsights(self, t_goal):
        """t_goal: last visited state"""
        for transition in self.history:
            t_last_ob, t_action, t_ob = transition
            t_done = (t_ob == t_goal).all()
            t_reward = 1 if t_done else -0.1
            new_transition = (
                torch.cat([t_last_ob, t_goal], 0),
                t_action,
                torch.tensor(t_reward),
                torch.cat([t_ob, t_goal], 0),
                t_done 
            )
            self.agent.memory.add(new_transition)


class HER(Wrapper):
    def __init__(self, agent):
        super().__init__(agent)
        self.name = "HER" + str(agent.name)
    
    def act(self, ob, reward, done, truncated, goal):
        to_act = (ob, reward, done, truncated, goal)
        ob = self.get_features(ob)
        if self.agent.action is not None:
            transition = (self.last_ob, torch.tensor(self.agent.action), ob)
            self.history.append(transition)
        self.last_ob = ob
        if done:
            self.store_hinsights(ob)
            self.history = []
            self.last_ob = None
        return self.agent.act(*to_act)


class IGS(Wrapper):
    def __init__(self, agent):
        super().__init__(agent)
        self.name = "IGS" + str(agent.name)
        self.beta = self.agent.opt.beta
        self.alpha = self.agent.opt.alpha
        self.ng = agent.opt.goal_buffer_size
        assert (self.beta >= 0) and (self.beta <= 1)
        self.goals = Memory(mem_size=self.ng, items=3, replace=True, 
                            weighted=True, replace_type="fifo")
        self.goal, self.goal_idx = None, None
        self.goal_freq = self.agent.opt.goalFreq
        self.goal_time = True
        self.b = False
        self.j = 0

    def entropy(self, ratio):
        if ratio == 1:
            ratio -= 1e-4
        if ratio == 0:
            ratio += 1e-4
        try:
            with np.errstate(invalid='raise'):
                entropy = - ratio * np.log(ratio) - (1 - ratio) * np.log(1 - ratio)
        except FloatingPointError:
            entropy = 0
        return entropy

    def is_in_buffer(self, goal, buffer):
        for g in buffer:
            if (goal == g).all():
                return True
        return False
     
    def act(self, ob, reward, done, truncated, goal):
        if self.goal is None:
            if self.goals.size > 1 and np.random.random() < self.beta:
                # Iterative goal sampling
                (self.goal, _, _), self.goal_idx = self.goals.sample(1, return_indices=True)
                self.goal, self.goal_idx = self.goal.squeeze(), int(self.goal_idx)
                self.b = True
            else:
                self.goal = goal
                self.b = False

        if self.agent.action is not None:
            transition = (self.last_ob, torch.tensor(self.agent.action), ob)
            self.history.append(transition)
        
        self.last_ob = ob
        
        if done:
            self.store_hinsights(ob)
            self.history = []
            self.last_ob = None

        goal = self.goal
        win = (ob == self.goal).all()
        if done or win:
            if self.b:
                # Update weights
                self.goals.memory[1][self.goal_idx] += 1
                if win:
                    reward = 1
                    self.goals.memory[2][self.goal_idx] += 1
                entropy = self.entropy(
                    self.goals.memory[2][self.goal_idx] / self.goals.memory[1][self.goal_idx]
                )
                self.goals.update_weight(self.goal_idx, self.alpha * entropy)
            if done:
                self.j += 1
                
            self.goal_idx = None
            self.goal = None
            if self.j % self.goal_freq == 0:
                self.goal_time = True          
            if self.goal_time:
                if not self.is_in_buffer(ob, self.goals.memory[0]):
                    self.goals.add((ob, torch.tensor(0), torch.tensor(0)), weight=0)
                    self.goal_time = False
        return self.agent.act(ob, reward, done, truncated, goal)

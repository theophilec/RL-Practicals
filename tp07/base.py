import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseAgent(object):
    """
    BaseAgent class for all RL agents.
    """
    def __init__(self, env, opt):
        self.name = "BaseAgent"
        self.test = False

        self.opt = opt
        self.featureExtractor = opt.featExtractor(env)

        self.action_space = env.action_space
        

    def get_features(self, ob):
        return torch.tensor(self.featureExtractor.getFeatures(ob), dtype=torch.float32).squeeze()

class ApproxFunction(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims, 
                    output_activation=None, hidden_activation=F.relu):
        super().__init__()
        self.fcinput = nn.Linear(input_size, hidden_dims[0])
        self.fchidden = nn.ModuleList()
        for i, d in enumerate(hidden_dims[1:]):
            self.fchidden.append(nn.Linear(hidden_dims[i], d))
        self.fcoutput = nn.Linear(hidden_dims[-1], output_size)
        self.act_output = output_activation
        self.act_hidden = hidden_activation

    def forward(self, x):
        x = self.act_hidden(self.fcinput(x))
        for layer in self.fchidden:
            x = self.act_hidden(layer(x))
        x = self.fcoutput(x)
        if self.act_output:
            x = self.act_output(x) 
        return x
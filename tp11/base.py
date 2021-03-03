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


class BaseImitation(BaseAgent):
    def __init__(self, env, opt):
        super().__init__(env, opt)
        self.expert_states, self.expert_actions = self.loadExpertTransitions(opt.expert_path)
        self.nbActions = self.action_space.n

    def loadExpertTransitions(self, file):
        with open(file, 'rb') as handle:
            expertdata = pickle.load(handle).to(torch.float)
            expertstates = expertdata[:, :self.featureExtractor.outSize]
            expertactions = expertdata[:, self.featureExtractor.outSize:]
            expertstates = expertstates.contiguous()
            expertactions = expertactions.contiguous()
            return expertstates, expertactions

    def toOneHot(self, actions):
        actions = actions.view(-1).to(torch.long)
        oneHot = torch.zeros(actions.size()[0], self.nbActions).to(torch.float)
        oneHot[range(actions.size()[0]),actions] = 1
        return oneHot

    def toIndexAction(self, oneHot):
        ac = torch.LongTensor(range(self.nbActions)).view(1, -1)
        ac = ac.expand(oneHot.size()[0], -1).contiguous().view(-1)
        actions = ac[oneHot.view(-1)>0].view(-1)
        return actions


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
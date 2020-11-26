import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden1=512, hidden2=256,init_w=3e-3):
    	super(Actor,self).__init__()
    	self.fc1 = nn.Linear(n_states,hidden1)
    	self.fc2 = nn.Linear(hidden1,hidden2)
    	self.fc3 = nn.Linear(hidden2,n_actions)
    	self.relu  = nn.ReLU()
    	self.tanh = nn.Tanh()
    	self.norm = nn.BatchNorm1d(n_states)
    	# self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self,x):
        # print(x,x.size())
        # print(x.float(),x.size())
        x = x.float()
        x = self.norm(x)
        x = self.relu(self.fc1(x))
        # print(x,x.size())
        x = self.relu(self.fc2(x))
        # print(x,x.size())
        x = self.tanh(self.fc3(x))

        return x

class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden1=512, hidden2=256, init_w=3e-3):
    	super(Critic,self).__init__()
    	self.fc1 = nn.Linear(n_states,hidden1)
    	self.fc2 = nn.Linear(hidden1 + n_actions,hidden2)
    	self.fc3 = nn.Linear(hidden2,1)
    	self.relu  = nn.ReLU()
    	self.norm = nn.BatchNorm1d(n_states)
        self.norm_a = nn.BatchNorm1d(n_actions)
    	# self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self,x,a):

    	x = x.float()
    	a = a.float()
        # print(a.size(),x.size())
    	x = self.norm(x)
    	x = self.relu(self.fc1(x))
        a = self.norm_a(a)
    	x = self.relu(self.fc2(torch.cat([x,a],1)))
    	x = self.fc3(x)
    	return x


    	
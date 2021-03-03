import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ObservationType

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
epsilon = 1e-6
hidden_size = 512
encoder_size = 512

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Encoder(nn.Module):
    def __init__(self, state_dim):
        super(Encoder, self).__init__()
        self.state_dim = state_dim
        self.detach = False
        self.features = nn.Sequential(
            nn.Conv2d(state_dim, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32*425, encoder_size),
            nn.LayerNorm(encoder_size)
        )

    def forward(self, x):
        x = self.features(x.transpose(1,3))
        x = x.view(x.size(0), -1)
        if self.detach:
            x.detach()
        x = self.fc(x)
        return x

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for layer in range(len(self.features),2):
            self.features[layer].weight = source.features[layer].weight
            self.features[layer].bias = source.features[layer].bias


class ActorCNN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCNN, self).__init__()
        self.encoder = Encoder(num_inputs)
        self.linear1 = nn.Linear(encoder_size+num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)

        self.apply(weights_init_)

        # action rescaling
        # if action_space is None:
        #     self.action_scale = torch.tensor(1.)
        #     self.action_bias = torch.tensor(0.)
        # else:
        self.action_scale = torch.tensor(
            1.)
        self.action_bias = torch.tensor(
            1.)

    def forward(self, state, exp_action):
        x = self.encoder(state)
        x = torch.cat([state,exp_action],axis=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, exp_action):
        mean, log_std = self.forward(state, exp_action)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(ActorCNN, self).to(device)

class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Actor, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs+num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)

        self.apply(weights_init_)

        # action rescaling
        # if action_space is None:
        #     self.action_scale = torch.tensor(1.)
        #     self.action_bias = torch.tensor(0.)
        # else:
        self.action_scale = torch.tensor(
            1.)
        self.action_bias = torch.tensor(
            1.)

    def forward(self, state, exp_action):
        x = torch.cat([state,exp_action],axis=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, exp_action):
        mean, log_std = self.forward(state, exp_action)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)



class Variational(object):
    def __init__(self, args, agent, state_dim, action_dim):
        self.args = args
        self.agent = agent
        self.state_dim = state_dim
        self.action_dim = action_dim
        if args.act==ObservationType.KIN:
            self.approx = Actor(state_dim, action_dim).to(self.device)
        else:
            self.approx = ActorCNN(state_dim, action_dim).to(self.device)
        self.approx.encoder.copy_conv_weights_from(self.agent.encoder)
        self.approx_optimizer = torch.optim.Adam(self.approx.parameters(), lr=self.args.lr)
    
    def train(self, state, exp_action):
        action, log_pi, _ = self.agent.sample(state)
        _, log_q, _ = self.approx.sample(state, exp_action)

        obj = log_q + self.alpha*log_pi

        self.approx_optimizer.zero_grad()
        obj.backward()
        self.approx_optimizer.step()
        
        return obj.item()












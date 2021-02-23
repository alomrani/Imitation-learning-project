import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ObservationType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]


class ActorCNN(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(ActorCNN, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.features = nn.Sequential(
			nn.Conv2d(state_dim, 32, kernel_size=8, stride=4, padding=0),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
			nn.ReLU()
		)
		self.fc = nn.Sequential(
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, action_dim)
		)
		self.max_action = max_action

	def forward(self, x):
		x = self.features(x.transpose(1,3))
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return self.max_action * torch.tanh(x)
    


class CriticCNN(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(CriticCNN, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.features = nn.Sequential(
			nn.Conv2d(state_dim, 32, kernel_size=8, stride=4, padding=0),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
			nn.ReLU()
		)
		self.fc = nn.Sequential(
			nn.Linear(512+self.action_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 1)
		)
        
	def forward(self, x, action):
		x = self.features(x.transpose(1,3))
		x = x.view(x.size(0), -1)
		x = torch.cat([x, action], axis=1)
		x = self.fc(x)
		return x
    

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 128)
		self.l3 = nn.Linear(128, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256 + action_dim, 128)
		self.l3 = nn.Linear(128, 1)


	def forward(self, state, action):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(torch.cat([q, action], 1)))
		return self.l3(q)


class DDPG(object):
	def __init__(self, args, state_dim, action_dim, max_action, discount=0.99, tau=0.001, policy_noise=0.2, noise_clip=0.5,policy_freq=1):
		self.args = args
		if args.obs==ObservationType.KIN:
			self.actor = Actor(state_dim, action_dim, max_action).to(device)
			self.critic = Critic(state_dim, action_dim).to(device)
		else:
			self.actor = ActorCNN(state_dim, action_dim, max_action).to(device)
			self.critic = CriticCNN(state_dim, action_dim).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

		self.discount = discount
		self.tau = tau


	def select_action(self, state):
		state = torch.FloatTensor(state).to(device) #.reshape(1, -1)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=64):
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute the target Q value
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		target_Q = reward + (not_done * self.discount * target_Q).detach()

		# Get current Q estimate
		current_Q = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(state, self.actor(state)).mean()
		
		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		
		return actor_loss.item(), critic_loss.item()


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "/_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "/_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "/_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "/_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "/_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "/_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "/_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "/_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)



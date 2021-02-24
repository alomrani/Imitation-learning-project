import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ObservationType

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

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
		self.fc1 = nn.Sequential(
			nn.Linear(512+self.action_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 1)
		)
		self.fc2 = nn.Sequential(
			nn.Linear(512+self.action_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 1)
		)
        
	def forward(self, x, action):
		x = self.features(x.transpose(1,3))
		x = x.view(x.size(0), -1)
		sa = torch.cat([x, action], axis=1)
		q1 = self.fc1(sa)
		q2 = self.fc2(sa)
		return q1, q2

	def Q1(self, x, action):
		x = self.features(x.transpose(1,3))
		x = x.view(x.size(0), -1)
		sa = torch.cat([x, action], axis=1)
		q1 = self.fc1(sa)
		return q1

class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, action_space=None):
        super(Actor, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, 1024)
        self.linear2 = nn.Linear(1024, 1024)

        self.mean_linear = nn.Linear(1024, num_actions)
        self.log_std_linear = nn.Linear(1024, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (1 - (-1))) / 2.)
            self.action_bias = torch.FloatTensor(
                (1 + (-1))) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
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


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 1024)
		self.l2 = nn.Linear(1024, 1024)
		self.l3 = nn.Linear(1024, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 1024)
		self.l5 = nn.Linear(1024, 1024)
		self.l6 = nn.Linear(1024, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


class SAC(object):
    def __init__(
		self,
		args, 
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

        self.args = args
        self.gamma = discount
        self.tau = tau
        self.alpha = args.alpha

		self.total_it = 0

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = torch.device("cuda" if args.cuda else "cpu")


		if args.obs==ObservationType.KIN:
            self.actor = Actor(state_dim, action_dim).to(self.device)
            self.critic = Critic(state_dim, action_dim).to(device=self.device)
		# else:
		# 	self.actor = ActorCNN(state_dim, action_dim).to(device)
		# 	self.critic = CriticCNN(state_dim, action_dim).to(device)
	
    	self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device=self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
    
        self.critic_target = copy.deepcopy(self.critic)

        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=1e-4)


    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def train(self, replay_buffer, batch_size):
        self.total_it += 1
        # Sample a batch from memory
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_state)
            qf1_next_target, qf2_next_target = self.critic_target(next_state, next_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_pi
            next_q_value = reward + not_done * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state, action)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        critic_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(state)

        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)


        if self.total_it % self.target_update_interval == 0:
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))










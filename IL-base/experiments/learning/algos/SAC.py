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
hidden_size = 1024
encoder_size = 50

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Encoder(nn.Module):
    def __init__(elf, state_dim):
        super(Encoder, self).__init__()
		self.state_dim = state_dim
        self.detach = True
		self.features = nn.Sequential(
			nn.Conv2d(state_dim, 32, kernel_size=3, stride=2),
            nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, stride=1)
            nn.ReLU(),
		)
		self.fc = nn.Sequential(
			nn.Linear(32*35*35, encoder_size),
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
        for i in range(self.num_layers):
            self.convs[i].weight = source.convs[i].weight
            self.convs[i].bias = source.convs[i].bias


class ActorCNN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCNN, self).__init__()
        self.encoder = Encoder(num_inputs)
        self.linear1 = nn.Linear(encoder_size, hidden_size)
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

    def forward(self, state):
        x = self.encoder(state)
        x = F.relu(self.linear1(x))
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


class CriticCNN(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(CriticCNN, self).__init__()
        self.encoder = Encoder(num_inputs)
		# Q1 architecture
		self.l1 = nn.Linear(encoder_size + action_dim, hidden_size)
		self.l2 = nn.Linear(hidden_size, hidden_size)
		self.l3 = nn.Linear(hidden_size, 1)

		# Q2 architecture
		self.l4 = nn.Linear(encoder + action_dim, hidden_size)
		self.l5 = nn.Linear(hidden_size, hidden_size)
		self.l6 = nn.Linear(hidden_size, 1)


	def forward(self, state, action):
        enc_state = self.encoder(state)
		sa = torch.cat([enc_state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Actor, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
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
		self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
		self.l2 = nn.Linear(hidden_size, hidden_size)
		self.l3 = nn.Linear(hidden_size, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, hidden_size)
		self.l5 = nn.Linear(hidden_size, hidden_size)
		self.l6 = nn.Linear(hidden_size, 1)


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        if args.obs==ObservationType.KIN:
            self.actor = Actor(state_dim, action_dim).to(self.device)
            self.critic = Critic(state_dim, action_dim).to(device=self.device)
        else:
        	self.actor = ActorCNN(state_dim, action_dim).to(device)
        	self.critic = CriticCNN(state_dim, action_dim).to(device)
            self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr)

        self.critic = Critic(state_dim, action_dim).to(device=self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr)

        self.critic_target = copy.deepcopy(self.critic)

        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.args.lr)


    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def train(self, replay_buffer, batch_size):
        self.total_it += 1
        # Sample a batch from memory
        for _ in range(self.args.num_updates):
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

            actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
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










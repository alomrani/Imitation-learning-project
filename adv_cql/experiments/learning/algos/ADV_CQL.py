import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Normal
from algos.variational import Variational
from algos.SAC import Actor as SACActor
from algos.SAC import ActorCNN as SACActorCNN
from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.ZigZagAviary import ZigZagAviary
from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.envs.single_agent_rl.TuneAviary import TuneAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from stable_baselines3.common.cmd_util import make_vec_env
import shared_constants
import algos
torch.autograd.set_detect_anomaly(True)
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
        self.linear1 = nn.Linear(encoder_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)

        self.apply(weights_init_)

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
        return super(ActorCNN, self).to(device)


class CriticCNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticCNN, self).__init__()
        self.encoder = Encoder(state_dim)
        # Q1 architecture
        self.l1 = nn.Linear(encoder_size + action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

        # Q2 architecture
        self.l4 = nn.Linear(encoder_size + action_dim, hidden_size)
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


class ADV_CQL(object):
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
        self.temp = 1.0
        self.min_q_weight = 1.0
        self.lagrange_thresh = 0.0
        self.with_lagrange = True
        self.args = args
        self.gamma = discount
        self.tau = tau
        self.alpha = args.alpha
        self.num_exp_episodes = 50
        self.total_it = 0
        self.min_q_version = 3
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_random = 10

        env_name = self.args.env+"-aviary-v0"
        sa_env_kwargs = dict(aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=self.args.obs, act=self.args.act)

        if env_name == "takeoff-aviary-v0":
            train_env_name = TakeoffAviary

        elif env_name == "hover-aviary-v0":
            train_env_name = HoverAviary

        elif env_name == "zigzag-aviary-v0":
            train_env_name = ZigZagAviary

        elif env_name == "flythrugate-aviary-v0":
            train_env_name = FlyThruGateAviary

        elif env_name == "tune-aviary-v0":
            train_env_name = TuneAviary

        self.train_env = make_vec_env(train_env_name,
                                    env_kwargs=sa_env_kwargs,
                                    n_envs=self.args.cpu,
                                    seed=0
                                    )

        self.action_dim = self.train_env.action_space.shape[0] 
        if self.args.obs== ObservationType.KIN:
            self.state_dim = self.train_env.observation_space.shape[0]
            self.train_env = algos.utils.Base(self.train_env)
            self.actor = Actor(state_dim, self.action_dim).to(self.device)
            self.critic = Critic(state_dim, self.action_dim).to(device=self.device)
            self.expert= SACActor(state_dim, self.action_dim).to(self.device)
            self.expert.load_state_dict(torch.load("experts/"+self.args.env+"_kin"))
        else:
            self.state_dim = 4
            self.train_env = algos.utils.Normalize(self.train_env)
            self.actor = Actor(state_dim, self.action_dim).to(self.device)
            self.critic = Critic(state_dim, self.action_dim).to(device=self.device)
            self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
            self.expert= SACActorCNN(state_dim, self.action_dim).to(self.device)
            self.expert.load_state_dict(torch.load("experts/"+self.args.env+"_rgb"))

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr)

        self.critic_target = copy.deepcopy(self.critic)
        self.var_obj = Variational(self.args, self.actor, state_dim, action_dim)

        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.args.lr)
        if self.with_lagrange:
            self.target_action_gap = torch.FloatTensor([self.lagrange_thresh]).to(self.device)
            self.log_alpha_prime = torch.zeros(1, requires_grad=True)
            self.alpha_prime_optimizer = torch.optim.Adam(
                [self.log_alpha_prime],
                lr=self.args.lr,
            )


    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def approx_logsumexp(self, act_q):
        q = act_q.squeeze(-1)
        func = torch.logsumexp(q / self.temp, dim=1,).mean()
        q_dim = q.shape[1]
        grads = torch.exp(q - func)
        ones = torch.ones((q_dim,q_dim))
        hess_approx = 0.5*(torch.eye(q_dim) - (1/(q_dim+1))*ones*ones).to(device=self.device)
        return (func + q*grads + 0.5*torch.mm(q*q,hess_approx)).mean()
        
    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions.to(self.device))
        return preds[0].view(obs_shape, num_repeat, 1), preds[1].view(obs_shape, num_repeat, 1)

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        new_obs_actions, new_obs_log_pi, _ = network.sample(obs_temp)
        return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)

    def collect_data(self, replay_buffer):
        for _ in range(self.num_exp_episodes):
            state, done = self.train_env.reset(), False
            total_reward = 0
            while done==False:
                action, _, _ = self.expert.sample(torch.FloatTensor(state).to(self.device))
                next_state, reward, done, _ = self.train_env.step(action.detach().cpu().numpy()[0])
                done = float(done)
                action = action.detach().cpu().numpy()[0]
                replay_buffer.add(state, action, next_state, reward, done)
                state = next_state
                total_reward += reward
        print("Expert Data Collected")

    def train(self, replay_buffer, batch_size):
        # Collect data from expert
        if self.total_it%10000==0:
            self.prefill_buffer(replay_buffer, batch_size)
        self.total_it += 1
        # Sample a batch from memory
        for _ in range(self.args.num_updates):
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            int_mot = self.var_obj.train(state, action)

            with torch.no_grad():
                next_action, next_log_pi, _ = self.actor.sample(next_state)
                qf1_next_target, qf2_next_target = self.critic_target(next_state, next_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_pi
                next_q_value = reward + not_done * self.gamma * (min_qf_next_target)
            qf1, qf2 = self.critic(state, action)  # Two Q-functions to mitigate positive bias in the policy improvement step # .unsqueeze(1)
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            critic_loss = qf1_loss + qf2_loss

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = self.log_alpha.exp()
            else:
                alpha_loss = torch.tensor(0.).to(self.device)

            ## add CQL
            random_actions_tensor = torch.FloatTensor(qf2.shape[0] * self.num_random, self.action_dim).uniform_(-1, 1)
            curr_actions_tensor, curr_log_pis = self._get_policy_actions(state, num_actions=self.num_random, network=self.actor)
            new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_state, num_actions=self.num_random, network=self.actor)
            q1_rand, q2_rand = self._get_tensor_values(state, random_actions_tensor, network=self.critic)
            q1_curr_actions, q2_curr_actions = self._get_tensor_values(state, curr_actions_tensor, network=self.critic)
            q1_next_actions, q2_next_actions = self._get_tensor_values(state, new_curr_actions_tensor, network=self.critic)

            q1_next_actions -=  - self.args.alpha*int_mot
            q2_next_actions -=  - self.args.alpha*int_mot

            cat_q1 = torch.cat(
                [q1_rand, qf1.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
            )
            cat_q2 = torch.cat(
                [q2_rand, qf2.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
            )
            # std_q1 = torch.std(cat_q1, dim=1)
            # std_q2 = torch.std(cat_q2, dim=1)

            if self.min_q_version == 3:
                # importance sammpled version
                random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
                cat_q1 = torch.cat(
                    [q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
                )
                cat_q2 = torch.cat(
                    [q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
                )

            # # bohning approximation as an upper bound on logsumexp
            # min_qf1_loss = self.approx_logsumexp(cat_q1) * self.min_q_weight * self.temp
            # min_qf2_loss = self.approx_logsumexp(cat_q2) * self.min_q_weight * self.temp

            min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
            min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp

            """Subtract the log likelihood of data"""
            min_qf1_loss = min_qf1_loss - qf1.mean() * self.min_q_weight
            min_qf2_loss = min_qf2_loss - qf2.mean() * self.min_q_weight

            if self.with_lagrange:
                alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0).to(self.device)
                min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
                min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

                self.alpha_prime_optimizer.zero_grad()
                alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5 
                alpha_prime_loss.backward(retain_graph=True)
                self.alpha_prime_optimizer.step()

            qf1_loss = qf1_loss + min_qf1_loss
            qf2_loss = qf2_loss + min_qf2_loss

            critic_loss = qf1_loss + qf2_loss

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            """
            Update networks
            """
            pi, log_pi, _ = self.actor.sample(state)

            qf1_pi, qf2_pi = self.critic(state, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            """
            Soft Updates
            """

            if self.total_it % self.target_update_interval == 0:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "/_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "/_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "/_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "/_actor_optimizer")

    def prefill_buffer(self, replay_buffer, batch_size):
        print("Filling Dataset...")
        self.collect_data(replay_buffer)
    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "/_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "/_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "/_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "/_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
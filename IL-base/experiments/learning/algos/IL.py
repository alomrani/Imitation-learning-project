import copy
import numpy as np
import torch
import algos
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.ZigZagAviary import ZigZagAviary
from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.envs.single_agent_rl.TuneAviary import TuneAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from stable_baselines3.common.cmd_util import make_vec_env
import shared_constants

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
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.apply(weights_init_)


    def forward(self, state):
        x = self.encoder(state)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Actor, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x


class IL(object):
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
        self.num_exp_episodes = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            self.actor = Actor(state_dim, action_dim).to(self.device)
        else:
            self.state_dim = 4#train_env.observation_space.shape[2]
            self.train_env = algos.utils.Normalize(self.train_env)
            self.actor = ActorCNN(state_dim, action_dim).to(self.device)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr)


    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]
    
    def collect_data(self, replay_buffer):
        for _ in range(self.num_exp_episodes):
            state, done = self.train_env.reset(), False
            total_reward = 0
            steps = 0
            while done==False:
                steps += 1
                next_state, reward, done, _ = self.train_env.step(action)
                replay_buffer.add(state, action, next_state, reward, done)
                state = next_state
                total_reward += reward


    def train(self, replay_buffer, batch_size):
        # Collect data from expert
        with torch.no_grad():
            self.collect_data(replay_buffer)
        # Sample a batch from memory
        state, exp_action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        action = self.actor(state)
        
        actor_loss = F.mse_loss(action, exp_action.detach())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        print(actor_loss)

        return actor_loss.item(), _

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "/_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "/_actor_optimizer")


    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "/_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "/_actor_optimizer"))










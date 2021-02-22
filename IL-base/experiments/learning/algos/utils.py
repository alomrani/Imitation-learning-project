import numpy as np
import torch
import time
import csv
import gym
import random
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.buffer = deque(maxlen=self.max_size)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.buffer.append((state, action, reward, next_state, done))

	def sample(self, batch_size):
		state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
		return (
			torch.FloatTensor(np.concatenate(state)).to(self.device),
			torch.FloatTensor(action).to(self.device),
			torch.FloatTensor(np.concatenate(next_state)).to(self.device),
			torch.FloatTensor(reward).to(self.device),
			torch.FloatTensor(np.expand_dims(done, 1)).to(self.device)
		)

	def __len__(self):
		return len(self.buffer)

def args_type(default):
  def parse_string(x):
    if default is None:
      return x
    if isinstance(default, bool):
      return bool(['False', 'True'].index(x))
    if isinstance(default, int):
      return float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, (list, tuple)):
      return tuple(args_type(default[0])(y) for y in x.split(','))
    return type(default)(x)
  def parse_object(x):
    if isinstance(default, (list, tuple)):
      return tuple(x)
    return x
  return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def save_logs(filename, log_dir):
    with open(filename+'logs.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in log_dir.items():
            writer.writerow([key, value])

class Logger():
	def __init__(self, args):
		self.args = args
		self.log = {}
		self.log['rewards'] = []
		self.log['average_rewards'] = []
		self.log['actor_loss'] = []
		self.log['critic_loss'] = []
		self.log['average_actor_loss'] = []
		self.log['average_critic_loss'] = []

	def log_episode(self, reward, actor_loss, critic_loss):
		self.log['rewards'].append(reward)
		self.log['actor_loss'].append(actor_loss)
		self.log['critic_loss'].append(critic_loss)

	def log_data(self):
		if len(self.log['rewards']) > self.args.window_size:
			self.log['average_rewards'].append(np.mean(self.log['rewards'][-self.args.window_size:]))
			self.log['average_actor_loss'].append(np.mean(self.log['actor_loss'][-self.args.window_size:]))
			self.log['average_critic_loss'].append(np.mean(self.log['critic_loss'][-self.args.window_size:]))

	def res_plot(self, filename):
		# plot rewards
		plt.figure()
		plt.title('Average Returns', fontsize=24)
		plt.plot(self.log['average_rewards'])
		plt.xlabel('Steps', fontsize=18)
		plt.ylabel('Returns', fontsize=18)
		plt.savefig(filename+'/plot_rewards.png', dpi=600, bbox_inches='tight')
		# plot actor loss
		plt.figure()
		plt.title('Average Actor Loss', fontsize=24)
		plt.plot(self.log['average_actor_loss'])
		plt.xlabel('Steps', fontsize=18)
		plt.ylabel('Loss', fontsize=18)
		plt.savefig(filename+'/plot_actor_loss.png', dpi=600, bbox_inches='tight')
		# plot critic loss
		plt.figure()
		plt.title('Average Critic Loss', fontsize=24)
		plt.plot(self.log['average_critic_loss'])
		plt.xlabel('Steps', fontsize=18)
		plt.ylabel('Loss', fontsize=18)
		plt.savefig(filename+'/plot_critic_loss.png', dpi=600, bbox_inches='tight')

	def save_logs(self, filename):
		with open(filename+'/logs.csv', 'w') as csv_file:
			writer = csv.writer(csv_file)
			for key, value in self.log.items():
				writer.writerow([key, value])
	

def record_video(args, policy, eval_env, seed, shared_constants, filename):
	eval_env.seed(seed + 100)
	state, done = eval_env.reset(), False
	images = []
	while not done:
		action = policy.select_action(np.array(state))
		state, reward, done, _ = eval_env.step(action)
		im = Image.fromarray(state[0,:,:,0:3])
		images.append(im.convert('P'))
	images[0].save(filename+'/'+str(time.time())+'-.gif', save_all=True, append_images=images[1:], optimize=False, duration=20, loop=0)






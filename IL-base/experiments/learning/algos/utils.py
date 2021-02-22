import numpy as np
import torch
import time
import matplotlib.pyplot as plt

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		if len(state_dim)==3:
			self.state = np.zeros((max_size, state_dim[0], state_dim[1], state_dim[2]), dtype=np.float32)
			self.next_state = np.zeros((max_size, state_dim[0], state_dim[1], state_dim[2]), dtype=np.float32)
		else:
			self.state = np.zeros((max_size, state_dim[0]), dtype=np.float32)
			self.next_state = np.zeros((max_size, state_dim[0]), dtype=np.float32)

		self.action = np.zeros((max_size, action_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


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

	def log_episode(self, reward):
		self.log['rewards'].append(reward)

	def log_data(self):
		if len(self.log['rewards'] > self.args.window_size):
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
	
def record_video(env, recorder):
	env.render()
	recorder.capture_frame()







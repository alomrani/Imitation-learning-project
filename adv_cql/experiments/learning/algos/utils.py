import numpy as np
import torch
import time
import csv
import gym
import random
import pybullet as pb
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
		self.log = self.normalize_rewards()
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

	def normalize_rewards(self):	
		self.log['average_rewards'] -= min(self.log['average_rewards'])
		self.log['rewards'] -= min(self.log['rewards'])
		return self.log

def record_video(args, policy, eval_env, seed, shared_constants, filename):
	eval_env.seed(seed + 100)
	state, done = eval_env.reset(), False
	images = []
	images_scene = []
	while not done:
		action = policy.select_action(np.array(state))
		state, reward, done, _ = eval_env.step(action)
		im = Image.fromarray((state[0,:,:,0:3]*255).astype(np.uint8))
		images.append(im.resize((240,240)).convert('P'))
		scene_img = eval_env.render()
		img = Image.fromarray(scene_img)
		images_scene.append(scene_img)

	images[0].save(filename+'/'+str(time.time())+'-.gif', save_all=True, append_images=images[1:], optimize=False, duration=20, loop=0)
	images_scene[0].save(filename+'/'+str(time.time())+'-scene.gif', save_all=True, append_images=images_scene[1:], optimize=False, duration=20, loop=0)


class Base():
	def __init__(self, inp):
		self.inp = inp
		self.action_space = inp.action_space
		self.observation_space = inp.observation_space

	def step(self, action):
		action = np.expand_dims(action, axis=0)
		state, reward, done, _ = self.inp.step(action)
		return state, reward, done, _

	def reset(self):
		return self.inp.reset()

	def seed(self, val):
		self.inp.seed(val)
	
	def render(self, mode="rgb_array", close=False):
		if mode != "rgb_array":
			return np.array([])
		base_pos = [0,0,0]
		_cam_dist = 2.5  #.3
		_cam_yaw = 50
		_cam_pitch = -35
		_render_width=240
		_render_height=240

		view_matrix = pb.computeViewMatrixFromYawPitchRoll(
			cameraTargetPosition=base_pos,
			distance=_cam_dist,
			yaw=_cam_yaw,
			pitch=_cam_pitch,
			roll=0,
			upAxisIndex=2)
		proj_matrix = pb.computeProjectionMatrixFOV(
			fov=90, aspect=float(_render_width)/_render_height,
			nearVal=0.01, farVal=100.0)
		(_, _, px, _, _) = pb.getCameraImage(
			width=_render_width, height=_render_height, viewMatrix=view_matrix,
			projectionMatrix=proj_matrix, renderer=pb.ER_TINY_RENDERER) #ER_BULLET_HARDWARE_OPENGL)
		rgb_array = np.array(px, dtype=np.uint8)
		rgb_array = np.reshape(rgb_array, (_render_height, _render_width, 4))
		rgb_array = rgb_array[:, :, :3]
		return rgb_array

	def record_video(self, args, policy, seed, shared_constants, filename):
		self.seed(seed + 100)
		state, done = self.reset(), False
		images_scene = []
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = self.step(action)
			scene_img = self.render()
			scene_img = Image.fromarray(scene_img)
			images_scene.append(scene_img)

		images_scene[0].save(filename+'/'+str(time.time())+'-scene.gif', save_all=True, append_images=images_scene[1:], optimize=False, duration=20, loop=0)


class Normalize():
	def __init__(self, inp):
		self.inp = inp
		self.action_space = inp.action_space
		self.observation_space = inp.observation_space
		
	def step(self, action):
		action = np.expand_dims(action, axis=0)
		state, reward, done, _ = self.inp.step(action)
		state = self.rgb2gray(state)
		self.state = self.stack_frames(state)
		return self.state / 255., reward, done, _

	def rgb2gray(self, rgb):
		r, g, b, a = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2], rgb[:,:,:,3]
		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
		return np.expand_dims(gray, axis=-1)

	def reset(self):
		self.state = self.rgb2gray(self.inp.reset())
		state = self.rgb2gray(self.inp.reset())
		self.state = np.concatenate((self.state, state, state, state), axis=3) / 255.
		return self.state

	def stack_frames(self, next_state):
		return np.concatenate((self.state[:,:,:,-3:], next_state), axis=3)

	def seed(self, val):
		self.inp.seed(val)
	
	def render(self, mode="rgb_array", close=False):
		if mode != "rgb_array":
			return np.array([])
		base_pos = [0,0,0]
		_cam_dist = 2.5  #.3
		_cam_yaw = 50
		_cam_pitch = -35
		_render_width=240
		_render_height=240

		view_matrix = pb.computeViewMatrixFromYawPitchRoll(
			cameraTargetPosition=base_pos,
			distance=_cam_dist,
			yaw=_cam_yaw,
			pitch=_cam_pitch,
			roll=0,
			upAxisIndex=2)
		proj_matrix = pb.computeProjectionMatrixFOV(
			fov=90, aspect=float(_render_width)/_render_height,
			nearVal=0.01, farVal=100.0)
		(_, _, px, _, _) = pb.getCameraImage(
			width=_render_width, height=_render_height, viewMatrix=view_matrix,
			projectionMatrix=proj_matrix, renderer=pb.ER_TINY_RENDERER) #ER_BULLET_HARDWARE_OPENGL)
		rgb_array = np.array(px, dtype=np.uint8)
		rgb_array = np.reshape(rgb_array, (_render_height, _render_width, 4))
		rgb_array = rgb_array[:, :, :3]
		return rgb_array

	def record_video(self, args, policy, seed, shared_constants, filename):
		self.seed(seed + 100)
		state, done = self.reset(), False
		images = []
		images_scene = []
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = self.step(action)
			im = Image.fromarray((state[0,:,:,-1]*255).astype(np.uint8))
			images.append(im.resize((240,240)).convert('P'))
			scene_img = self.render()
			scene_img = Image.fromarray(scene_img)
			images_scene.append(scene_img)

		images[0].save(filename+'/'+str(time.time())+'-.gif', save_all=True, append_images=images[1:], optimize=False, duration=20, loop=0)
		images_scene[0].save(filename+'/'+str(time.time())+'-scene.gif', save_all=True, append_images=images_scene[1:], optimize=False, duration=20, loop=0)


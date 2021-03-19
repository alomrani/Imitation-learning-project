## Advice-based Conservative Q-Learning (Adv-CQL)

Karush Suri, Mhd Ali Alomrani, Reza Moravej  

This repository contains the original implementation for _No Imitation for Me: Advice-based Offline Reinforcement Learning for Aerial Control_. Code is based on the [Gym-Pybullet-Drones](https://github.com/utiasDSL/gym-pybullet-drones) framework.  

## Requirements
Our implementation is written using _Python 3.7_ and tested on _Ubuntu 18.04_ using _PyTorch_. Use the following command to setup the required dependencies-  

```
setup.sh
```

## Usage

Adv-CQL resides in the [`ADV_CQL.py`](experiments/learning/algos/ADV.py) file. The [`singleagent.py`](experiments/learning/singleagent.py) file runs experiments and saves results in the [`results`](experiments/learning/results/) folder.  

To run an `ADV_CQL` agent on the `takeoff` task with `kin` states use the following-

```
python singleagent.py --configs ADV_CQL --env takeoff --obs kin
```

This will train the agent for `2e5` timesteps. Default settings train an `SAC` agent on the `hover` task with `kin` feature inputs as per the following-

```
python singleagent.py
```

RL agent implementations can be found in the [`algos`](experiments/learning/algos/) folder with their configurations in the [`config`](experiments/learning/configs/) folder. Additionally, expert weights can be found in the [`experts`](experiments/learning/experts/) folder.  

## Development 

So what is a good place to start your work? Have a look at the following-  

|Directory|Description|Use Case|
|:-------:|:---------:|:------:|
|[`algos`](experiments/learning/algos/)|RL agents|Design custom agents|
|[`configs`](experiments/learning/configs/)|Agent configs|Set/tune custom configs|
|[`experts`](experiments/learning/experts/)|Expert weights|Add new experts|
|[`single_agent_rl`](gym_pybullet_drones/envs/single_agent_rl/)|Drone control envs|Add custom envs|



## Advice-based Conservative Q-Learning (Adv-CQL)

This repository contains the original implementation for _No Imitation for Me: Advice-based Offline Reinforcement Learning for Aerial Control_. Code is based on the [Gym-Pybullet-Drones](https://github.com/utiasDSL/gym-pybullet-drones) framework.  

## Requirements
Our implementation is written using _Python 3.7_ and tested on _Ubuntu 18.04_ using _PyTorch_. Use the following command to setup the required dependencies-  

```
setup.sh
```

## Usage

Adv-CQL resides in the [`ADV_CQL.py`](experiments/learning/algos/ADV.py) file. The [`singleagent.py`](experiments/learning/singleagent.py) file runs experiments and saves results in the ['results'](experiments/learning/results/) folder.  

To run an `ADV_CQL` agent on the `takeoff` task with `kin` states use the following-

```
python singleagent.py --configs ADV_CQL --env takeoff --obs kin
```

This will train the agent for `2e5` timesteps. Default settings train an `SAC` agent on the `hover` task with `kin` feature inputs as per the following-

```
python singleagent.py
```

Custom implementations can be trained using config files in their respective directories in the `config` folder.  


## Development 

So what is a good place to start your work? Have a look at the following-  

* `algos`- Follow a similar line of coding as in the `algos` folder as this will lead to easier integration and faster progress.  
* `configs`- Make sure that your arguments are clean and tuned. A `configs.yaml` is a great way to tune your parameters.
* New files- Incase you wish to make a new file for your code, then please do so in the `algos` folder. This will keep the directory consistent.  






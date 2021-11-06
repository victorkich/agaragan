#! /usr/bin/env python3

import rospy
import comet_ml
from utils.utils import OUNoise, ReplayBuffer, random_crop
from tensorboardX import SummaryWriter
from algorithms.curl import CurlSAC
from algorithms.rigan import RiGAN
from torchvision import transforms
from utils.logger import Logger
from itertools import count
import gym_turtlebot3
from tqdm import tqdm
import numpy as np
import algorithms
import argparse
import torch
import time
import yaml
import gym
import os

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', default='SAC', type=str, help='Select the algorithm: DDPG or SAC(default).')
args = parser.parse_args()

path = os.path.dirname(os.path.abspath(__file__))
with open(path + '/config.yml', 'r') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)  # Loading configs from config.yaml

print('Using:', config['device'])
os.system('gnome-terminal --tab --working-directory=WORK_DIR -- zsh -c "roslaunch turtlebot3_gazebo '
          'turtlebot3_stage_{}.launch"'.format(config['ros_environment']))
print('Waiting ROS environment starts...')
time.sleep(5)

rospy.init_node('rlccgan')
env = gym.make(config['env_name'])

# Create directory for saved models and Logger
save_dir = path + '/saved_models'
logger = Logger(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if config['seed']:
    env.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_low = [-1.5, -0.1]
action_high = [1.5, 0.12]
obs_shape = (3, config['image_size'], config['image_size'])
# pre_aug_obs_shape = (3, 100, 100)
max_action = float(env.action_space.high[0])

# Create a memory replay
replay_buffer = ReplayBuffer(obs_shape=obs_shape, action_shape=action_dim, config=config)

# Image transformations
transform = [
    transforms.ToPILImage(),
    transforms.Resize((obs_shape[1], obs_shape[2])),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform = transforms.Compose(transform)

if __name__ == '__main__':
    # Create environment
    ou_noise = OUNoise(dim=action_dim, low=action_low, high=action_high)
    ou_noise.reset()

    os.environ['COMET_API_KEY'] = config['api_key']
    comet_ml.init(project_name=config['project_name'])
    writer = SummaryWriter(comet_config={"disabled": True if config['disabled'] else False})
    writer.add_hparams(hparam_dict=config, metric_dict={})

    assert args.algorithm == 'SAC' or args.algorithm == 'DDPG'  # Available algorithms: DDPG, SAC.
    agent = CurlSAC(obs_shape=obs_shape, action_shape=action_dim, config=config, save_dir=save_dir, writer=writer,
                    algorithm=args.algorithm)
    if config['rigan']:
        rigan = RiGAN(obs_shape, config, path, writer=writer)

    ep_r = 0
    if not config['train']:
        agent.load()
        if config['rigan']:
            rigan.load()

        for i in tqdm(range(config['num_eps_test'])):
            state = env.reset()
            for t in count():
                state = transform(state).unsqueeze(0)
                if config['rigan']:
                    state = rigan.forward(state)
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                if done or t >= config['max_ep_length']:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state

    elif config['train']:
        if config['load']:
            agent.load()
            if config['rigan']:
                rigan.load()

        total_step = 0
        for i in tqdm(range(config['num_eps_train'])):
            total_reward = 0
            step = 0
            state = env.reset(new_random_goals=True)
            state = transform(state).unsqueeze(0)
            ep_start_time = time.time()
            for t in count():
                if config['rigan']:
                    pos = rigan.forward(state)
                else:
                    pos = random_crop(state, config['image_size'])
                    state = random_crop(state, config['image_size'])
                action = agent.select_action(pos)
                action = ou_noise.get_action(action, t)

                next_state, reward, done, info = env.step(action)
                next_state = transform(next_state).unsqueeze(0)
                pos = pos.cpu().detach().numpy()
                state = state.cpu().detach().numpy()
                replay_buffer.add(state, action, reward, next_state, np.float(done), pos)

                state = next_state
                step += 1
                total_reward += reward

                # if t >= 256:
                loss_rl = agent.update(replay_buffer=replay_buffer, step=step)
                if config['rigan']:
                    rigan.update(agent, loss_rl, state)

                if done or t >= config['max_ep_length']:
                    break

            total_step += step
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))

            # Log metrics
            writer.add_scalars(main_tag="agent", tag_scalar_dict={"reward": total_reward,
                                                                  "episode_timing": time.time() - ep_start_time,
                                                                  "total_step": total_step}, global_step=i)

            if i % config['num_episode_save'] == 0:
                agent.save()
                if config['rigan']:
                    rigan.save()

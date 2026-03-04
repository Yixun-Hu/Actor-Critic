'''
    File name: a2c_train.py
    Author: Jayson Ng
    Email: iamjaysonph@gmail.com
    Date created: 15/7/2021
    Python Version: 3.7
'''

import torch
import numpy as np
from tqdm import tqdm
from model import A2CNetwork
from agent import TDAgent
from utils import save_training_curves, save_policy_gif

import gym
from pathlib import Path


if __name__ == '__main__':
    env = gym.make('CartPole-v1', new_step_api=True)

    HIDDEN_LAYER = 128  # NN hidden layer size

    # Hyper-parameters
    log_intv = 10
    capacity = 10000
    target_update_intv = 500  # in terms of iterations
    max_episodes = 5000
    max_steps = 500
    lr = 0.008
    discount_factor = 0.99
    batch_size = 256
    goal = 200

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    net = A2CNetwork(env.observation_space.high.shape[0], env.action_space.n, hid_size=HIDDEN_LAYER).to(device)
    agent = TDAgent(net, capacity, env.action_space.n, batch_size, discount_factor, lr, target_update_intv)

    losses = []
    reward_hist = []
    avg_reward_hist = []
    avg_reward = 8
    best_avg_reward = 0
    for episode_i in tqdm(range(max_episodes)):
        s, _ = env.reset(return_info=True)

        if len(reward_hist) >= 100 and np.mean(reward_hist[-100:]) >= goal:  # benchmark of cartpole-v0 problem
            print(f'Solved! Average Reward reaches {goal} over the past 100 runs')
            break
        ep_reward = 0
        ep_loss = 0
        loss = None
        episode_done = False
        for si in range(max_steps):

            a, a_prob = agent.select_action(s)
            new_s, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            if done:
                r = -1

            agent.store_transition(s, a, new_s, r, a_prob)

            if done:
                episode_done = True
                reward_hist.append(ep_reward)
                break

            s = new_s
            ep_reward += 1

            loss = agent.learn()
            if loss is not None:
                ep_loss += loss

        if not episode_done:
            reward_hist.append(ep_reward)

        losses.append(ep_loss/(si+1))
        avg_reward = int(0.95 * avg_reward + 0.05 * ep_reward)
        avg_reward_hist.append(avg_reward)

        if episode_i % log_intv == 0:
            print(f'Episode {episode_i} | Reward: {ep_reward} | Avg Reward: {avg_reward} | Loss: {loss}')

    env.close()

    output_dir = Path(__file__).resolve().parent
    plot_path = output_dir / 'td_a2c_training_curves.png'
    gif_path = output_dir / 'td_a2c_cartpole_v1.gif'

    save_training_curves(reward_hist, losses, plot_path)
    print(f'Training curves saved to: {plot_path}')

    saved_frames = save_policy_gif(agent.net, agent.device, gif_path, max_steps=max_steps, fps=50)
    if saved_frames > 0:
        print(f'Evaluation GIF saved to: {gif_path}')
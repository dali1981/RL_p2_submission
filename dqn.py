from collections import namedtuple, deque

import torch

import Environment
from dqn_agent import Agent
import numpy as np

Params = namedtuple("Params", "n_episodes max_t eps_start eps_end eps_decay")


def dqn(agent: Agent, params: Params):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = params.eps_start  # initialize epsilon
    for i_episode in range(1, params.n_episodes + 1):
        agent.init_episode()
        score = 0  # initialize the score
        for t in range(params.max_t):
            agent.act(eps)  # to be defined in the agent
            agent.step()    # to be defined in the agent
            score += agent.get_reward()
            if agent.get_done():
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(params.eps_end, params.eps_decay * eps)  # decrease epsilon
        # print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

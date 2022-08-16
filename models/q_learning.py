
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../envs'))
from causal_env_v0 import CausalEnv_v0
from math import comb
from itertools import combinations
import numpy as np

import tqdm
import gym
import numpy as np
import itertools
import random
import argparse

from gym import spaces

from typing import Dict, Any, Protocol, Tuple, List, Set
from collections import defaultdict


class History_Env(CausalEnv_v0):

    def __init__(self, env_config: Dict[str, Any]) -> None:
        super().__init__(env_config)
        self.actions = list(itertools.product([0,1],repeat=self._n_blickets))
        self.n_actions = len(self.actions)

    def reset(self) -> np.ndarray:
        self.history = np.array([])
        rtn = super().reset()
        return rtn


    def step(self, action: [int, np.ndarray]) -> Tuple[np.ndarray, float, bool, dict]:
        if type(action) == int:
            action = np.array(self.actions[action])
        observation, reward, done, info = super().step(action)
        self.history = np.concatenate([self.history, observation])
        return self.history, reward, done, info


class Q_learner():

    def __init__(self, env, test_env, alpha, discount, epsilon):
        self.env = env
        self.test_env = test_env
        self.alpha = alpha
        self.discount = discount
        self.epsilon = epsilon
        self.q = defaultdict(lambda: np.zeros(self.env.n_actions))
        self.training_steps = 0
        self.episodes = 0


    def train(self, episodes):

        converged = False

        with tqdm.tqdm(total=episodes) as pbar:
            for i in range(episodes):
                if converged:
                    break
                reward_ = 0

                obs = self.env.reset()
                self.episodes += 1
                while True:
                    self.training_steps += 1
                    action = self.act(obs)
                    next_obs, reward, done, info = self.env.step(int(action))
                    reward_ += reward
                    best_next_action = self.act(next_obs, False)
                    td_target = reward + self.discount * self.Q(next_obs)[best_next_action]
                    td_delta = td_target - self.Q(obs)[action]
                    self.Q(obs)[action] += self.alpha * td_delta
                    obs = next_obs
                    if done:
                        break

                if i% 20 == 0:
                    rewards = []
                    steps = []
                    for _ in range(100):
                        reward_ = 0
                        step_ = 0
                        obs = self.test_env.reset()
                        while True:
                            action = self.act(obs, False)
                            next_obs, reward, done, info = self.test_env.step(int(action))
                            reward_ += reward
                            step_ += 1
                            if done:
                                rewards.append(reward_)
                                steps.append(step_)
                                break
                    avg_reward = sum(rewards)/100
                    pbar.set_description_str('Training | Episode: {} | Reward: {} | Steps: {}'.format(self.episodes, avg_reward, sum(steps)/100))
                    if abs(avg_reward - 3.99) < 1e-5:
                        converged = True
                        return self.episodes, self.training_steps
                pbar.update(1)
        if not converged:
            print('Failed to converge')
            return self.episodes, self.training_steps

    def Q(self, obs):
        return self.q[tuple(obs)]


    def act(self, obs, explore=True):
        roll = random.random()
        if explore == False or roll > self.epsilon:
            q = self.Q(obs)
            max_ = np.max(q)
            max_ = np.where(q == max_)[0]
            action = random.choice(max_)
        else:
            action = random.randint(0,self.env.n_actions-1)
        return action


def main(args):

    env = History_Env({'reward_structure':'quiz'})
    test_env = History_Env({'reward_structure':'quiz', 'testing':True})

    episodes = []
    steps = []
    i = args.num
    for _ in range(i):
        learner = Q_learner(env, test_env, args.alpha, args.discount, args.epsilon)
        episode, step = learner.train(10000)
        episodes.append(episode)
        steps.append(step)
    print(f"Over {i} trials, the optimal strategy took on average {sum(episodes)/i} episodes and {sum(steps)/i} steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a q-learner')
    parser.add_argument('--num', type=int, default=100, help='Number of times to experiment')
    parser.add_argument('--alpha', type=float, default=0.95, help='Learning rate')
    parser.add_argument('--discount', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Eepsilon-greedy exploration rate')
    args = parser.parse_args()
    main(args)

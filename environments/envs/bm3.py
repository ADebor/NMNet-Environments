import gym
from gym import spaces
import pygame
import numpy as np
from random import uniform, choice
import copy

class Benchmark_3(gym.Env):

    metadata = {
        "render_modes": [
            "human",
        ],
        "render_fps": 25,
    }

    def __init__(self, render_mode=None):

        # spaces
        self.observation_space = spaces.Box(
            low=-2.5,
            high=2.5,
            shape=(4,),
        )
        self.action_space = spaces.Box(low=-np.inf, high=np.inf)

        # rewards
        rew_pos = 100
        rew_neg = -50
        rew_mid = 0.5 * (rew_pos + rew_neg)
        rew_mid + (rew_pos - rew_mid) * self._alpha[4]
        self._dist_1 = np.inf
        self._dist_2 = np.inf
        self._reward_dict = {
            (1, 0): rew_mid
            + (rew_pos - rew_mid) * self._alpha[4],  # if +1 : r=100, if -1 : r=-50
            (0, 1): rew_mid
            - (rew_pos - rew_mid) * self._alpha[4],  # if -1 : r=100, if +1 : r=-50
            (0, 0): 0,
            (1, 1): self._rew_closest(rew_pos, rew_mid),
        }

        # render
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _rew_closest(
        self,
        rew_pos,
        rew_mid,
    ):
        if self._dist_1 <= self._dist_2:
            return rew_mid + (rew_pos - rew_mid) * self._alpha[4]
        else:
            return rew_mid - (rew_pos - rew_mid) * self._alpha[4]

    def _get_obs(self):
        obs = copy.copy(self._x)
        return obs

    def _get_info(self):
        return None

    def reset(self):

        # draw MDP in D
        self._alpha = None
        self._draw_alpha()

        # init pos
        self._size = [-2.0, 2.0]  # size of the space
        self._u_x = uniform(-1.5, 1.5)
        self._u_y = uniform(-1.5, 1.5)
        self._x = None
        self._compute_pos()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        is_target = self._is_target()
        if is_target[0] or is_target[0]:
            self._u_x = uniform(-1.5, 1.5)
            self._u_x = uniform(-1.5, 1.5)
        else:
            self._u_x += 0.25 * np.sin(action * np.pi)
            self._u_y += 0.25 * np.cos(action * np.pi)

        self._boundaries()
        self._compute_pos()

        observation = self._get_obs()
        reward = self._compute_reward(is_target)
        done = False
        info = self._get_info()

        return observation, reward, done, info

    def _compute_reward(self, is_target):
        return self._reward_dict[is_target]

    def _compute_pos(self):
        self._x = [
            self._alpha[0] - self._u_x,
            self._alpha[1] - self._u_y,
            self._alpha[2] - self._u_x,
            self._alpha[3] - self._u_y,
        ]

    def _draw_alpha(self):
        self._alpha = (
            uniform(-1.0, 1.0),
            uniform(-1.0, 1.0),
            uniform(-1.0, 1.0),
            uniform(-1.0, 1.0),
            choice([-1, 1]),
        )

    def _is_target(self):
        self._dist_1 = np.sqrt(
            (self._u_x - self._alpha[0]) ** 2 + (self._u_y - self._alpha[1]) ** 2
        )
        self._dist_2 = np.sqrt(
            (self._u_x - self._alpha[2]) ** 2 + (self._u_y - self._alpha[3]) ** 2
        )
        return int(self._dist_1 <= 0.4), int(self._dist_2 <= 0.4)

    def _boundaries(self):
        if self._u_x > self._size[1]:
            self._u_x -= 4
        if self._u_x < self._size[0]:
            self._u_x += 4
        if self._u_y > self._size[1]:
            self._u_y -= 4
        if self._u_y < self._size[0]:
            self._u_y += 4
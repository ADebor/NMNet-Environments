import gym
from gym import spaces
import pygame
import numpy as np
from random import uniform, choice
import copy

class MultipleReferencesEnv(gym.Env):

    metadata = {
        "render_modes": [
            "human",
        ],
        "render_fps": 25,
    }

    def __init__(self, render_mode=None):

        # spaces
        self.observation_space = spaces.Box(
            low=-3.0,
            high=3.0,
            shape=(2,),
        )
        self.action_space = spaces.Box(low=-np.inf, high=np.inf)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

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
        if self._is_target():
            reward = 100  # TODO: weird but defined like this (reward depends on t and not t+1)
            self._u_x = uniform(-1.5, 1.5)
            self._u_x = uniform(-1.5, 1.5)
        else:
            reward = -2
            n = uniform(-np.pi / 4, np.pi / 4)
            self._u_x += 0.25 * (np.sin(action) + np.sin(self._alpha[2] + n))
            self._u_y += 0.25 * (np.cos(action) + np.cos(self._alpha[2] + n))

        self._boundaries()
        self._compute_pos()

        observation = self._get_obs()
        # reward = self._compute_reward()
        done = False
        info = self._get_obs()

        return observation, reward, done, info

    def _compute_pos(self):
        self._x = [self._alpha[0] - self._u_x, self._alpha[1] - self._u_y]

    def _draw_alpha(self):
        self._alpha = uniform(-1.0, 1.0), uniform(-1.0, 1.0), uniform(np.pi, np.pi)

    def _is_target(self):
        dist = np.sqrt(
            (self._u_x - self._alpha[0]) ** 2 + (self._u_y - self._alpha[1]) ** 2
        )
        return dist <= 0.4

    def _boundaries(self):
        if self._u_x > self._size[1]:
            self._u_x -= 4
        if self._u_x < self._size[0]:
            self._u_x += 4
        if self._u_y > self._size[1]:
            self._u_y -= 4
        if self._u_y < self._size[0]:
            self._u_y += 4

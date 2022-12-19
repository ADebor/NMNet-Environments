import gym
from gym import spaces
import pygame
import numpy as np
from random import uniform, choice
import copy

class MovingTargetEnv(gym.Env):

    metadata = {
        "render_modes": [
            "human",
        ],
        "render_fps": 25,
    }

    def __init__(self, render_mode=None, alpha=None):

        # rendering specs
        self.window_size = 512
        
        # env specs
        self._alpha = 0.0
        if alpha:
            self._alpha = alpha
        else:
            self._draw_alpha()
            
        self.pos_dom = [-5.0 - self._alpha, 5.0 - self._alpha]

        self._p = 0.0  # ground truth (true position)
        self._x = 0.0  # biased position
        self._r = 0.0  # reward

        # info
        self._dist = 0.0

        # action space
        self.action_space = spaces.Box(low=-20, high=20)

        # observation space
        self.observation_space = spaces.Box(low=-5, high=5)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    ## Mendatory methods ##

    def _get_obs(self):
        self._compute_biased_pos()
        return {
            "x": self._x,
        }

    def _get_info(self):
        return {
            "distance": self._dist,
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._draw_pos()  # draw p
        self._draw_alpha()  # draw alpha
        self._compute_biased_pos()  # draw x

        observation = self._get_obs()

        # reset dist
        self._dist = 0.0
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        reward = self._compute_reward(action)
        self._update_true_pos()
        observation = self._get_obs()
        info = self._get_info()
        done = False  # no terminal state, position is resampled (adaptation) if close

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, info

    def render(self):
        super().render()  # to be implemented

    ## Helper methods ##

    def _draw_alpha(self):
        alpha = uniform(-10, 10)
        self._alpha = alpha
        print("in draw alpha", self._alpha)

    def _draw_pos(self):
        self._p = uniform(self.pos_dom[0], self.pos_dom[1])

    def _compute_biased_pos(self):
        self._x = self._p + self._alpha

    def _compute_reward(self, a):
        self._dist = np.abs(a - self._p)
        if self._dist < 1:
            self._r = 10
        else:
            self._r = -self._dist
        return self._r

    def _update_true_pos(self):
        if self._r > 0:
            self._draw_pos()

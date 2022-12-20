import gym
from gym import spaces
import pygame
import numpy as np
from random import uniform, choice
import copy
import time

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
        self.max_steps = 1400 #should be passed as argument (maybe)
        self.first_render = True
        self.dt = 0.1

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
        self._a = 0.0 # action taken

        # info
        self._dist = 0.0

        # action space
        self._a_low = -20
        self._a_high = 20
        self.action_space = spaces.Box(low=self._a_low, high=self._a_high)

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
        self._a = 0.0
        
        observation = self._get_obs()

        # reset dist
        self._dist = 0.0
        info = self._get_info()

        if self.render_mode == "human":
            #self._render_frame()
            self.render()
        return observation, info

    def step(self, action):
        self._a = action
        reward = self._compute_reward(action)
        self._update_true_pos()
        observation = self._get_obs()
        info = self._get_info()
        done = False  # no terminal state, position is resampled (adaptation) if close TODO: change with max_steps

        if self.render_mode == "human":
            self.render()
            #self._render_frame()

        return observation, reward, done, info

    def render(self):
        super().render()  # to be implemented

    ## Helper methods ##

    def _draw_alpha(self):
        alpha = uniform(-10, 10)
        self._alpha = alpha

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

    def render(self):
        rectangle_width = np.floor(self.window_size / self.max_steps)
        if self.first_render:
            self.first_render = False
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            self.previous_time = time.time()
        self.screen.fill((255,255,255))
        
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        textsurface = myfont.render("MovingTarget", False, (0, 0, 0))
        self.screen.blit(textsurface, (int(self.window_size / 2), 0))

        middle_screen_height = int(self.window_size / 2)
        p, a, x = self._p, self._a, self._x
        print(a)
        pixel = int(((p + 0.5 * self._a_high) / self._a_high) * self.window_size) - 1
        points = [(pixel, middle_screen_height), (pixel, middle_screen_height + 100)]
        pygame.draw.lines(self.screen, (0, 255, 0), False, points, 5)
        
        pixel = int(((x + 0.5 * self._a_high) / self._a_high) * self.window_size) - 1
        points = [(pixel, middle_screen_height), (pixel, middle_screen_height + 100)]
        pygame.draw.lines(self.screen, (0, 0, 255), False, points, 5)
        
        pixel = int(((x + 0.5 * self._a_high + 1.0) / self._a_high) * self.window_size) - 1
        points = [(pixel, middle_screen_height), (pixel, middle_screen_height + 100)]
        pygame.draw.lines(self.screen, (0, 0, 0), False, points, 5)
        
        pixel = int(((x + 0.5 * self._a_high - 1.0) / self._a_high) * self.window_size) - 1
        points = [(pixel, middle_screen_height), (pixel, middle_screen_height + 100)]
        pygame.draw.lines(self.screen, (0, 0, 0), False, points, 5)
        
        pixel = int(((a + 0.5 * self._a_high) / self._a_high) * self.window_size)
        points = [(pixel, middle_screen_height), (pixel, middle_screen_height - 100)]
        pygame.draw.lines(self.screen, (255, 0, 0), False, points, 5)


        cur_time = time.time()
        pygame.display.update()
        for event in pygame.event.get():
            pass
        while not (cur_time - self.previous_time >= self.dt):
            cur_time = time.time()
        self.previous_time = cur_time
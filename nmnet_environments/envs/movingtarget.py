import gym
import pygame
import numpy as np
import random

class CustomEnv(gym.Env):
    def __init__(self):
        # Define the action and observation spaces
        self.action_space = gym.spaces.Box(low=-20, high=20, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-5, high=5, shape=(1,))

        # Set the initial state and target position
        self.state = np.zeros(1)
        self.target_pos = 0
        self.alpha = 0
        
        # Set the rendering parameters
        self.bar_length = 10
        self.arrow_size = 20
        self.window_size = (self.arrow_size, self.arrow_size * self.bar_length)
        self.window = pygame.display.set_mode(self.window_size)
        
    def reset(self):
        # Reset the state and target position
        self.state = np.zeros(1)
        self.target_pos = 0
        
        # Draw a new value for alpha
        self.alpha = random.uniform(-10, 10)
        
        return self.state
    
    def step(self, action):
        # Compute the reward
        reward = 10 if abs(action - self.target_pos) < 1 else -abs(action - self.target_pos)
        
        # Update the target position
        if reward > 0:
            self.target_pos = random.uniform(-5 - self.alpha, 5 - self.alpha)
        else:
            self.target_pos = self.target_pos
        
        # Update the state
        self.state = self.target_pos + self.alpha
        
        return self.state, reward, False, {}
    
    def render(self, mode='human'):
        # Clear the window
        self.window.fill((255, 255, 255))

        # Compute the positions of the arrows
        target_pos = int((self.target_pos + 5) / (self.bar_length / 2) * self.bar_length)
        action_pos = int((self.state + 5) / (self.bar_length / 2) * self.bar_length)
        state_pos = int((self.state + 5) / (self.bar_length / 2) * self.bar_length)

        # Draw the arrows
        pygame.draw.polygon(self.window, (255, 0, 0), [(0, target_pos * self.arrow_size), (self.arrow_size // 2, target_pos * self.arrow_size + self.arrow_size // 2), (self.arrow_size, target_pos * self.arrow_size)])
        pygame.draw.polygon(self.window, (0, 255, 0), [(0, action_pos * self.arrow_size), (self.arrow_size // 2, action_pos * self.arrow_size + self.arrow_size // 2), (self.arrow_size, action_pos * self.arrow_size)])
        pygame.draw.polygon(self.window, (0, 0, 255), [(0, state_pos * self.arrow_size), (self.arrow_size // 2, state_pos * self.arrow_size + self.arrow_size // 2), (self.arrow_size, state_pos * self.arrow_size)])

        # Update the display
        pygame.display.update()


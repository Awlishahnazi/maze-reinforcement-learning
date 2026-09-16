import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MazeEnv(gym.Env):
    
    def __init__(self, grid_size=10, walls=None, start=None, goal=None):
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = 200
        
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Discrete(grid_size * grid_size)
        
        if walls is None:
            self.walls = {(3,3), (3,4), (3,5), (5,2), (5,3), (5,4), (6,6)}
        else:
            self.walls = walls
        
        self.start = start if start else (0, 0)
        self.goal = goal if goal else (grid_size-1, grid_size-1)
        
        self.agent_pos = None
        self.steps = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = list(self.start)
        self.steps = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        return self.agent_pos[0] * self.grid_size + self.agent_pos[1]
    
    def step(self, action):
        self.steps += 1
        
        moves = {0: (-1,0), 1: (0,1), 2: (1,0), 3: (0,-1)}
        dr, dc = moves[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        
        if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size 
            and (nr, nc) not in self.walls):
            self.agent_pos = [nr, nc]
        
        terminated = (tuple(self.agent_pos) == self.goal)
        truncated = self.steps >= self.max_steps
        
        if terminated:
            reward = 1.0
        else:
            
            dist = abs(self.agent_pos[0] - self.goal[0]) + abs(self.agent_pos[1] - self.goal[1])
            reward = -0.01 - 0.001 * dist
        
        return self._get_obs(), reward, terminated, truncated, {}
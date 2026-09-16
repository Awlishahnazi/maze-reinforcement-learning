import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
    
    def choose_action(self, state: int) -> int:

        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        else:
            return int(np.argmax(self.q_table[state]))
    
    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        
        if done:
            target = reward
            
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state][action] += self.lr * (target - current_q)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_policy(self, n_states):
        policy = {}
        for s in range(n_states):
            policy[s] = int(np.argmax(self.q_table[s]))
        return policy
    
    def get_q_values_grid(self, grid_size):
        q_grid = np.zeros((grid_size, grid_size))
        for s in range(grid_size * grid_size):
            q_grid[s // grid_size, s % grid_size] = np.max(self.q_table[s])
        return q_grid
# test_env.py
from maze_env import MazeEnv
import random

env = MazeEnv(grid_size=10)
obs, _ = env.reset()

for step in range(20):
    action = random.randint(0, 3)
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"Step {step}: action = {action}, pos = {env.agent_pos}, reward = {reward:.3f}")
    
    if terminated:
        print("Goal Reached")
        break
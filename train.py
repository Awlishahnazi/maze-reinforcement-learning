import numpy as np
from maze_env import MazeEnv
from q_learning import QLearningAgent
import matplotlib.pyplot as plt

def train(episodes=2000, grid_size=10, verbose=True):
    env = MazeEnv(grid_size=grid_size)
    agent = QLearningAgent(
        n_states=grid_size * grid_size,
        n_actions=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
    )
    
    rewards_history = []
    steps_history = []
    success_count = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
        if terminated:
            success_count += 1
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            success_rate = sum(1 for s in steps_history[-100:] if s < env.max_steps) / 100
            print(f"Episode {episode+1}: "
                  f"Avg Reward = {avg_reward:.3f}, "
                  f"Epsilon = {agent.epsilon:.3f}, "
                  f"Success Rate (last 100) = {success_rate:.0%}")
    
    return agent, env, rewards_history, steps_history


if __name__ == "__main__":
    print("Starting Q-Learning training...")
    agent, env, rewards, steps = train(episodes=2000)
    
    print("\nTraining complete!")
    print(f"Best reward: {max(rewards):.3f}")
    print(f"Average of last 100 episodes: {np.mean(rewards[-100:]):.3f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
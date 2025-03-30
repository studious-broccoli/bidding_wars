import gym
from gym import spaces
import numpy as np
import random
from stable_baselines3 import PPO

# Define a custom Gym environment for the bidding problem.
class BiddingEnv(gym.Env):
    """
    Custom Environment for RL Bidding Agent.
    
    - State: Simulated viewer metadata, time, and history as a feature vector.
    - Action: Discrete bid/ad choices (0: no bid, 1: low bid, 2: medium bid, 3: high bid).
    - Reward: Engagement metric (simulated as a function of the action).
    """
    def __init__(self):
        super(BiddingEnv, self).__init__()
        # Observation space: for example, 5 continuous features representing metadata.
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        # Action space: 4 discrete actions corresponding to different bid levels.
        self.action_space = spaces.Discrete(4)
        self.current_step = 0
        self.max_steps = 100  # Episode length

    def reset(self):
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        # Simulate state as a random feature vector.
        return np.random.rand(5).astype(np.float32)

    def step(self, action):
        self.current_step += 1

        # Simulate reward: if no bid (action==0) no engagement, else engagement scales with bid level.
        if action == 0:
            reward = 0.0
        else:
            # Higher bids may yield higher engagement (but in real systems, the function would be more complex)
            reward = random.uniform(0, 1) * action

        done = self.current_step >= self.max_steps
        info = {}
        return self._get_obs(), reward, done, info

# Simple contextual bandit agent using an epsilon-greedy policy.
class EpsilonGreedyAgent:
    def __init__(self, n_actions, epsilon=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.q_values = np.zeros(n_actions)  # Estimated value for each action
        self.action_counts = np.zeros(n_actions)

    def select_action(self, state):
        # With probability epsilon, explore; otherwise, exploit.
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return int(np.argmax(self.q_values))

    def update(self, action, reward):
        self.action_counts[action] += 1
        alpha = 1.0 / self.action_counts[action]  # Learning rate update
        self.q_values[action] += alpha * (reward - self.q_values[action])

# Training loop for the contextual bandit agent.
def train_contextual_bandit(env, episodes=500):
    agent = EpsilonGreedyAgent(n_actions=env.action_space.n, epsilon=0.1)
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(action, reward)
            total_reward += reward
            state = next_state
        rewards.append(total_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward:.2f}")
    return agent, rewards

# Training loop for the PPO agent for scaling to policy gradient methods.
def train_ppo(env, timesteps=5000):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model

if __name__ == "__main__":
    env = BiddingEnv()

    # 1. Offline RL: Train a simple contextual bandit as a baseline using offline data.
    #    (In practice, one would use historical bid data for offline training.)
    print("Training contextual bandit agent (offline RL)...")
    bandit_agent, bandit_rewards = train_contextual_bandit(env, episodes=500)

    # 2. Scaling to Policy Gradient: Use PPO to account for delayed/sparse rewards and for budget pacing.
    print("Training PPO agent (policy gradient for online rollout)...")
    ppo_model = train_ppo(env, timesteps=5000)

    # Example online rollout using the trained PPO model.
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = ppo_model.predict(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print("Total reward from online rollout:", total_reward)

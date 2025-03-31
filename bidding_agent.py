"""
    Script: bidding_agent.py
    Description: (1) Create Gym Environment simulating a bidding process
                 (2) Metadata [viewer_age, viewer_gender, time_of_day, history, bid, reward] is simulated and
                     artificially correlated in simulate_data.py
                 (3) There are 4 discrete actions / bids randomly chosen
                 (4) Reward is artificially correlated to metadata to get something learnable
                 (5) Train simple contextual bandit with epsilon‐greedy approach (offline RL)
                 (6) Then train with policy gradient method (PPO from stable-baselines3) for budget pacing
                     and handling delayed/sparse rewards
"""
import pdb
import gym
from gym import spaces
import random
from stable_baselines3 import PPO
import pandas as pd
import numpy as np
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from simulate_data import create_simulated
from plotter import *
from utils import test_oracle_policy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# export LC_ALL="en_US.UTF-8"
# export LANG="en_US.UTF-8"


# ------------------------------------------------------------------------------------------------------------
# Simulated Data
# Older users respond better to high bids
# Certain times of day have higher engagement
# Users with higher "history" respond more
# Bids matter, but only in the right context
# ------------------------------------------------------------------------------------------------------------
df = create_simulated('simulated_bid_data.csv')
print(df.head())
print(df.describe())


# ------------------------------------------------------------------------------------------------------------
# Define a custom Gym environment
# ------------------------------------------------------------------------------------------------------------
class BiddingEnv(gym.Env):
    def __init__(self, data):
        super(BiddingEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.data = data.reset_index(drop=True)
        self.num_samples = len(self.data)
        self.current_index = 0
        self.max_steps = 100
        self.state_embeddings = []  # Track states

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self):
        self.start_index = random.randint(0, self.num_samples - self.max_steps)
        self.current_index = self.start_index
        self.steps_taken = 0
        self.state_embeddings = []
        self.labels = {"time_of_day": [], "bid": []}
        return self._get_obs()

    def _get_obs(self):
        row = self.data.iloc[self.current_index]
        obs = np.array([
            row['viewer_age'] / 100.0,
            {'male': 0, 'female': 1, 'non-binary': 2}[row['viewer_gender']] / 2.0,
            ['morning', 'afternoon', 'evening', 'night'].index(row['time_of_day']) / 3.0,
            row['history'] / 20.0
        ], dtype=np.float32)

        # Track embeddings and labels
        self.state_embeddings.append(obs.tolist())
        self.labels["time_of_day"].append(row['time_of_day'])
        self.labels["bid"].append(row['bid'])
        return obs

    def step(self, action):
        # We are going to make it so that the reward is correlated to
        # age, history, time, and gender

        # Use agent's action as the actual bid
        bid = int(action)

        # Grab the current data
        row = self.data.iloc[self.current_index]

        # Determine normalized demographics to be correlated
        age_factor = (row['viewer_age'] - 18) / 52
        history_factor = row['history'] / 20
        time_factor = ['morning', 'afternoon', 'evening', 'night'].index(row['time_of_day']) / 3
        gender_factor = {'male': 0, 'female': 1, 'non-binary': 2}[row['viewer_gender']] / 2

        # Create mapping function
        context_score = (0.3 * age_factor +
                         0.3 * history_factor +
                         0.2 * time_factor +
                         0.2 * gender_factor) * 3

        # Add noise to not overfit
        noise = np.random.normal(loc=0.0, scale=0.05)

        # Calculate reward, which will be a function of the
        # encoded demographics plus noise
        reward = 1.0 - abs(bid - context_score) / 3
        reward += noise
        reward = np.clip(reward, 0.0, 1.0)

        self.current_index = (self.current_index + 1) % self.num_samples
        # done = (self.current_index == 0) or (self.current_index >= self.max_steps)
        self.steps_taken += 1
        done = self.steps_taken >= self.max_steps
        return self._get_obs(), reward, done, {}

    def get_batch(self, batch_size=32):
        indices = np.random.choice(self.num_samples, size=batch_size, replace=False)
        batch = self.data.iloc[indices]
        return batch


# ------------------------------------------------------------------------------------------------------------
# EpsilonGreedyAgent: Non-contextual bandit (scalar lookup)
# ------------------------------------------------------------------------------------------------------------
class EpsilonGreedyAgent:
    def __init__(self, n_actions, epsilon=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.q_values = np.zeros(n_actions)  # single scalar Q-value per action
        self.action_counts = np.zeros(n_actions)

    def select_action(self, state):
        # With probability epsilon, explore; otherwise, exploit.
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return int(np.argmax(self.q_values))

    def update(self, action, reward):
        self.action_counts[action] += 1
        # alpha = 1.0 / self.action_counts[action]  # Learning rate update
        # self.q_values[action] += alpha * (reward - self.q_values[action])
        self.q_values[action] += 0.1 * (reward - self.q_values[action])  # Updated using a simple average


# ------------------------------------------------------------------------------------------------------------
# ContextualBanditAgent: Contextual bandit (dot product with weights + bias)
# ------------------------------------------------------------------------------------------------------------
class ContextualBanditAgent:
    def __init__(self, n_actions, context_dim, epsilon=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.learning_rate = 0.1
        self.models = [np.zeros(context_dim) for _ in range(n_actions)]
        self.biases = [0.0 for _ in range(n_actions)]  # NEW

    def select_action(self, state):
        q_values = [np.dot(w, state) + b for w, b in zip(self.models, self.biases)]
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        return int(np.argmax(q_values))

    def update(self, state, action, reward):
        prediction = np.dot(self.models[action], state) + self.biases[action]
        error = reward - prediction
        self.models[action] += self.learning_rate * error * state
        self.biases[action] += self.learning_rate * error  # update bias too


# ------------------------------------------------------------------------------------------------------------
# Training loop for the contextual bandit agent.
# ------------------------------------------------------------------------------------------------------------
def train_contextual_bandit(env, episodes=500, log_path='bandit_logs.csv', agent_type="epsilon"):
    print("Epsilon-Greedy contextual bandit")
    if agent_type == "contextual":
        agent = ContextualBanditAgent(n_actions=env.action_space.n,
                                      context_dim=env.observation_space.shape[0], epsilon=0.1)
    else:
        agent = EpsilonGreedyAgent(n_actions=env.action_space.n, epsilon=0.1)

    rewards = []
    log_data = []

    all_embeddings = []
    all_time_of_day = []
    all_bids = []

    for episode in range(episodes):
        state = env.reset()
        state = state / np.linalg.norm(state)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.select_action(state)

            # simulates environments response
            """
                In reality, we would be requesting online response
                send_bid_request_to_ad_server(action)
                observe_user_response() → reward
            """
            next_state, reward, done, _ = env.step(action)
            next_state = next_state / (np.linalg.norm(next_state) + 1e-8)

            if agent_type == "contextual":
                agent.update(state, action, reward)
            else:
                agent.update(action, reward)

            total_reward += reward
            state = next_state
            steps += 1

        rewards.append(total_reward)
        log_data.append({
            "episode": episode,
            "reward": total_reward,
            "steps": steps
        })

        # Accumulate data
        all_embeddings.extend(env.state_embeddings)
        all_time_of_day.extend(env.labels["time_of_day"])
        all_bids.extend(env.labels["bid"])

        print(f"Episode {episode}: Total Reward = {total_reward:.2f}, Steps = {steps}")

        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward:.2f}")
            if agent_type == "contextual":
                print(f"Q-values: {[np.round(np.dot(w, state), 3) for w in agent.models]}")

    # Save logs
    keys = log_data[0].keys()
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(log_data)

    print(f"[✓] Training log saved to {log_path}")

    # Save all accumulated data
    np.save("state_embeddings.npy", np.array(all_embeddings))
    np.save("labels_time_of_day.npy", np.array(all_time_of_day))
    np.save("labels_bid.npy", np.array(all_bids))
    print(f"[✓] Saved {len(all_embeddings)} state embeddings across {episodes} episodes.")

    return agent, rewards


# ------------------------------------------------------------------------------------------------------------
# Training loop for the PPO agent for scaling to policy gradient methods.
# ------------------------------------------------------------------------------------------------------------
def train_ppo(env, timesteps=5000):
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)

    rewards = []
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(state)
        state, reward, done, _ = env.step(action)
        reward = float(reward)
        total_reward += reward
        rewards.append(total_reward)

        if len(rewards) % 100 == 0:
            print(f"Step {len(rewards)}: Avg Reward = {np.mean(rewards[-100:]):.2f}")

    print(f"Total PPO reward: {total_reward:.2f}")
    return model, rewards


# ------------------------------------------------------------------------------------------------------------
# Autoencoder (for plotting embeddings)
# ------------------------------------------------------------------------------------------------------------
class StateAutoencoder(nn.Module):
    def __init__(self, input_dim=5, encoding_dim=3):
        super(StateAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(embeddings_path='state_embeddings.npy', encoding_dim=3, epochs=50, batch_size=32):
    X = np.load(embeddings_path).astype(np.float32)
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val)), batch_size=batch_size)

    model = StateAutoencoder(input_dim=X.shape[1], encoding_dim=encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}")

    # Generate encoded embeddings
    model.eval()
    with torch.no_grad():
        all_encoded = model.encoder(torch.tensor(X)).numpy()

    np.save("encoded_states.npy", all_encoded)
    print("[*] Encoded embeddings saved as 'encoded_states.npy'")
    return all_encoded


# ------------------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("[*] Building Custom Gym environment")

    # (1) Train Contextual Bandit (Offline RL)
    bandit_env = BiddingEnv(df)
    print("Training contextual bandit agent (offline RL)...")
    bandit_agent, bandit_rewards = train_contextual_bandit(bandit_env, episodes=500)
    plot_reward_curve(bandit_rewards)
    test_oracle_policy(bandit_env)

    # (2)  Train Autoencoder
    encoded_states = train_autoencoder(embeddings_path='state_embeddings.npy')

    # (3) Visualize Encoded Embeddings
    visualize_embeddings(method="umap", embedding_file="encoded_states.npy")
    visualize_embeddings(method="umap", embedding_file="encoded_states.npy", label_type="time_of_day")
    visualize_embeddings(method="umap", embedding_file="encoded_states.npy", label_type="bid")

    # (4) Clustering on Encoded Embeddings
    cluster_embeddings(method="kmeans", embedding_file="encoded_states.npy", n_clusters=5)
    cluster_embeddings(method="dbscan", embedding_file="encoded_states.npy")

    # (5) PPO Agent (Online RL)
    from stable_baselines3.common.vec_env import DummyVecEnv
    ppo_env = DummyVecEnv([lambda: BiddingEnv(df)])
    print("Training PPO agent (policy gradient for online rollout)...")
    ppo_model, ppo_rewards = train_ppo(ppo_env, timesteps=20000)

    # (6) Compare Bandit vs PPO Rewards
    compare_rewards(bandit_rewards, ppo_rewards)

    # (7) Run PPO Online Rollout
    rollout_env = BiddingEnv(df)  # unwrapped env for rollout
    state = rollout_env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = ppo_model.predict(state)
        state, reward, done, _ = rollout_env.step(action)
        total_reward += reward

    print("Total reward from online PPO rollout:", total_reward)

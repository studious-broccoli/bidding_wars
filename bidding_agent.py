import pdb
import gym
from gym import spaces
import random
from stable_baselines3 import PPO
import pandas as pd
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from plotter import *



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
The code creates a custom Gym environment simulating a bidding process 
where the state includes “viewer metadata” (here simulated by a random feature vector), time, and history. 
The actions represent different bid/ad choices, and the reward is based on a simulated 
engagement metric (for example, view time or conversions). 

The code first trains a simple contextual bandit
using an epsilon‐greedy approach (offline RL) and then scales to a 
policy gradient method (using PPO from stable-baselines3) for budget pacing and handling delayed/sparse rewards.

export LC_ALL="en_US.UTF-8"
export LANG="en_US.UTF-8"

"""


# ------------------------------------------------------------------------------------------------------------
# Simulated Data
# 👵 Older users respond better to high bids
# 🌅 Certain times of day have higher engagement
# 📈 Users with higher "history" respond more
# 🧠 Bids matter, but only in the right context
# ------------------------------------------------------------------------------------------------------------
np.random.seed(42)
num_samples = 1000

# Viewer metadata
viewer_age = np.random.randint(18, 70, size=num_samples)
viewer_gender = np.random.choice(['male', 'female', 'non-binary'], size=num_samples)
time_of_day = np.random.choice(['morning', 'afternoon', 'evening', 'night'], size=num_samples)
history = np.random.randint(0, 20, size=num_samples)
bid = np.random.choice([0, 1, 2, 3], size=num_samples)

# Encoding helper
def encode_gender(g):
    return {'male': 0, 'female': 1, 'non-binary': 2}[g]

def encode_time(t):
    return {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}[t]

# Reward model with correlation
reward = []
for i in range(num_samples):
    base = 0.1

    # Positively correlated factors
    age_factor = (viewer_age[i] - 18) / 52  # normalized
    history_factor = history[i] / 20
    time_factor = encode_time(time_of_day[i]) / 3
    gender_factor = encode_gender(viewer_gender[i]) / 2

    # Add correlation: high bid + older age + active history + evening time
    signal = (
        0.4 * bid[i] / 3 +
        0.2 * age_factor +
        0.2 * history_factor +
        0.1 * time_factor +
        0.1 * gender_factor
    )

    noise = np.random.normal(loc=0.0, scale=0.05)
    reward_val = max(0.0, min(1.0, signal + noise))  # clamp to [0, 1]
    reward.append(reward_val)

# Build DataFrame
df = pd.DataFrame({
    'viewer_age': viewer_age,
    'viewer_gender': viewer_gender,
    'time_of_day': time_of_day,
    'history': history,
    'bid': bid,
    'reward': reward
})

csv_filename = 'simulated_bid_data.csv'
df.to_csv(csv_filename, index=False)

print(f"Simulated dataset saved as {csv_filename}")
print(df.head())
print(df.describe())

# ------------------------------------------------------------------------------------------------------------
# Define a custom Gym environment for the bidding problem.
# ------------------------------------------------------------------------------------------------------------
class BiddingEnv(gym.Env):
    def __init__(self, data):
        super(BiddingEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.data = data.reset_index(drop=True)
        self.num_samples = len(self.data)
        self.current_index = 0
        self.max_steps = 100
        self.state_embeddings = []  # Track states

    def reset(self):
        self.current_index = random.randint(0, self.num_samples - self.max_steps)
        self.state_embeddings = []
        self.labels = {"time_of_day": [], "bid": []}  # NEW
        return self._get_obs()

    def _get_obs(self):
        row = self.data.iloc[self.current_index]
        obs = np.array([
            row['viewer_age'] / 100.0,
            {'male': 0, 'female': 1, 'non-binary': 2}[row['viewer_gender']] / 2.0,
            ['morning', 'afternoon', 'evening', 'night'].index(row['time_of_day']) / 3.0,
            row['history'] / 20.0,
            row['bid'] / 3.0
        ], dtype=np.float32)

        # Track embeddings and labels
        self.state_embeddings.append(obs.tolist())
        self.labels["time_of_day"].append(row['time_of_day'])
        self.labels["bid"].append(row['bid'])
        return obs

    def step(self, action):
        reward = 0.0 if action == 0 else random.uniform(0, 1) * action
        self.current_index = (self.current_index + 1) % self.num_samples
        done = (self.current_index == 0) or (self.current_index >= self.max_steps)
        return self._get_obs(), reward, done, {}

    def get_batch(self, batch_size=32):
        indices = np.random.choice(self.num_samples, size=batch_size, replace=False)
        batch = self.data.iloc[indices]
        return batch



# ------------------------------------------------------------------------------------------------------------
# Simple contextual bandit agent using an epsilon-greedy policy.
# ------------------------------------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------------------------------------
# Training loop for the contextual bandit agent.
# ------------------------------------------------------------------------------------------------------------
def train_contextual_bandit(env, episodes=500, log_path='bandit_logs.csv'):
    print("🧠 Epsilon-Greedy contextual bandit")
    agent = EpsilonGreedyAgent(n_actions=env.action_space.n, epsilon=0.1)
    rewards = []
    log_data = []

    all_embeddings = []
    all_time_of_day = []
    all_bids = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
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

        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward:.2f}")

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
        total_reward += reward
        rewards.append(total_reward)

    print(f"Total PPO reward: {total_reward:.2f}")
    return model, rewards



# ------------------------------------------------------------------------------------------------------------
# Autoencoder
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

def train_autoencoder_pytorch(embeddings_path='state_embeddings.npy', encoding_dim=3, epochs=50, batch_size=32):
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

    np.save("encoded_states_pytorch.npy", all_encoded)
    print("[✓] Encoded embeddings saved as 'encoded_states_pytorch.npy'")
    return all_encoded


# ------------------------------------------------------------------------------------------------------------
# Main
# 🎯 Custom Gym environment
# 🧠 Epsilon-Greedy contextual bandit
# 🧬 State embedding extraction
# 🔥 PyTorch autoencoder
# 🌈 UMAP visualizations with color-coding
# 🔍 Clustering (KMeans + DBSCAN)
# 🦾 PPO agent training
# 📊 Reward comparison plots
# ------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("🎯 Building Custom Gym environment")
    # Initialize environment
    env = BiddingEnv(df)

    # === 1. Train Contextual Bandit ===
    print("Training contextual bandit agent (offline RL)...")
    bandit_agent, bandit_rewards = train_contextual_bandit(env, episodes=500)
    plot_reward_curve(bandit_rewards)

    # === 2. Save state embeddings and labels ===
    # np.save("state_embeddings.npy", np.array(env.state_embeddings))
    # np.save("labels_time_of_day.npy", np.array(env.labels["time_of_day"]))
    # np.save("labels_bid.npy", np.array(env.labels["bid"]))

    # === 3. Train Autoencoder (PyTorch) ===
    encoded_states = train_autoencoder_pytorch(embeddings_path='state_embeddings.npy')

    # === 4. Visualize Encoded Embeddings ===
    visualize_embeddings(method="umap", embedding_file="encoded_states_pytorch.npy")
    visualize_embeddings(method="umap", embedding_file="encoded_states_pytorch.npy", label_type="time_of_day")
    visualize_embeddings(method="umap", embedding_file="encoded_states_pytorch.npy", label_type="bid")

    # === 5. Clustering on Encoded Embeddings ===
    cluster_embeddings(method="kmeans", embedding_file="encoded_states_pytorch.npy", n_clusters=5)
    cluster_embeddings(method="dbscan", embedding_file="encoded_states_pytorch.npy")

    # === 6. PPO Training (Online RL) ===
    print("Training PPO agent (policy gradient for online rollout)...")
    ppo_model, ppo_rewards = train_ppo(env, timesteps=5000)

    # === 7. Compare Bandit vs PPO Rewards ===
    compare_rewards(bandit_rewards, ppo_rewards)

    # === 8. Run PPO Online Rollout ===
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = ppo_model.predict(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print("Total reward from online rollout:", total_reward)




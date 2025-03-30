import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
# ------------------------------------------------------------------------------------------------------------
# Plotting Function
# ------------------------------------------------------------------------------------------------------------
def plot_reward_curve(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Curve over Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reward_curve.png")
    plt.close('all')
    print("[✓] Reward curve saved as 'reward_curve.png'")


# def visualize_embeddings(method="umap", embedding_file="encoded_states_pytorch.npy", label_type="time_of_day"):
#     X = np.load(embedding_file)
#     if X.shape[0] < 5:
#         print(f"[⚠️] Not enough data to visualize embeddings: only {X.shape[0]} samples.")
#         return
#
#     labels = np.load(f"labels_{label_type}.npy", allow_pickle=True)
#     if len(labels) != len(X):
#         print(f"[⚠️] Label and embedding size mismatch: {len(labels)} vs {len(X)}")
#         return

def visualize_embeddings(method="umap", embedding_file="encoded_states_pytorch.npy", label_type="time_of_day"):
    X = np.load(embedding_file)
    if X.shape[0] < 5:
        print(f"[⚠️] Not enough data to visualize embeddings: only {X.shape[0]} samples.")
        return

    labels = np.load(f"labels_{label_type}.npy", allow_pickle=True)
    if len(labels) != len(X):
        print(f"[⚠️] Label and embedding size mismatch: {len(labels)} vs {len(X)}")
        return

    if method == "umap":
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        reduced = reducer.fit_transform(X)
    else:
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        reduced = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(reduced[mask, 0], reduced[mask, 1], label=str(label), s=15, alpha=0.6)

    plt.title(f"{method.upper()} Projection Colored by {label_type}")
    plt.legend(title=label_type)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"{method}_{label_type}_plot.png")
    plt.close()


def cluster_embeddings(method="kmeans", embedding_file="encoded_states_pytorch.npy", n_clusters=5):
    X = np.load(embedding_file)

    # Optional: normalize for DBSCAN
    X_scaled = StandardScaler().fit_transform(X)

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X)
    else:
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(X_scaled)

    # Visualize the clusters in 2D
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    reduced = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        color = 'gray' if label == -1 else None  # DBSCAN noise
        plt.scatter(reduced[mask, 0], reduced[mask, 1], s=15, alpha=0.6, label=f"Cluster {label}", c=color)

    plt.title(f"{method.upper()} Clustering on Encoded Embeddings")
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{method}_clusters.png")
    plt.close()

    print(f"[✓] Cluster plot saved as '{method}_clusters.png'")
    return labels


def compare_rewards(bandit_rewards, ppo_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(bandit_rewards, label="Bandit (Offline)", alpha=0.8)
    plt.plot(ppo_rewards, label="PPO (Online Rollout)", alpha=0.8)
    plt.xlabel("Episode / Step")
    plt.ylabel("Reward")
    plt.title("Reward Comparison: Bandit vs PPO")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ppo_vs_bandit_rewards.png")
    plt.close()
    print("[✓] Comparison plot saved as 'ppo_vs_bandit_rewards.png'")
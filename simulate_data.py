import numpy as np
import pandas as pd


# Encoding helper
def encode_gender(g):
    return {'male': 0, 'female': 1, 'non-binary': 2}[g]


def encode_time(t):
    return {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}[t]


def create_simulated(csv_filename):
    np.random.seed(42)
    num_samples = 1000

    # Viewer metadata
    viewer_age = np.random.randint(18, 70, size=num_samples)
    viewer_gender = np.random.choice(['male', 'female', 'non-binary'], size=num_samples)
    time_of_day = np.random.choice(['morning', 'afternoon', 'evening', 'night'], size=num_samples)
    history = np.random.randint(0, 20, size=num_samples)
    bid = np.random.choice([0, 1, 2, 3], size=num_samples)


    # Reward model with correlation
    reward = []
    for i in range(num_samples):
        base = 0.1

        # Positively correlated factors
        age_factor = (viewer_age[i] - 18) / 52  # normalized
        history_factor = history[i] / 20
        time_factor = encode_time(time_of_day[i]) / 3
        gender_factor = encode_gender(viewer_gender[i]) / 2

        # Define a context-dependent optimal bid level
        context_score = (
                                0.3 * age_factor +
                                0.3 * history_factor +
                                0.2 * time_factor +
                                0.2 * gender_factor
                        ) * 3  # Scale to same range as bid (0–3)

        # Reward is highest when action is close to the "right" bid
        reward_val = 1.0 - abs(bid - context_score) / 3  # 1.0 if perfect, 0.0 if far off
        reward_val += np.random.normal(0, 0.01)  # small noise
        reward_val = np.clip(reward_val, 0.0, 1.0)

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

    # csv_filename = 'simulated_bid_data.csv'
    df.to_csv(csv_filename, index=False)

    print(f"Simulated dataset saved as {csv_filename}")

    return df

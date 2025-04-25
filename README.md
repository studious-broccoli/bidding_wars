# bidding_wars
RL Awakens

- (1) Create Gym Environment simulating a bidding process
- (2) Metadata [viewer_age, viewer_gender, time_of_day, history, bid, reward] is simulated and
        artificially correlated in simulate_data.py
- (3) There are 4 discrete actions / bids randomly chosen
- (4) Reward is artificially correlated to metadata to get something learnable
- (5) Train simple contextual bandit with epsilon‐greedy approach (offline RL)
- (6) Then train with policy gradient method (PPO from stable-baselines3) for budget pacing
        and handling delayed/sparse rewards

<figure>
    <img src="reward_curve.png" alt="Screenshot">
    <figcaption>Figure 1: Reward Curve.</figcaption>
</figure>

- Model learns quite quickly.
- Should add early stopping to avoid unncessary training.

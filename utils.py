
def test_oracle_policy(env):
    env.reset()
    total_reward = 0
    for _ in range(100):
        obs = env._get_obs()
        row = env.data.iloc[env.current_index]
        age_factor = (row['viewer_age'] - 18) / 52
        history_factor = row['history'] / 20
        time_factor = ['morning', 'afternoon', 'evening', 'night'].index(row['time_of_day']) / 3
        gender_factor = {'male': 0, 'female': 1, 'non-binary': 2}[row['viewer_gender']] / 2
        context_score = (0.3 * age_factor + 0.3 * history_factor +
                         0.2 * time_factor + 0.2 * gender_factor) * 3
        optimal_bid = int(np.clip(round(context_score), 0, 3))
        _, reward, done, _ = env.step(optimal_bid)
        total_reward += reward
        if done:
            break
    print(f"Oracle policy reward (perfect bidding): {total_reward:.2f}")

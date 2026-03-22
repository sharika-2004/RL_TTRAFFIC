import numpy as np
from sumo_env import SumoEnvironment

# ==============================
# CONFIG
# ==============================
EPISODES = 100
STEPS = 100

alpha = 0.1
gamma = 0.95
epsilon = 0.9
epsilon_decay = 0.995
epsilon_min = 0.05

# ==============================
# Q-TABLE
# ==============================
q_table = {}

def get_q(state):
    if state not in q_table:
        q_table[state] = [0, 0]
    return q_table[state]

# ==============================
# STATE SIMPLIFICATION
# ==============================
def simplify_state(state, prev_action=None):
    half = len(state) // 2
    ns = sum(state[:half])
    ew = sum(state[half:])
    # finer binning with more granularity: 21x21 grid for better state resolution
    ns_bin = min(ns // 2, 20)
    ew_bin = min(ew // 2, 20)
    action_flag = -1 if prev_action is None else prev_action
    return (ns_bin, ew_bin, action_flag)


def evaluate_policy(env, q_table, episodes=5, steps=100):
    total = 0.0
    for _ in range(episodes):
        state_raw = env.reset()
        state = simplify_state(state_raw, prev_action=None)
        prev_action = None
        for _ in range(steps):
            action = np.argmax(q_table.get(state, [0, 0]))
            next_state_raw, reward = env.step(action)
            prev_action = action
            state = simplify_state(next_state_raw, prev_action)
            total += reward
    return total / episodes


# ==============================
# INIT ENV
# ==============================
env = SumoEnvironment("sumo_files/config.sumocfg")  # ⚠️ change path
env.start()

# Track metrics for analysis
episode_rewards = []
episode_evals = []

# ==============================
# TRAINING LOOP
# ==============================
best_total = -1e9
rolling = []
for episode in range(EPISODES):

    print(f"\n===== EPISODE {episode+1} =====")

    state_raw = env.reset()
    state = simplify_state(state_raw, prev_action=None)

    total_reward = 0
    prev_action = None

    for step in range(STEPS):

        # epsilon-greedy
        if np.random.rand() < epsilon:
            action = np.random.choice([0, 1])
        else:
            action = np.argmax(get_q(state))

        # penalty for switching signal too often (skip first step)
        if prev_action is not None and action != prev_action:
            switch_penalty = 0.1
        else:
            switch_penalty = 0.0

        # take action
        next_state_raw, reward = env.step(action)
        
        # do NOT rescale reward here; env reward is now baseline-shifted
        reward -= switch_penalty

        next_state = simplify_state(next_state_raw, action)

        # Q update with standard TD
        q = get_q(state)
        q_next = get_q(next_state)

        target = reward + gamma * max(q_next)
        
        # update with learning rate
        current_q = q[action]
        q[action] = current_q + alpha * (target - current_q)

        prev_action = action
        state = next_state
        total_reward += reward


    avg_queue = np.mean(state_raw)
    print(f"--- Episode {episode+1} Summary ---")
    print(f"Total Reward = {total_reward:.2f}")
    print(f"Avg waiting vehicles (last state) = {avg_queue:.2f}")
    print(f"Reward per step = {total_reward / STEPS:.3f}")

    episode_rewards.append(total_reward)

    rolling.append(total_reward)
    if len(rolling) > 25:
        rolling.pop(0)

    if total_reward > best_total:
        best_total = total_reward

    print(f"Best Total so far = {best_total}")
    print(f"Rolling mean (25) = {np.mean(rolling):.2f}")

    # extended exploration schedule for sufficient exploration
    if episode < 200:
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    elif episode < 400:
        epsilon = max(epsilon_min, epsilon * 0.998)
    else:
        epsilon = epsilon_min
    
    # decay alpha per episode for convergence
    alpha = max(0.01, alpha * 0.998)

    if (episode + 1) % 100 == 0:
        print(f">>> Epsilon = {epsilon:.6f}, Alpha = {alpha:.4f}, Q-table states = {len(q_table)}")

    # evaluate greedy policy every 25 episodes
    if (episode + 1) % 25 == 0:
        eval_reward = evaluate_policy(env, q_table, episodes=3, steps=STEPS)
        episode_evals.append((episode + 1, eval_reward))
        greedy_vs_train = eval_reward - total_reward
        print(f"[EVAL] Ep{episode+1}: Greedy={eval_reward:.2f}, Train={total_reward:.2f}, Gap={greedy_vs_train:.2f}")
        if greedy_vs_train > 30:
            print(f"  *** High gap detected: policy found but exploration dominates ***")

# ==============================
# CLOSE
# ==============================
env.close()

# Save training metrics
import json
metrics = {
    "total_episodes": EPISODES,
    "best_reward": best_total,
    "final_rolling_mean": float(np.mean(rolling)),
    "avg_reward_last_100": float(np.mean(episode_rewards[-100:])) if len(episode_rewards) > 100 else float(np.mean(episode_rewards)),
    "q_table_states": len(q_table),
    "evaluations": episode_evals
}

with open("training_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print(f"Best Total Reward: {best_total:.2f}")
print(f"Final Rolling Mean (25): {np.mean(rolling):.2f}")
print(f"Final Epsilon: {epsilon:.6f}")
print(f"Q-Table States Learned: {len(q_table)}")
print(f"Metrics saved to training_metrics.json")
print("="*50)
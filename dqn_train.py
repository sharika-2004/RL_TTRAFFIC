# train_dqn.py
import numpy as np
import torch
from sumo_env import SumoEnvironment
from dqn_agent import DQNAgent

EPISODES = 1000
STEPS = 100
EVAL_EVERY = 25
UPDATE_TARGET = 20  # update target network every 20 episodes

env = SumoEnvironment("sumo_files/config.sumocfg")
state_dim = len(env.reset())
action_dim = 2  # change based on your traffic light phases

agent = DQNAgent(state_dim, action_dim, lr=0.0004, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=128, tau=0.01)

episode_rewards = []
episode_eval_rewards = []

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for step in range(STEPS):
        action = agent.act(state)
        next_state, reward = env.step(action)
        done = step == STEPS - 1
        agent.remember(state, action, reward, next_state, done)

        # do one replay pass every step after memory warmup
        agent.replay()

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)

    if (episode + 1) % UPDATE_TARGET == 0:
        agent.update_target()

    if (episode + 1) % EVAL_EVERY == 0:
        eval_reward = 0
        for _ in range(5):
            s = env.reset()
            rsum = 0
            for _ in range(STEPS):
                a = np.argmax(agent.model(torch.FloatTensor(s).unsqueeze(0).to(agent.device)).detach().cpu().numpy())
                s, r = env.step(a)
                rsum += r
            eval_reward += rsum
        eval_reward /= 5
        episode_eval_rewards.append((episode + 1, eval_reward))

    print(f"Episode {episode+1} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f} | AvgReward(25): {np.mean(episode_rewards[-25:]):.2f}")

# final diagnostic
print("Training complete")
print(f"Best reward: {max(episode_rewards):.2f}")

# save metrics to JSON for later analysis
import json
metrics = {
    "total_episodes": EPISODES,
    "best_reward": float(max(episode_rewards)),
    "avg_reward_last_100": float(np.mean(episode_rewards[-100:])),
    "episode_rewards": [float(r) for r in episode_rewards],
    "eval_rewards": [{"episode": e, "eval_reward": float(r)} for e, r in episode_eval_rewards],
    "final_epsilon": float(agent.epsilon),
    "q_table_states": None
}
with open("dqn_training_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved to dqn_training_metrics.json")

env.close()
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Optional: nicer plots
sns.set(style="whitegrid")

# Load metrics saved from your training
with open("training_metrics.json", "r") as f:
    metrics = json.load(f)

# -----------------------------
# 1. Reward vs Episodes
# -----------------------------
episode_rewards = metrics.get("evaluations", [])
episodes = [e["episode"] for e in episode_rewards]
rewards = [e["reward"] for e in episode_rewards]

plt.figure(figsize=(8,5))
plt.plot(episodes, rewards, marker='o', color='b')
plt.title("Reward vs Episodes")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.savefig("reward_vs_episodes.png")
plt.show()

# -----------------------------
# 2. Waiting Time Comparison
# -----------------------------
waiting_times = [e["waiting"] for e in episode_rewards]

plt.figure(figsize=(8,5))
plt.plot(episodes, waiting_times, marker='o', color='r')
plt.title("Waiting Time vs Episodes")
plt.xlabel("Episode")
plt.ylabel("Total Waiting Time")
plt.grid(True)
plt.savefig("waiting_time_vs_episodes.png")
plt.show()

# -----------------------------
# 3. Queue Length Comparison
# -----------------------------
queue_lengths = [e["queue"] for e in episode_rewards]

plt.figure(figsize=(8,5))
plt.plot(episodes, queue_lengths, marker='o', color='g')
plt.title("Queue Length vs Episodes")
plt.xlabel("Episode")
plt.ylabel("Total Queue Length")
plt.grid(True)
plt.savefig("queue_length_vs_episodes.png")
plt.show()

# -----------------------------
# 4. Throughput
# -----------------------------
throughputs = [e["throughput"] for e in episode_rewards]

plt.figure(figsize=(8,5))
plt.plot(episodes, throughputs, marker='o', color='orange')
plt.title("Throughput vs Episodes")
plt.xlabel("Episode")
plt.ylabel("Vehicles Cleared per Simulation Step")
plt.grid(True)
plt.savefig("throughput_vs_episodes.png")
plt.show()

import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size=10, action_size=2):
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = 0.1      # learning rate
        self.gamma = 0.9      # discount factor
        self.epsilon = 1.0    # exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.q_table.shape[1]-1)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action]
        )
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
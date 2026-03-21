from sumo_env import SumoEnvironment
import time

env = SumoEnvironment("sumo_files/config.sumocfg")

env.start()

state = env.reset()

for step in range(20):
    action = step % 2  # alternate signals

    state, reward = env.step(action)

    print("Step:", step)
    print("State:", state)
    print("Reward:", reward)
    print("-" * 30)

    time.sleep(0.5)

env.close()
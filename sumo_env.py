import traci
import numpy as np


class SumoEnvironment:
    def __init__(self, config_path):
        self.config = config_path
        self.tls_id = None

    def start(self):
        # Start SUMO GUI
        traci.start(["sumo-gui", "-c", self.config])

        # Get traffic light ID
        self.tls_id = traci.trafficlight.getIDList()[0]

    def reset(self):
        # Reload simulation
        traci.load(["-c", self.config])
        return self.get_state()

    def step(self, action):
        # 🔥 Get correct signal format dynamically
        current_state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
        length = len(current_state)

        half = length // 2

        # Generate safe patterns
        pattern1 = "G" * half + "r" * (length - half)
        pattern2 = "r" * half + "G" * (length - half)

        # Apply action
        if action == 0:
            traci.trafficlight.setRedYellowGreenState(self.tls_id, pattern1)
        else:
            traci.trafficlight.setRedYellowGreenState(self.tls_id, pattern2)

        # Move simulation forward
        traci.simulationStep()

        # Get new state
        state = self.get_state()

        # Reward = minimize congestion
        reward = -sum(state)

        return state, reward

    def get_state(self):
        # Get vehicle count on all lanes
        lanes = traci.lane.getIDList()
        state = []

        for lane in lanes:
            count = traci.lane.getLastStepVehicleNumber(lane)
            state.append(count)

        return state

    def close(self):
        traci.close()
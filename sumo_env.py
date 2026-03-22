import traci


class SumoEnvironment:
    def __init__(self, config_path):
        self.config = config_path
        self.tls_id = None
        self.is_running = False
        self.prev_total = 0   # for reward calculation

    def start(self):
        if not self.is_running:
            traci.start(["sumo", "-c", self.config])
            self.tls_id = traci.trafficlight.getIDList()[0]
            self.is_running = True

    def reset(self):
        if self.is_running:
            traci.load(["-c", self.config])   # reload same simulation
        else:
            self.start()

        state = self.get_state()
        self.prev_total = sum(state)   # initialize reward baseline
        return state

    def step(self, action):
        current_state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
        length = len(current_state)
        half = length // 2

        # generate safe patterns
        pattern1 = "G" * half + "r" * (length - half)
        pattern2 = "r" * half + "G" * (length - half)

        # apply action
        if action == 0:
            traci.trafficlight.setRedYellowGreenState(self.tls_id, pattern1)
        else:
            traci.trafficlight.setRedYellowGreenState(self.tls_id, pattern2)

        # run simulation for few steps (IMPORTANT)
        for _ in range(5):
            traci.simulationStep()

        # get new state
        state = self.get_state()

        current_total = sum(state)

        # reward is change in total waiting vehicles (positive = improvement)
        reward = self.prev_total - current_total

        # additional small steady reward for keeping absolute queues low
        reward -= 0.1 * current_total

        # clip to stabilize learning, prevent explosion
        reward = max(-10.0, min(10.0, reward))

        self.prev_total = current_total

        return state, reward

    def get_state(self):
        lanes = traci.lane.getIDList()
        state = []
        for lane in lanes:
            waiting = traci.lane.getLastStepHaltingNumber(lane)
            state.append(waiting)

        return state  # returns list like [3, 5, 2, 7]

    def close(self):
        if self.is_running:
            traci.close()
            self.is_running = False
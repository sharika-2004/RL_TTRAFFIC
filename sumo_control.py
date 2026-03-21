import traci
import time

# Path to SUMO config
sumo_config = "sumo_files/config.sumocfg"

# Start SUMO
traci.start(["sumo-gui", "-c", sumo_config])

step = 0

print("Starting SUMO control...")

# Get traffic light ID
tls_id = traci.trafficlight.getIDList()[0]

# Get correct signal format automatically
initial_state = traci.trafficlight.getRedYellowGreenState(tls_id)
print("Detected signal format:", initial_state)

length = len(initial_state)

# Create two alternating patterns
half = length // 2

pattern1 = "G" * half + "r" * (length - half)
pattern2 = "r" * half + "G" * (length - half)

print("Pattern 1:", pattern1)
print("Pattern 2:", pattern2)

while step < 100:
    traci.simulationStep()

    # Alternate traffic lights
    if step % 20 < 10:
        traci.trafficlight.setRedYellowGreenState(tls_id, pattern1)
    else:
        traci.trafficlight.setRedYellowGreenState(tls_id, pattern2)

    print(f"Step {step}")

    step += 1
    time.sleep(0.3)  # slow down so you can see changes

traci.close()
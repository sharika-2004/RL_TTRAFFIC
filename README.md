# 🚦 RL-Based Smart Traffic Signal Optimization using SUMO

## 📌 Overview

This project focuses on developing an intelligent traffic signal control system using **Reinforcement Learning (RL)** integrated with the **SUMO (Simulation of Urban Mobility)** traffic simulator.

The system dynamically controls traffic signals to reduce congestion and improve traffic flow efficiency at intersections.

---

## 🎯 Objectives

* Simulate real-world traffic using SUMO
* Implement adaptive traffic signal control
* Minimize vehicle waiting time and congestion
* Provide an RL-compatible environment for training agents

---

## 🧩 Project Structure

```
RL-Traffic-Signal/
 ┣ ┣ Training/
 ┣ ┣ dqn_train.py.py
 ┣ ┣ dqn_training_metrics.json
 ┣ ┣ qlearning_agent.py
 ┣ ┣ train.py
 ┣ ┣ training_metrics.json
 ┣ ┣ dqn_agent.py
 ┣ sumo_env.py          # RL-ready SUMO environment (Person 1)
 ┣ sumo_control.py      # Manual traffic signal control using TraCI
 ┣ test_sumo_env.py     # Test script for environment
 ┣ environment.ipynb    # Initial custom simulation (Phase 1)
 ┣ sumo_files/          # SUMO configuration files
 ┃ ┣ config.sumocfg
 ┃ ┣ nodes.nod.xml
 ┃ ┣ edges.edg.xml
 ┃ ┣ routes.rou.xml
 ┃ ┣ intersection.net.xml
 ┣ requirements.txt
 ┗ README.md
```

---

## ⚙️ Technologies Used

* Python
* SUMO (Simulation of Urban Mobility)
* TraCI (Traffic Control Interface)
* NumPy

---

## 🚀 Setup Instructions

### 1. Install SUMO

Download and install SUMO from:
https://www.eclipse.org/sumo/

Add SUMO to system PATH:

```
C:\Program Files (x86)\Eclipse\Sumo\bin
```

---

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Run SUMO Environment Test

```bash
python test_sumo_env.py
```

This will:

* Open SUMO GUI
* Simulate traffic
* Control traffic lights dynamically

---

## 🧠 Person 1 Contribution (Environment + SUMO Integration)

* Developed a custom traffic simulation environment
* Built SUMO-based traffic intersection
* Implemented vehicle flow and routing
* Generated traffic light system using `netconvert`
* Integrated Python with SUMO using TraCI
* Created RL-compatible environment (`sumo_env.py`)

---
Person 2 Contributions – Reinforcement Learning Agent

Role: Develop and integrate a Reinforcement Learning (RL) agent to optimize traffic signals using the SUMO environment.

Key Contributions
Agent Development
Implemented QLearningAgent in agent.py for controlling traffic lights.
Designed state-action-reward logic for traffic signal optimization.
Training Pipeline
Created train.py to train the RL agent within the SUMO simulation environment.
Integrated environment methods (reset(), step(), get_state()) to interact with the agent.
Traffic Optimization
Tuned reward functions to reduce vehicle congestion.
Verified intelligent signal switching and observed improved traffic flow over episodes.
Documentation & Testing
Ensured the RL agent runs correctly with the existing SUMO-based environment (sumo_env.py) without modifying core logic.
Provided usage instructions for cloning, installing dependencies, and running the RL training.
Optional Enhancements
Produced graphs for reward progression and traffic metrics (if applicable).

Impact:

Successfully integrated a Reinforcement Learning agent (Q-learning) into the SUMO environment, enabling intelligent traffic signal control and providing a foundation for further improvements such as DQN or advanced RL algorithms.

---

## 📊 Expected Outcomes

* Reduced vehicle waiting time
* Improved traffic flow efficiency
* Adaptive signal control based on real-time traffic

---



---

## 🏆 Resume Highlight

Developed a SUMO-based traffic simulation environment integrated with Python using TraCI, enabling real-time traffic signal control and reinforcement learning compatibility.

---

## 👥 Team Roles

* **Person 1**: Environment + SUMO Integration
* **Person 2**: Reinforcement Learning Model
* **Person 3**: Visualization + Dashboard

---

## 📌 Notes

* Ensure SUMO is properly installed before running
* Use `sumo_env.py` for RL integration
* Simulation parameters can be modified in `sumo_files/`

---

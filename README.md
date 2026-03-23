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
 ┣ Training/
 ┃ ┣ dqn_train.py
 ┃ ┣ dqn_training_metrics.json
 ┃ ┣ qlearning_agent.py
 ┃ ┣ train.py
 ┃ ┣ training_metrics.json
 ┃ ┣ dqn_agent.py
 ┃
 ┣ graphs/
 ┃ ┣ graph.py
 ┃ ┣ generate_plots.py             
 ┃ ┣ dqn.png
 ┃ ┣ evaluation_qlearn.png
 ┃ ┣ queue_qlearn.png
 ┃ ┣ reward_qlearn.png
 ┃ ┣ summary_dashboard_qlearn.png
 ┃ ┣ throughput_qlearn.png
 ┃ ┣ training_curves_qlearn.png
 ┃ ┣ waiting_qlearn.png
 ┃
 ┣ gui/                     
 ┃ ┣ dqn_ui.py                
 ┃ ┣ q_learn_ui.py
 ┃
 ┣ assets/                  
 ┃ ┣ dqn_dash1.png
 ┃ ┣ dqn_dash2.png
 ┃ ┣ dqn_dash3.png
 ┃ ┣ qlearn_dash1.png
 ┃ ┣ qlearn_dash2.png
 ┃ ┣ qlearn_dash3.png
 ┃
 ┣ sumo_env.py
 ┣ sumo_control.py
 ┣ test_sumo_env.py
 ┣ environment.ipynb
 ┣ sumo_files/
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
Person 3 Contributions – Visualization & Dashboard

Role: Develop visualization tools and a user interface to analyze and present RL model performance.

Key Contributions

Data Handling
Processed training outputs from Q-Learning and DQN (training_metrics.json, dqn_training_metrics.json) for analysis.

Visualization
Created graphs for reward progression and evaluation metrics.
Compared Q-Learning and DQN performance using clear plots.

Dashboard Development
Built a UI to display training results, graphs, and evaluation summaries.
Integrated multiple views such as training curves and model comparison.

Analysis
Observed learning trends and convergence behavior.
Highlighted performance differences between Q-Learning and DQN.

Impact:

Developed a visualization and dashboard system that converts training results into clear insights, enabling effective analysis and presentation of RL model performance.

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

📌 Final Conclusion & Results

This project successfully demonstrates the application of Reinforcement Learning for adaptive traffic signal control using the SUMO simulation environment.

Both Q-Learning and Deep Q-Network (DQN) approaches were implemented and evaluated:

Q-Learning
Learned a discrete policy using state binning
Showed gradual improvement over episodes
Limited by state-space discretization and scalability
DQN (Deep Reinforcement Learning)
Achieved significantly better performance and stability
Converged to near-optimal reward values (~ -32 range)
Handled continuous state representation more effectively
Demonstrated strong generalization across episodes

📊 Key Observations

Reduction in congestion-related penalties over time
Stable policy convergence in DQN compared to Q-learning
Improved traffic throughput and reduced queue lengths in later episodes
Exploration–exploitation balance played a critical role in performance

📈 Visualization & Analysis

Training and evaluation metrics were tracked across episodes
Reward curves, evaluation trends, and performance dashboards were developed
A user interface was built to visualize model performance and compare algorithms

⚠️ Limitations

Simplified traffic intersection (single junction)
Limited action space (basic signal switching)
Reward function primarily based on vehicle count (can be extended)

🚀 Future Improvements

Multi-intersection traffic control (multi-agent RL)
Advanced reward shaping (waiting time, fairness, emissions)
Integration of real-world traffic datasets
Deployment of more advanced RL algorithms (Double DQN, PPO)
---

## 📌 Notes

* Ensure SUMO is properly installed before running
* Use `sumo_env.py` for RL integration
* Simulation parameters can be modified in `sumo_files/`

---

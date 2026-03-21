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

## 🔄 Future Work (Person 2 & 3)

* Integrate Reinforcement Learning (Q-Learning / DQN)
* Optimize traffic signal policies
* Add visualization dashboard and performance metrics
* Compare RL vs fixed-time traffic signals

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

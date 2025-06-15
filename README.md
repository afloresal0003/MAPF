# MAPF Framework — Deep Reinforcement Learning for Multi-Agent Path Finding

This repository contains my research work at the ACT Lab (USC Viterbi School of Engineering), where I developed a decentralized Multi-Agent Path Finding (MAPF) framework using deep reinforcement learning (DRL) to optimize collision-free navigation for teams of autonomous agents operating in obstacle-dense environments.

MAPF is an NP-hard problem with growing relevance to real-world robotics, warehouse logistics, and multi-robot coordination. This project explores the application of deep reinforcement learning to overcome scalability challenges found in traditional MAPF solvers.

---

## 📈 Visual Demo

Example run of trained MAPF agents navigating to their destinations without collisions:  
![MAPF Demo GIF](https://media.giphy.com/media/ThAouXKuiW0Z6Aswba/giphy.gif)

---

## 🔬 Research Context

- **Program:** USC Viterbi School of Engineering — SHINE Research Program  
- **Duration:** 3-month full-time research internship + 4 months part-time continuation  
- **Lab:** ACT Lab (Autonomous Collective Transportation Lab), USC  
- **Mentor:** Eric Ewing (PhD Candidate, USC)

---

## 🧠 Core Methods

- Developed a decentralized MAPF planner using:
  - **Deep Reinforcement Learning (DRL)** — Advantage Actor-Critic (A2C) algorithm
  - **Deep Convolutional Neural Networks (CNNs)** for policy learning
  - **Vector-based observation spaces** combining obstacle maps, agent locations, goal vectors, and magnitude scaling
- Designed experiments varying:
  - Number of agents
  - Obstacle densities
  - Environment sizes
- Evaluated success rates, solution lengths, and generalization performance across MAPF scenarios.

---

## 📊 Additional Work

- Trained DNN models on the EMNIST dataset (240K handwritten digits) using PyTorch, achieving 98% classification accuracy.
- Prototyped a second MAPF framework variant ("SMP") using TensorFlow to explore additional DRL architectures and heuristics.
- Generated internal technical reports analyzing MAPF framework inefficiencies, DRL policy behavior, and framework improvement directions.

---

## 📂 Technologies Used

- Python 3
- PyTorch
- TensorFlow
- NumPy
- OpenAI Gym
- Google Colab
- Git


---

## 🖼️ Research Poster

Full project poster summarizing methods and results:  
[MAPF Framework Research Poster (PDF)](./poster.png)

---

## 👥 Acknowledgments

- **Eric Ewing** — PhD Mentor (USC Viterbi ACT Lab)
- **Anthony Flores-Alvarez** — Research Intern & Developer

---

## 🚀 Future Work

- Extend to continuous environments and 3D navigation
- Explore centralized learning with decentralized execution (CLDE) frameworks
- Apply curriculum learning to improve policy generalization for high agent counts

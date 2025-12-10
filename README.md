# Strawdew_Valley_Mining_Bot

This project implements a **Stardew-like mining environment** using `gymnasium` and trains a **PPO agent** (from `stable-baselines3`) to mine ore efficiently.  
The agent can be trained and evaluated in two variants of the environment:

- **Without weeds**
- **With weeds**

You can also **render and save MP4 videos** of the trained agent exploring the mine.

---

## 1. Requirements

- **Python**: 3.10.x (recommended)
- **pip**: recent version (`python -m pip install --upgrade pip`)

Core Python packages:

- `numpy`
- `gymnasium`
- `stable-baselines3`
- `stable-baselines3[extra]`
- `opencv-python` (optional; for visualization scripts)

All required packages are listed in [`requirements.txt`].

---

## 2. Installation & Environment Setup

### 2.1. Clone the repository

```bash
git clone <your-repo-url> Strawdew_Valley_Mining_Bot
cd Strawdew_Valley_Mining_Bot

### 2.2. Install dependencies

pip install -r requirements.txt

This will install the packages listed in the Requirements section.

---

# 3 Usage

All commands below assume that you are in the root directory.

## 3.1 No-weed environment

The following is the step-by-step instruction on how to train, evaluate, and visualize the agent's performance on the no-weed environment.

### 3.1.1 Train the agent

```python agents/ppo_train.py```

This script creates the environment `StardewMineEnv(size=10, spawn_weed=False)`, trains the PPO with policy `MultiInputPolicy` and `total_timesteps = 400_000`, and saves the model to `models/ppo_mine_no_weeds.zip`.

### 3.1.2 Evaluate the agent

```python evaluation/evaluate.py models/ppo_mine_no_weeds.zip```

This script runs several episodes and report the average number of ores collected, average amound of energy used, exploration rate (fraction of grid visited), and number of max floor reached.

The output will be in the following format:

```Using model file: <path-to-model>
=== Evaluation Metrics ===
Avg Ore Collected:      X.XX
Avg Energy Used:        Y.YY
Avg Exploration Rate:   0.ZZZ
Avg Max Floor Reached:  W.WW```

### 3.1.3 Visualize the agent

```python visualize_agent_noweeds.py --model models/ppo_mine_no_weeds.zip --episodes 3 --output agent_demo_noweeds.mp4```

This script loads the PPO model, renders frames with OpenCV, and saves as MP4 file.

## 3.2 With-weed environment

The following is the step-by-step instruction on how to train, evaluate, and visualize the agent's performance on the with-weed environment.

### 3.2.1 Train the agent

```python agents/ppo_train.py --weeds```

This script creates the environment `StardewMineEnv(size=10, spawn_weed=True)`, trains the PPO with policy `MultiInputPolicy` and `total_timesteps = 400_000`, and saves the model to `models/ppo_mine_weeds.zip`.

### 3.2.2 Evaluate the agent

```python evaluation/evaluate.py models/ppo_mine_weeds.zip```

This script runs several episodes and report the average number of ores collected, average amound of energy used, exploration rate (fraction of grid visited), and number of max floor reached.

The output will be in the following format:

```Using model file: <path-to-model>
=== Evaluation Metrics ===
Avg Ore Collected:      X.XX
Avg Energy Used:        Y.YY
Avg Exploration Rate:   0.ZZZ
Avg Max Floor Reached:  W.WW```

### 3.2.3 Visualize the agent

```python visualize_agent_weeds.py --model models/ppo_mine_weeds.zip --episodes 3 --output agent_demo_weeds.mp4```

This script loads the PPO model, renders frames with OpenCV, and saves as MP4 file.
# Enhancing Urban Traffic Management through Intelligent Control Systems

## Project Overview

Urban traffic congestion is a pervasive problem, leading to significant delays, increased fuel consumption, and environmental pollution. This project introduces an innovative approach to urban traffic management by leveraging **Long Short-Term Memory (LSTM) networks** for predictive analysis and **Reinforcement Learning (RL) agents** for adaptive traffic signal control. Our intelligent control system aims to dynamically optimize traffic flow, reduce congestion, and improve overall urban mobility.

## How It Works

The core idea revolves around a decentralized data collection mechanism combined with a centralized intelligent controller:

1. **Vehicle Data Transmission:** Vehicles equipped with communication capabilities continuously send real-time data to a central controller. This data includes critical parameters such as:

   * **Turn Intent:** Indication of upcoming turns (left, right, straight).

   * **Lane ID:** Current lane occupied by the vehicle.

   * **Distance to Intersection:** Remaining distance to the next intersection.

   * **Entry Time:** Timestamp of when the vehicle entered the system or a specific segment.

2. **LSTM-based Prediction:** Upon receiving the raw vehicle data, an LSTM neural network analyzes this time-series information. LSTMs are particularly well-suited for sequence prediction tasks, allowing the system to:

   * Learn complex patterns and dependencies in traffic flow.

   * Predict future traffic conditions, such as expected vehicle arrivals, queue lengths, and turning movements at upcoming intersections.

3. **Reinforcement Learning for Policy Allocation:** With the current and predicted future traffic states, a Reinforcement Learning agent comes into play. The RL agent's role is to:

   * Evaluate various traffic light phasing and timing policies.

   * Learn optimal strategies through trial and error within a simulated environment.

   * Allocate the most efficient traffic policy (e.g., green light duration for specific directions) to minimize congestion, reduce waiting times, and maximize throughput at intersections. The agent's decisions are based on maximizing a reward signal, which is typically inversely proportional to congestion metrics.

## Project Structure

The repository is organized into several key Python files, each serving a distinct purpose in the overall system:

```

.
├── app.py
├── data\_preprocess.py
├── lstm\_prediction.py
├── rl\_agent.py
├── rl\_env.py
├── rl\_model.py
└── signal\_log.csv

```

* **`app.py`**: This file simulates the behavior of vehicles sending data to the central controller. It acts as the primary interface for generating synthetic traffic data that feeds into the intelligent control system.

* **`data_preprocess.py`**: Contains helper functions and scripts for cleaning, transforming, and preparing the raw vehicle data before it is fed into the LSTM model for prediction. This ensures data quality and consistency.

* **`lstm_prediction.py`**: Implements the LSTM neural network model responsible for analyzing historical and real-time traffic data to predict future traffic conditions and patterns at intersections.

* **`rl_agent.py`**: Orchestrates the training and deployment of the Reinforcement Learning agent. This script defines the training loop, reward function, and manages the interaction between the agent and its environment. It also logs the traffic signal decisions and relevant metrics into `signal_log.csv`.

* **`rl_env.py`**: Defines the traffic simulation environment in which the RL agent operates. This environment models intersections, vehicle movements, and traffic light states, providing the necessary state observations and accepting actions from the RL agent.

* **`rl_model.py`**: Specifies the architecture and components of the Reinforcement Learning model (e.g., Q-network, policy network) used by the `rl_agent.py` for learning optimal traffic control policies.

* **`signal_log.csv`**: A CSV file used by `rl_agent.py` to log the traffic signal allocation policies, their timings, and other relevant data during the RL agent's training and evaluation phases. This serves as a record for analysis and debugging.

## Technologies Used

* **Python 3.x**

* **PyTorch** (for LSTM and RL model implementation)

* **Pandas/NumPy** (for data manipulation)

## Getting Started

To get started with the project, follow these steps:

1. **Clone the repository:**

```

git clone https://github.com/Manohara-Ai/Enhancing_Urban_Traffic_Management.git
cd Enhancing_Urban_Traffic_Management

```

2. **Run the vehicle data simulation:**

```

python app.py

```

This will simulate vehicles sending data to the controller.

3. **Run the RL agent for training and traffic control:**

```

python rl_agent.py

```

This will start the RL agent, which will use the simulated data to learn and apply traffic control policies.

## Contributing

We welcome contributions! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
```

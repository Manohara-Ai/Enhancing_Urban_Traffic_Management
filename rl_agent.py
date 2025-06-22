import torch
import random
import itertools
import os
import numpy as np
import pandas as pd
from collections import deque
from rl_env import Junction, SignalPlan
from rl_model import Linear_QNet, QTrainer
from lstm_prediction import main

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

def generate_lane_options():
    # These are safe movement types for Indian roads
    return [
        (0, 0, 0),  # All red
        (1, 0, 0),  # Left only (free)
        (0, 1, 0),  # Straight only
        (0, 0, 1),  # Right only
        (1, 1, 0),  # Left + Straight
        (0, 1, 1),  # Straight + Right
        (1, 0, 1),  # Left + Right
        (1, 1, 1)   # All allowed
    ]

def is_non_conflicting(plan):
    # Add your custom rules here — for now just prevent all being (0,0,0)
    if all(sig == (0, 0, 0) for sig in plan):
        return False
    
    num_lanes = len(plan)
    # Base conflict pairs for 4 lanes (N, E, S, W)
    base_conflicts = [
        # (lane_i, move_i), (lane_j, move_j)
        ((0, 0), (2, 1)),  # N left vs S straight
        ((0, 0), (3, 2)),  # N left vs W right
        ((0, 1), (2, 1)),  # N straight vs S straight
        ((0, 2), (1, 0)),  # N right vs E left

        ((1, 0), (3, 1)),  # E left vs W straight
        ((1, 0), (0, 2)),  # E left vs N right
        ((1, 1), (3, 1)),  # E straight vs W straight
        ((1, 2), (2, 0)),  # E right vs S left

        ((2, 0), (0, 1)),  # S left vs N straight
        ((2, 0), (1, 2)),  # S left vs E right
        ((2, 1), (0, 1)),  # S straight vs N straight
        ((2, 2), (3, 0)),  # S right vs W left

        ((3, 0), (1, 1)),  # W left vs E straight
        ((3, 0), (2, 2)),  # W left vs S right
        ((3, 1), (1, 1)),  # W straight vs E straight
        ((3, 2), (0, 0)),  # W right vs N left
    ]

    # For lanes beyond 4, we add generic conflicts (simplified)
    # Just prevent simultaneous green for same movements from adjacent lanes
    # You can customize more sophisticated rules per your intersection

    # Helper to get lane indices in a circular manner
    def prev_lane(l):
        return (l - 1) % num_lanes

    def next_lane(l):
        return (l + 1) % num_lanes

    # Add conflicts for lanes 4 and 5 (if present)
    extra_conflicts = []

    if num_lanes > 4:
        # Lane 4 conflicts with lane 1 and lane 3 (diagonal?)
        extra_conflicts += [
            ((4, 0), (1, 1)),  # Example conflict: lane 4 left vs lane 1 straight
            ((4, 1), (3, 0)),  # lane 4 straight vs lane 3 left
            ((4, 2), (2, 0)),  # lane 4 right vs lane 2 left
        ]

    if num_lanes > 5:
        # Lane 5 conflicts with lane 0 and lane 2 (example)
        extra_conflicts += [
            ((5, 0), (0, 1)),
            ((5, 1), (2, 0)),
            ((5, 2), (1, 0)),
        ]

    all_conflicts = base_conflicts + extra_conflicts

    # Filter conflicts only relevant for current lanes
    filtered_conflicts = [
        (c1, c2) for c1, c2 in all_conflicts
        if c1[0] < num_lanes and c2[0] < num_lanes
    ]

    for (lane_i, move_i), (lane_j, move_j) in filtered_conflicts:
        if plan[lane_i][move_i] == 1 and plan[lane_j][move_j] == 1:
            return False  # Conflict detected

    return True

def generate_all_plans(num_lanes, limit=None):
    lane_opts = generate_lane_options()
    all_combos = itertools.product(lane_opts, repeat=num_lanes)
    
    safe_plans = []
    for plan in all_combos:
        if is_non_conflicting(plan):
            duration = random.randint(15, 60)
            safe_plans.append((list(plan), duration))

        if limit and len(safe_plans) >= limit:
            break

    return safe_plans

class Agent:
    # Agent class
    def __init__(self, num_lanes):
        self.n_games = 0
        self.last_rr_lane = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.all_possible_plans = generate_all_plans(num_lanes, limit=5000)
        self.model = Linear_QNet(16*num_lanes, 256, (num_lanes*3)+1)

        # Define the model path
        model_path = r'Model/Classic Model.pth'

        # Check if the file exists before loading
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, weights_only=True),)
            print("Model loaded successfully.")
        else:
            print("Model file not found. Initializing with default weights.")

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, junction):
        state = []

        # Load past data from CSV files
        past_data = {
            key: pd.read_csv(path)
            for key, path in junction.file_paths.items()
        }

        future= 10 if  8 * 60 <= junction.TIME_INSTANT <= 22 * 60 else 60 

        # Get future predictions
        future_data = {
            key: main(df, time_instant=junction.TIME_INSTANT, future=future)
            for key, df in past_data.items()
        }

        for lane in range(1, junction.num_lanes + 1):
            # Get last known (past) vehicle count
            total_past = past_data['total'][f'Lane_{lane}'].iloc[-1]
            priority_past = past_data['priority'][f'Lane_{lane}'].iloc[-1]
            left_past = past_data['left'][f'Lane_{lane}'].iloc[-1]
            right_past = past_data['right'][f'Lane_{lane}'].iloc[-1]
            straight_past = past_data['straight'][f'Lane_{lane}'].iloc[-1]

            # Get next predicted (future) vehicle count
            total_future = future_data['total'][lane - 1][0]
            priority_future = future_data['priority'][lane - 1][0]
            left_future = future_data['left'][lane - 1][0]
            right_future = future_data['right'][lane - 1][0]
            straight_future = future_data['straight'][lane - 1][0]

            # Combine into state
            state.extend([
                total_past, priority_past, left_past, right_past, straight_past,
                total_future, priority_future, left_future, right_future, straight_future
            ])

        # Encode previous and current signal states (hot-encoded tuple)
        for signal_tuple in junction.prev_state + junction.current_state:
            state.extend(signal_tuple)

        return np.array(state, dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        # remember previous states
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached


    def train_long_memory(self):
        # train agent from previous long memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        for state, action, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done)


    def train_short_memory(self, state, action, reward, next_state, done):
        # train agent from previous short memory
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state, num_lanes):
        self.epsilon = 80 - self.n_games

        if random.randint(0, 200) < self.epsilon:
            # Exploration: Random safe plan
            signals, duration = random.choice(self.all_possible_plans)
            return SignalPlan(signals, duration)

        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0).detach().numpy()

            best_score = -float('inf')
            best_plan = None

            for signals, duration in self.all_possible_plans:
                features = []
                for sig in signals:
                    features.extend(sig)
                features.append(duration)

                score = np.dot(prediction, features)
                if score > best_score:
                    best_score = score
                    best_plan = SignalPlan(signals, duration)

            # ✅ Fallback if model failed
            if best_plan is None:
                signals, duration = random.choice(self.all_possible_plans)
                return SignalPlan(signals, duration)

            return best_plan

def train():
    # start training the agent
    num_lanes = len([f for f in os.listdir("train_simulation") if os.path.isfile(os.path.join("train_simulation", f))])
    agent = Agent(num_lanes)
    junction = Junction(num_lanes)
    
    while True:
        # get old state
        state_old = agent.get_state(junction)

        # get move
        final_move = agent.get_action(state_old, num_lanes)

        # perform move and get new state
        print(final_move)
        reward, done = junction.allocate(final_move)
        state_new = agent.get_state(junction)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            agent.model.save(junction.__repr__()+' Model.pth')
            junction.reset()
            print(f"epoch {agent.n_games} done")
            agent.n_games += 1
            agent.train_long_memory()


if __name__ == '__main__':
    train()
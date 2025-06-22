import csv
from collections import namedtuple
from data_preprocess import generate_all_vehicle_statistics

SignalPlan = namedtuple('SignalPlan', ['signals', 'duration'])
generate_all_vehicle_statistics()

class Junction:
    def __init__(self, num_lanes, output_file='signal_log.csv'):
        self.num_lanes = num_lanes
        self.output_file = output_file
        self._initialize_state()
        print(f"Initialized controller for {self.num_lanes} lanes.")

    def _initialize_state(self):
        self.TIME_INSTANT = 10
        self.total_duration = 2 * 60 
        self.prev_state = [(0, 0, 0)] * self.num_lanes
        self.current_state = [(0, 0, 0)] * self.num_lanes

        self.file_paths = {
            "total": 'train/vehicle_counts.csv',
            "priority": 'train/vehicle_counts_priority.csv',
            "left": 'train/vehicle_counts_left.csv',
            "right": 'train/vehicle_counts_right.csv',
            "straight": 'train/vehicle_counts_straight.csv'
        }

        self.lanes = {
            i + 1: {
                "total_past": [],
                "total_future": [],
                "priority_past": [],
                "priority_future": [],
                "left_past": [],
                "left_future": [],
                "right_past": [],
                "right_future": [],
                "straight_past": [],
                "straight_future": []
            }
            for i in range(self.num_lanes)
        }

        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['TIME_INSTANT'] + [f'Lane_{i+1}' for i in range(self.num_lanes)]
            writer.writerow(headers)

    def reset(self):
        self._initialize_state()

    def __repr__(self):
        return "Classic"

    def _decode_signal(self, signal_tuple):
        if signal_tuple == (1, 1, 1):
            return 'GREEN'
        elif signal_tuple == (1, 0, 0):
            return 'LEFT'
        elif signal_tuple == (0, 1, 0):
            return 'STRAIGHT'
        elif signal_tuple == (0, 0, 1):
            return 'RIGHT'
        elif signal_tuple == (1, 0, 1):
            return 'LEFT+RIGHT'
        elif signal_tuple == (1, 1, 0):
            return 'LEFT+STRAIGHT'
        elif signal_tuple == (0, 1, 1):
            return 'STRAIGHT+RIGHT'
        else:
            return 'RED'

    def allocate(self, signal_plan: SignalPlan):
        new_state = signal_plan.signals  # hot-encoded (l, s, r) per lane

        # Compute how many vehicles the signal plan could clear
        vehicles_cleared = 0
        priority_passed = 0

        for i, signal in enumerate(new_state):
            if signal[0] == 1:
                vehicles_cleared += sum(self.lanes[i+1]["left_past"][-1:])  # recent left vehicles
            if signal[1] == 1:
                vehicles_cleared += sum(self.lanes[i+1]["straight_past"][-1:])
            if signal[2] == 1:
                vehicles_cleared += sum(self.lanes[i+1]["right_past"][-1:])

            # Priority vehicle passage
            priority_passed += sum(self.lanes[i+1]["priority_past"][-1:]) if any(signal) else 0

        # more robust reward ------------------------------------------
        new_state = signal_plan.signals  # list of (l, s, r) tuples
        num_lanes = len(new_state)

        vehicles_cleared = 0
        priority_passed = 0
        idle_penalty = 0
        starvation_penalty = 0
        red_penalty = 0
        no_change_penalty = 0

        all_red = all(sig == (0, 0, 0) for sig in new_state)

        for i, signal in enumerate(new_state):
            lane_id = i + 1

            # Traffic counts
            left = sum(self.lanes[lane_id]["left_past"][-1:])
            straight = sum(self.lanes[lane_id]["straight_past"][-1:])
            right = sum(self.lanes[lane_id]["right_past"][-1:])
            priority = sum(self.lanes[lane_id]["priority_past"][-1:])
            total = left + straight + right

            # Count cleared vehicles
            cleared = 0
            if signal[0]: cleared += left
            if signal[1]: cleared += straight
            if signal[2]: cleared += right

            vehicles_cleared += cleared
            if any(signal): priority_passed += priority

            # Penalty if vehicles are waiting but no signal
            if total > 0 and not any(signal):
                idle_penalty += 1

            # Starvation: 5 time steps with traffic but no green
            recent_total = self.lanes[lane_id]["total_past"][-5:]
            if sum(recent_total) > 0 and not any(signal):
                starvation_penalty += 1

        # Penalize if signals didn't change
        if new_state == self.prev_state:
            no_change_penalty += 1

        # Penalize if all red and traffic is present
        total_traffic_present = sum(sum(self.lanes[i+1]["total_past"][-1:]) for i in range(num_lanes))
        if all_red and total_traffic_present > 0:
            red_penalty += 1

        # Final reward
        reward = (
            -num_lanes  # base time cost
            + vehicles_cleared * 1.0
            + priority_passed * 5.0
            - idle_penalty * 10.0
            - starvation_penalty * 8.0
            - red_penalty * 8.0
            - no_change_penalty * 8.0
        )

        # Strong negative if nothing helpful is done
        if vehicles_cleared == 0 and priority_passed == 0:
            reward -= 2.0

        # Terminal condition
        done = self.TIME_INSTANT >= self.total_duration

        # Logging
        decoded_state = [self._decode_signal(sig) for sig in new_state]
        with open(self.output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.TIME_INSTANT] + decoded_state)

        print(f"TIME_INSTANT {self.TIME_INSTANT}: Logged new state {decoded_state}")
        
        self.prev_state = self.current_state.copy()
        self.current_state = new_state.copy()
        self.TIME_INSTANT += signal_plan.duration

        return reward, done
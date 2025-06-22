from flask import Flask, render_template, request, jsonify
import csv
import random
import math
import os

app = Flask(__name__)

# Global Variables
lanes = 3
functions = {i: "10 + 5*t" for i in range(lanes)}
high_priority_vehicles = {i: 0 for i in range(lanes)}
simulation_time = 0
directions = ["Left", "Right", "Straight"]

# Safe evaluation context
safe_globals = {
    "__builtins__": None,
    "math": math
}

@app.route("/", methods=["GET", "POST"])
def home():
    global lanes, functions, high_priority_vehicles

    if request.method == "POST":
        lanes = int(request.form.get("lanes", 1))

        functions = {
            i: request.form.get(f"function[{i}]", "0")
            for i in range(lanes)
        }

        high_priority_vehicles = {
            i: int(request.form.get(f"high_priority[{i}]", 0))
            for i in range(lanes)
        }

        return jsonify({"message": "Simulation updated!"})

    return render_template("index.html", lanes=lanes,
                           functions=functions,
                           high_priority_vehicles=high_priority_vehicles)

@app.route("/simulate", methods=["POST"])
def simulate():
    global simulation_time, lanes, functions, high_priority_vehicles

    simulation_time += 1
    vehicles = {}
    high_priority_counts = {}

    for i in range(lanes):
        safe_locals = {"t": simulation_time}
        try:
            val = eval(functions.get(i, "0"), safe_globals, safe_locals)
            vehicles[i] = max(0, int(val))  # prevent negative values
        except:
            vehicles[i] = 0

        high_priority_counts[i] = high_priority_vehicles.get(i, 0)

    return jsonify({
        "time": simulation_time,
        "vehicles": vehicles,
        "high_priority": high_priority_counts
    })

@app.route("/simulate_hour", methods=["POST"])
def simulate_hour():
    global lanes, functions, high_priority_vehicles

    lane_logs = {i: [] for i in range(lanes)}

    for t in range(1, 61):  # 60 minutes
        for i in range(lanes):
            safe_locals = {"t": t}
            try:
                val = eval(functions.get(i, "0"), safe_globals, safe_locals)
                num_vehicles = max(0, int(val))  # avoid negatives
            except:
                num_vehicles = 0

            num_high_priority = high_priority_vehicles.get(i, 0)

            for v in range(num_vehicles):
                vehicle_id = f"Lane{i+1}_V{v+1}_T{t}"
                priority = "High" if v < num_high_priority else "Normal"
                direction = random.choice(directions)
                distance = round(random.uniform(10.0, 100.0), 2)  # in meters

                lane_logs[i].append([
                    vehicle_id, i + 1, priority, direction, t, distance
                ])

    for i in range(lanes):
        directory = 'train_simulation'
        os.makedirs(directory, exist_ok=True)
        filename = f"train_simulation/lane_{i+1}.csv"
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Vehicle ID", "Lane", "Priority", "Direction", "Entry Time", "Distance"])
            writer.writerows(lane_logs[i])

    return jsonify({"message": "1-hour simulation logged successfully!"})

@app.route("/generate_function")
def generate_function():
    func_type = random.choice(["linear", "quadratic", "cubic", "sine", "cosine", "mixed"])

    if func_type == "linear":
        func = f"{random.randint(10, 30)} + {random.randint(5, 10)}*t"
    elif func_type == "quadratic":
        func = f"{random.randint(1, 3)}*t**2 + {random.randint(5, 10)}*t + {random.randint(10, 20)}"
    elif func_type == "cubic":
        func = f"{random.randint(1, 2)}*t**3 + {random.randint(2, 4)}*t**2 + {random.randint(5, 10)}*t + {random.randint(5, 15)}"
    elif func_type == "sine":
        func = f"{random.randint(15, 25)} + {random.randint(5, 15)}*math.fabs(math.sin(t / {random.randint(2, 6)}))"
    elif func_type == "cosine":
        func = f"{random.randint(15, 25)} + {random.randint(5, 15)}*math.fabs(math.cos(t / {random.randint(2, 6)}))"
    elif func_type == "mixed":
        func = (
            f"{random.randint(10, 20)} + "
            f"{random.randint(2, 5)}*t + "
            f"{random.randint(1, 3)}*math.fabs(math.sin(t/5)) + "
            f"{random.randint(1, 2)}*t**2"
        )
    else:
        func = "10 + 5*t"

    return jsonify({"function": func})

if __name__ == "__main__":
    app.run(debug=True)

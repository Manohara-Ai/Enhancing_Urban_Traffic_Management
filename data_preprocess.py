import os
import glob
import pandas as pd
from typing import Optional
import csv
import re

# Function to create the output folder if it doesn't exist
def create_output_folder(folder_name="train"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")

# Extract lane number from the file name (e.g., lane_1.csv -> 1)
def extract_lane_number(filename):
    base = os.path.basename(filename)
    return base.split("_")[1].split(".")[0]

# Count vehicles per lane, with optional filtering by vehicle type and direction
def count_vehicles_per_lane(file_path: str,
                            vehicle_type: Optional[str] = None,
                            direction: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    lane_num = extract_lane_number(file_path)

    # Safely filter on vehicle priority (Priority == "High")
    if vehicle_type:
        if 'Priority' not in df.columns:
            print(f"Warning: Skipping '{file_path}' - missing 'Priority' column.")
            return pd.DataFrame(columns=["Entry Time", f"Lane_{lane_num}"])
        df = df[df['Priority'] == vehicle_type]

    # Safely filter on direction
    if direction:
        if 'Direction' not in df.columns:
            print(f"Warning: Skipping '{file_path}' - missing 'Direction' column.")
            return pd.DataFrame(columns=["Entry Time", f"Lane_{lane_num}"])
        df = df[df['Direction'] == direction]

    if df.empty:
        return pd.DataFrame(columns=["Entry Time", f"Lane_{lane_num}"])

    count = df.groupby("Entry Time").size().reset_index(name=f"Lane_{lane_num}")
    return count

# Merge lane data for all files and save as a CSV
def merge_lane_data(folder_path: str = "simulation",
                    output_file: str = "vehicle_counts.csv",
                    vehicle_type: Optional[str] = None,
                    direction: Optional[str] = None,
                    output_folder: str = "train") -> pd.DataFrame:
    # Create the output folder if it doesn't exist
    create_output_folder(output_folder)
    
    csv_files = sorted(glob.glob(os.path.join(folder_path, "lane_*.csv")))
    if not csv_files:
        raise FileNotFoundError("No lane CSV files found in the specified folder.")

    # Get all unique timestamps across all lane files
    all_timestamps = set()
    for file in csv_files:
        df = pd.read_csv(file)
        all_timestamps.update(df["Entry Time"].unique())

    # Sort timestamps to get a full range (if any are missing)
    all_timestamps = sorted(all_timestamps)

    lane_dfs = []

    # Process each lane file
    for file in csv_files:
        df = count_vehicles_per_lane(file, vehicle_type, direction)

        # If no data, we will create an empty frame with the full list of timestamps
        if df.empty:
            df = pd.DataFrame({"Entry Time": all_timestamps, f"Lane_{extract_lane_number(file)}": [0] * len(all_timestamps)})
        
        # Reindex the data to include all timestamps (add missing timestamps with 0s)
        df = df.set_index("Entry Time").reindex(all_timestamps, fill_value=0).reset_index()
        df.columns = ["Entry Time", f"Lane_{extract_lane_number(file)}"]
        lane_dfs.append(df)

    # Merge all lane dataframes on 'Entry Time'
    merged_df = lane_dfs[0]
    for df in lane_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="Entry Time", how="outer")

    # Save the merged data to the specified folder
    output_path = os.path.join(output_folder, output_file)
    merged_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    return merged_df

# Generate all vehicle statistics (all, priority, by direction)
def generate_all_vehicle_statistics(folder_path: str = "train_simulation", output_folder: str = "train"):
    # All vehicles (normal + priority)
    merge_lane_data(folder_path=folder_path, output_file="vehicle_counts.csv", output_folder=output_folder)

    # Priority vehicles (high priority only)
    merge_lane_data(folder_path=folder_path, output_file="vehicle_counts_priority.csv", vehicle_type="High", output_folder=output_folder)

    # Vehicles by direction
    merge_lane_data(folder_path=folder_path, output_file="vehicle_counts_straight.csv", direction="Straight", output_folder=output_folder)
    merge_lane_data(folder_path=folder_path, output_file="vehicle_counts_left.csv", direction="Left", output_folder=output_folder)
    merge_lane_data(folder_path=folder_path, output_file="vehicle_counts_right.csv", direction="Right", output_folder=output_folder)

# Process vehicle data 
def process_vehicle_data(vehicle_id, info, lane_number):
    direction = info.get("indicatorStatus", "Straight")
    if direction == "Neutral":
        direction = "Straight"
    priority = "High" if info.get("priorityVehicle", "No") == "Yes" else "Normal"
    entry_time = info.get("timestamp", "N/A")
    distance = info.get("distance", "N/A")
    return [vehicle_id, lane_number, priority, direction, entry_time, distance]

# Write data to csv
def write_lane_csv(lane_number, lane_rows, output_folder, header):
    os.makedirs(output_folder, exist_ok=True)
    lane_filename = os.path.join(output_folder, f"lane_{lane_number}.csv")
    with open(lane_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(lane_rows)
    print(f"Saved: {lane_filename}")

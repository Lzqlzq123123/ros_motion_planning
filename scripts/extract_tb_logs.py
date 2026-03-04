#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tensorboard_data(log_dir, base_output_dir='/data/lzq/data'):
    if not os.path.exists(log_dir):
        print(f"Error: Log directory '{log_dir}' does not exist.")
        return

    # Extract the run name from the log directory path (e.g., 'run8')
    run_name = os.path.basename(os.path.normpath(log_dir))
    output_dir = os.path.join(base_output_dir, run_name)
    
    print(f"Loading TensorBoard logs from {log_dir}...")
    
    # Load the event accumulator, size_guidance=0 means load all data
    event_acc = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    event_acc.Reload()
    
    # Get all scalar tags
    tags = event_acc.Tags().get('scalars', [])
    
    if not tags:
        print("No scalar data found in the provided log directory.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Save each tag as a separate CSV file
    for tag in tags:
        events = event_acc.Scalars(tag)
        
        steps = [e.step for e in events]
        values = [e.value for e in events]
        wall_times = [e.wall_time for e in events]
        
        df = pd.DataFrame({
            'wall_time': wall_times,
            'step': steps,
            'value': values
        })
        
        # Format tag name to be a valid filename
        safe_tag = tag.replace('/', '_').replace(' ', '_')
        csv_path = os.path.join(output_dir, f"{safe_tag}.csv")
        
        df.to_csv(csv_path, index=False)
        print(f"Saved {tag} to {csv_path}")
        
    print(f"\nAll data successfully extracted to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract TensorBoard logs to CSV')
    parser.add_argument('log_dir', type=str, help='Path to the TensorBoard log directory (e.g., src/rl_training/logs/experiment_name/run8)')
    parser.add_argument('--out_base', type=str, default='src/rl_training/data/', help='Base directory to save the CSV files (default: /data/lzq/data)')
    
    args = parser.parse_args()
    extract_tensorboard_data(args.log_dir, args.out_base)

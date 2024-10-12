#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:18:38 2024

@author: brianhirschfield
"""
from mining_oop2 import *

#if __name__ == "__main__":
miner_list = [
        Miner('S19jPro', 499, 100, 3068),
        Miner('S21XP_Hyd', 10004, 473, 5676),
        Miner('S21XP_Imm', 7439, 300, 4050),
        Miner('S21XP', 6729, 270, 3645),
        Miner('S21Pro', 4979, 234, 3531),
        Miner('S21', 4109, 200, 3550),
        Miner('M63S', 8004, 390, 7215),
        Miner('M63', 6354, 270, 6646),
        Miner('M60S+', 4924, 212, 3600),
        Miner('Bitaxe-nano3', 300, 3, 140)
        # Add other miners as necessary
    ]
    
simulator = MiningSimulator(miner_list, network_eh=600, btc_price=60000, epochs=10)
results = MiningResults(miner_list, simulator)

    # Generate the data with assumptions
results_df = results.calculate_cope_multiple(diff_adj=0.01, 
                                             fees=0.6, 
                                             per_kwh_cost=0.06, 
                                             btc_offset=False, 
                                             clocking=True, 
                                             clock_ratio=0.05, 
                                             clockingblocks=40000)

    # Plot the results without needing to re-enter the assumptions
plotter = MiningPlotter()
plotter.plot_profitability(results_df)

import os
import pandas as pd

# Define a function to handle plotting and saving
def plot_and_save(results_df, save_path):
    plotter = MiningPlotter()
    plotter.plot_profitability(results_df, save_path=save_path)

# Define the sets of assumptions
diff_adj_values = [0.01, 0.02]  # Example values for difficulty adjustment
per_kwh_cost_values = [0.06, 0.08]  # Example values for electricity cost
btc_offset_values = [True, False]  # Toggle BTC offset
clocking_values = [True, False]  # Toggle clocking
clock_ratio_values = [0.05, 0.10]  # Clocking ratios
clockingblocks_values = [20000, 40000]  # Clocking blocks
fees = 0.6  # Fixed value for fees

# Directory to save plots
save_dir = 'miner_profitability_plots'
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

# List to store all DataFrames for concatenation
all_results_list = []

# Loop through combinations of assumptions and calculate results
for diff_adj in diff_adj_values:
    for per_kwh_cost in per_kwh_cost_values:
        for btc_offset in btc_offset_values:
            for clocking in clocking_values:
                for clock_ratio in clock_ratio_values:
                    for clockingblocks in clockingblocks_values:
                            # Generate a descriptive key for the assumptions
                        key = (f"diff_adj_{diff_adj}_per_kwh_{per_kwh_cost}_btc_offset_{btc_offset}"
                               f"_clocking_{clocking}_clock_ratio_{clock_ratio}_clockingblocks_{clockingblocks}")

                        # Calculate results for this set of assumptions
                        results_df = results.calculate_cope_multiple(diff_adj=diff_adj, 
                                                                     fees=fees, 
                                                                     per_kwh_cost=per_kwh_cost, 
                                                                     btc_offset=btc_offset, 
                                                                     clocking=clocking, 
                                                                     clock_ratio=clock_ratio, 
                                                                     clockingblocks=clockingblocks)
                        
                        # Add assumption columns to the results_df
                        results_df['diff_adj'] = diff_adj
                        results_df['per_kwh_cost'] = per_kwh_cost
                        results_df['btc_offset'] = btc_offset
                        results_df['clocking'] = clocking
                        results_df['clock_ratio'] = clock_ratio
                        results_df['clockingblocks'] = clockingblocks

                        # Append the results_df to the all_results_list
                        all_results_list.append(results_df.copy())  # Ensure a copy is added, not a reference

                        # Generate a file name for the plot
                        file_name = f"{key}.png"
                        save_path = os.path.join(save_dir, file_name)

                        # Call the plotting function and save the plot
                        plot_and_save(results_df, save_path=save_path)
                            
if all_results_list:  # Ensure the list is not empty
    combined_results_df = pd.concat(all_results_list, ignore_index=True)

    # Save combined results to CSV
    combined_results_df.to_csv('combined_results.csv', index=False)

    # Optionally, display the combined DataFrame
    print(combined_results_df.head())
else:
    print("No results were generated.")

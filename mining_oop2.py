#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:50:39 2024

@author: brianhirschfield
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Miner Class
class Miner:
    def __init__(self, name, price, hashrate, power):
        self.name = name
        self.price = price
        self.hashrate = hashrate
        self.power = power

# Class to handle CSV processing and Miner object creation
class MinerDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def csv_to_miner_list(self):
        # Load the CSV file
        df = pd.read_csv(self.file_path)
        
        # Extract relevant columns (modify according to your CSV structure)
        df_miners = df[['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 4']]
        df_miners.columns = ['model', 'hashrate', 'joules_per_th', 'unit_price']
        
        # Drop rows with missing miner model, hashrate, J/T, or unit price
        df_miners = df_miners.dropna(subset=['model', 'hashrate', 'joules_per_th', 'unit_price'])

        # Convert relevant columns to numeric
        df_miners['hashrate'] = pd.to_numeric(df_miners['hashrate'], errors='coerce')
        df_miners['joules_per_th'] = pd.to_numeric(df_miners['joules_per_th'], errors='coerce')
        # Clean and convert unit price (remove $ signs, commas, etc.)
        df_miners['unit_price'] = df_miners['unit_price'].replace({'\$': '', ',': ''}, regex=True)
        df_miners['unit_price'] = pd.to_numeric(df_miners['unit_price'], errors='coerce')  # Convert to float

        # Remove non-numeric rows in the unit price
        df_miners = df_miners[df_miners['unit_price'].notna()]
        
        # Create a list of Miner objects
        miner_list = []
        for index, row in df_miners.iterrows():
            power = row['hashrate'] * row['joules_per_th']  # Calculate power in watts
            miner = Miner(name=row['model'], price=row['unit_price'], hashrate=row['hashrate'], power=power)
            miner_list.append(miner)
        
        return miner_list

    # Example usage:
        # file_path = 'path_to_your_csv_file.csv'
        # processor = MinerDataProcessor(file_path)
        # miner_list = processor.csv_to_miner_list()

# Mining Simulator Class
class MiningSimulator:
    def __init__(self, miners, network_eh, btc_price, epochs):
        self.miners = miners
        self.network_eh = network_eh
        self.btc_price = btc_price
        self.epochs = epochs

    def generate_mining_df(self, 
                           miner, 
                           diff_adj, 
                           per_kwh_cost, 
                           fees, 
                           btc_offset, 
                           clocking,
                           clock_ratio,
                           clockingblocks):
        
        network_hash = self.network_eh * 1e6  # Convert EH to TH
        
        # Build the initial DataFrame for mining simulation
        block = {'blockheight': range(1, 210000 * self.epochs)}
        df = pd.DataFrame(block)
        df['miner'] = miner
        df['miner_name'] = miner.name
        df['difficulty_epoch'] = np.floor((df.blockheight - 1) / 2016.0) + 1
        df['mining_epoch'] = np.floor((df.blockheight - 1) / 210000.0) + 1
        df['day'] = np.floor((df.blockheight - 1) / 144.0) + 1
        df['year'] = np.floor((df.day - 1) / 365.0) + 1

        # Difficulty Adjustment
        diff = [1] * 2 + [1 * (1 + diff_adj)**i for i in range(2, max(df.difficulty_epoch.astype(int)))]
        difficulty = pd.DataFrame({'difficulty_adj': diff})
        difficulty.index.names = ['difficulty_epoch']
        df = pd.merge(df, difficulty, on='difficulty_epoch', how='inner')
        df['adj_network_hash'] = network_hash * df['difficulty_adj']

        # Mining Block Reward
        reward = [6.5, 3.125] + [3.125 / (2**i) for i in range(2, max(df['mining_epoch'].astype(int)))]
        blk_reward = pd.DataFrame({'blk_reward': reward})
        blk_reward.index.names = ['mining_epoch']

        # Merge block reward and calculate earnings
        df = pd.merge(df, blk_reward, on='mining_epoch', how='inner')
        df['fees'] = fees
        df['raw_earnings_sats'] = (miner.hashrate / df['adj_network_hash']) * (df['blk_reward'] + fees) * 1e8
  
        # Clocking Logic
        if clocking:
        # Apply clocking for the first `clockingblocks` blocks
            df.loc[:clockingblocks-1, 'earnings_sats'] = df['raw_earnings_sats'][:clockingblocks] * (1 + clock_ratio)
        
        # Apply unclocked rewards for the rest of the blocks
            df.loc[clockingblocks:, 'earnings_sats'] = df['raw_earnings_sats'][clockingblocks:]
        else:
        # No clocking, just use raw earnings
            df['earnings_sats'] = df['raw_earnings_sats']
    
    # Calculate earnings in dollars
        df['earnings_dols'] = df['earnings_sats'] / 1e8 * self.btc_price

    # Electricity cost
        df['elec_cost_dols'] = per_kwh_cost * miner.power / 6000
        if btc_offset:
            df['elec_cost_sats'] = df['elec_cost_dols'] * 1e8 / self.btc_price / df['difficulty_adj']
        else:
            df['elec_cost_sats'] = df['elec_cost_dols'] * 1e8 / self.btc_price

        # Electricity cost
        df['elec_cost_dols'] = per_kwh_cost * miner.power / 6000
        if btc_offset:
            df['elec_cost_sats'] = df['elec_cost_dols'] * 1e8 / self.btc_price / df['difficulty_adj']
        else:
            df['elec_cost_sats'] = df['elec_cost_dols'] * 1e8 / self.btc_price

        
    # Profitability
        df['blocks_profitable'] = np.sum(df['earnings_sats'] > df['elec_cost_sats'])
    
        return df

# Mining Results Class
class MiningResults:
    def __init__(self, miners, simulator):
        self.miners = miners
        self.simulator = simulator

    def query_results(self, df):
        # Filter profitable blocks
        profitable_df = df[df['earnings_sats'] > df['elec_cost_sats']]
        
        # Aggregate mining data
        grouped_df = profitable_df.groupby(['miner_name']).agg({
            'blocks_profitable': 'mean',
            'earnings_sats': 'sum',
            'earnings_dols': 'sum',
            'elec_cost_sats': 'sum',
            'elec_cost_dols': 'sum'
        }).reset_index()

        # Calculate net profits
        grouped_df['net_profit_sats'] = grouped_df['earnings_sats'] - grouped_df['elec_cost_sats']
        grouped_df['net_profit_dols'] = grouped_df['net_profit_sats'] / 1e8 * self.simulator.btc_price

        return grouped_df

    def calculate_cope_multiple(self,
                            diff_adj, 
                            per_kwh_cost, 
                            fees, 
                            btc_offset, 
                            clocking,
                            clock_ratio,
                            clockingblocks):
        result_list = []
        for miner in self.miners:
            # Correct the argument passing for the simulator method, remove `self`
            df = self.simulator.generate_mining_df(miner, 
                                               diff_adj, 
                                               per_kwh_cost, 
                                               fees, 
                                               btc_offset, 
                                               clocking,
                                               clock_ratio,
                                               clockingblocks)

            result_df = self.query_results(df)
        
            # Calculate cope_multiple
            earnings_dols = result_df['net_profit_dols'].sum()
            cope_multiple = miner.price / earnings_dols if earnings_dols > 0 else float('inf')

            # Store results along with assumptions
            miner_data = {
                'miner_name': miner.name,
                'price': miner.price,
                'hashrate': miner.hashrate,
                'diff_adj': diff_adj,
                'fees': fees,
                'btc_offset': btc_offset,
                'clocking': clocking,
                'clock_ratio': clock_ratio,
                'clockingblocks': clockingblocks,
                'blocks_profitable': np.mean(df['blocks_profitable']),
                'per_kwh_cost': per_kwh_cost,
                'net_profit_sats': result_df['net_profit_sats'].sum(),
                'net_profit_dols': earnings_dols,
                'cope_multiple': cope_multiple
            }
            result_list.append(miner_data)

        return pd.DataFrame(result_list)
    
# Mining Plotter Class
class MiningPlotter:
    @staticmethod
    def plot_profitability(df, save_path=None):
        # Close any open figures
        plt.close('all')
        
        # Extract unique assumption values from the DataFrame
        diff_adj = df['diff_adj'].unique()[0]
        per_kwh_cost = df['per_kwh_cost'].unique()[0]
        fees = df['fees'].unique()[0]
        btc_offset = df['btc_offset'].unique()[0]
        clocking = df['clocking'].unique()[0]
        clock_ratio = df['clock_ratio'].unique()[0]
        clockingblocks = df['clockingblocks'].unique()[0]
        
        # Group the DataFrame by miner_name
        df_grouped = df.groupby('miner_name').agg({
            'price': 'mean',
            'hashrate': 'mean',
            'net_profit_dols': 'mean',
            'net_profit_sats': 'mean',
            'cope_multiple': 'mean',
            'blocks_profitable': 'mean'
        }).reset_index()

        # Plotting
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = sns.color_palette("deep", n_colors=len(df_grouped))
        
        bars = ax.bar(df_grouped['miner_name'], df_grouped['net_profit_dols'], color=colors)
        
        # Labels
        ax.set_xticks(np.arange(len(df_grouped)))
        ax.set_xticklabels(df_grouped['miner_name'], rotation=45, ha='right', fontsize=16)

        ax.set_xlabel('Miner (Hashrate)', fontsize=16)
        ax.set_ylabel('Net Profit (USD)', fontsize=16)
        ax.set_title('Miner Profitability', fontsize=38)
        ax.grid(True, axis='y', linestyle='--')

        for bar, cope_multiple, net_profit_sats, price in zip(bars, df_grouped['cope_multiple'], df_grouped['net_profit_sats'], df_grouped['price']):
            yval = bar.get_height()
            
            # Position the price inside the bar (white text)
            ax.text(bar.get_x() + bar.get_width() / 2, yval / 2,  # Centered inside the bar
                    f'price:\n${price:,.0f} ',  # price label
                    ha='center', va='center', fontsize=12, fontweight='bold', color='white')  # White text

            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'Cope: {cope_multiple:.2f}\n{net_profit_sats:,.0f} sats',
                    ha='center', va='bottom')

        # Adjust layout before adding text
        plt.tight_layout()

        # Update btc_offset description
        if btc_offset:
            btcoffset_str = f"Bitcoin offsets electricity costs by 30% a year"
        else:
            btcoffset_str = f"Bitcoin does not offset electricity costs"

        # Add overclocking information
        if clocking:
            overclocking_str = f"Overclock at {clock_ratio*100:.1f}% for {clockingblocks} blocks"
        else:
            overclocking_str = "No overclocking"

        # Now add assumptions text box after layout adjustment
        assumptions_text = (
            f"Assumptions:\n"
            f"- Difficulty Adjustment: {diff_adj * 100:.1f}% per 2016 blocks\n"
            f"- Fees: {fees:.1f} BTC per block\n"
            f"- Electricity Cost: ${per_kwh_cost:.2f} per kWh\n"
            f"- {btcoffset_str}\n"
            f"- {overclocking_str}"  # Add overclocking information here
        )

        # Place the assumptions box
        plt.gcf().text(0.1, 0.785, assumptions_text, fontsize=17, va='center', ha='left',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

        # Save the plot if a save_path is provided
        if save_path:
            plt.savefig(save_path, format='png')

        plt.show()
        


# Example Usage
if __name__ == "__main__":
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

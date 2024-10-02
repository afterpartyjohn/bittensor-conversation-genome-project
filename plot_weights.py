import matplotlib.pyplot as plt
import numpy as np
import bittensor as bt
import torch
import csv
from datetime import datetime
import re

class MockValidator():
    def __init__(self, hotkey, config=None):
        self.hotkey = hotkey
        self.stake = 0
        self.weights = torch.zeros(256)
        self.scores = torch.zeros(256)
        self.ema_scores = torch.zeros(256)
        self.real_weights = torch.zeros(256)

    def __str__(self):
        return f"MockValidator(hotkey={self.hotkey}, stake={self.stake})"

    def __repr__(self):
        return self.__str__()

# Initialize variables to calculate stake-weighted averages
total_stake = 0
weighted_sums = None
real_weighted_sums = None

# Load the CSV files
weights_csv = 'weights_400_1vali_25percentsplit.csv'
stake_csv = 'stakes.csv'

# Initialize a dictionary to store validators by hotkey
validators_dict = {}

# Read the stake CSV and initialize MockValidators
with open(stake_csv, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        hotkey = row['hotkey']
        stake_str = row['stake']
        # Use regex to extract the numeric value from the string
        match = re.search(r'tensor\(([\d.]+)\)', stake_str)
        if match:
            stake = float(match.group(1))
        else:
            stake = float(stake_str)
        if hotkey not in validators_dict.keys():
            validators_dict[hotkey] = MockValidator(hotkey)
            validators_dict[hotkey].stake = stake
        else:
            print(f"Duplicate hotkey found in stake CSV: {hotkey}. Skipping.")

# Read the weights CSV and assign weights to the corresponding MockValidators
with open(weights_csv, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        hotkey = row['hotkey']
        weights = np.array(eval(row['weights']))  # Convert string representation of list to numpy array
        if hotkey in validators_dict:
            if np.any(validators_dict[hotkey].weights.numpy() != 0):
                print(f"Duplicate hotkey found in weights CSV: {hotkey}. Skipping.")
            else:
                validators_dict[hotkey].weights = weights
        else:
            print(f"Hotkey {hotkey} found in weights CSV but not in stake CSV. Skipping.")

# Convert the dictionary to a list
validator_list = list(validators_dict.values())

subtensor = bt.subtensor(network="finney")
metagraph = bt.metagraph(netuid=33,lite=False)

for validator in validator_list:
    print(f"examining validator: {hotkey}")


    uid = subtensor.get_uid_for_hotkey_on_subnet(validator.hotkey, 33)
    #print(f"REAL WEIGHTS: {metagraph.W[uid]}")
    if uid is None:
        continue
    validator.real_weights = metagraph.W[uid]


    # Ensure weights and stakes are numpy arrays for element-wise operations
    weights = np.array(validator.weights)
    stakes = np.array(validator.stake)

    realweights = np.array(validator.real_weights)

    # Initialize weighted_sums if it's the first iteration
    if weighted_sums is None:
        weighted_sums = np.zeros_like(weights, dtype=float)
    
    if real_weighted_sums is None:
        real_weighted_sums = np.zeros_like(realweights, dtype=float)
    

    # Update total stake and weighted sums
    total_stake += stakes
    weighted_sums += weights * stakes
    real_weighted_sums += realweights * stakes

    # Extract weights and sort them in ascending order
    sorted_weights = sorted(weights)
    # Exclude 0's from sorted weights
    sorted_weights = [weight for weight in sorted_weights if weight != 0]
    #print(sorted_weights)

    # Plot the sorted weights
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_weights, marker='o', linestyle='-', color='b')
    plt.xlabel('Index')
    plt.ylabel('Weight Value')
    plt.title(f'Validator Weights for hotkey: {validator.hotkey}')
    plt.grid(True)

# Calculate the overall stake-weighted average for each index
stake_weighted_averages = weighted_sums / total_stake if total_stake != 0 else np.zeros_like(weighted_sums)
#print(f"Overall TEST stake-weighted averages: {stake_weighted_averages}")
#print(f"\n\n LENGTH OF SWA TABLE {len(stake_weighted_averages)}\n\n")
# Order the stake-weighted averages in ascending order

real_stake_weighted_averages = real_weighted_sums / total_stake if total_stake != 0 else np.zeros_like(real_weighted_sums)
#print(f"Overall REAL stake-weighted averages: {real_stake_weighted_averages}")
#print(f"\n\n LENGTH OF SWA TABLE {len(real_stake_weighted_averages)}\n\n")
# Order the stake-weighted averages in ascending order

# Normalize stake_weighted_averages
if np.sum(stake_weighted_averages) > 0:
    normalized_stake_weighted_averages = stake_weighted_averages / np.sum(stake_weighted_averages)
else:
    normalized_stake_weighted_averages = np.zeros_like(stake_weighted_averages)

# Normalize real_stake_weighted_averages
if np.sum(real_stake_weighted_averages) > 0:
    normalized_real_stake_weighted_averages = real_stake_weighted_averages / np.sum(real_stake_weighted_averages)
else:
    normalized_real_stake_weighted_averages = np.zeros_like(real_stake_weighted_averages)

#print(f"Normalized TEST stake-weighted averages: {normalized_stake_weighted_averages}")
#print(f"Normalized REAL stake-weighted averages: {normalized_real_stake_weighted_averages}")



sorted_stake_weighted_averages = sorted(normalized_stake_weighted_averages)

real_sorted_stake_weighted_averages = sorted(normalized_real_stake_weighted_averages)

# Plot the sorted stake-weighted averages
plt.figure(figsize=(10, 6))
plt.plot(sorted_stake_weighted_averages, marker='o', linestyle='-', color='g')
plt.xlabel('Index')
plt.ylabel('TEST Stake-weighted Average Value')
plt.title('TEST Stake-weighted Averages')
plt.grid(True)

# Plot the sorted stake-weighted averages
plt.figure(figsize=(10, 6))
plt.plot(real_sorted_stake_weighted_averages, marker='o', linestyle='-', color='g')
plt.xlabel('Index')
plt.ylabel('REAL Stake-weighted Average Value')
plt.title('REAL Stake-weighted Averages')
plt.grid(True)
plt.show()
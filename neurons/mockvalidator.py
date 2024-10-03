# The MIT License (MIT)
# Copyright © 2024 Conversation Genome Project

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import time
import os
import hashlib
import random
import pprint
import csv
import sys
import torch
import tracemalloc
import hashlib
import pickle
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import shutil
import mmap
from typing import Optional, Tuple


import bittensor as bt

from conversationgenome.base.validator import BaseValidatorNeuron

import conversationgenome.utils
import conversationgenome.validator

from conversationgenome.ConfigLib import c
from conversationgenome.utils.Utils import Utils

from conversationgenome.analytics.WandbLib import WandbLib

from conversationgenome.validator.ValidatorLib import ValidatorLib
from conversationgenome.validator.evaluator import Evaluator
from conversationgenome.llm.LlmLib import LlmLib

from conversationgenome.protocol import CgSynapse

def normalize_max_weight(x: torch.Tensor, limit: float) -> torch.Tensor:
    """
    Normalizes the input tensor so that the sum is equal to 1 and the max value is limited.

    Args:
        x (torch.Tensor): Input tensor to normalize.
        limit (float): The maximum allowed value for any element in the tensor.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    if x.sum() > 0:
        x = x / x.sum()
    max_weight = x.max()
    if max_weight > limit:
        x = x * (limit / max_weight)
    return x

def process_weights_for_netuid(
    uids: torch.Tensor,
    weights: torch.Tensor,
    netuid: int,
    subtensor: "Subtensor",
    metagraph: None,
    exclude_quantile: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Processes weight tensors for a given subnet id using the provided weight and UID tensors,
    applying constraints and normalization based on the subtensor and metagraph data.

    Args:
        uids (torch.Tensor): Tensor of unique identifiers of the neurons.
        weights (torch.Tensor): Tensor of weights associated with the user IDs.
        netuid (int): The network uid to process weights for.
        subtensor (Subtensor): Subtensor instance to access blockchain data.
        metagraph (Optional[Metagraph]): Metagraph instance for additional network data.
            If None, it is fetched from the subtensor using the netuid.
        exclude_quantile (int): Quantile threshold for excluding lower weights. Defaults to 0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the tensor of user IDs and
        the corresponding normalized weights.
    """
    # Get latest metagraph from chain if metagraph is None.
    #if metagraph is None:
        #metagraph = subtensor.metagraph(netuid)

    # Ensure weights are float32
    weights = weights.to(torch.float32)

    # Network configuration parameters from subtensor
    U16_MAX = 65535  # Constant for maximum value of uint16
    quantile = exclude_quantile / U16_MAX
    min_allowed_weights = 1
    max_weight_limit = 65535

    # Find all non-zero weights
    non_zero_weight_idx = torch.nonzero(weights > 0).squeeze(dim=1)
    non_zero_weight_uids = uids[non_zero_weight_idx]
    non_zero_weights = weights[non_zero_weight_idx]

    if non_zero_weights.numel() == 0 or 256 < min_allowed_weights:
        logging.warning("No non-zero weights returning all ones.")
        final_weights = torch.ones(256, "cuda") / 256
        final_weights_count = torch.arange(len(final_weights), "cuda")
        return final_weights_count, final_weights

    elif non_zero_weights.numel() < min_allowed_weights:
        logging.warning("Non-zero weights less than min allowed weight, returning all ones.")
        weights = torch.ones(256, device="cuda") * 1e-5
        weights[non_zero_weight_idx] += non_zero_weights
        normalized_weights = normalize_max_weight(weights, max_weight_limit)
        return torch.arange(len(normalized_weights), device="cuda"), normalized_weights

    # Compute the exclude quantile and find the weights in the lowest quantile
    max_exclude = max(0, len(non_zero_weights) - min_allowed_weights) / len(non_zero_weights)
    exclude_quantile = min(quantile, max_exclude)
    lowest_quantile = torch.quantile(non_zero_weights, exclude_quantile)

    # Exclude all weights below the allowed quantile
    mask = lowest_quantile <= non_zero_weights
    non_zero_weight_uids = non_zero_weight_uids[mask]
    non_zero_weights = non_zero_weights[mask]

    # Normalize weights and return
    normalized_weights = normalize_max_weight(non_zero_weights, max_weight_limit)
    
    return non_zero_weight_uids, normalized_weights



def get_weights(scores,subtensor):
    weight_time_zero = time.time()
    #print("setting weights starting:")
    """
    Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
    """
    # Check if self.scores contains any NaN values and log a warning if it does.
    if torch.isnan(scores).any():
        bt.logging.warning(
            f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
        )

    # Calculate the average reward for each uid across non-zero values.
    # Replace any NaN values with 0.
    raw_weights = torch.nn.functional.normalize(scores, p=1, dim=0)
    weight_time_one = time.time()
    #print(f"normalize raw weights took: {weight_time_one-weight_time_zero}")

    #print("raw_weights", raw_weights)
    #bt.logging.debug("raw_weight_uids", metagraph.uids.to("cpu"))
    # Process the raw weights to final_weights via subtensor limitations.
    #(
    #    processed_weight_uids,
    #    processed_weights,
    #) = bt.utils.weight_utils.process_weights_for_netuid(
    #    uids=torch.arange(256, dtype=torch.int64).to("cuda"),
    #    weights=raw_weights.to("cuda"),
    #    netuid=33,
    #    subtensor=subtensor,
    #    metagraph=None
    #)

    #weight_time_two = time.time()
    (
        processed_weight_uids,
        processed_weights,
    ) = process_weights_for_netuid(
        uids=torch.arange(256, dtype=torch.int64).to("cuda"),
        weights=raw_weights.to("cuda"),
        netuid=33,
        subtensor=subtensor,
        metagraph=None
    )
    #weight_time_local = time.time()
    #print(f"local Process weights function took: {weight_time_local-weight_time_two}")

    #weight_time_two = time.time()
    #print(f"bt.utils.process_weights_for_netuid took: {weight_time_two-weight_time_local}")
    #print("processed_weights", processed_weights)
    #print("processed_weight_uids", processed_weight_uids)

    # Convert to uint16 weights and uids.
    #(
    #    uint_uids,
    #    uint_weights,
    #) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
    #    uids=processed_weight_uids, weights=processed_weights
    #)
    #weight_time_three = time.time()
    #print(f"bt.utils.weight_utils.convert_weights_and_uids_for_emit took: {weight_time_three - weight_time_two}")
    #bt.logging.debug("uint_weights", uint_weights)
    #bt.logging.debug("uint_uids", uint_uids)
    
    #return the weights

    return processed_weight_uids,processed_weights


def update_scores(rewards: torch.FloatTensor, uids, scores, ema_scores):
    """
    Performs exponential moving average on the scores based on the rewards received from the miners,
    then normalizes, applies a non-linear transformation, and renormalizes the scores.
    """
    ema_scores = ema_scores.to("cpu")

    #print(f"\n\n in update_scores.")
    #print(f"\n\n rewards: {rewards}")
    #print(f"\n\n uids: {uids}")

    rewards = rewards.to("cuda")
    uids = uids.to("cuda")
    scores = scores.to("cuda")
    ema_scores = ema_scores.to("cuda")


    vl = ValidatorLib()
    updated_scores, updated_ema_scores = vl.update_scores(rewards, uids, ema_scores, scores, 0.1, "cpu", 256, 3)

    if torch.numel(updated_scores) > 0 and torch.numel(updated_ema_scores) > 0 and not torch.isnan(updated_scores).any() and not torch.isnan(updated_ema_scores).any():
        scores=updated_scores
        ema_scores=updated_ema_scores
    else: 
        bt.logging.error("Error 2378312: Error with Nonlinear transformation and Renormalization in update_scores. self.scores not updated")

    #print(f"Updated final scores: {scores}")
    return (scores,ema_scores)


async def get_vector_embeddings_set(tags):
        llml = LlmLib()
        #print(f"Starting vector gen for tags: {tags}")
        response = await llml.get_vector_embeddings_set(tags)
        return response


class MockValidator():
    def __init__(self, hotkey,config=None):
        self.hotkey = hotkey
        self.stake = 0
        self.weights = torch.zeros(256)
        self.scores = torch.zeros(256)
        self.ema_scores = torch.zeros(256)
        self.window_count = 0

    def __str__(self):
        return f"MockValidator(hotkey={self.hotkey}, stake={self.stake})"

    def __repr__(self):
        return self.__str__()
    

def process_and_plot_validators(validators, rows_processed, interval, start_time):
    if rows_processed % interval != 0 or validators is None or len(validators) ==0:
        return

    # Convert the dictionary to a list if needed
    validator_list = list(validators.values())

    print(f"Total unique validators processed: {len(validator_list)}")

    # Initialize variables to calculate stake-weighted averages
    total_stake = 0
    weighted_sums = None

    for validator in validator_list:
        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"validator_weights_{timestamp}.csv"

        # Open the CSV file in append mode
        with open(csv_filename, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Write the header if the file is empty
            if csv_file.tell() == 0:
                header = ['hotkey', 'weights']
                csv_writer.writerow(header)
            
            # Write the current validator's hotkey and weights as a single row
            row = [validator.hotkey, validator.weights.tolist()]
            csv_writer.writerow(row)

        # Ensure weights and stakes are numpy arrays for element-wise operations
        weights = np.array(validator.weights)
        stakes = np.array(validator.stake)

        # Initialize weighted_sums if it's the first iteration
        if weighted_sums is None:
            weighted_sums = np.zeros_like(weights, dtype=float)

        # Update total stake and weighted sums
        total_stake += stakes
        weighted_sums += weights * stakes

        # Extract weights and sort them in ascending order
        sorted_weights = sorted(weights)
        # Exclude 0's from sorted weights
        sorted_weights = [weight for weight in sorted_weights if weight != 0]
        print(sorted_weights)
        # Sort the weights from largest to smallest but display the index of the sorted weights
        sorted_indices = np.argsort(weights)[::-1]  # Get indices of sorted weights in descending order
        sorted_weights_with_indices = [(index, weights[index]) for index in sorted_indices]
        print(f"WINDOW COUNT: {validator.window_count}")
        non_zero_weight_count = np.count_nonzero(validator.weights)
        windows_per_id = validator.window_count / non_zero_weight_count if non_zero_weight_count != 0 else 0
        print(f"WINDOWS PER ID: {windows_per_id}")
        print(f"SORTED WEIGHTS BY INDEX: {sorted_weights_with_indices}")

        # Plot the sorted weights
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_weights, marker='o', linestyle='-', color='b')
        plt.xlabel('Index')
        plt.ylabel('Weight Value')
        plt.title(f"Weight distro for vali: {validator.hotkey} RP: {rows_processed}")
        plt.grid(True)

    # Calculate the overall stake-weighted average for each index
    stake_weighted_averages = weighted_sums / total_stake if total_stake != 0 else np.zeros_like(weighted_sums)
    print(f"Overall stake-weighted averages: {stake_weighted_averages}")
    print(f"\n\n LENGTH OF SWA TABLE {len(stake_weighted_averages)}\n\n")
    # Order the stake-weighted averages in ascending order
    sorted_stake_weighted_averages = sorted(stake_weighted_averages)
    
    # Plot the sorted stake-weighted averages
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_stake_weighted_averages, marker='o', linestyle='-', color='g')
    plt.xlabel('Index')
    plt.ylabel('Stake-weighted Average Value')
    plt.title('Final Stake-weighted Averages')
    # Create parent folder with start time
    parent_folder = f"validator_weights_{start_time}"
    os.makedirs(parent_folder, exist_ok=True)

    # Create subfolder for the current number of rows processed
    subfolder = os.path.join(parent_folder, f"{rows_processed} Rows")
    os.makedirs(subfolder, exist_ok=True)

    # Save all open figures to the subfolder
    for i in plt.get_fignums():
        plt.figure(i)
        plot_file_path = os.path.join(subfolder, f"weights_plot_{rows_processed}_{i}.png")
        plt.savefig(plot_file_path)
        plt.close(i)


def save_hash_to_score(hash_to_score):
    hash_score_file_path = "hash_to_score.pkl"
    temp_file_path = None
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile('wb', delete=False) as temp_file:
            temp_file_path = temp_file.name
            pickle.dump(hash_to_score, temp_file)
        
        # Rename the temporary file to the target file
        shutil.move(temp_file_path, hash_score_file_path)
        #print(f"Successfully saved hash_to_score to {hash_score_file_path}")
    except Exception as e:
        print(f"Failed to save hash_to_score: {e}")
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    
async def check_hash(hash_input_string, hash_to_score, tags):
    hash_lookup_starttime = time.time()
    print(f"STARTING HASH LOOKUP on dictionary length: {len(hash_to_score)}")
    if not (hash_input_string) or not tags:
        print("error with hash input string")
        return None
    
    miner_hash_input = hash_input_string
    miner_hash_object = hashlib.sha256(miner_hash_input.encode())
    miner_hash_hex = miner_hash_object.hexdigest()

    # Ensure the hash is correctly generated and checked
    if miner_hash_hex in hash_to_score:
        print("loading stored vector hash")
        # Load the vectors mapped to by that hash
        if hash_to_score[miner_hash_hex] == {}:
            vectors = await get_vector_embeddings_set(tags)
            print(f"in check has, vectors= {vectors}")
            if vectors:
                hash_to_score[miner_hash_hex] = vectors
            else:
                print("Emtpy Vector List loaded, vector retrieval failed")
                return None
        else:
            vectors = hash_to_score[miner_hash_hex]
    else:
        # Generate the vectors
        vectors = await get_vector_embeddings_set(tags)
        # Optionally, you might want to store the generated vectors back to hash_to_score
        if vectors:
            hash_to_score[miner_hash_hex] = vectors
        else:
            print("vector retrieval failed")
            return None
    finish_time = time.time()
    print(f"HASH LOOKUP TOOK: {finish_time-hash_lookup_starttime} SECONDS")

    return vectors
    
    
def check_and_update_validator_list(validators, current_validator_hotkey,stakes, subtensor, metagraph, stakes_file_path ):
    if current_validator_hotkey not in validators:
        validators[current_validator_hotkey] = MockValidator(current_validator_hotkey)

        if current_validator_hotkey in stakes:
            stake = stakes[current_validator_hotkey]
        else:
            uid = subtensor.get_uid_for_hotkey_on_subnet(current_validator_hotkey, 33)
            if uid is None:
                stake = 0
            else:
                stake = metagraph.stake[uid]
            # Write the hotkey/stake combo to the csv stakes.csv
            with open(stakes_file_path, 'a', newline='') as stakes_file:
                csv_writer = csv.writer(stakes_file)
                csv_writer.writerow([current_validator_hotkey, stake])
            stakes[current_validator_hotkey] = stake
        
        validators[current_validator_hotkey].stake = stake


def load_stakes_file(stakes_file_path):
    stakes ={}
    # Check if the file exists
    if os.path.exists(stakes_file_path):
        # Load the stakes from the CSV file
        with open(stakes_file_path, 'r', newline='') as stakes_file:
            csv_reader = csv.DictReader(stakes_file)
            for row in csv_reader:
                hotkey = row['hotkey']
                stake_str = row['stake']
                # Use regex to extract the numeric value from the string
                match = re.search(r'tensor\(([\d.]+)\)', stake_str)
                if match:
                    stake = float(match.group(1))
                else:
                    stake = float(stake_str)
                stakes[hotkey] = stake
    else:
        print(f"File {stakes_file_path} does not exist. Initializing empty stakes dictionary.")
        with open(stakes_file_path, 'w', newline='') as stakes_file:
            csv_writer = csv.writer(stakes_file)
            csv_writer.writerow(['hotkey', 'stake'])
        print(f"File {stakes_file_path} created with columns 'hotkey' and 'stake'.")
    return stakes

def backup_pickle_file(file_path):
    backup_dir = os.path.join(os.path.dirname(file_path), 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    backup_file = []
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir, exist_ok=True)
    else:
        backup_files = sorted(
            [f for f in os.listdir(backup_dir) if os.path.isfile(os.path.join(backup_dir, f))],
            key=lambda x: os.path.getctime(os.path.join(backup_dir, x))
        ) 

    # Check if there are more than 5 files and delete the least recent
    if len(backup_files) > 5:
        files_to_delete = backup_files[:-5]  # Keep only the 5 most recent files
        for file in files_to_delete:
            os.remove(os.path.join(backup_dir, file))

    # Create a backup file with a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file_path = os.path.join(backup_dir, f'hash_to_score_{timestamp}.pkl')

    # Copy the original file to the backup location
    shutil.copy(file_path, backup_file_path)
    print(f"Backup created at {backup_file_path}")

def load_hash_to_score(hash_score_file_path):
    hash_to_score = {}
    if os.path.exists(hash_score_file_path):
        try:
            # Load the hash to score mapping from the pickle file
            with open(hash_score_file_path, 'rb') as hash_score_file:
                hash_to_score = pickle.load(hash_score_file)
        except EOFError:
            print(f"File {hash_score_file_path} is empty or corrupted. Initializing empty hash_to_score dictionary.")
    else:
        print(f"File {hash_score_file_path} does not exist. Initializing empty hash_to_score dictionary.")
    return hash_to_score



async def main():
    import ast
    interval= 100
    subtensor= bt.subtensor(network="finney")
    metagraph=subtensor.metagraph(netuid=33)
    graph_time = time.time()
    print(f"Process started at: {datetime.fromtimestamp(graph_time).strftime('%Y-%m-%d %H:%M:%S')}")

    validators = {}  # Use a dictionary for O(1) lookup
    el = Evaluator()
    valilib = ValidatorLib()
    all_miners = []
    i=0

    # Initialize a dictionary to store the hash to score mapping
    hash_score_file_path = os.path.abspath('hash_to_score_20241002_231636.pkl')  # Ensure the file path is absolute
    hash_to_score = {}
    hash_to_score = load_hash_to_score(hash_score_file_path)

    # Initialize a dictionary to store the stakes
    stakes_file_path = os.path.abspath('./stakes.csv')  # Ensure the file path is absolute
    stakes = {}
    stakes = load_stakes_file(stakes_file_path)


    #isolate desired rows from Subnet Data spreadsheet
    
    validator_rows = []
    miner_index = {}
    csv_file_path = './output.csv'
    with open(csv_file_path, 'r', newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            if row['type'] == 'validator' and row['hotkey'] == '5EhvL1FVkQPpMjZX4MAADcW42i3xPSF1KiCpuaxTYVr28sux':
                validator_rows.append(row)
            elif row['type'] == 'miner':
                keyref = row['keyRef']
                if keyref not in miner_index:
                    miner_index[keyref] = []
                miner_index[keyref].append(row)

        

    #begin iteration through validator rows (simulating full Conversation)
    for row in validator_rows:
        print(i)

        current_validator_hotkey = ""

        #get row data for convo and validator
        keyref = row['keyRef']
        current_validator_hotkey = row['hotkey']

        #check current validator dict and stake dict for validator data
        check_and_update_validator_list(validators, current_validator_hotkey,stakes, subtensor, metagraph, stakes_file_path)
    
        #get Full convo tags from csv row
        full_convo_tags = row['tags']

        #generate List from csv String
        try:
            full_convo_tags = ast.literal_eval(full_convo_tags)
        except ValueError as e:
            print(f"Error parsing tags for validator {current_validator_hotkey}: {e}")
            continue  # Skip this row and move to the next one

        #check hash dict for existence of full convo tag vectors
        validator_output = await check_hash(str(full_convo_tags) + keyref, hash_to_score, full_convo_tags)
        if validator_output:
            full_convo_vectors = validator_output
        else: 
            print("Error in Check Hash - Validator")
            continue

        full_convo_metadata = {}
        full_convo_metadata["vectors"] = full_convo_vectors
        full_convo_metadata["tags"] = full_convo_tags

        # get 3 miners at a time with matching keyref until all miners with that keyref have been queried (simulating Windows)
        three_miners = []
        start_time = time.time()
        for miner_row in miner_index[keyref]:
            print("starting new miner window")
            time_zero= time.time()

            #confirm that it is a miner and they were given same convo
            if miner_row['type'] == 'miner' and miner_row['keyRef'] == keyref:

                # if not already in, add to miner "metagraph"
                if miner_row['hotkey'] not in all_miners:
                    all_miners.append(miner_row['hotkey'])

                #add to current "window" simulation
                three_miners.append(miner_row)
                time_two = time.time()
                print(f"Found miner {miner_row['hotkey']} at index {len(three_miners)} in {time_two-time_zero} seconds")

                #when window is full, score all of them
                if len(three_miners) == 3:

                    miner_uids = []
                    # loop through and generate vectors for each
                    find_vector_start_time = time.time()
                    for miner in three_miners:
                        
                        #generate List from csv String
                        try:
                            miner['tags'] = ast.literal_eval(miner['tags'])
                        except ValueError as e:
                            print(f"Error parsing tags for miner {miner['hotkey']}: {e}")
                            continue  # Skip this miner and move to the next one
                        
                        #check hash dict for existence of this miner's tags for this convo
                        output = await check_hash(str(miner['tags']) + miner['keyRef'], hash_to_score, miner['tags'])
                        if output:
                            miner['vectors'] = output
                        else: 
                            print("Error in Check Hash - Miner")
                            continue

                        miner['uid'] = all_miners.index(miner['hotkey'])
                        miner_uids.append(miner['uid'])

                        if not miner['vectors']or miner['vectors']== {}:
                            print(f"Didn't find Miner Vectors")

                    time_three = time.time()
                    print(f"Generated Three miner vectors in {time_three-find_vector_start_time} seconds")

                    #save hash to score data
                    save_hash_to_score(hash_to_score)


                    # GET SCORES
                    final_scores = await el.mock_evaluate(full_convo_metadata, three_miners)

                    # get fixed scores
                    final_scores, rank_scores = await valilib.assign_fixed_scores(final_scores)

                    #optional - bypass fixed scores by commenting line above, and set rank scores below by uncommenting
                    #rank_scores = []
                    #for score in final_scores:
                        #rank_scores.append(score['final_miner_score'])
                    #print(rank_scores)
                    #rank_scores_tensor = torch.tensor(rank_scores, dtype=torch.float32)

                    #reset three_miners as this is no longer needed
                    three_miners = []


                    rank_scores_tensor = torch.tensor(rank_scores, dtype=torch.float32).to("cuda")  # Move to GPU
                    miner_uids_tensor = torch.tensor(miner_uids, dtype=torch.int64).to("cuda")  # Move to GPU
                    #get updated validator score and ema_score tensors
                    (new_scores, new_ema_scores) = update_scores(rank_scores_tensor, miner_uids_tensor, validators[current_validator_hotkey].scores, validators[current_validator_hotkey].ema_scores)
                    
                    time_four = time.time()
                    print(f"Scored and updated scores {time_four-time_three} seconds")
                    #update current validator's data
                    validators[current_validator_hotkey].scores = new_scores
                    validators[current_validator_hotkey].ema_scores = new_ema_scores
                    validators[current_validator_hotkey].window_count +=1
                
        # Save the hash_to_score dictionary to a file
        save_hash_to_score(hash_to_score)
        backup_pickle_file("hash_to_score.pkl")

        time_five = time.time()
        print(f"Scored and updated scores {time_five-time_four} seconds")

        # Get and set validator weights
        (new_weight_uids,new_weights) = get_weights(validators[current_validator_hotkey].scores,subtensor)
        for uid, weight in zip(new_weight_uids, new_weights):
            validators[current_validator_hotkey].weights[uid] = weight

        time_six = time.time()
        print(f"set weights: {time_six-time_five} seconds")

        i += 1
        if i % 5 == 0:
            print(f"Processed {i} rows so far.")
        process_and_plot_validators(validators, i, interval, graph_time)
            
        if i>1600:
            print("exiting now")
            break
        

    # graph individual validator weight curves
    process_and_plot_validators(validators, i, interval, graph_time)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
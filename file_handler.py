# file_handler.py
# This script's purpose is to handle all file operations.

import os
import numpy as np
import shutil
import torch
import pickle

def write_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)

# Define the function to copy directory contents
def copy_directory_contents(source_dir, dest_dir):
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)

def create_folders():
    # Create the folders if they don't exist
    if not os.path.exists("pkl_files"):
        os.makedirs("pkl_files")
    if not os.path.exists("text_files"):
        os.makedirs("text_files")

def update_text_files(total_rewards):
    previous_rewards_file = 'text_files/previous_rewards.txt'
    average_file = 'text_files/average.txt'

    previous_rewards = 0
    if os.path.exists(previous_rewards_file):
        with open(previous_rewards_file, 'r') as f:
            previous_rewards = int(f.read())

    write_to_file(average_file, f"Previous Reward Avg: {previous_rewards} | Current Reward Avg: {int(np.mean(total_rewards))}")
    write_to_file(previous_rewards_file, str(int(np.mean(total_rewards)))) 

def save_model(agent, save_dir, ending_position, num_in_queue):
    torch.save(agent.local_net.state_dict(), f"{save_dir}/DQN1.pt")
    torch.save(agent.target_net.state_dict(), f"{save_dir}/DQN2.pt")
    torch.save(agent.STATE_MEM, f"{save_dir}/STATE_MEM.pt")
    torch.save(agent.ACTION_MEM, f"{save_dir}/ACTION_MEM.pt")
    torch.save(agent.REWARD_MEM, f"{save_dir}/REWARD_MEM.pt")
    torch.save(agent.STATE2_MEM, f"{save_dir}/STATE2_MEM.pt")
    torch.save(agent.DONE_MEM, f"{save_dir}/DONE_MEM.pt")
    with open(os.path.join(save_dir, "ending_position.pkl"), "wb") as f:
        pickle.dump(ending_position, f)
    with open(os.path.join(save_dir, "num_in_queue.pkl"), "wb") as f:
        pickle.dump(num_in_queue, f)
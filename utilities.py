# utilites.py contains functions that are used in the train.py file.

import os
import gym_super_mario_bros
import numpy as np
import wrapper
import torch
from model import DQNAgent
from rewards import CustomRewardEnv
from file_handler import write_to_file, save_model
import datetime

# Define the total number of worlds and stages in the game.
WORLDS = 8
STAGES_PER_WORLD = 4

# Main function to progress levels.
def progress_level(ep_num, STAGES_PER_WORLD, agent, passed=True, world=None, stage=None, 
                   stage_episode=None, ending_position=None, num_in_queue=None,
                   save_dir=None, num_episodes_per_stage=None):
    if passed:
        save_model(agent, save_dir, ending_position, num_in_queue)
        print(f"Stage {stage} completed in {ep_num} episodes!")
        log_message = f"World {world}-{stage} completed on retry #{stage_episode}.\n"
        with open('text_files/lvl_complete_log.txt', 'a') as f:
            f.write(log_message)
    else:
        print(f"Failed to complete stage {world}-{stage} within {stage_episode} tries!")
        save_model(agent, save_dir, ending_position, num_in_queue)

        # create a log file for the completed level
        log_message = f"World {world}-{stage} skipped after {stage_episode} tries!\n"
        with open('text_files/lvl_complete_log.txt', 'a') as f:
            f.write(log_message)
    
    # Progress to the next stage
    if stage == STAGES_PER_WORLD:
        if world == WORLDS:
            # Reset to World 1 Stage 1
            world = 1
            stage = 1
        else:
            # Progress to the next world
            world += 1
            stage = 1
    else:
        # Progress to the next stage
        stage += 1

    new_save_dir = f"pkl_files/world_{world}_stage_{stage}"
    if not os.path.exists(new_save_dir):
        os.makedirs(new_save_dir)
    save_dir = new_save_dir
    if os.listdir(save_dir):
        agent.pretrained_flag = True
    else:
        agent.pretrained_flag = False
    stage_episode = 0

    # Save the current world and stage to a file
    write_to_file('text_files/current_level.txt', f"{world},{stage}")

    print(f"Progressing to World {world} Stage {stage}")

    # Reset the number of episodes for each new stage
    num_episodes = num_episodes_per_stage

    # Set the new level environment
    level_env = f"SuperMarioBros-{world}-{stage}-v0"
    env = gym_super_mario_bros.make(level_env)
    env = CustomRewardEnv(env)
    env = wrapper.create_mario_env(env)

    return stage, world, stage_episode, save_dir, num_episodes, env

def reset_parameters(agent, optimizer, lr=0.001, explore_rate=0.8):
    # Reset the exploration parameters for each stage
    agent.exploration_rate = explore_rate

    # Reset the learning rate parameters
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def should_stop_training():
    # Get the current day of the week and time
    current_time = datetime.datetime.now().time()
    date = datetime.datetime.now()
    current_day = date.weekday()  # Monday is 0 and Sunday is 6
    current_hour = current_time.hour
    target_time = datetime.time(16, 00)  # 16:00 hours

    # Check if it's Monday to Friday, the time is after 16:00 hours, and before 21:00 hours
    if current_day >= 0 and current_day <= 4 and target_time <= current_time < datetime.time(21, 00):
        return True
    else:
        return False



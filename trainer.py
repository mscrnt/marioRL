# trainer.py
# This script contains the main function for training the agent on all levels.

import os
import gym_super_mario_bros
from tqdm import tqdm
import numpy as np
import wrapper
import torch
from model import DQNAgent
from rewards import CustomRewardEnv
import pickle
from file_handler import update_text_files, write_to_file, save_model, create_folders
from utilities import progress_level, reset_parameters, should_stop_training
import sys


# Do not adjust these parameters. They are used to keep track of the levels.
WORLDS = 8
STAGES_PER_WORLD = 4

# Set the frequency for saving the model
SAVE_FREQUENCY = 5001

# Create directories for saving models and text files
create_folders()

# Main function for training the agent on all levels.
def train_all_levels(num_episodes_per_stage=1000000, start_world=1, start_stage=1):
    # Set the initial level environment
    level_env = f"SuperMarioBros-{start_world}-{start_stage}-v0"
    env = gym_super_mario_bros.make(level_env)
    env = CustomRewardEnv(env)
    env = wrapper.create_mario_env(env)

    while True:
        # Train the agent on each level.
        for world in range(start_world, WORLDS + 1):
            start_stage_idx = start_stage if world == start_world else 1
            for stage in range(start_stage_idx, STAGES_PER_WORLD + 1):
                print(f"Training on World {world} Stage {stage}")
                train_on_level(world, stage, num_episodes_per_stage, env)

        # Reset start_world and start_stage for the next round of training
        start_world = 1
        start_stage = 1

        # Optional: Add a break condition if you want to stop training after one round
        # break

        # Set the initial level environment for the next round of training
        level_env = f"SuperMarioBros-{start_world}-{start_stage}-v0"
        env = gym_super_mario_bros.make(level_env)
        env = CustomRewardEnv(env)
        env = wrapper.create_mario_env(env)


def train_on_level(world, stage, num_episodes, env):
    # Set the save directory for this level.
    save_dir = f"pkl_files/world_{world}_stage_{stage}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create text file directories if they don't exist
    if not os.path.exists("text_files"):
        os.makedirs("text_files")

    # Create the agent
    observation_space = env.observation_space.shape
    action_space = env.action_space.n

    # Check if the save directory is empty
    save_files = os.listdir(save_dir)
    ending_position = 0
    num_in_queue = 0
    if save_files:

    # else:
        ending_position_file = os.path.join(save_dir, "ending_position.pkl")
        num_in_queue_file = os.path.join(save_dir, "num_in_queue.pkl")
        if os.path.exists(ending_position_file):
            with open(ending_position_file, "rb") as f:
                ending_position = pickle.load(f)
        else:
            ending_position = 0
        if os.path.exists(num_in_queue_file):
            with open(num_in_queue_file, "rb") as f:
                num_in_queue = pickle.load(f)
        else:
            num_in_queue = 0

    # Set initial exploration parameters. These will be reset if the agent is pretrained.
    exploration_max = .8  
    exploration_min = 0.02  
    exploration_decay = 0.99  

    # Set initial learning rate parameters. These will be reset if the agent is pretrained.
    learning_rate = 0.001  
    learning_rate_decay = 0.99 
    learning_rate_min = 0.00025 

    # Check if the save directory is empty to determine if the agent is pretrained
    if os.listdir(save_dir):
        pretrained_flag = True
    else:
        pretrained_flag = False

    agent = DQNAgent(
        state_space=observation_space,
        action_space=action_space,
        max_memory_size= 30000, # Adjust this parameter if you want to change the size of the memory
        batch_size=200, # Adjust this parameter if you want to change the batch size
        gamma=0.9, # Adjust this parameter if you want to change the discount factor
        lr=learning_rate,
        dropout=0.2, # Adjust this parameter if you want to change the dropout rate
        exploration_max=exploration_max,
        exploration_min=exploration_min,
        exploration_decay=exploration_decay,
        double_dqn=True,
        pretrained=pretrained_flag,
        save_dir=save_dir,
        ending_position=ending_position,
        num_in_queue=num_in_queue
    )


    # Load the pretrained model for the specific stage
    # Check if directory is empty
    if agent.pretrained:
        agent.local_net.load_state_dict(torch.load(f"{save_dir}/DQN1.pt", map_location=torch.device(agent.device)))
        agent.target_net.load_state_dict(torch.load(f"{save_dir}/DQN2.pt", map_location=torch.device(agent.device)))

        # Load the memory for the specific stage
        agent.STATE_MEM = torch.load(f"{save_dir}/STATE_MEM.pt")
        agent.ACTION_MEM = torch.load(f"{save_dir}/ACTION_MEM.pt")
        agent.REWARD_MEM = torch.load(f"{save_dir}/REWARD_MEM.pt")
        agent.STATE2_MEM = torch.load(f"{save_dir}/STATE2_MEM.pt")
        agent.DONE_MEM = torch.load(f"{save_dir}/DONE_MEM.pt")

        # Reset the exploration parameters for each stage
        agent.exploration_rate = .2

        # Reset the learning rate parameters
        agent.learning_rate = 0.0005

    # Do not touch these parameters
    completed_episodes = 0
    stage_episode = 0

    # Load the total number of completed episodes if it exists.
    if os.path.exists('text_files/attempts.txt'):
        with open('text_files/attempts.txt', 'r') as f:
            completed_episodes = int(f.read())

    num_episodes_per_stage = num_episodes  # Number of episodes for each stage

    # Run the trainer for this level.
    for ep_num in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0

        while True:
            action = agent.act(state)
            steps += 1
            env.render()

            # Take a step in the environment
            state_next, reward, terminal, info = env.step(int(action[0]))
            total_reward += reward
            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)
            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            agent.remember(state, action, reward, state_next, terminal)
            agent.experience_replay()

            state = state_next

            agent.write_last_actions('text_files/last_actions.txt')

            # Stop conditions: Set passed to True if the agent has completed the level, otherwise set it to False

            # If the agent has completed the level
            if info['flag_get']:
                # Close the environment, progress to the next level, and reset the exploration parameters
                env.close()
                stage, world, stage_episode, ending_position, num_in_queue, save_dir, num_episodes_per_stage, env = progress_level(
                    ep_num, STAGES_PER_WORLD, agent, passed=True, world=world, stage=stage, stage_episode=stage_episode, ending_position=ending_position, num_in_queue=num_in_queue, save_dir=save_dir, num_episodes_per_stage=num_episodes_per_stage
                )
                # Reset the exploration parameters for each stage
                reset_parameters(agent, agent.optimizer, lr=0.001, explore_rate=0.8)
                break

            # If the agent has died too many times in the level
            if stage_episode == 10000:
                # Close the environment, progress to the next level, and reset the exploration parameters
                env.close()
                stage, world, stage_episode, ending_position, num_in_queue, save_dir, num_episodes_per_stage, env = progress_level(
                    ep_num, STAGES_PER_WORLD, agent, passed=False, world=world, stage=stage, stage_episode=stage_episode, ending_position=ending_position, num_in_queue=num_in_queue, save_dir=save_dir, num_episodes_per_stage=num_episodes_per_stage
                )
                # Reset the exploration parameters for each stage
                reset_parameters(agent, agent.optimizer, lr=0.001, explore_rate=0.8)
                break

            # Miscellaneous stop conditions
            # Looping level takes too long to train
            if stage_episode == 500 and world == 4 and stage == 4:
                # Close the environment, progress to the next level, and reset the exploration parameters
                env.close()
                stage, world, stage_episode, ending_position, num_in_queue, save_dir, num_episodes_per_stage, env = progress_level(
                    ep_num, STAGES_PER_WORLD, agent, passed=False, world=world, stage=stage, stage_episode=stage_episode, ending_position=ending_position, num_in_queue=num_in_queue, save_dir=save_dir, num_episodes_per_stage=num_episodes_per_stage
                )
                # Reset the exploration parameters for each stage
                reset_parameters(agent, agent.optimizer, lr=0.001, explore_rate=0.8)
                break

            if should_stop_training():
                # Save the model and quit the training loop
                save_model(agent, save_dir, ending_position, num_in_queue)
                print("Quitting training because it's 4pm!")
                sys.exit()

            # If the agent has died, reset the level
            if terminal:
                break

            agent.episodes += 1

        # Update only the current episode count after each episode
        completed_episodes += 1
        stage_episode += 1
        write_to_file('text_files/attempts.txt', str(completed_episodes))
        write_to_file('text_files/message.txt', f"Mario has died {completed_episodes} times total.")
        write_to_file('text_files/stage_episode.txt', f"Mario has died {stage_episode} times on this stage.")

        # Update the exploration rate
        if (stage_episode + 1) % 5 == 0:
            agent.exploration_rate *= agent.exploration_decay
            agent.exploration_rate = max(agent.exploration_rate, agent.exploration_min)

        # Update the learning rate
        if (stage_episode + 1) % 5 == 0:
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] *= learning_rate_decay
                param_group['lr'] = max(param_group['lr'], learning_rate_min)


        if (stage_episode + 1) % 100 == 0:
            # Update the rewards files
            update_text_files(total_reward)

        if (stage_episode + 1) % SAVE_FREQUENCY == 0:
            # Save the agent's model and memory
            save_model(agent, save_dir)

        # Update the scalars
        agent.writer.add_scalar(f'{world}-{stage} Episode', stage_episode, agent.global_step / steps)
        agent.writer.add_scalar(f'{world}-{stage} Exploration Rate', agent.exploration_rate, stage_episode)
        agent.writer.add_scalar(f'{world}-{stage} Total Reward', total_reward, stage_episode)
        agent.writer.add_scalar('Completed Episodes', agent.global_step, completed_episodes)
        agent.writer.add_scalar(f'{world}-{stage} Episode Reward', total_reward, stage_episode)
        agent.writer.add_scalar(f'{world}-{stage} Average Reward', total_reward / (ep_num + 1), stage_episode)
        agent.writer.add_scalar(f'{world}-{stage} Learning Rate', agent.lr, stage_episode)

        agent.global_step += 1

    env.close()

# Load level from text file if it exists, otherwise start at World 1-1
if os.path.exists("text_files/current_level.txt"):
    with open("text_files/current_level.txt", "r") as f:
        world_stage = f.read().strip().split(",")
    start_world, start_stage = map(int, world_stage)
    # Validate the start_world and start_stage values
    if start_world < 1 or start_world > WORLDS:
        start_world = 1
    if start_stage < 1 or start_stage > STAGES_PER_WORLD:
        start_stage = 1
else:
    start_world = 1
    start_stage = 1

train_all_levels(1000000, start_world, start_stage)

# Train the agent on all levels.
#train_all_levels()

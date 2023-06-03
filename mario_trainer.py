# trainer.py 
# This script's purpose is to train the DQN agent. 

import torch
import gym_super_mario_bros
from tqdm import tqdm
import pickle 
import numpy as np
import os
import wrapper
import model
from rewards import CustomRewardEnv

def show_state(env, ep=0, info=""):
    env.render()
    #time.sleep(0.001)

# Write to a file. If the file exists, delete it and create a new one.
def write_to_file(filename, data, mode='w', encoding='utf-8'):
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, mode, encoding=encoding) as outfile:
        outfile.write(data)


def create_folders():
    # Create the folders if they don't exist
    if not os.path.exists("pkl_files"):
        os.makedirs("pkl_files")
    if not os.path.exists("text_files"):
        os.makedirs("text_files")


def create_checkpoint_image():
    # TODO Check to see if this is needed or not
    image_data = "Please Wait.jpg"
    with open("checkpoint_image.jpg", "wb") as image_file:
        image_file.write(image_data)

def update_text_files(total_rewards, completed_episodes):
    attempts_file = 'text_files/attempts.txt'
    message_file = 'text_files/message.txt'
    previous_rewards_file = 'text_files/previous_rewards.txt'
    average_file = 'text_files/average.txt'

    current_episode = completed_episodes

    write_to_file(attempts_file, str(completed_episodes))

    write_to_file(message_file, f"Mario has used {current_episode} continues.")

    previous_rewards = 0
    if os.path.exists(previous_rewards_file):
        with open(previous_rewards_file, 'r') as f:
            previous_rewards = int(f.read())

    write_to_file(average_file, f"Previous Reward Avg: {previous_rewards} | Current Reward Avg: {int(np.mean(total_rewards))}")
    write_to_file(previous_rewards_file, str(int(np.mean(total_rewards)))) 

def run(training_mode, pretrained, double_dqn, num_episodes=1000, exploration_max=.3):
    create_folders()  # Create the folders if they don't exist

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = CustomRewardEnv(env)  # Wrap the environment with the custom reward wrapper
    env = wrapper.create_mario_env(env) # Wrap the environment with the custom wrapper
    observation_space = env.observation_space.shape
    action_space = env.action_space.n 
    agent = model.DQNAgent(
        state_space=observation_space, 
        action_space=action_space, 
        max_memory_size=30000, 
        batch_size=64, 
        gamma=0.9, 
        lr=0.00025, 
        dropout=0.2, 
        exploration_max=.03, 
        exploration_min=0.001, 
        exploration_decay=0.99, 
        double_dqn=double_dqn, 
        pretrained=pretrained 
    )

    # Restart the environment for each episode
    env.reset()

    completed_episodes = 0
    total_steps = []

    if os.path.exists('text_files/attempts.txt'):
        with open('text_files/attempts.txt', 'r') as f:
            completed_episodes = int(f.read())

    total_rewards = []

    if training_mode and pretrained:
        with open("pkl_files/total_rewards.pkl", 'rb') as f:
            total_rewards = pickle.load(f)

    # reward_stall_threshold = 0.05  # Minimum improvement threshold for reward stall detection
    # reward_stall_episodes = 10  # Number of episodes to consider for reward stall detection
    # exploration_increase_rate = 0.1  # Rate of exploration rate increase when rewards stall
    # exploration_increase_frequency = 100  # Frequency of checking for reward stall (every 100 episodes)


    for ep_num in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0
        episode_num = completed_episodes + ep_num + 1

        while True:
            if training_mode:
                show_state(env, ep_num)

            action = agent.act(state)
            steps += 1

            state_next, reward, terminal, info = env.step(int(action[0]))
            total_reward += reward
            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)
            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            if training_mode:
                agent.remember(state, action, reward, state_next, terminal)
                agent.experience_replay()

            state = state_next

            # Write last actions after each action is taken
            agent.write_last_actions('text_files/last_actions.txt')

            # if (ep_num + 1) % exploration_increase_frequency == 0:
            #     # Check for reward stall and increase exploration rate if needed
            #     if len(total_rewards) >= reward_stall_episodes:
            #         average_reward_last_n = np.mean(total_rewards[-reward_stall_episodes:])
            #         average_reward_current = np.mean(total_rewards[-(reward_stall_episodes + 1):-1])
            #         reward_improvement = average_reward_last_n - average_reward_current
            #         if reward_improvement <= reward_stall_threshold:
            #             agent.exploration_rate += exploration_increase_rate


            if terminal:
                break

        total_rewards.append(total_reward)
        total_steps.append(steps)

        # Update only the current episode count after each episode
        current_episode = episode_num
        write_to_file('text_files/attempts.txt', str(current_episode))
        write_to_file('text_files/message.txt', f"Mario has used {current_episode} continues.")

        agent.writer.add_scalar('Exploration Rate', agent.exploration_rate, episode_num) 
        agent.writer.add_scalar('Total Reward', total_reward, episode_num)  
        agent.writer.add_scalar('Steps', steps, episode_num)  
        agent.writer.add_scalar('Average Reward', np.mean(total_rewards[-10:]), episode_num)  
        agent.writer.add_scalar('Average Steps', np.mean(total_steps[-10:]), episode_num)

        if (ep_num + 1) % 1 == 0:
            # Decay exploration rate
            agent.exploration_rate *= agent.exploration_decay
            agent.exploration_rate = max(agent.exploration_rate, agent.exploration_min)

        # Create a checkpoint every 100 episodes
        if (ep_num + 1) % 100 == 0:
            with open("pkl_files/ending_position.pkl", "wb") as f:
                pickle.dump(agent.ending_position, f)
            with open("pkl_files/num_in_queue.pkl", "wb") as f:
                pickle.dump(agent.num_in_queue, f)
            with open("pkl_files/total_rewards.pkl", "wb") as f:
                pickle.dump(total_rewards, f)
            with open("pkl_files/total_steps.pkl", "wb") as f:
                pickle.dump(total_steps, f)

            if agent.double_dqn:
                torch.save(agent.local_net.state_dict(), "pkl_files/DQN1.pt")
                torch.save(agent.target_net.state_dict(), "pkl_files/DQN2.pt")
            else:
                torch.save(agent.dqn.state_dict(), "pkl_files/DQN.pt")

            torch.save(agent.STATE_MEM, "pkl_files/STATE_MEM.pt")
            torch.save(agent.ACTION_MEM, "pkl_files/ACTION_MEM.pt")
            torch.save(agent.REWARD_MEM, "pkl_files/REWARD_MEM.pt")
            torch.save(agent.STATE2_MEM, "pkl_files/STATE2_MEM.pt")
            torch.save(agent.DONE_MEM, "pkl_files/DONE_MEM.pt")

            # Update averages and other information every 100 episodes
            update_text_files(total_rewards, current_episode)


            print("Episode {} score = {}, average score = {}".format(
                current_episode, total_rewards[-1], np.mean(total_rewards))
            )

    env.close()


# For training and to create checkpoint every 100 episodes
run(training_mode=True, pretrained=True, double_dqn=True, num_episodes=200000, exploration_max=.3)

# trainer.py 
# This script's purpose is to train the DQN agent. 

import torch
import gym_super_mario_bros
from tqdm import tqdm
import pickle 
import gym
import numpy as np
import os
import wrapper
import model

def show_state(env, ep=0, info=""):
    env.render()
    #time.sleep(0.001)

class CustomRewardEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardEnv, self).__init__(env)
        self.previous_score = 0
        self.previous_coins = 0
        self.previous_lives = env.unwrapped._life
        self.gone_down_pipe = False
        self.gone_up_vine = False
        self.start_y_position = 0
        self.timer = 0
        self.moved_right = False

    def reward(self, reward):
        # Reward based on increased score
        if self.env.unwrapped._score > self.previous_score:
            reward += self.env.unwrapped._score - self.previous_score
        self.previous_score = self.env.unwrapped._score

        # Reward based on collected coins
        if self.env.unwrapped._coins > self.previous_coins:
            reward += (self.env.unwrapped._coins - self.previous_coins) * 12
        self.previous_coins = self.env.unwrapped._coins

        # Penalty for lost life
        if self.env.unwrapped._life < self.previous_lives:
            reward -= 50
        self.previous_lives = self.env.unwrapped._life

        # Reward based on distance traveled
        distance_reward = self.env.unwrapped._x_position - self.env.unwrapped._x_position_last
        self.env.unwrapped._x_position_last = self.env.unwrapped._x_position
        reward += distance_reward * 0.13

        # Reward for transforming from small to large
        if self.env.unwrapped._player_state == 0x0A and self.previous_player_state == 0x01:
            reward += 1000

        # Penalty for transforming from large to small
        if self.env.unwrapped._player_state == 0x09 and self.previous_player_state == 0x0A:
            reward -= 1000

        # Reward for transforming to Fire Mario
        if self.env.unwrapped._player_state == 0x0C and self.previous_player_state != 0x0C:
            reward += 2000

        # Reward based on going down a pipe
        if self.env.unwrapped._player_state == 0x03 and not self.gone_down_pipe:
            reward += 50000  
            self.gone_down_pipe = True
        elif self.env.unwrapped._player_state != 0x03:
            self.gone_down_pipe = False

        # Reward based on going up a vine
        if self.env.unwrapped._player_state == 0x01 and not self.gone_up_vine:
            reward += 100  
            self.gone_up_vine = True
        elif self.env.unwrapped._player_state != 0x01:
            self.gone_up_vine = False

        # Reward for pressing down after landing
        if (
            self.env.unwrapped._y_position > self.start_y_position and
            self.env.unwrapped._player_state == 0x00 and
            self.env.unwrapped._action == 10
        ):
            reward += 500

        # Large reward for completing a level
        if self.env.unwrapped._flag_get:
            reward += 20000

        # Small penalty for NOOP
        if self.env.unwrapped._action == 0:
            reward -= 10

        # Large penalty for game over
        if self.env.unwrapped._is_game_over:
            reward -= 500

        # Clip the reward to a specific range if needed
        reward = max(-100, min(reward, 100))

        # Check if Mario has moved right
        if self.env.unwrapped._action == 1:
            self.moved_right = True

        # Move left if Mario hasn't moved right for 20 seconds and has already moved left
        if not self.moved_right and self.timer >= 20000:  # Adjust the timer based on your desired time limit
            reward += 10  # Gain 10 points for moving left
            self.env.unwrapped._x_position -= 2  # Move left by 2 spaces

        # Increment the timer if Mario hasn't moved right
        if not self.moved_right:
            self.timer += 1

        return reward


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


def update_text_files(total_rewards, completed_episodes, total_episodes):
    total_episodes_file = 'text_files/total_episodes.txt'
    attempts_file = 'text_files/attempts.txt'
    message_file = 'text_files/message.txt'
    previous_rewards_file = 'text_files/previous_rewards.txt'
    average_file = 'text_files/average.txt'

    current_episode = completed_episodes

    write_to_file(attempts_file, str(completed_episodes))
    write_to_file(total_episodes_file, str(total_episodes))

    write_to_file(message_file, f"Mario has completed {current_episode}+ episodes.")

    previous_rewards = 0
    if os.path.exists(previous_rewards_file):
        with open(previous_rewards_file, 'r') as f:
            previous_rewards = int(f.read())

    write_to_file(average_file, f"Previous Reward Avg: {previous_rewards} | Current Reward Avg: {int(np.mean(total_rewards))}")
    write_to_file(previous_rewards_file, str(int(np.mean(total_rewards)))) 


def run(training_mode, pretrained, double_dqn, num_episodes=1000, exploration_max=1):
    create_folders()  # Create the folders if they don't exist

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = CustomRewardEnv(env)  # Wrap the environment with the custom reward wrapper
    env = wrapper.create_mario_env(env) # Wrap the environment with the custom wrapper
    observation_space = env.observation_space.shape
    action_space = env.action_space.n 
    agent = model.DQNAgent(
        state_space=observation_space, 
        action_space=action_space, 
        max_memory_size=30000, # 30000
        batch_size=64, # 64
        gamma=0.9, # 0.9
        lr=0.00025, # 0.00025
        dropout=0.2, # 0.2
        exploration_max=1.0, # 1.0
        exploration_min=0.05, # 0.05
        exploration_decay=0.99, # 0.99
        double_dqn=double_dqn, # True
        pretrained=pretrained # True
    )

    # Restart the environment for each episode
    env.reset()

    total_episodes = 0
    completed_episodes = 0

    if os.path.exists('text_files/total_episodes.txt'):
        with open('text_files/total_episodes.txt', 'r') as f:
            total_episodes = int(f.read())

    if os.path.exists('text_files/attempts.txt'):
        with open('text_files/attempts.txt', 'r') as f:
            completed_episodes = int(f.read())

    total_rewards = []

    if training_mode and pretrained:
        with open("pkl_files/total_rewards.pkl", 'rb') as f:
            total_rewards = pickle.load(f)

    for ep_num in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0
        episode_num = completed_episodes + ep_num + 1

        agent.writer.add_scalar('Exploration Rate', agent.exploration_rate, episode_num) 
        agent.writer.add_scalar('Total Reward', total_reward, episode_num)  
        agent.writer.add_scalar('Steps', steps, episode_num)  
        agent.writer.add_scalar('Average Reward', np.mean(total_rewards[-10:]), episode_num)  
        agent.writer.add_scalar('Average Steps', np.mean(total_rewards[-10:]), episode_num) 

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

            if terminal:
                break

        total_rewards.append(total_reward)

        # Create a checkpoint every 100 episodes
        if (ep_num + 1) % 100 == 0:
            with open("pkl_files/ending_position.pkl", "wb") as f:
                pickle.dump(agent.ending_position, f)
            with open("pkl_files/num_in_queue.pkl", "wb") as f:
                pickle.dump(agent.num_in_queue, f)
            with open("pkl_files/total_rewards.pkl", "wb") as f:
                pickle.dump(total_rewards, f)

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

            update_text_files(total_rewards, episode_num, total_episodes) 

            print("Episode {} score = {}, average score = {}".format(
                episode_num, total_rewards[-1], np.mean(total_rewards))
            )

    env.close()



# For training and to create checkpoint every 100 episodes
run(training_mode=True, pretrained=True, double_dqn=True, num_episodes=200000, exploration_max=1)

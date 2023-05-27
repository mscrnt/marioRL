# Super Mario DQN Agent

This repository contains code for training a Deep Q-Network (DQN) agent to play the Super Mario Bros game using Gym and OpenAI baselines.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Code Explanation](#code-explanation)
5. [Key Features](#key-features)
6. [License](#license)

## Introduction

The Super Mario DQN Agent is designed to train an agent using Deep Q-Networks to play the Super Mario Bros game. The agent uses reinforcement learning to learn an optimal policy for playing the game.

## Installation

To run the Super Mario DQN Agent, you need to install the required dependencies. You can do this by following these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/mscrnt/marioRL.git && cd marioRL
   ``` 

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train the agent, you can run the following command:

```bash
python mario_trainer.py
```

To test the agent, you can run the following command:

In mario_trainer.py, Change:
run(training_mode=True, pretrained=False, double_dqn=True, num_episodes=200000, exploration_max=1)
To:
run(training_mode=False, pretrained=False, double_dqn=True, num_episodes=200000, exploration_max=1) 

```bash
python mario_trainer.py
``` 

To resume training the agent from checkpoint, change the following in mario_trainer.py:
run(training_mode=True, pretrained=True, double_dqn=True, num_episodes=200000, exploration_max=1)
To:
run(training_mode=True, pretrained=True, double_dqn=True, num_episodes=200000, exploration_max=1) 
    
    ```bash
    python mario_trainer.py
    ```

## Code Explanation

The code is divided into 3 files:

1. `mario_trainer.py`: This file contains the code for training the agent.
2. `model.py`: This file contains the code for the agent.
3. `wrapper.py`: This file contains the code for the environment wrapper.

## Key Features

The code is broken up so you shouldn't need to touch the model.py or wrapper.py files. You can simply run the mario_trainer.py file to train the agent.

mario_trainer.py contains the following functions:

```python
def run(training_mode=True, pretrained=False, double_dqn=True, num_episodes=200000, exploration_max=1)
```

This function is used to train the agent. It takes in the following parameters:

1. `training_mode`: This parameter is used to specify whether the agent should be trained or not. If this parameter is set to `True`, the agent will be trained. If this parameter is set to `False`, the agent will not be trained.
2. `pretrained`: This parameter is used to specify whether the agent should be trained from scratch or from a checkpoint. If this parameter is set to `True`, the agent will be trained from a checkpoint. If this parameter is set to `False`, the agent will be trained from scratch.
3. `double_dqn`: This parameter is used to specify whether the agent should use Double DQN or not. If this parameter is set to `True`, the agent will use Double DQN. If this parameter is set to `False`, the agent will not use Double DQN.
4. `num_episodes`: This parameter is used to specify the number of episodes the agent should be trained for.
5. `exploration_max`: This parameter is used to specify the maximum exploration rate for the agent.

The run function contains the following code:

```python
max_memory_size=30000,
batch_size=64,
gamma=0.9,
lr=0.00025,
dropout=0.2,
exploration_max=1.0,
exploration_min=0.05,
exploration_decay=0.99,
```

These parameters are used to specify the following:

1. `max_memory_size`: This parameter is used to specify the maximum size of the replay memory.
2. `batch_size`: This parameter is used to specify the batch size for training the agent.
3. `gamma`: This parameter is used to specify the discount factor for the agent.
4. `lr`: This parameter is used to specify the learning rate for the agent.
5. `dropout`: This parameter is used to specify the dropout rate for the agent.
6. `exploration_max`: This parameter is used to specify the maximum exploration rate for the agent.
7. `exploration_min`: This parameter is used to specify the minimum exploration rate for the agent.
8. `exploration_decay`: This parameter is used to specify the exploration decay rate for the agent.

Modify these parameters to change the behavior of the agent.

```python
# Create a checkpoint every 100 episodes
if (ep_num + 1) % 100 == 0:
```

This code is used to create a checkpoint every 100 episodes. You can modify this code to create a checkpoint every `n` episodes.

```python
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
```

This code is used to create the folders and update the text files. You can modify this code to create the folders and update the text files as per your requirements. These will update the text files with the number of episodes completed, the average reward, and the current reward. These files will update everytime a checkpoint is created.

```python
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
            self.env.unwrapped._x_position -= 2  # Move left by 10 spaces

        # Increment the timer if Mario hasn't moved right
        if not self.moved_right:
            self.timer += 1

        return reward
```

This code is used to calculate the reward for the agent. You can modify this code to change the reward calculation for the agent. The reward is calculated based on the following:
# TODO: Need to figure out if these are actually being used.
1. Increased score
2. Collected coins
3. Lost life
4. Distance traveled
5. Transformed from small to large
6. Transformed from large to small
7. Transformed to Fire Mario
8. Went down a pipe
9. Went up a vine
10. Pressed down after landing  # To promote finding secret pipes. 
11. Completed a level
12. NOOP






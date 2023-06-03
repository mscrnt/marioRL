import gym
import numpy as np
import logging
from logging.handlers import RotatingFileHandler

class CustomRewardEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardEnv, self).__init__(env)
        self.previous_score = 0
        self.previous_coins = 0
        self.previous_player_state = env.unwrapped._player_state
        self.gone_up_vine = False
        self.start_y_position = 150
        self.timer = 0
        self.previous_x_position = None
        self.mario_is_tall = False
        self.previous_time = None


        # # Configure logging with RotatingFileHandler
        # log_file = 'logs/reward_log.txt'
        # max_file_size = 1024 * 1024  # 1 MB
        # backup_count = 5

        # self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.INFO)

        # formatter = logging.Formatter('%(asctime)s - %(message)s')

        # # Create a file handler and set its formatter
        # file_handler = RotatingFileHandler(log_file, maxBytes=max_file_size, backupCount=backup_count)
        # file_handler.setFormatter(formatter)
        # self.logger.addHandler(file_handler)

        self.current_episode = 0

    def reward(self, reward, action, info):
        self.current_episode += 1

        # Reward based on increased score
        if self.env.unwrapped._score > self.previous_score:
            score_reward = (self.env.unwrapped._score - self.previous_score) * 0.015
            reward += score_reward
            self.previous_score = self.env.unwrapped._score
            #self.logger.info(f"Step: {self.current_episode} - Increased score reward: {score_reward}")

        # Reward based on collected coins
        if self.env.unwrapped._coins > self.previous_coins:
            coins_reward = (self.env.unwrapped._coins - self.previous_coins) * 0.015
            reward += coins_reward
            self.previous_coins = self.env.unwrapped._coins
            #self.logger.info(f"Step: {self.current_episode} - Collected coins reward: {coins_reward}")

        # Penalty for lost life
        if self.env.unwrapped._player_state == 6:
            life_penalty = 12.5
            reward -= life_penalty
            #self.logger.info(f"Step: {self.current_episode} - Lost life penalty: -{life_penalty}")

        # Reward based on distance traveled
        if self.previous_x_position is not None:
            distance_reward = self.env.unwrapped._x_position - self.previous_x_position
            if distance_reward < -5 or distance_reward > 5:
                distance_reward = 0
            reward += distance_reward
            #self.logger.info(f"Step: {self.current_episode} - Distance traveled reward: {distance_reward}")
            self.previous_x_position = self.env.unwrapped._x_position  # Save the current position for the next step

        if self.previous_x_position is None:
            self.previous_x_position = self.env.unwrapped._x_position


        # Large reward for completing a level
        if self.env.unwrapped._flag_get:
            level_completion_reward = 200
            reward += level_completion_reward
            #self.logger.info(f"Step: {self.current_episode} - Level completion reward: {level_completion_reward}")

        # Small penalty for NOOP
        if action == 0:
            noop_penalty = 0.1
            reward -= noop_penalty
            #self.logger.info(f"Step: {self.current_episode} - NOOP penalty: -{noop_penalty}")

        # Large penalty for game over
        if self.env.unwrapped._is_game_over:
            game_over_penalty = 16
            reward -= game_over_penalty
            #self.logger.info(f"Step: {self.current_episode} - Game over penalty: -{game_over_penalty}")

        # Reward for transforming from small to large
        if info['status'] == 'tall' and self.previous_player_state == 'small':
            reward += 5
            self.mario_is_tall = True
            #self.logger.info(f"Step: {self.current_episode} - Transform reward: 5")
        else:
            self.mario_is_tall = False

        if info['status'] == 'small' and self.previous_player_state == 'tall':
            self.mario_is_tall = False
            reward -= 2
            #self.logger.info(f"Step: {self.current_episode} - Transform penalty: -2")

        # # Reward for transforming to Fire Mario
        # if self.env.unwrapped._player_state == 0x0C and self.previous_player_state != 0x0C:
        #     transform_fire_reward = 20
        #     reward += transform_fire_reward
        #     self.logger.info(f"Step: {self.current_episode} - Transformed to Fire Mario reward: {transform_fire_reward}")

        # # Reward based on going down a pipe
        # if self.env.unwrapped._player_state == 3:
        #     print(f"player state: {self.env.unwrapped._player_state}")
        #     go_down_pipe_reward = 500
        #     reward += go_down_pipe_reward
        #     self.gone_down_pipe = True
        #     self.logger.info(f"Step: {self.current_episode} - Went down a pipe reward: {go_down_pipe_reward}")


        # # Reward based on going up a vine
        # if self.env.unwrapped._player_state == 0x01 and not self.gone_up_vine:
        #     go_up_vine_reward = 10
        #     reward += go_up_vine_reward
        #     self.gone_up_vine = True
        #     self.logger.info(f"Step: {self.current_episode} - Went up a vine reward: {go_up_vine_reward}")
        # elif self.env.unwrapped._player_state != 0x01:
        #     self.gone_up_vine = False

        # Reward for pressing down after landing
        if self.env.unwrapped._y_position > self.start_y_position and action == 32: # 32 is the action for pressing down         
            press_down_reward = .1
            reward += press_down_reward
            #self.logger.info(f"Step: {self.current_episode} - Pressed down after landing reward: {press_down_reward}")

        # Large reward for completing a level
        if self.env.unwrapped._flag_get:
            level_completion_reward = 200
            reward += level_completion_reward
            #self.logger.info(f"Step: {self.current_episode} - Level completion reward: {level_completion_reward}")

        # # Reward for entering a reversed-L pipe
        # if self.env.unwrapped._player_state == 2:
        #     enter_reversed_pipe_reward = 100
        #     reward += enter_reversed_pipe_reward
        #     self.logger.info(f"Step: {self.current_episode} - Entered a reversed-L pipe reward: {enter_reversed_pipe_reward}")

        # self.previous_player_state = self.env.unwrapped._player_state  # Update the previous player state

        # Reward for the in-game clock ticking
        if self.previous_time is not None:
            time_reward = self.env.unwrapped._time - self.previous_time
            if time_reward >= 0:
                time_reward = 0
            reward += time_reward
            #self.logger.info(f"Step: {self.current_episode} - Time reward: {time_reward}")
        self.previous_time = self.env.unwrapped._time


        # Clip the reward to a specific range if needed
        reward = max(-100, min(reward, 100))


        self.previous_player_state = info['status']  # Update the previous player state
        return reward

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward = self.reward(reward, action, info)  # This is calling your custom reward function
        return state, reward, done, info
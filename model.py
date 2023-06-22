# model.py
# This script contains the DQN agent and the neural network model.

import os
import torch
import torch.nn as nn
import random
from torch.utils.tensorboard import SummaryWriter
from wrapper import DQNSolver
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import pickle


class DQNAgent:
    def __init__(
        self,
        state_space,
        action_space,
        max_memory_size,
        batch_size,
        gamma,
        lr,
        dropout,
        exploration_max,
        exploration_min,
        exploration_decay,
        double_dqn,
        pretrained,
        save_dir,
        ending_position=None,
        num_in_queue=None
    ):
        self.state_space = state_space
        self.action_space = action_space
        self.double_dqn = double_dqn
        self.pretrained = pretrained
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.stuck_counter = 0
        self.ACTION_NAMES = COMPLEX_MOVEMENT
        self.prev_state = None
        self.writer = SummaryWriter("runs/train")
        self.global_step = 0
        self.last_actions = []
        self.total_reward = 0
        self.steps = 0
        self.episodes = 0

        self.lr = lr

        if self.double_dqn:
            self.local_net = DQNSolver(state_space, action_space, dropout).to(
                self.device
            )
            self.target_net = DQNSolver(state_space, action_space, dropout).to(
                self.device
            )

            if self.pretrained:
                dqn1_file = os.path.join(save_dir, "DQN1.pt")
                dqn2_file = os.path.join(save_dir, "DQN2.pt")
                self.local_net.load_state_dict(
                    torch.load(dqn1_file, map_location=torch.device(self.device))
                )
                self.target_net.load_state_dict(
                    torch.load(dqn2_file, map_location=torch.device(self.device))
                )
                

            self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)
            self.copy = 5000  # Copy the local model weights into the target network every 5000 steps
            self.step = 0
        else:
            self.dqn = DQNSolver(state_space, action_space, dropout).to(self.device)

            if self.pretrained:
                dqn_file = os.path.join(save_dir, "DQN.pt")
                self.dqn.load_state_dict(
                    torch.load(dqn_file, map_location=torch.device(self.device))
                )
            self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

        self.max_memory_size = max_memory_size
        if self.pretrained:
            state_mem_file = os.path.join(save_dir, "STATE_MEM.pt")
            action_mem_file = os.path.join(save_dir, "ACTION_MEM.pt")
            reward_mem_file = os.path.join(save_dir, "REWARD_MEM.pt")
            state2_mem_file = os.path.join(save_dir, "STATE2_MEM.pt")
            done_mem_file = os.path.join(save_dir, "DONE_MEM.pt")

            self.STATE_MEM = torch.load(state_mem_file)
            self.ACTION_MEM = torch.load(action_mem_file)
            self.REWARD_MEM = torch.load(reward_mem_file)
            self.STATE2_MEM = torch.load(state2_mem_file)
            self.DONE_MEM = torch.load(done_mem_file)

            if ending_position is not None and num_in_queue is not None:
                self.ending_position = ending_position
                self.num_in_queue = num_in_queue
            else:
                ending_position_file = os.path.join(save_dir, "ending_position.pkl")
                num_in_queue_file = os.path.join(save_dir, "num_in_queue.pkl")
                with open(ending_position_file, "rb") as f:
                    self.ending_position = pickle.load(f)
                with open(num_in_queue_file, "rb") as f:
                    self.num_in_queue = pickle.load(f)
        else:
            self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.ACTION_MEM = torch.zeros(max_memory_size, 1)
            self.REWARD_MEM = torch.zeros(max_memory_size, 1)
            self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.DONE_MEM = torch.zeros(max_memory_size, 1)
            self.ending_position = 0
            self.num_in_queue = 0

        self.memory_sample_size = batch_size

        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device)
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

    def remember(self, state, action, reward, state2, done):
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def batch_experiences(self):
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]
        return STATE, ACTION, REWARD, STATE2, DONE

    def act(self, state):
        if self.double_dqn:
            self.step += 1

        if random.random() < self.exploration_rate:
            action = random.randrange(self.action_space)
        else:
            with torch.no_grad():
                net = self.local_net if self.double_dqn else self.dqn
                action = torch.argmax(net(state.to(self.device))).item()

        action_name = self.ACTION_NAMES[action]

        self.last_actions.append(action_name)
        if len(self.last_actions) > 20:
            self.last_actions = self.last_actions[-20:]

        return torch.tensor([action], dtype=torch.long)

    def write_last_actions(self, filename):
        with open(filename, "w") as f:
            for action in self.last_actions:
                f.write(str(action) + "\n")

    def copy_model(self):
        self.target_net.load_state_dict(self.local_net.state_dict())

    def experience_replay(self):
        if self.double_dqn and self.step % self.copy == 0:
            self.copy_model()

        if self.memory_sample_size > self.num_in_queue:
            return

        STATE, ACTION, REWARD, STATE2, DONE = self.batch_experiences()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)

        self.optimizer.zero_grad()
        if self.double_dqn:
            target = REWARD + torch.mul(
                (self.gamma * self.target_net(STATE2).max(1).values.unsqueeze(1)),
                1 - DONE,
            )
            current = self.local_net(STATE).gather(1, ACTION.long())
        else:
            target = REWARD + torch.mul(
                (self.gamma * self.dqn(STATE2).max(1).values.unsqueeze(1)), 1 - DONE
            )
            current = self.dqn(STATE).gather(1, ACTION.long())

        loss = self.l1(current, target)
        self.writer.add_scalar("Loss", loss, self.step)
        #self.writer.add_scalar('Exploration Rate', self.exploration_rate, self.step)
        # self.writer.add_scalar('Total Reward', self.total_reward, self.step)
        # self.writer.add_scalar('Steps', self.steps, self.step)
        # self.writer.add_scalar('Average Reward', self.total_reward / self.steps, self.step) 
        # self.writer.add_scalar('Average Steps', self.steps / self.episodes, self.step)
        self.global_step += 1
        loss.backward()
        self.optimizer.step()



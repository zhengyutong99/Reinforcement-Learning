import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
import random

class AtariDQNAgent(DQNBaseAgent):
    def __init__(self, config):
        super(AtariDQNAgent, self).__init__(config)

        # gymnasium.wrappers.AtariPreprocessing(
        # env: Env, noop_max: int = 30,
        # frame_skip: int = 4,
        # screen_size: int = 84,
        # terminal_on_life_loss: bool = False,
        # grayscale_obs: bool = True,
        # grayscale_newaxis: bool = False,
        # scale_obs: bool = False)
  
        ### TODO ###
        # initialize env
        self.env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
        self.env = gym.wrappers.AtariPreprocessing(self.env,30,1,84,False,True,False,False)
        self.env = gym.wrappers.FrameStack(self.env,4)
        ### TODO ###
        # initialize test_env
        self.test_env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
        self.test_env = gym.wrappers.AtariPreprocessing(self.test_env,30,1,84,False,True,False,False)
        self.test_env = gym.wrappers.FrameStack(self.test_env,4)

        # initialize behavior network and target network
        self.behavior_net = AtariNetDQN(self.env.action_space.n)
        self.behavior_net.to(self.device)
        self.target_net = AtariNetDQN(self.env.action_space.n)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.behavior_net.state_dict())
        # initialize optimizer
        self.lr = config["learning_rate"]
        self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)

    def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
        ### TODO ###
        # get action from behavior net, with epsilon-greedy selection

        # def lazy_frames_to_tensor(lazy_frames, device):
        # 	frame_list = [frame for frame in lazy_frames]
        # 	stacked_frames = np.stack(frame_list, axis=-1)
        # 	stacked_frames = stacked_frames.transpose(2, 0, 1)
        # 	state_tensor = torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0).to(device)

        # 	return state_tensor

        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            # observation = lazy_frames_to_tensor(observation, device = self.device)
            # action = self.behavior_net(observation).argmax()
            obs_tensor = torch.tensor(np.asarray(observation), dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.behavior_net(obs_tensor).argmax(dim=1).item()
            
        return action
    
    def update_behavior_network(self):
        # sample a minibatch of transitions
        if self.load_warmup < self.batch_size:
            self.load_warmup += 1
            state, action, reward, next_state, done = self.replay_buffer.sample(self.load_warmup, self.device)
        else:
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

        q_value = self.behavior_net(state).gather(1, action.to(torch.int64))
        # q_next = torch.zeros(self.batch_size, self.device)
        
        with torch.no_grad():
            next_q_values = self.behavior_net(next_state)
            next_q_state_values = self.target_net(next_state)
            next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1))
            # if episode terminates at next_state, then q_target = reward
            q_target = reward + (1-done) * self.gamma * next_q_value
        
        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)

        self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
import numpy as np
import torch
import torch.nn as nn
import gym
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from models.atari_model import AtariNetDQN

def decide_agent_actions(env, model, observation, epsilon=0.0, action_space=None, device="cpu"):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        obs_tensor = torch.tensor(np.asarray(observation), dtype=torch.float32, device=device).unsqueeze(0)
        action = model(obs_tensor).argmax(dim=1).item()
    return action

def evaluate(env, model):
    print("==============================================")
    print("Evaluating...")
    observation, info = env.reset()
    total_reward = 0
    while True:
        # env.render()
        action = decide_agent_actions(env, model, observation, 0.001, env.action_space)
        next_observation, reward, terminate, truncate, info = env.step(action)
        total_reward += reward
        if terminate or truncate:
            print(f"episode reward: {total_reward}")
            break

        observation = next_observation
    
    # avg = sum(all_rewards) / self.eval_episode
	# print(f"average score: {avg}")
    print("==============================================")
    env.close()


# if you don't have GUI, you can use the following code to record video
if __name__ == "__main__":
    env = gym.make("ALE/Enduro-v5", render_mode="rgb_array")
    env = gym.wrappers.AtariPreprocessing(env,30,1,84,False,True,False,False)
    env = gym.wrappers.FrameStack(env, 4)
    
    env = gym.wrappers.RecordVideo(env, 'Enduro-video')
    
    model = AtariNetDQN(env.action_space.n)
    model.load_state_dict(torch.load("/home/zhengyutong/data/Reinforce_Learning/Lab2/Enduro-v5/log/model_1993336_248.pth"))
    evaluate(env, model)
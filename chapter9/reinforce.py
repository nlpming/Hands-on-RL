# coding:utf-8
import os
import sys
import json

sys.path.append("..")

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state, is_training: bool = True):
        state = torch.from_numpy(state).float().unsqueeze(0).to('cpu')
        probs = self.policy_net(state).cpu()

        if is_training:
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            return action.item()
        else:
            max_prob_action = probs.argmax().item()
            return max_prob_action

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        # 版本1：每一步都计算梯度，之后的梯度会覆盖之前的；所以只保留了最后一步的梯度值。
        # G = 0
        # self.optimizer.zero_grad()
        # for i in reversed(range(len(reward_list))):  # 从最后一步算起
        #     reward = reward_list[i]
        #     state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
        #     action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
        #     log_prob = torch.log(self.policy_net(state).gather(1, action))
        #     G = self.gamma * G + reward
        #     loss = -log_prob * G
        #     loss.backward()
        # self.optimizer.step()  # 梯度下降

        # 版本2：修正后版本，根据整个轨迹计算loss
        G = 0
        loss = []
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))  # 预测得到log_prob
            G = self.gamma * G + reward
            loss.append(-log_prob * G)
        loss = torch.cat(loss).sum()
        loss.backward()
        self.optimizer.step()  # 梯度下降



def train():
    """开始训练模型"""

    learning_rate = 0.005
    num_episodes = 1000
    gamma = 1.0
    hidden_dim = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 倒立摆游戏
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)

    # 开始模型训练
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }

                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)

                    # next_state: 下一步状态
                    # reward: 奖励值
                    # done: 游戏是否结束
                    next_state, reward, done, _ = env.step(action)

                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward

                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    # 保持模型
    model_file = "./reinforce.pth"
    torch.save(agent.policy_net.state_dict(), model_file)
    print(f"save model file: {model_file} Done.")

    # 保持总的回报
    return_file = "./return_list.json"
    with open(return_file, "w", encoding="utf-8") as fo:
        json.dump(return_list, fo)
    print(f"save return file: {return_file} Done.")


def test(max_step: int = 10000):
    env = gym.make('CartPole-v0')
    env._max_episode_steps = max_step
    env.seed(0)

    # 加载模型
    learning_rate = 0.005
    gamma = 1.0
    hidden_dim = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)
    agent.policy_net.load_state_dict(torch.load("./reinforce.pth"))

    state = env.reset()
    for t in range(max_step):
        action = agent.take_action(state, is_training=False)
        env.render()
        state, reward, done, _ = env.step(action)
        print(f"t = {t}: state: {state} | reward = {reward} | done = {done}")
        if done:
            break

    env.close()


if __name__ == "__main__":
    # train()
    # rl_utils.reward_visualize(return_file="./return_list.json")
    test()





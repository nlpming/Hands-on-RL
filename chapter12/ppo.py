# coding:utf-8
import os
import sys
import json

sys.path.append("..")

import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)  # 输出是一维

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state, is_training: bool = True):
        """下一步动作"""
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)  # actor网络获取下一步动作的概率

        if is_training:
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()  # 采样一个动作
            return action.item()
        else:
            max_prob_action = probs.argmax().item()
            return max_prob_action

    def update(self, transition_dict):
        """更新agent参数"""
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # rewards: [n_step, 1]；
        # self.critic(next_states): [n_step, 1]
        # dones: [n_step, 1]
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)  # 下一个状态放入ValueNet
        td_delta = td_target - self.critic(states)  # 当前状态放入ValueNet
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()  # 老的策略用于做示范

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))  # 新的策略
            ratio = torch.exp(log_probs - old_log_probs)  # 新策略/老策略
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))  # 此处如何理解？
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

def train_ppo():
    """ppo算法训练过程"""
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500  # 与环境交互次数
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10  # 每次交互更新次数
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)

    # 开始训练模型
    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    # 保存模型
    model_file = "./cartpole-ppo.pth"
    torch.save(agent.actor.state_dict(), model_file)
    print(f"save model file: {model_file} Done.")

    # 保存总的回报
    return_file = "./return_list.json"
    with open(return_file, "w", encoding="utf-8") as fo:
        json.dump(return_list, fo)
    print(f"save return file: {return_file} Done.")


def test_ppo(max_step: int = 10000):
    env = gym.make('CartPole-v0')
    env._max_episode_steps = max_step
    env.seed(0)

    # 加载模型
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)
    agent.actor.load_state_dict(torch.load("./cartpole-ppo.pth"))

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
    # train_ppo()
    # rl_utils.reward_visualize("./return_list.json", env_name="CartPole-v0")
    test_ppo()


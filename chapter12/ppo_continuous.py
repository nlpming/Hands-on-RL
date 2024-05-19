# coding:utf-8
import os
import sys
import json

sys.path.append("..")

import torch
import torch.nn.functional as F
import rl_utils
import gym


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state, is_training: bool = True):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)

        if is_training:
            action = action_dist.sample()
            return [action.item()]
        else:
            return [mu.item()]

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(actions)  # 动作是正态分布

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

def train():
    actor_lr = 1e-4
    critic_lr = 5e-3
    num_episodes = 2000
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  # 连续动作空间
    agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device)

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    # 保持模型
    model_file = "./pendulum-ppo.pth"
    torch.save(agent.actor.state_dict(), model_file)
    print(f"save model file: {model_file} Done.")

    # 保持总的回报
    return_file = "./return_list.json"
    with open(return_file, "w", encoding="utf-8") as fo:
        json.dump(return_list, fo)
    print(f"save return file: {return_file} Done.")

def test():
    # 创建Pendulum-v0环境
    env = gym.make('Pendulum-v0')
    env._max_episode_steps = 10000

    # 使用Monitor包装器记录视频
    # env = gym.wrappers.Monitor(env, './videos', force=True)

    actor_lr = 1e-4
    critic_lr = 5e-3
    num_episodes = 2000
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  # 连续动作空间
    agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                          lmbda, epochs, eps, gamma, device)
    agent.actor.load_state_dict(torch.load("./pendulum-ppo.pth"))

    # 重置环境并获取初始状态
    observation = env.reset()

    # 执行100个随机动作
    for t in range(10000):
        # 随机选择一个动作（在Pendulum-v0中，动作是一个连续的值，范围是[-2, 2]）
        # action = env.action_space.sample()
        action = agent.take_action(observation, is_training=False)
        env.render()

        # 执行动作并获取下一个状态、奖励、是否终止和额外信息
        observation, reward, done, info = env.step(action)

        print(f"t = {t}: state: {observation} | reward = {reward} | done = {done}")

        # 如果游戏结束，重置环境
        if done:
            break

    # 关闭环境
    env.close()


if __name__ == "__main__":
    # train()
    # rl_utils.reward_visualize("./return_list.json", env_name="Pendulum-v0")
    test()


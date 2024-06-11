# coding:utf-8
import os
import sys
import gym
from gym.wrappers import Monitor


def test1():
    # 创建Pendulum-v0环境
    env = gym.make('Pendulum-v0')

    # 重置环境并获取初始状态
    observation = env.reset()

    # 执行100个随机动作
    for _ in range(100):
        # 随机选择一个动作（在Pendulum-v0中，动作是一个连续的值，范围是[-2, 2]）
        action = env.action_space.sample()

        # 执行动作并获取下一个状态、奖励、是否终止和额外信息
        observation, reward, done, info = env.step(action)

        # 打印当前状态、奖励和是否终止
        print(f"Observation: {observation}, Reward: {reward}, Done: {done}")

        # 如果游戏结束，重置环境
        if done:
            print("Episode finished after {} timesteps".format(_+1))
            observation = env.reset()

    # 关闭环境
    env.close()


def test2():
    # 创建Pendulum-v0环境
    env = gym.make('Pendulum-v0')

    # 使用Monitor包装器记录视频
    env = Monitor(env, './videos', force=True)

    # 重置环境并获取初始状态
    observation = env.reset()

    # 执行100个随机动作
    for _ in range(100):
        # 随机选择一个动作（在Pendulum-v0中，动作是一个连续的值，范围是[-2, 2]）
        action = env.action_space.sample()

        # 执行动作并获取下一个状态、奖励、是否终止和额外信息
        observation, reward, done, info = env.step(action)

        # 如果游戏结束，重置环境
        if done:
            observation = env.reset()

    # 关闭环境
    env.close()


if __name__ == "__main__":
    # test1()
    test2()


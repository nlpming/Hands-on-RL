# coding:utf-8
import os
import sys
import time
import pygame
import random
import gymnasium as gym


def test1():
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()


def test2():
    print('\n'.join([env for env in gym.envs.registry.keys()]))


def test3():
    pygame.init()  # 初始化pygame
    env = gym.make("CartPole-v1", render_mode="human")  # 创建CartPole游戏环境
    state, _ = env.reset()

    cart_position = state[0]  # 小车位置
    cart_speed = state[1]  # 小车速度
    pole_angle = state[2]  # 杆的角度
    pole_speed = state[3]  # 杆的角度苏

    print(f"Begin state: {state}")
    print(f"cart_position: {cart_position:.2f}")
    print(f"cart_speed: {cart_speed:.2f}")
    print(f"pole_angle: {pole_angle:.2f}")
    print(f"pole_speed: {pole_speed:.2f}")
    time.sleep(3)

    start_time = time.time()
    max_action = 1000  # 最大执行次数

    step = 0
    fail = False
    for step in range(1, max_action+1):
        time.sleep(0.3)

        # 以非阻塞的方式接收用户的键盘输入
        keys = pygame.key.get_pressed()
        action = 0

        if not keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
            action = random.choice([0, 1])
        if keys[pygame.K_LEFT]:
            action = 0
        elif keys[pygame.K_RIGHT]:
            action = 1

        state, _, done, _, _ = env.step(action)
        if done:  # 代表游戏结束
            fail = True
            break
        print(f"step:{step} action:{action} angle:{state[2]:.2f} position:{state[0]:.2f}")

    end_time = time.time()
    game_time = end_time - start_time
    if fail:
        print(f"游戏结束：时长 {game_time}s, 步数 {step}")
    else:
        print(f"通关游戏：时长 {game_time}s, 步数 {step}")
    env.close()


if __name__ == "__main__":
    # test1()
    # test2()
    test3()


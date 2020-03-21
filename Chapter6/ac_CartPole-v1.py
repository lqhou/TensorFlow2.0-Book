import gym
import numpy as np


def policy_function(observation, theta):
    x = np.dot(theta, observation)
    s = 1 / (1 + np.exp(-x))
    # 根据策略函数的输出进行动作选择
    if s > 0.5:
        action = 1
    else:
        action = 0
    return s, action


def actor(env, observation, theta, pre_phi, phi, df_gamma, df_lambda):
    # 学习率
    learning_rate = 0.001
    while True:
        s, action = policy_function(observation, theta)

        pre_observation = observation
        observation, reward, done, info = env.step(action)

        env.render()

        delta, pre_phi, phi = critic(pre_phi, phi, pre_observation, observation, reward, df_gamma, df_lambda)

        theta += learning_rate * df_lambda * delta * s * (1 - s) * (-pre_observation)

        df_lambda *= df_gamma

        if done:
            observation = env.reset()
            # 随机初始化策略函数和状态价值函数的参数
            theta = np.random.rand(4)
            phi = np.random.rand(4)
            pre_phi = phi

            # 折扣因子
            df_gamma = 0.9
            df_lambda = 1
            print("Faile")


def critic(pre_phi, phi, pre_observation, observation, reward, df_gamma, df_lambda):
    # 学习率
    learning_rate = 0.001

    v = np.dot(phi, observation)
    pre_v = np.dot(pre_phi, pre_observation)

    delta = reward + df_gamma * v - pre_v

    pre_phi = phi
    phi += learning_rate * df_lambda * delta * pre_observation

    return delta, pre_phi, phi


def actor_critic(env):
    observation = env.reset()

    # 随机初始化策略函数和状态价值函数的参数
    theta = np.random.rand(4)
    phi = np.random.rand(4)
    pre_phi = phi

    # 折扣因子
    discount_factor_gamma = 0.9
    discount_factor_lambda = 1

    actor(env, observation, theta, pre_phi, phi, discount_factor_gamma, discount_factor_lambda)


if __name__ == "__main__":
    # 注册游戏环境
    game_env = gym.make('CartPole-v1')
    # 取消限制
    game_env = game_env.unwrapped
    # 让agent开始学习玩“CartPole-v1”游戏
    actor_critic(game_env)
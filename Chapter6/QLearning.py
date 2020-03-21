import gym
import gym.spaces
from gym.envs.registration import register
import numpy as np
import random as rd


# 注册游戏环境（is_slippery的值为Flase）
register(
    id='FrozenLake8x8-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'is_slippery': True}
)

# 注册游戏环境
env = gym.make('FrozenLake8x8-v3')

# 定义Q值表，初始值设为0
Q = np.zeros([env.observation_space.n, env.action_space.n])
# 设置参数
learningRate = .85
discountFactor = .95

# 定义一个数组，用于保存每一回合得到的奖励
rewardList = []


def epsilon_greedy(q_table, s, num_episodes):
    rand_num = rd.randint(0, 20000)
    if rand_num > num_episodes:
        # 随机选择一个动作
        action = rd.randint(0, 3)
    else:
        # 选择一个最优的动作
        action = np.argmax(q_table[s, :])
    return action


def train():
    for i_episodes in range(20000):
        # 重置游戏环境
        s = env.reset()

        i = 0
        # 学习 Q-Table
        while i < 2000:
            i += 1
            # 使用带探索的策略（）选择动作
            a = epsilon_greedy(Q, s, i_episodes)

            # 执行动作，并得到新的环境状态、奖励等
            observation, reward, done, info = env.step(a)
            # 更新 Q-Table
            Q[s, a] = (1-learningRate) * Q[s, a] + learningRate * (
                    reward + discountFactor * np.max(Q[observation, :]))

            s = observation
            if done:
                break


def test():
    for i_episodes in range(100):
        # 重置游戏环境
        s = env.reset()
        i = 0
        total_reward = 0
        while i < 500:
            i += 1
            # 选择一个动作
            a = np.argmax(Q[s, :])
            # 执行动作，并得到新的环境状态、奖励等
            observation, reward, done, info = env.step(a)
            # 可视化游戏画面（重绘一帧画面）
            env.render()
            # 计算当前回合的总奖励值
            total_reward += reward
            s = observation
            if done:
                break
        rewardList.append(total_reward)


train()
test()

print("Final Q-Table Values：")
print(Q)
print("Success rate: " + str(sum(rewardList) / len(rewardList)))

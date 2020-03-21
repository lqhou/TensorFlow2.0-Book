#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: lqhou
@file: dqn1.py
@time: 2019/09/08
"""

import gym
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers

np.random.seed(1)
tf.random.set_seed(1)


class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__(name='q_network')
        self.fc1 = layers.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.fc2 = layers.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.logits = layers.Dense(num_actions, name='q_values')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.logits(x)
        return x

    def action_value(self, obs):
        q_values = self.predict(obs)
        best_action = np.argmax(q_values, axis=-1)
        return best_action[0], q_values[0]


class DQNAgent:
    def __init__(self, model, target_model, env, buffer_size=100,
                 learning_rate=.00001, epsilon=.1, epsilon_dacay=0.995,
                 min_epsilon=.01, gamma=.95, batch_size=5,
                 target_update_iter=400, train_nums=20000, start_learning=10):
        self.model = model
        self.target_model = target_model

        optimizer = optimizers.Adam(clipvalue=10.0)
        self.model.compile(optimizer=optimizer, loss='mse')

        self.env = env                              # gym环境
        self.lr = learning_rate                     # 学习率
        self.epsilon = epsilon                      # epsilon-greedy
        self.epsilon_decay = epsilon_dacay          # epsilon衰减因子
        self.min_epsilon = min_epsilon              # epsilon最小值
        self.gamma = gamma                          # 折扣因子
        self.batch_size = batch_size                # batch_size
        self.target_update_iter = target_update_iter    # 目标网络参数的更新周期
        self.train_nums = train_nums                # 总训练步数
        self.num_in_buffer = 0                      # 经验池中已经保存的经验数
        self.buffer_size = buffer_size              # 经验池的大小
        self.start_learning = start_learning        # 开始训练之前要先确保经验池中有一定量数据

        # 经验池参数 [(s, a, r, ns, done), ...]
        self.obs = np.empty((self.buffer_size,) + self.env.reset().shape)
        self.actions = np.empty(self.buffer_size, dtype=np.int8)
        self.rewards = np.empty(self.buffer_size, dtype=np.float32)
        self.dones = np.empty(self.buffer_size, dtype=np.bool)
        self.next_states = np.empty((self.buffer_size,) + self.env.reset().shape)
        self.next_idx = 0

    def train(self):
        """模型训练"""
        # 初始化环境信息
        obs = self.env.reset()
        for t in range(1, self.train_nums):
            best_action, q_values = self.model.action_value(obs[None])
            # 采取epsilon-greedy策略对环境进行探索，得到最终执行的动作
            action = self.get_action(best_action)
            # 执行动作，获取反馈信息
            next_obs, reward, done, info = self.env.step(action)
            print(reward)
            # 将经验保存到经验池
            self.store_transition(obs, action, reward, next_obs, done)
            self.num_in_buffer = min(self.num_in_buffer + 1, self.buffer_size)

            # 开始学习
            if t > self.start_learning:
                losses = self.train_step()
                if t % 1000 == 0:
                    print('{}/{} loss: {}'.format(self.train_nums, t, losses))

            if t % self.target_update_iter == 0:
                # 更新目标网络的参数
                self.update_target_model()
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs

    def train_step(self):
        """逐步训练，经验回放"""
        idxes = self.replay_transition(self.batch_size)
        s_batch = self.obs[idxes]
        a_batch = self.actions[idxes]
        r_batch = self.rewards[idxes]
        ns_batch = self.next_states[idxes]
        done_batch = self.dones[idxes]

        # 使用Target-Q网络计算目标值
        target_q = r_batch + self.gamma * np.amax(self.get_target_value(ns_batch), axis=1) * (1 - done_batch)
        # 使用Q网络产生预测值
        q_value = self.model.predict(s_batch)
        for i, val in enumerate(a_batch):
            q_value[i][val] = target_q[i]

        # 使用train_on_batch方法进行训练
        losses = self.model.train_on_batch(s_batch, q_value)

        return losses

    def evalation(self, env, render=True):
        """模型验证"""
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, q_values = self.model.action_value(obs[None])
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        env.close()
        return ep_reward

    def store_transition(self, obs, action, reward, next_state, done):
        """存储经验"""
        n_idx = self.next_idx % self.buffer_size
        self.obs[n_idx] = obs
        self.actions[n_idx] = action
        self.rewards[n_idx] = reward
        self.next_states[n_idx] = next_state
        self.dones[n_idx] = done
        self.next_idx = (self.next_idx + 1) % self.buffer_size

    def replay_transition(self, n):
        """经验回放"""
        assert n < self.num_in_buffer
        res = []
        while True:
            num = np.random.randint(0, self.num_in_buffer)
            if num not in res:
                res.append(num)
            if len(res) == n:
                break
        return res

    def get_action(self, best_action):
        """epsilon-greedy"""
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return best_action

    def update_target_model(self):
        # 将Q网络的参数拷贝给目标Q网络
        self.target_model.set_weights(self.model.get_weights())

    def get_target_value(self, obs):
        return self.target_model.predict(obs)

    def e_decay(self):
        self.epsilon *= self.epsilon_decay


if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    num_actions = env.action_space.n
    model = Model(num_actions)
    target_model = Model(num_actions)
    agent = DQNAgent(model, target_model,  env)

    agent.train()
    # test
    rewards_sum = agent.evalation(env)
    print("Test Result: %d out of 200" % rewards_sum)

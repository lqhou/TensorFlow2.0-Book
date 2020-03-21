import numpy as np

# 输入数据，总共三个time step
dataset = np.array([[1, 2], [2, 3], [3, 4]])

# 初始化相关参数
state = [0.0, 0.0]  # 记忆单元

# 给定随机数种子，每次产生相同的随机数
np.random.seed(2)
W_h = np.random.rand(4, 2)  # 隐藏层权重矩阵
b_h = np.random.rand(2)     # 隐藏层偏置项

np.random.seed(3)
W_o = np.random.rand(2)     # 输出层权重矩阵
b_o = np.random.rand()      # 输出层偏置项

for i in range(len(dataset)):

    # 将前一时刻的状态和当前的输入拼接
    value = np.append(state, dataset[i])

    # 隐藏层计算
    h_in = np.dot(value, W_h) + b_h  # 隐藏层的输入
    h_out = np.tanh(h_in)            # 隐藏层的输出
    state = h_out                    # 保存当前状态

    # 输出层
    y_in = np.dot(h_out, W_o) + b_o  # 输出层的输入
    y_out = np.tanh(y_in)            # 输出层的输出（即最终神经网络的输出）

    print(y_out)

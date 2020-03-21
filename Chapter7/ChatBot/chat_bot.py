#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: lqhou
@file: chat_bot.py
@time: 2019/09/07
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
import jieba
import os
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def preprocess_sentence(sentence):
    """为句子添加开始和结束标记"""
    sentence = '<start> ' + sentence + ' <end>'
    return sentence


def max_length(tensor):
    # 计算问答序列的最大长度
    return max(len(t) for t in tensor)


def tokenize(sentences):
    # 初始化分词器，并生成词典
    sentences_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    sentences_tokenizer.fit_on_texts(sentences)

    # 利用词典将文本数据转为id表示
    tensor = sentences_tokenizer.texts_to_sequences(sentences)
    # 将数据pad成统一长度，以所有数据中最大长度为准，长度不够的补零
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=30, padding='post')

    return tensor, sentences_tokenizer


def load_dataset(file_path):
    """加载数据集"""
    with open(file_path, "r") as file:
        lines = file.readlines()
        q = ''
        a = ''
        qa_pairs = []
        for i in range(len(lines)):
            if i % 3 == 0:
                q = " ".join(jieba.cut(lines[i].strip()))
            elif i % 3 == 1:
                a = " ".join(jieba.cut(lines[i].strip()))
            else:  # 组合
                pair = [preprocess_sentence(q), preprocess_sentence(a)]
                qa_pairs.append(pair)
    # zip操作删除重复问答
    # zip返回格式：[(q,a),(q,a),...]
    q_sentences, a_sentences = zip(*qa_pairs)

    q_tensor, q_tokenizer = tokenize(q_sentences)
    a_tensor, a_tokenizer = tokenize(a_sentences)

    return q_tensor, a_tensor, q_tokenizer, a_tokenizer


class Encoder(tf.keras.Model):
    """编码器"""
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.Model):
    """Bahdanau attention"""
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query为Encoder最后一个时间步的隐状态（hidden）
        # values为Encoder的输出，形状为(batch_size, max_length, hidden size)
        # query的形状为：(batch_size, hidden size)
        # 为了后续计算，需要将query的形状转为(batch_size, 1, hidden size)
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # 计算Score和Attention Weights
        # score的形状：(batch_size, max_length, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights的形状：(batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # 计算Context Vector
        context_vector = attention_weights * values
        # 求和之后的形状：(batch_size, hidden_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    """解码器"""
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        # attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # 获得Context Vector和Attention Weights
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # 编码之后x的形状：(batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # 将context_vector和输入x拼接，
        # 拼接后形状：(batch_size, 1, embedding_dim + hidden_size)
        # 这里的hidden_size即context_vector向量的长度
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # 拼接后输入GRU网络
        output, state = self.gru(x)

        # reshape前output形状为：(batch_size, 1, hidden_size)
        # reshape后output形状为：(batch_size, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # x的形状为：(batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


def loss_function(real, pred):
    """交叉熵损失函数"""
    # 返回非零值（去掉了序列不够长时填补的零）
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    # 交叉熵损失
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss_ = loss_object(real, pred)
    # 将mask转为loss_.dtype类型
    mask = tf.cast(mask, dtype=loss_.dtype)
    # 计算loss
    loss_ *= mask

    # 每次计算的是一个batch_size的数据，因此要求平均loss
    return tf.reduce_mean(loss_)


# 使用Adam优化器
optimizer = tf.keras.optimizers.Adam()


@tf.function
def train_step(q, a, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(q, enc_hidden)
        dec_hidden = enc_hidden
        # Decoder第一个时间步的输入
        dec_input = tf.expand_dims([a_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        # 逐个时间步进行Decoder
        for t in range(1, a.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            # 计算当前时间步的loss
            loss += loss_function(a[:, t], predictions)
            # 使用Teacher Forcing方法，该方法要求模型的生成结果必须和参考句一一对应
            dec_input = tf.expand_dims(a[:, t], 1)
    # 要输出的一个batch的loss(取Decoder中所有时间步loss的均值)
    batch_loss = (loss / int(a.shape[1]))

    # 优化参数
    variables = encoder.trainable_variables + decoder.trainable_variables
    # 计算梯度
    gradients = tape.gradient(loss, variables)
    # 使用Adam优化器更新参数
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def train():
    """模型训练"""
    # 设置模型的存储路径
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(),
                                     encoder=encoder,
                                     decoder=decoder)

    epochs = 10
    for epoch in range(epochs):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (q, a)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(q, a, enc_hidden)
            total_loss += batch_loss

            # 每隔100个batch输出一次loss信息
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # 每2个回合保存一次checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def test(sentence):
    """测试"""
    # 加载模型
    checkpoint_dir = './training_checkpoints'
    checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(),
                                     encoder=encoder,
                                     decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    sentence = " ".join(jieba.cut(sentence.strip()))
    sentence = preprocess_sentence(sentence)

    inputs = [q_tokenizer.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           # maxlen=max_q_tensor,
                                                           maxlen=30,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([a_tokenizer.word_index['<start>']], 0)

    for t in range(max_q_tensor):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()
        result += a_tokenizer.index_word[predicted_id] + ' '

        if a_tokenizer.index_word[predicted_id] == '<end>':
            break

        dec_input = tf.expand_dims([predicted_id], 0)

    print('Q: %s' % sentence)
    print('A: {}'.format(result))


if __name__ == '__main__':
    corpus_path = "./corpus_data/corpus.txt"
    q_tensor, a_tensor, q_tokenizer, a_tokenizer = load_dataset(corpus_path)
    max_a_tensor, max_q_tensor = max_length(a_tensor), max_length(q_tensor)

    BUFFER_SIZE = len(q_tensor)
    BATCH_SIZE = 10
    steps_per_epoch = len(q_tensor) // BATCH_SIZE
    embedding_dim = 128
    units = 512

    vocab_q_size = len(q_tokenizer.word_index) + 1
    vocab_a_size = len(a_tokenizer.word_index) + 1

    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(q_tensor,
                                                                                                    a_tensor,
                                                                                                    test_size=0.5)
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    encoder = Encoder(vocab_q_size, embedding_dim, units, BATCH_SIZE)
    attention_layer = BahdanauAttention(10)
    decoder = Decoder(vocab_a_size, embedding_dim, units, BATCH_SIZE)

    print("Start Training")

    start_train = time.time()

    train()

    print('Time taken for train {} sec\n'.format(time.time() - start_train))

    # 测试
    # input_sentence = "Start chatting"
    # while input_sentence != "stop":
    #     print("请输入：")
    #     input_sentence = input()
    #     test(input_sentence)
    #     print("-------------------")

"""
数据预处理
@author：lqhou
@date：2018/8/10
"""

from collections import Counter
import tensorflow as tf
import os

DOCUMENTS = list()


class DataConfig:
    # 词汇表路径
    vocab_path = "./vocab.txt"
    # 词汇表大小
    vocab_size = 5000
    # 待分类文本的最大长度
    max_length = 200


def build_vocab():
    """根据数据集构建词汇表"""
    all_data = []
    for content in DOCUMENTS:
        all_data.extend(content)

    # 选出出现频率最高的前dict_size个字
    counter = Counter(all_data)
    count_pairs = counter.most_common(DataConfig.vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 作为填充字符
    words = ['<PAD>'] + list(words)
    # 保存词汇表
    open(DataConfig.vocab_path, mode='w').write('\n'.join(words) + '\n')


def read_file(dir_path):
    global DOCUMENTS
    # 列出当前目录下的所有子目录
    dir_list = os.listdir(dir_path)
    # 遍历所有子目录
    for sub_dir in dir_list:
        # 组合得到子目录的路径
        child_dir = os.path.join('%s/%s' % (dir_path, sub_dir))
        if os.path.isfile(child_dir):
            # 获取当前目录下的数据文件
            with open(child_dir, 'r') as file:
                document = ''
                lines = file.readlines()
                for line in lines:
                    # 将文件内容组成一行，并去掉换行和空格等字符
                    document += line.strip()
            DOCUMENTS.append(dir_path[dir_path.rfind('/')+1:] + "\t" + document)
        else:
            read_file(child_dir)


def load_data(dir_path):
    global DOCUMENTS
    data_x = []
    data_y = []

    # 读取所有数据文件
    read_file(dir_path)

    # 读取词汇表，词汇表不存在时重新构建
    if not os.path.exists(DataConfig.vocab_path):
        build_vocab()

    with open(DataConfig.vocab_path, 'r') as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))

    # 构建类标
    categories = ['科技', '股票', '体育', '娱乐', '时政',
                  '社会', '教育', '财经', '家居', '游戏']
    cat_to_id = dict(zip(categories, range(len(categories))))

    # contents, labels = read_file(data_path)
    for document in DOCUMENTS:
        y_, x_ = document.split("\t", 1)
        data_x.append([word_to_id[x] for x in x_ if x in word_to_id])
        data_y.append(cat_to_id[y_])

    # 将文本pad为固定长度
    data_x = tf.keras.preprocessing.sequence.pad_sequences(data_x, DataConfig.max_length)
    # 将标签转换为one-hot表示
    data_y = tf.keras.utils.to_categorical(data_y, num_classes=len(cat_to_id))

    return data_x, data_y


if __name__ == '__main__':
    # 遍历指定目录，显示目录下的所有文件名
    data_path = "./news_data"
    load_data(data_path)

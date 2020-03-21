'''
功能：利用双偏振数据生成图像
'''


import pandas as pd
import cv2
import numpy as np

INPUT_PATH='./data/'

def create_train():
    # Read the json files into a pandas dataframe
    df_train = pd.read_json(INPUT_PATH + 'train.json')
    print("df_train.size", len(df_train))

    # for ix, row in df_train.iterrows():
    #     img = np.array(row['band_1']).reshape((75, 75))
    #
    #     img2 = np.array(row['band_2']).reshape((75, 75))
    #     #img3 = img + img2
    #     img3 = img2
    #     img3 -= img3.min()
    #     img3 /= img3.max()
    #     img3 *= 255
    #     img3 = img3.astype(np.uint8)
    #     if row['is_iceberg']==0:
    #         cv2.imwrite("./data/train/00/f{}.png".format(ix), img3)
    #
    #     elif row['is_iceberg']==1:
    #         cv2.imwrite("./data/train/11/f{}.png".format(ix), img3)

create_train()

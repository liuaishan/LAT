'''
@author: liuaishan
@contact: liuaishan@buaa.edu.cn
@file: utils.py
@time: 2018/10/28 12:30
@desc:
'''

import os
import numpy as np
import pickle


# read data from files
# @return data,label (numpy array)
def read_data(file_path):

    if not os.path.exists(file_path):
        return None, None

    with open(file_path, 'rb') as fr:
        data_set = pickle.load(file_path)
        size = len(data_set)

        # illegal data
        if not len(data_set['data']) == len(data_set['label']):
            return None, None
        
        data = data_set['data'][:size] / 255.
        label = data_set['label'][:size] / 255.
        data = np.asarray(data)
        label = np.asarray(label)
        return data, label

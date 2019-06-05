# _*_ coding:utf-8 _*_

# https://blog.csdn.net/weixin_42097808/article/details/80172824

import numpy as np
import pandas as pd

def calc_score_median(sample_set, var):
    '''
    计算相邻评分的中位数，以便进行决策树二元切分
    param sample_set: 待切分样本
    param var: 分割变量名称
    '''

    var_list = list(np.unique(sample_set[var]))
    var_median_list = []
    for i in range(len(var_list) - 1):
        var_median = (var_list[i] + var_list[i + 1]) / 2
        var_median_list.append(var_median)
    return var_median_list


# 读取数据集，至少包含变量和target两列
sample_set = pd.read_csv('data_ceshi_fx.csv', encoding='utf-8')
var = sample_set.columns[1:]
var_median_list = calc_score_median(sample_set=sample_set, var=var)

print(var_median_list)
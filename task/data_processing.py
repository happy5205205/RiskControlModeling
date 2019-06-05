# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:15:50 2019

"""

import numpy as np
from sklearn import metrics


def find_threshold(start, end, step, y_true, y_pred_prob):
    '''
        start, step, end: 阈值网格
        y_true: 二分类真实标签
        y_pred_prob: 二分类概率
    '''
    f1_max = 0
    threshold = start
    for x in np.arange(start, end, step):
        y_pred = [1 if i >= x else 0 for i in y_pred_prob]
        f1 = metrics.f1_score(y_true, y_pred)
        print('threshold : %f, f1 : %f' %(x, f1))
        if f1 > f1_max:
            f1_max = f1
            threshold = x
    return threshold, f1_max


def classification_metrics(y_true, y_pred):
    '''
        二分类指标
    '''
    f1 = metrics.f1_score(y_true, y_pred)
    return f1

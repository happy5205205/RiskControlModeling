# _*_coding:utf-8 _*_

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from scipy.stats import ks_2samp


def ks_calc_cross(data, pred, y_label):

    """
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    输入值:
    data: 二维数组或dataframe，包括模型得分和真实的标签
    pred: 一维数组或series，代表模型得分（一般为预测正类的概率）
    y_label: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    输出值:
    'ks': KS值，'crossdens': 好坏客户累积概率分布以及其差值gap
    """
    crossfreq = pd.crosstab(data[pred[0]], data[y_label[0]])
    crossdens = crossfreq.cumsum(axis=0) / crossfreq.sum()
    crossdens['gap'] = abs(crossdens[0] - crossdens[1])
    ks = crossdens[crossdens['gap'] == crossdens['gap'].max()]
    return ks, crossdens


def ks_calc_auc(data, pred, y_label):
    '''
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    输入值:
    data: 二维数组或dataframe，包括模型得分和真实的标签
    pred: 一维数组或series，代表模型得分（一般为预测正类的概率）
    y_label: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    输出值:
    'ks': KS值
    '''
    fpr, tpr, thresholds = roc_curve(data[y_label[0]], data[pred[0]])
    ks = max(tpr - fpr)
    return ks


def ks_calc_2samp(data, pred, y_label):
    '''
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    输入值:
    data: 二维数组或dataframe，包括模型得分和真实的标签
    pred: 一维数组或series，代表模型得分（一般为预测正类的概率）
    y_label: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    输出值:
    'ks': KS值，'cdf_df': 好坏客户累积概率分布以及其差值gap
    '''
    Bad = data.loc[data[y_label[0]] == 1, pred[0]]
    Good = data.loc[data[y_label[0]] == 0, pred[0]]
    data1 = Bad.values
    data2 = Good.values
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / (1.0 * n1)
    cdf2 = (np.searchsorted(data2, data_all, side='right')) / (1.0 * n2)
    ks = np.max(np.absolute(cdf1 - cdf2))
    cdf1_df = pd.DataFrame(cdf1)
    cdf2_df = pd.DataFrame(cdf2)
    cdf_df = pd.concat([cdf1_df, cdf2_df], axis=1)
    cdf_df.columns = ['cdf_Bad', 'cdf_Good']
    cdf_df['gap'] = cdf_df['cdf_Bad'] - cdf_df['cdf_Good']
    return ks, cdf_df


data = {'y_label': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        'pred': [0.5, 0.6, 0.7, 0.6, 0.6, 0.8, 0.4, 0.2, 0.1, 0.4, 0.3, 0.9]}

data = pd.DataFrame(data)
ks1, crossdens = ks_calc_cross(data, ['pred'], ['y_label'])

ks2 = ks_calc_auc(data, ['pred'], ['y_label'])

ks3 = ks_calc_2samp(data, ['pred'], ['y_label'])

get_ks = lambda y_pred, y_true: ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic
ks4 = get_ks(data['pred'], data['y_label'])
print('KS1:', ks1['gap'].values)
print('KS2:', ks2)
print('KS3:', ks3[0])
print('KS4:', ks4)
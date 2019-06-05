# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:54:58 2018
@author: wolfly_fu
解决的问题：
1、实现了二分类的卡方分箱
2、实现了最大分组限定停止条件，和最小阈值限定停止条件；
问题，
1、自由度k,如何来确定？
算法扩展：
1、卡方分箱除了用阈值来做约束条件，还可以进一步的加入分箱数约束，以及最小箱占比，坏人率约束等。
2、需要实现更多分类的卡方分箱算法
"""
# https://blog.csdn.net/m0_37235489/article/details/84590656
import pandas as pd
import numpy as np
from scipy.stats import chi2


# 计算卡方统计量
def cal_chi2(input_df, var_name, Y_name):  # 二分类，，计算每个变量值的卡方统计量
    '''
    df = input_df[[var_name, Y_name]]
    var_values = sorted(list(set(df[var_name])))
    Y_values = sorted(list(set(df[Y_name])))
    #用循环的方式填充
    chi2_result = pd.DataFrame(index=var_values, columns=Y_values)
    for var_value in var_values:
        for Y_value in Y_values:
            chi2_result.loc[var_value][Y_value] = \
            df[(df[var_name]==var_value)&(df[Y_name]==Y_value)][var_name].count()
    '''
    # input_df = input_df[[var_name, Y_name]]  # 取数据
    all_cnt = input_df[Y_name].count()  # 样本总数
    all_0_cnt = input_df[input_df[Y_name] == 0].shape[0]  # 二分类的样本数量
    all_1_cnt = input_df[input_df[Y_name] == 1].shape[0]
    expect_0_ratio = all_0_cnt * 1.0 / all_cnt  # 样本分类比例
    expect_1_ratio = all_1_cnt * 1.0 / all_cnt

    # 对变量的每个值计算实际个数，期望个数，卡方统计量
    var_values = sorted(list(set(input_df[var_name])))
    actual_0_cnt = []  # actual_0  该值，类别为0的数量
    actual_1_cnt = []  # actual_1  该值，类别为1的数量
    actual_all_cnt = []
    expect_0_cnt = []  # expect_0 类别0 的卡方值
    expect_1_cnt = []  # expect_1 类别1 的卡方值
    chi2_value = []  # chi2_value 该组的卡方值

    for value in var_values:
        actual_0 = input_df[(input_df[var_name] == value) & (input_df[Y_name] == 0)].shape[0]  # 该值，类别为0的数量
        actual_1 = input_df[(input_df[var_name] == value) & (input_df[Y_name] == 1)].shape[0]
        actual_all = actual_0 + actual_1  # 总数
        expect_0 = actual_all * expect_0_ratio  # 类别0 的 期望频率
        expect_1 = actual_all * expect_1_ratio

        chi2_0 = (expect_0 - actual_0) ** 2 / expect_0  # 类别0 的卡方值
        chi2_1 = (expect_1 - actual_1) ** 2 / expect_1

        actual_0_cnt.append(actual_0)  # 样本为0的，该值的数量
        actual_1_cnt.append(actual_1)

        actual_all_cnt.append(actual_all)  # 改组的总样本数
        expect_0_cnt.append(expect_0)  # 类别0 的 期望频率
        expect_1_cnt.append(expect_1)

        chi2_value.append(chi2_0 + chi2_1)  # 改变量值的卡方值

    chi2_result = pd.DataFrame({'actual_0': actual_0_cnt, 'actual_1': actual_1_cnt, 'expect_0': expect_0_cnt,
                                'expect_1': expect_1_cnt, 'chi2_value': chi2_value, var_name + '_start': var_values,
                                var_name + '_end': var_values},
                               columns=[var_name + '_start', var_name + '_end', 'actual_0', 'actual_1', 'expect_0',
                                        'expect_1', 'chi2_value'])

    return chi2_result, var_name

# 导入数据
df = pd.read_csv('data_ceshi_copy.csv')

df = df.drop(['card_name', 'id_card_no', 'loan_date'], axis=1)

cols = df.columns

var_name = list(df.columns[1 :])
Y_name = [df.columns[0]]

chi2_result, var_name = cal_chi2(input_df=df, var_name=var_name, Y_name=Y_name)




# 定义合并区间的方法
def merge_area(chi2_result, var_name, idx, merge_idx):
    # 按照idx和merge_idx执行合并
    chi2_result.ix[idx, 'actual_0'] = chi2_result.ix[idx, 'actual_0'] + chi2_result.ix[merge_idx, 'actual_0']
    chi2_result.ix[idx, 'actual_1'] = chi2_result.ix[idx, 'actual_1'] + chi2_result.ix[merge_idx, 'actual_1']
    chi2_result.ix[idx, 'expect_0'] = chi2_result.ix[idx, 'expect_0'] + chi2_result.ix[merge_idx, 'expect_0']
    chi2_result.ix[idx, 'expect_1'] = chi2_result.ix[idx, 'expect_1'] + chi2_result.ix[merge_idx, 'expect_1']
    chi2_0 = (chi2_result.ix[idx, 'expect_0'] - chi2_result.ix[idx, 'actual_0']) ** 2 / chi2_result.ix[idx, 'expect_0']
    chi2_1 = (chi2_result.ix[idx, 'expect_1'] - chi2_result.ix[idx, 'actual_1']) ** 2 / chi2_result.ix[idx, 'expect_1']

    chi2_result.ix[idx, 'chi2_value'] = chi2_0 + chi2_1  # 计算卡方值

    # 调整每个区间的起始值
    if idx < merge_idx:
        chi2_result.ix[idx, var_name + '_end'] = chi2_result.ix[merge_idx, var_name + '_end']  # 向后扩大范围
    else:
        chi2_result.ix[idx, var_name + '_start'] = chi2_result.ix[merge_idx, var_name + '_start']  ##，向前扩大范围

    chi2_result = chi2_result.drop([merge_idx])  # 删掉行
    chi2_result = chi2_result.reset_index(drop=True)

    return chi2_result


# 自动进行分箱，使用最大区间限制
def chiMerge_maxInterval(chi2_result, var_name, max_interval=5):  # 最大分箱数 为  5
    groups = chi2_result.shape[0]  # 各组的卡方值，数量
    while groups > max_interval:
        min_idx = chi2_result[chi2_result['chi2_value'] == chi2_result['chi2_value'].min()].index.tolist()[
            0]  # 寻找最小的卡方值
        if min_idx == 0:
            chi2_result = merge_area(chi2_result, var_name, min_idx, min_idx + 1)  # 合并1和2组
        elif min_idx == groups - 1:
            chi2_result = merge_area(chi2_result, var_name, min_idx, min_idx - 1)

        else:  # 寻找左右两边更小的卡方组
            if chi2_result.loc[min_idx - 1, 'chi2_value'] > chi2_result.loc[min_idx + 1, 'chi2_value']:
                chi2_result = merge_area(chi2_result, var_name, min_idx, min_idx + 1)
            else:
                chi2_result = merge_area(chi2_result, var_name, min_idx, min_idx - 1)
        groups = chi2_result.shape[0]

    return chi2_result


def chiMerge_minChiSquare(chi2_result, var_name):  # (chi_result, maxInterval=5):
    '''
    卡方分箱合并--卡方阈值法，，同时限制，最大组为6组，，可以去掉
    '''
    threshold = get_chiSquare_distribution(4, 0.1)
    min_chiSquare = chi2_result['chi2_value'].min()
    # min_chiSquare = chi_result['chi_square'].min()
    group_cnt = len(chi2_result)
    # 如果变量区间的最小卡方值小于阈值，则继续合并直到最小值大于等于阈值
    while (min_chiSquare < threshold and group_cnt > 6):
        min_idx = chi2_result[chi2_result['chi2_value'] == chi2_result['chi2_value'].min()].index.tolist()[
            0]  # 寻找最小的卡方值
        # min_index = chi_result[chi_result['chi_square']==chi_result['chi_square'].min()].index.tolist()[0]
        # 如果分箱区间在最前,则向下合并
        if min_idx == 0:
            chi2_result = merge_area(chi2_result, var_name, min_idx, min_idx + 1)  # 合并1和2组
        elif min_idx == group_cnt - 1:
            chi2_result = merge_area(chi2_result, var_name, min_idx, min_idx - 1)

        else:  # 寻找左右两边更小的卡方组
            if chi2_result.loc[min_idx - 1, 'chi2_value'] > chi2_result.loc[min_idx + 1, 'chi2_value']:
                chi2_result = merge_area(chi2_result, var_name, min_idx, min_idx + 1)
            else:
                chi2_result = merge_area(chi2_result, var_name, min_idx, min_idx - 1)

        min_chiSquare = chi2_result['chi2_value'].min()
        group_cnt = len(chi2_result)

    return chi2_result


# 分箱主体部分包括两种分箱方法的主体函数，其中merge_chiSquare()是对区间进行合并，
# get_chiSquare_distribution()是根据自由度和置信度得到卡方阈值。我在这里设置的是自由度为4
# ，置信度为10%。两个自定义函数如下


def get_chiSquare_distribution(dfree=4, cf=0.1):
    '''
    根据自由度和置信度得到卡方分布和阈值
    dfree:自由度k= (行数-1)*(列数-1)，默认为4     #问题，自由度k,如何来确定？
    cf:显著性水平，默认10%
    '''
    percents = [0.95, 0.90, 0.5, 0.1, 0.05, 0.025, 0.01, 0.005]
    df = pd.DataFrame(np.array([chi2.isf(percents, df=i) for i in range(1, 30)]))
    df.columns = percents
    df.index = df.index + 1
    # 显示小数点后面数字
    pd.set_option('precision', 3)
    return df.loc[dfree, cf]
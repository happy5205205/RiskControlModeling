# _*_ coding:utf-8 _*_

import pandas as pd
import numpy as np


def Chi2(df, total_col, bad_col, overallRate):
    '''
     #此函数计算卡方值
     :df dataFrame
     :total_col 每个值得总数量
     :bad_col 每个值的坏数据数量
     :overallRate 坏数据的占比
     : return 卡方值
    '''
    df2 = df.copy()
    df2['expected'] = df[total_col].apply(lambda x: x * overallRate)
    combined = zip(df2['expected'], df2[bad_col])
    chi = [(i[0] - i[1]) ** 2 / i[0] for i in combined]
    chi2 = sum(chi)
    return chi2


# 基于卡方阈值卡方分箱，有个缺点，不好控制分箱个数。
def ChiMerge_MinChisq(df, col, target, confidenceVal=3.841):
    '''
    #此函数是以卡方阈值作为终止条件进行分箱
    : df dataFrame
    : col 被分箱的特征
    : task 目标值,是0,1格式
    : confidenceVal  阈值，自由度为1， 自信度为0.95时，卡方阈值为3.841
    : return 分箱。
    这里有个问题，卡方分箱对分箱的数量没有限制，这样子会导致最后分箱的结果是分箱太细。
    '''
    # 对待分箱特征值进行去重
    colLevels = set(df[col])

    # count是求得数据条数
    total = df.groupby(col)[target].count()
    total = pd.DataFrame({'total': total})

    # sum是求得特征值的和
    # 注意这里的target必须是0,1。要不然这样求bad的数据条数，就没有意义，并且bad是1，good是0。
    bad = df.groupby(col)[target].sum()
    bad = pd.DataFrame({'bad': bad})
    # 对数据进行合并，求出col，每个值的出现次数（total，bad）
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)

    # 求出整的数据条数
    N = sum(regroup['total'])
    # 求出黑名单的数据条数
    B = sum(regroup['bad'])
    overallRate = B * 1.0 / N

    # 对待分箱的特征值进行排序
    colLevels = sorted(list(colLevels))
    groupIntervals = [[i] for i in colLevels]

    groupNum = len(groupIntervals)
    while (1):
        if len(groupIntervals) == 1:
            break
        chisqList = []
        for interval in groupIntervals:
            df2 = regroup.loc[regroup[col].isin(interval)]
            chisq = Chi2(df2, 'total', 'bad', overallRate)
            chisqList.append(chisq)

        min_position = chisqList.index(min(chisqList))

        if min(chisqList) >= confidenceVal:
            break

        if min_position == 0:
            combinedPosition = 1
        elif min_position == groupNum - 1:
            combinedPosition = min_position - 1
        else:
            if chisqList[min_position - 1] <= chisqList[min_position + 1]:
                combinedPosition = min_position - 1
            else:
                combinedPosition = min_position + 1
        groupIntervals[min_position] = groupIntervals[min_position] + groupIntervals[combinedPosition]
        groupIntervals.remove(groupIntervals[combinedPosition])
        groupNum = len(groupIntervals)
    return groupIntervals

# 读取数据集，至少包含变量和target两列

sample_set = pd.read_csv('data_ceshi_fx.csv')

var_median_list = ChiMerge_MinChisq(df=sample_set, col='申请准入置信度', target='task')
# print(var_median_list)
for i in range(len(var_median_list)):
    print(min(var_median_list[i]), '---', max(var_median_list[i]), '分箱个数', len(var_median_list[i]))
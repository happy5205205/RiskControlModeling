# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:31:02 2019

@author: yunying_wu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
import re
import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree  #导入决策树模块
import graphviz  #导入可视化模块
from ScoreCardModel.weight_of_evidence import WeightOfEvidence

import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def KS(df, score, target, asc=False):
    '''
    :param df: the dataset containing probability and bad indicator
    :param score:
    :param target:
    :return:
    '''
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total': total, 'bad': bad})
    all['good'] = all['total'] - all['bad']
    all[score] = all.index
    all.index.name = 'index'
    all = all.sort_values(by=score, ascending=asc)
    all.index = range(len(all))
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    all['totalPcnt'] = all['total'] / all['total'].sum()
    KS = abs(all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1))
    KS_value = max(KS)
    return KS_value


    ################################# 输出单调性分析 #################################
def monotone_analysis(df,
                      qcut_flag=True,
                      bin_count=10,
                      miss_flag=False,
                      missing_thres=0.4):
    '''
    bin_count           #默认等频分箱, 若False,则为等距分箱
    qcut_flag          #默认等频分箱, 若False,则为等距分箱
    miss_flag          #是否需要删除缺失过多的变量，不建议删除
    missing_thres      #运行缺失比率小于等于missing_thres
    '''
    if qcut_flag:
        file_cut_name = 'qcut'
    else:
        file_cut_name = 'cut'
    workbook = xlsxwriter.Workbook('./{}.xlsx'.format('test_' + file_cut_name +
                                                      str(bin_count)))
    worksheet = workbook.add_worksheet()
    bold_format = workbook.add_format({'bold': True})
    money_format = workbook.add_format({'num_format': '$#,##0'})
    date_format = workbook.add_format({'num_format': 'mmmm d yyyy'})
    worksheet.set_column(1, 1, 15)
    worksheet.write('A1', '标签名', bold_format)
    worksheet.write('B1', '统计值', bold_format)

    features = df.columns[4:].tolist()
    num_sample = df.shape[0]

    removed_col = []
    ks_dict = {}
    iv_dict = {}
    monotone_dict = {}
    woe = WeightOfEvidence()

    for i, col in enumerate(features):
        selected = df.loc[:, [col, 'task']]
        #去除缺失值计算iv和ks
        selected = selected[~selected[col].isna()]

        if selected[col].nunique() == 1:
            continue

        if miss_flag and selected.shape[0] / num_sample < 1 - missing_thres:
            removed_col.append(col)
            continue
        #连续性变量，10等频率分箱
        if selected[col].dtypes == 'float64' or selected[col].dtypes == 'int64':
            if qcut_flag:
                selected[col] = pd.qcut(selected[col],
                                        bin_count,
                                        duplicates='drop')
            else:
                selected[col] = pd.cut(selected[col],
                                       bin_count,
                                       right=False,
                                       duplicates='drop')

        #如果不是连续性变量，延用原来的分箱
        else:
            try:
                selected[col] = selected[col].apply(lambda x: float(
                    re.split(',', x)[0].strip('[]()>')))
            except:
                selected[col] = selected[col].apply(lambda x: re.split(',', x)[
                    0].strip('[]()>'))

        ks_dict[col] = KS(df, col, 'task', asc=False)
        woe.fit(selected[col], selected['task'])
        iv_dict[col] = woe.iv

        total = selected.groupby(col).count()['task']
        bad = selected.groupby(col).sum()['task']
        overdueRate = pd.DataFrame({'total': total, 'bad': bad})
        overdueRate['rate'] = overdueRate['bad'] / overdueRate['total']
        overdueRate['rate'] = overdueRate['rate'].fillna(0)  #等距可能存在空箱

        badRate = list(overdueRate['rate'])
        badRateMonotone = [
            badRate[i] <= badRate[i + 1] for i in range(len(badRate) - 1)
        ]
        monotone = badRateMonotone.count(True) <= 1 or badRateMonotone.count(False) <= 1
        monotone_dict[col] = monotone

        worksheet.write_string(4 * i + 1, 0, col)
        worksheet.write_string(4 * i + 2, 0, col + '_分箱人数')
        worksheet.write_string(4 * i + 3, 0, col + '_首逾率')
        worksheet.write_string(4 * i + 4, 0, col + '_单调性')
        for j in range(overdueRate.shape[0]):
            worksheet.write_string(4 * i + 1, j + 1, str(overdueRate.index[j]))
            worksheet.write_number(4 * i + 2, j + 1,
                                   overdueRate.iloc[j]['total'])
            worksheet.write_number(4 * i + 3, j + 1,
                                   overdueRate.iloc[j]['rate'])
        worksheet.write_string(4 * i + 4, 1, str(monotone))
    workbook.close()

    outFile = pd.merge(pd.DataFrame.from_dict(iv_dict, orient='index',columns=['IV']),\
                       pd.DataFrame.from_dict(ks_dict, orient='index',columns=['KS']),\
                       left_index=True,right_index=True)
    outFile = pd.merge(outFile,\
                       pd.DataFrame.from_dict(monotone_dict, orient='index',columns=['monotone']),\
                       left_index=True,right_index=True)
    #输出单调性分析
    outFile.to_csv('Result_iv_ks_monotone_' + file_cut_name + str(bin_count) +
                   '.csv',
                   encoding='ansi')


def DT_analysis(df, selected_col):
    selected = df.loc[:, ['task'] + selected_col]

    for col in selected_col:
        if selected[col].dtypes != 'float64' and selected[
                col].dtypes != 'int64':
            selected[col] = selected[col].fillna('99999')
            selected[col] = selected[col].apply(lambda x: re.split(
                ',', str(x))[0].strip('[]()>'))


#            selected[col] = selected[col].astype('float')

    selected = selected.fillna(-99999)

    X = selected.loc[:, selected_col]
    y = selected.loc[:, 'task']
    dtc = DecisionTreeClassifier(criterion='entropy', max_depth=5)  #创建基于信息熵的模型
    dtc.fit(X, y)

    dot_data = tree.export_graphviz(
        dtc,  #（决策树模型）
        out_file=None,
        feature_names=X.columns,
        class_names=['0', '1'],  #模型中对应标签名
        rounded=True,
        special_characters=True,
        filled=True)
    dot_data = dot_data.replace('helvetica', '"Microsoft Yahei"')
    graph = graphviz.Source(dot_data)  #选择要可视化的dot数据
    graph.render(r'tree', view=True)


def matrix_analysis(df,
                    col1,
                    col2,
                    target,
                    cut_num1=10,
                    cut_num2=10,
                    cut_value1=None,
                    cut_value2=None):
    if cut_value1 != None:
        df[col1 + '_bin'] = pd.cut(df[col1], cut_value1, duplicates='drop')
    else:
        df[col1 + '_bin'] = pd.qcut(df[col1], cut_num1, duplicates='drop')
    if cut_value2 != None:
        df[col2 + '_bin'] = pd.cut(df[col2], cut_value2, duplicates='drop')
    else:
        df[col2 + '_bin'] = pd.qcut(df[col2], cut_num2, duplicates='drop')

    total = df.groupby([col1 + '_bin', col2 + '_bin']).count()['task']
    bad = df.groupby([col1 + '_bin', col2 + '_bin']).sum()['task']
    overdueRate = pd.DataFrame({'total': total, 'bad': bad})
    overdueRate['rate'] = overdueRate['bad'] / overdueRate['total']

    pt = pd.DataFrame(columns=df[col2 + '_bin'].dtypes.categories,
                      index=df[col1 + '_bin'].dtypes.categories,
                      dtype='float64')
    num_interval1 = df[col1 + '_bin'].unique().shape[0]
    num_interval2 = df[col2 + '_bin'].unique().shape[0]
    for i in range(num_interval1):
        for j in range(num_interval2):
            pt.iloc[i, j] = overdueRate['rate'].iloc[i * num_interval2 + j]
    sns.heatmap(pt, annot=True, fmt='.2g', linewidths=0.05)
    plt.xlabel(col2)
    plt.ylabel(col1)
    plt.show()


def main():

    file_name = 'test_sample_processed_test.csv'
    test_size = 0.9
    oot_flag = False

    df = pd.read_csv(open(file_name))
    df = df.drop_duplicates()
    df['loan_date'] = pd.to_datetime(df['loan_date'], format='%Y-%m-%d')

    #划分训练集,测试集
    if oot_flag:
        splitDate = df['loan_date'].quantile(q=test_size)
        oot_data = df[df['loan_date'] >= splitDate]
        notoot_data = df[df['loan_date'] < splitDate]

    else:
        df = df.sample(frac=1).reset_index(drop=True)
        split_idx = round(df.shape[0] * test_size)
        notoot_data = df.iloc[:split_idx]
        oot_data = df.iloc[split_idx:]

    notoot_data.to_csv('train.csv', index=False, encoding='ansi')
    oot_data.to_csv('test.csv', index=False, encoding='ansi')

    ######################## 单调性分析 ########################
    monotone_analysis(notoot_data,
                      qcut_flag=False,
                      bin_count=5,
                      miss_flag=True,
                      missing_thres=0.4)

    ######################## 决策树分析 ########################

    #决策树选用变量
    selected_col = [
        '申请雷达_申请准入分', '申请雷达_申请准入置信度', '申请雷达_查询机构数', '申请雷达_查询消费金融类机构数',
        '申请雷达_查询网络贷款类机构数', '申请雷达_总查询次数', '申请雷达_近1个月总查询笔数', '申请雷达_近3个月总查询笔数',
        '申请雷达_近6个月总查询笔数'
    ]
    DT_analysis(notoot_data, selected_col)

    ######################## 矩阵分析 ########################

    col1 = '申请雷达_近1个月总查询笔数'
    col2 = '申请雷达_近6个月总查询笔数'
    selected = notoot_data.loc[:, [col1, col2, 'task']]
    selected = selected.dropna()
    #    selected = selected[selected[col1]!=-99999]
    #    selected = selected[selected[col2]!=-99999]
    matrix_analysis(selected, col1, col2, 'task', cut_num1=6, cut_num2=5)
    #    matrix_analysis(selected,col1,col2,'task',cut_value1 =[float('-inf'),1,2,5,10,15,20,float('inf')], cut_num2 =6)

    ######################## 单变量分析 ########################
    col = '场景雷达_中大额分期_近1个月查询多头'
    cut_value = [float('-inf'), 0, 1, 2, 5, 10, 20, 40, float('inf')]
    selected = notoot_data.loc[:, [col, 'task']]
    selected = selected[~selected[col].isna()]
    selected[col + '_bin'] = pd.cut(selected[col],
                                    cut_value,
                                    right=False,
                                    duplicates='drop')
    total = selected.groupby(col + '_bin').count()['task']
    bad = selected.groupby(col + '_bin').sum()['task']
    overdueRate = pd.DataFrame({'total': total, 'bad': bad})
    overdueRate['rate'] = overdueRate['bad'] / overdueRate['total']
    print(overdueRate)

    ######################## 拒绝率,逾期率预估 ########################
    num_sample = selected.shape[0]
    pass_group = selected[~(selected[col] >= 40)]
    #    pass_group = selected[~((selected[col1]>6) & (selected[col2]<=4))]
    num_refused = num_sample - pass_group.shape[0]
    refused_rate = num_refused / num_sample
    overdue_rate = pass_group.sum()['task'] / pass_group.count()['task']
    print('拒绝率:{0:.3f},规则设置后逾期率:{1:.3f}'.format(refused_rate, overdue_rate))

    ######################## 总结结果预估 ########################

    #规则中选用到的字段
    selected_col = [
        '申请雷达_近1个月总查询笔数', '场景雷达_中大额分期_近1个月查询多头', '场景雷达_中大额分期_近6个月查询多头',
        '申请雷达_近6个月总查询笔数', '场景雷达_中大额分期_最近查询时间距今天数', '行为雷达_近3个月贷款笔数',
        '行为雷达_贷款已结清订单数'
    ]
    #    data = notoot_data
    data = oot_data
    data = data.loc[:, ['task'] + selected_col]

    #视情况预处理
    for col in selected_col:
        if data[col].dtypes != 'float64' and selected[col].dtypes != 'int64':
            data[col] = data[col].fillna('99999')
            data[col] = data[col].apply(lambda x: re.split(',', str(x))[0].
                                        strip('[]()>'))


#            selected[col] = selected[col].astype('float')
    data = data.fillna(-99999)

    #设置规则
    num_rule = 6
    data['rule0'] = data['申请雷达_近1个月总查询笔数'].apply(lambda x: 1 if (
        x >= 30 and x != -99999) else 0)
    data['rule1'] = data['场景雷达_中大额分期_近1个月查询多头'].apply(lambda x: 1 if (
        x >= 30 and x != -99999) else 0)
    data['rule2'] = data['场景雷达_中大额分期_近6个月查询多头'].apply(lambda x: 1 if (
        x >= 45 and x != -99999) else 0)
    data['rule3'] = data.apply(lambda x: 1 if (x.申请雷达_近1个月总查询笔数 > 22 and x.
                                               申请雷达_近6个月总查询笔数 > 29) else 0,
                               axis=1)
    data['rule4'] = data.apply(lambda x :1 if x.场景雷达_中大额分期_最近查询时间距今天数 >1\
        and x.场景雷达_中大额分期_最近查询时间距今天数 <=5 and x.申请雷达_近1个月总查询笔数 >17  else 0,axis = 1 )
    data['rule5'] = data.apply(lambda x: 1 if x.行为雷达_近3个月贷款笔数 > 6 and x.
                               行为雷达_贷款已结清订单数 <= 4 else 0,
                               axis=1)

    #评估结果
    data['refuse_result'] = data['rule1']
    for i in range(1, num_rule):
        data['refuse_result'] = data.apply(lambda x: x['refuse_result'] or x[
            'rule' + str(i)],
                                           axis=1)
    pass_group = data[data['refuse_result'] == 0]

    num_sample = data.shape[0]
    num_refused = num_sample - pass_group.shape[0]
    refused_rate = num_refused / num_sample
    overdue_rate = pass_group.sum()['task'] / pass_group.count()['task']
    print('拒绝率:{0:.3f},规则设置后逾期率:{1:.3f}'.format(refused_rate, overdue_rate))

if __name__ == '__main__':
    main()
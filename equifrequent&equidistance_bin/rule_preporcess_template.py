# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:31:02 2019

@author: peng zhang
等频等距分箱计算每个变量的iv和ks
"""

import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter
import re
from ScoreCardModel.weight_of_evidence import WeightOfEvidence

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def KS(df, score, target, asc=False):
    """
        :param df: the dataset containing probability and bad indicator
        :param score:
        :param target:
        :return:
    """
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


def monotone_analysis(df, qcut_flag=True, bin_count=10, miss_flag=False, missing_thres=0.4):
    """
    bin_count          分箱个数
    qcut_flag          若True，等频分箱, 若False,则为等距分箱
    miss_flag          是否需要删除缺失过多的变量，不建议删除
    missing_thres      运行缺失比率小于等于missing_thres
    """
    if qcut_flag:
        file_cut_name = 'equifrequent'
    else:
        file_cut_name = 'equidistance'
    workbook = xlsxwriter.Workbook('./{}.xlsx'.format(file_cut_name + '_feature_bin_' + str(bin_count)))
    worksheet = workbook.add_worksheet()
    bold_format = workbook.add_format({'bold': True})
    # money_format = workbook.add_format({'num_format': '$#,##0'})
    # date_format = workbook.add_format({'num_format': 'mmmm d yyyy'})
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
        selected = df.loc[:, [col, 'target']]
        # 去除缺失值计算iv和ks
        selected = selected[~selected[col].isna()]

        if len(set(selected[col])) == 1:
            continue
        if miss_flag and selected.shape[0] / num_sample < 1 - missing_thres:
            removed_col.append(col)
            continue
        # 连续性变量，10等频率分箱
        if selected[col].dtypes == 'float64' or selected[col].dtypes == 'int64':
            if qcut_flag:
                selected[col] = pd.qcut(selected[col], bin_count, duplicates='drop')
            else:
                selected[col] = pd.cut(selected[col], bin_count, right=False, duplicates='drop')

        # 如果不是连续性变量，延用原来的分箱
        else:
            try:
                selected[col] = selected[col].apply(lambda x: float(re.split(',', x)[0].strip('[]()>')))
            except:
                selected[col] = selected[col].apply(lambda x: re.split(',', x)[0].strip('[]()>'))

        ks_dict[col] = KS(df, col, 'target', asc=False)
        woe.fit(selected[col], selected['target'])
        iv_dict[col] = woe.iv

        total = selected.groupby(col).count()['target']
        bad = selected.groupby(col).sum()['target']
        overdueRate = pd.DataFrame({'total': total, 'bad': bad})
        overdueRate['rate'] = overdueRate['bad'] / overdueRate['total']
        overdueRate['rate'] = overdueRate['rate'].fillna(0)  # 等距可能存在空箱

        badRate = list(overdueRate['rate'])
        badRateMonotone = [badRate[i] <= badRate[i + 1] for i in range(len(badRate) - 1)]
        monotone = badRateMonotone.count(True) <= 1 or badRateMonotone.count(False) <= 1
        monotone_dict[col] = monotone
        worksheet.write_string(4 * i + 1, 0, col)
        worksheet.write_string(4 * i + 2, 0, col + '_分箱人数')
        worksheet.write_string(4 * i + 3, 0, col + '_首逾率')
        worksheet.write_string(4 * i + 4, 0, col + '_单调性')
        for j in range(overdueRate.shape[0]):
            worksheet.write_string(4 * i + 1, j + 1, str(overdueRate.index[j]))
            worksheet.write_number(4 * i + 2, j + 1, overdueRate.iloc[j]['total'])
            worksheet.write_number(4 * i + 3, j + 1, overdueRate.iloc[j]['rate'])
        worksheet.write_string(4 * i + 4, 1, str(monotone))
    workbook.close()

    outFile = pd.merge(pd.DataFrame.from_dict(iv_dict, orient='index', columns=['IV']),
                       pd.DataFrame.from_dict(ks_dict, orient='index', columns=['KS']),
                       left_index=True, right_index=True)
    outFile = pd.merge(outFile, pd.DataFrame.from_dict(monotone_dict, orient='index', columns=['monotone']),
                       left_index=True, right_index=True)
    # 输出单调性分析
    outFile.to_csv('result_iv_ks_monotone_' + file_cut_name + '_' + str(bin_count) + '.csv', encoding='ansi')


def main():

    file_name = 'test_file.csv'
    test_size = 0.9
    oot_flag = False

    df = pd.read_csv(open(file_name, encoding='UTF-8'))
    df = df.drop_duplicates()
    # df = df.drop(['id_card_no', 'card_name', 'loan_date'], axis=1)
    # df['loan_date'] = pd.to_datetime(df['loan_date'], format='%Y-%m-%d')

    # 划分训练集, 测试集
    if oot_flag:
        splitDate = df['loan_date'].quantile(q=test_size)
        oot_data = df[df['loan_date'] >= splitDate]
        notoot_data = df[df['loan_date'] < splitDate]

    else:
        df = df.sample(frac=1).reset_index(drop=True)
        split_idx = round(df.shape[0] * test_size)
        notoot_data = df.iloc[:split_idx]
        oot_data = df.iloc[split_idx:]

    monotone_analysis(notoot_data, qcut_flag=True, bin_count=5, miss_flag=True, missing_thres=0.4)


if __name__ == '__main__':
    main()
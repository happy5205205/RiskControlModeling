# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:24:38 2019

@author: yunying_wu
"""
import pandas as pd
from dateutil.relativedelta import relativedelta

# #################################  data loading  ##################################
target_file = "test_sample.csv"

df = pd.read_csv(open(target_file))
df.columns = ['card_name', 'id_card_no', 'loan_date', 'task']
df = df.dropna()
df['loan_date'] = pd.to_datetime(df['loan_date'], format='%Y-%m-%d')

file_name = {
    '全景雷达（回溯）.xlsx': ['申请雷达回溯测试结果', '行为雷达回溯测试结果', '信用现状回溯测试结果'],
    '场景雷达_中大额分期（回溯）.xlsx': ['场景雷达_中大额分期回溯测试结果'],
    '场景雷达_信用卡代偿（回溯）.xlsx': ['场景雷达_信用卡代偿回溯测试结果'],
    '场景雷达_小额分期（回溯）.xlsx': ['场景雷达_小额分期回溯测试结果'],
    '场景雷达_小额网贷（回溯）.xlsx': ['场景雷达_小额网贷回溯测试结果'],
    '场景雷达_消费分期（回溯）.xlsx': ['场景雷达_消费分期回溯测试结果']
}

# ##############################  data preprocessing  ###############################
# 选取loan date前最近一次的回溯标签
# 对于'最近一次贷款时间','最近查询时间','最近逾期时间'等时间变量计算时间间隔
# 最终输出将所有测试报告特征关联在一起的csv文件

for k in file_name.keys():
    for sheet in file_name[k]:
        curr = pd.read_excel('./测试报告/' + k, sheetname=sheet, skiprows=2)
        if '姓名' not in curr.columns:
            curr = pd.read_excel('./测试报告/' + k, sheetname=sheet, skiprows=3)
        if '身份证' in curr.columns:
            curr.rename(columns={'身份证': 'id_card_no'}, inplace=True)
        if '身份证号' in curr.columns:
            curr.rename(columns={'身份证号': 'id_card_no'}, inplace=True)
        if '姓名' in curr.columns:
            curr.rename(columns={'姓名': 'card_name'}, inplace=True)
        curr_df = df.loc[:, ['id_card_no', 'card_name', 'loan_date']].merge(
            curr,
            left_on=['id_card_no', 'card_name'],
            right_on=['id_card_no', 'card_name'])
        if '回溯日期' in curr_df.columns:
            curr_df.rename(columns={'回溯日期': 'retrospective_date'},
                           inplace=True)
        if '回溯时间' in curr_df.columns:
            curr_df.rename(columns={'回溯时间': 'retrospective_date'},
                           inplace=True)

        curr_df['retrospective_date'] = pd.to_datetime(
            curr_df['retrospective_date'], format='%Y-%m-%d')

        curr_df = curr_df[curr_df.loc[:, 'loan_date'] >=
                          curr_df.loc[:, 'retrospective_date']]
        curr_df.sort_values(['id_card_no', 'retrospective_date'],
                            ascending=[0, 1],
                            inplace=True)

        curr_df = curr_df.groupby(['id_card_no']).head(1)
        curr_df = curr_df.replace({'/': None})
        curr_df = curr_df.replace({'-': None})

        for col in ['最近一次贷款时间', '最近查询时间', '最近逾期时间']:
            if col in curr_df.columns:
                curr_df[col] = curr_df[col].fillna('1970-01-01')
                curr_df[col] = pd.to_datetime(curr_df[col], format='%Y-%m-%d')
                if col == '最近逾期时间':
                    curr_df[col] = curr_df.apply(lambda row:\
                           relativedelta(row['retrospective_date'], row[col]).years*12 + \
                           relativedelta(row['retrospective_date'], row[col]).months ,axis=1)

                else:
                    curr_df[col] = curr_df.apply(lambda row:\
                           relativedelta(row['retrospective_date'], row[col]).years*360 + \
                           relativedelta(row['retrospective_date'], row[col]).months*30 + \
                           relativedelta(row['retrospective_date'], row[col]).days ,axis=1)
                    curr_df[col] = pd.cut(curr_df[col], [
                        float('-inf'), 0, 30, 60, 90, 180, 360, 40 * 360,
                        float('inf')
                    ],
                                          right=False,
                                          duplicates='drop')

        retain_col = ['id_card_no', 'card_name', 'loan_date']
        renamed_col = ['id_card_no', 'card_name', 'loan_date']

        for c in curr_df.columns:
            if 'Unnamed' in c or 'retrospective_date' in c:
                continue
            elif c not in ['id_card_no', 'card_name', 'loan_date']:
                retain_col.append(c)
                renamed_col.append(sheet.strip('回溯测试结果') + '_' + c)

        curr_df = curr_df.loc[:, retain_col]
        curr_df.columns = renamed_col

        df = df.merge(curr_df,
                      left_on=['id_card_no', 'card_name', 'loan_date'],
                      right_on=['id_card_no', 'card_name', 'loan_date'],
                      how='left')

# ##############################  outputting  ###############################
df.to_csv('test_sample_processed_test.csv', encoding='ansi', index=False)

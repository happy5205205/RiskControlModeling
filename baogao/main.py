# _*_ coding:utf-8 _*_

import pandas as pd
import os

data_path = './data'

modeldata_chu_train_0529 = pd.read_csv(os.path.join(data_path, 'modeldata_chu_train_800_result_0529.csv'))

data_a = pd.read_excel(os.path.join(data_path, '全景雷达（回溯）.xlsx'), sheetname=1)

data_b = pd.read_excel(os.path.join(data_path, '全景雷达（回溯）.xlsx'), sheetname=2)

data_c = pd.read_excel(os.path.join(data_path, '全景雷达（回溯）.xlsx'), sheetname=3)


data = pd.merge(
       pd.merge(
       pd.merge(modeldata_chu_train_0529, data_a, how='inner', left_on=['card_name', 'id_card_no', 'loan_date'],
                                                              right_on=['姓名', '身份证号', '回溯时间']),
                                          data_b, how='inner', left_on=['card_name', 'id_card_no', 'loan_date'],
                                                               right_on=['姓名', '身份证号', '回溯时间']),
                                          data_c,  how='inner', left_on=['card_name', 'id_card_no', 'loan_date'],
                                                                right_on=['姓名', '身份证号', '回溯时间'])
train_data = data.drop(['姓名', '身份证号', '回溯时间',
                        '姓名_x', '身份证号_x', '回溯时间_x',
                        '姓名_y', '身份证号_y', '回溯时间_y',
                        '最近查询时间', '最近一次贷款时间',
                        'Unnamed: 20', 'Unnamed: 15'], axis=1
                       )
# 查看target有无数据为空
data_miss = train_data['task'].isnull().sum(axis=0)

train_data.to_csv('data_800.csv', index=False)

# -*- coding: utf-8 -*-

"""
Created on Mon May 13 20:56:34 2019

@author: zhangpeng
"""
import pickle
import pandas as pd
import os

data_path_0509 = './data0509'
data_path_0515 = './data0515'


def main():
    # data1 = pickle.load(open(os.path.join(data_path_0509, 'keepFeaturesFinal1.pkl'), 'rb'), encoding='utf-8')
    # data2 = pickle.load(open(os.path.join(data_path_0509, 'keepFeaturesFinal2.pkl'), 'rb'), encoding='utf-8')
    # data3 = pickle.load(open(os.path.join(data_path_0509, 'keepFeaturesFinal3.pkl'), 'rb'), encoding='utf-8')
    # data4 = pickle.load(open(os.path.join(data_path_0509, 'keepFeaturesFinal4.pkl'), 'rb'), encoding='utf-8')
    # data5 = pickle.load(open(os.path.join(data_path_0509, 'keepFeaturesFinal5.pkl'), 'rb'), encoding='utf-8')
    data1 = pickle.load(open(os.path.join(data_path_0515, 'borrbasic1\keepFeaturesFinal.pkl'), 'rb'), encoding='utf-8')
    data2 = pickle.load(open(os.path.join(data_path_0515, 'borrbasic2\keepFeaturesFinal.pkl'), 'rb'), encoding='utf-8')
    data3 = pickle.load(open(os.path.join(data_path_0515, 'borrex\keepFeaturesFinal.pkl'), 'rb'), encoding='utf-8')
    data4 = pickle.load(open(os.path.join(data_path_0515, 'paybasic\keepFeaturesFinal.pkl'), 'rb'), encoding='utf-8')
    data5 = pickle.load(open(os.path.join(data_path_0515, 'payex\keepFeaturesFinal.pkl'), 'rb'), encoding='utf-8')

    data_all = data1 + data2 + data3 + data4 + data5
    # print(data_all)
    data_list = [item.split('_')[-2] + '_' + item.split('_')[-1] for item in data_all]
    feature_date = pd.read_excel('借贷还贷标签名V2.0标签字典.xlsx', sheet_name='new_merge', index_col=0,
                                 usecols=[0, 1], header=None)
    feature_dict = feature_date.to_dict()
    results_df = pd.DataFrame()
    for i in data_list:
        results_df.loc[i, 'feature'] = feature_dict[1][i]
    results_df['feature'].to_csv(os.path.join(data_path_0515, 'result.csv'))
    print('处理完成，结果保存至{}'.format(os.path.abspath('result.csv')))

if __name__ == '__main__':
    main()

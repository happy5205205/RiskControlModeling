# -*- coding: utf-8 -*-

"""
Created on Mon May 13 20:56:34 2019

@author: zhangpeng
"""
import pickle
import pandas as pd
import os
import csv


def main():
    data_path_0509 = './data0509'
    data_path_0515 = './data0515'
    tailing_data = './sample_zaoyi_6w_0704'

    # data1 = pickle.load(open(os.path.join(data_path_0509, 'keepFeaturesFinal1.pkl'), 'rb'), encoding='utf-8')
    # data2 = pickle.load(open(os.path.join(data_path_0509, 'keepFeaturesFinal2.pkl'), 'rb'), encoding='utf-8')
    # data3 = pickle.load(open(os.path.join(data_path_0509, 'keepFeaturesFinal3.pkl'), 'rb'), encoding='utf-8')
    # data4 = pickle.load(open(os.path.join(data_path_0509, 'keepFeaturesFinal4.pkl'), 'rb'), encoding='utf-8')
    # data5 = pickle.load(open(os.path.join(data_path_0509, 'keepFeaturesFinal5.pkl'), 'rb'), encoding='utf-8')
    data1 = pickle.load(open(os.path.join(tailing_data, 'borrbasic1/featureWoeMap.pkl'), 'rb'),
                        encoding='utf-8')
    data2 = pickle.load(open(os.path.join(tailing_data, 'borrbasic2/featureWoeMap.pkl'), 'rb'),
                        encoding='utf-8')
    data3 = pickle.load(open(os.path.join(tailing_data, 'borrex/featureWoeMap.pkl'), 'rb'), encoding='utf-8')
    data4 = pickle.load(open(os.path.join(tailing_data, 'paybasic/featureWoeMap.pkl'), 'rb'), encoding='utf-8')
    data5 = pickle.load(open(os.path.join(tailing_data, 'payex/featureWoeMap.pkl'), 'rb'), encoding='utf-8')

    data = {}
    data.update(data1)
    data.update(data2)
    data.update(data3)
    data.update(data4)
    data.update(data5)

    # 将字典保存成csv
    with open('woe_values.xlsx', 'w') as f:
        [f.write('{0}\t{1}\n'.format('feature', 'woe'))]
        [f.write('{0}\t{1}\n'.format(key.split('_')[-2] + '_' + key.split('_')[-1], value)) for key, value in
         data.items()]
    f.close()

    new_data = pd.read_excel('woe_values.xlsx')
    feature_date = pd.read_excel('借贷还贷标签名V2.0标签字典.xlsx', sheet_name='new_merge', index_col=0, usecols=[0, 1],
                                 header=None)
    featureCutOffPointMap_data = pd.merge(new_data, feature_date, how='left', left_on='feature', right_on=0).loc[:, [1, 'woe', ]]

    featureCutOffPointMap_data.to_csv('featureWoeMap_values.csv', index=False)


if __name__ == '__main__':
    main()

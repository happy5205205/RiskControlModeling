# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:41:22 2019

@author: peng_zhang
"""

import os
import pandas as pd
from newWork import config
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm.sklearn import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from newWork import utils_2


def main():
    print('==================加载数据集==================')
    print('正在加载数据集', end=',')
    train_row_data = pd.read_csv(os.path.join(config.data_path, 'cyj_helloworld_result_pingtai_suning_800_0702.csv'))

    train_row_data = train_row_data.drop_duplicates(subset=['id_card_no', 'card_name', 'loan_date', 'target'],
                                                    keep='first')
    train_row_data = train_row_data[train_row_data['target'].isin([0, 1])]

    train_row_data = train_row_data.reset_index(drop=True)
    print('数据加载完成')

    # 最终的特征
    print('==================开始筛选特征==================')
    imp_feature = utils_2.clean_feature(row_data=train_row_data)
    train_data_df = train_row_data[imp_feature]

    # 数据清洗
    print('==================开始数据清洗和数据切割==================')
    train_data = utils_2.clean_data(train_data=train_data_df)
    print('数据清洗完成', end=',')

    # 划分数据集
    X_data, val_data = train_test_split(train_data, test_size=1 / 4, random_state=10)
    X_train, y_train = X_data[:, : -1], X_data[:, -1]
    X_val, y_val = val_data[:, : -1], val_data[:, -1]
    print('数据切割完成')
    # 训练模型
    # 单模型训练
    print('==================开始数据建模==================')
    if config.model_select_button == 'oneModel':
        model_name_param_dict = {
            # 'KNN': (KNeighborsClassifier(), {'n_neighbors': range(3, 10, 2)}),
            # 'LR': (LogisticRegression(), {'C': range(1, 10, 1), 'penalty': ['l1', 'l2'],
            #                               'solver': ['liblinear']}),
            # 'DT': (DecisionTreeClassifier(), {'max_leaf_nodes': range(2, 4, 1), 'max_depth': range(4, 10, 2)}),
            # 'RF': (RandomForestClassifier(), {'n_estimators': range(50, 150, 10),
            #                                   'criterion': ['entropy', 'gini'],
            #                                   'max_depth': range(50, 200, 10),
            #                                   'min_samples_split': range(2, 5, 1),
            #                                   'min_weight_fraction_leaf': list(np.arange(0, 0.5, 0.1))}),
            # 'Adboost': (AdaBoostClassifier(), {'n_estimators': range(50, 100, 10),
            #                                    'learning_rate': list(np.arange(0.01, 0.1, 0.01))}),
            # 'GBDT': (GradientBoostingClassifier(), {'learning_rate': list(np.arange(0.01, 0.1, 0.01)),
            #                                         'subsample': list(np.arange(0.5, 0.8, 0.1)),
            #                                         'loss': ['deviance', 'exponential'],
            #                                         'n_estimators': range(50, 100, 10),
            #                                         'max_leaf_nodes': range(2, 4, 1),
            #                                         'max_depth': range(4, 10, 2)}),
            'LGBM': (LGBMClassifier(), {'objective': ['binary'], 'boosting_type': ['gbdt', 'rf'],
                                        'num_leaves': range(11, 51, 2), 'max_depth': range(-1, 10, 1),
                                        'learning_rate': list(np.arange(0.01, 0.1, 0.01)),
                                        'n_estimators': range(50, 100, 10), 'silent': [False],
                                        'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6], 'lambda_l2': [0, 10, 15, 35, 40],
                                        'cat_smooth': [1, 10, 15, 20, 35], 'min_split_gain': range(-1, 10, 1)})
                                }

        result_df = pd.DataFrame(columns=['train_ks', 'val_ks'], index=model_name_param_dict.keys())
        # 单个模型训练
        print('单模型自动调参训练')
        for mode_name, (model, param_range) in model_name_param_dict.items():
            train_y_pred, val_y_pred, train_ks, val_ks = utils_2.train_one_model(
                                                                                    mode_name=mode_name,
                                                                                    model=model,
                                                                                    param_range=param_range,
                                                                                    X_train=X_train,
                                                                                    y_train=y_train,
                                                                                    X_val=X_val,
                                                                                    y_val=y_val
                                                                                )

            result_df.loc[mode_name, 'train_ks'] = ('%.2f' % train_ks)
            result_df.loc[mode_name, 'val_ks'] = ('%.2f' % val_ks)

            # KS图形绘制
            print('==================KS图形绘制==================')
            utils_2.ks_plot(train_y_pred=train_y_pred, y_train=y_train, val_y_pred=val_y_pred,
                            y_val=y_val, mode_name=mode_name)
            print('图像绘制完成保存至{}'.format(config.out_path))

            # 计算分数保存
            # val_y_pred_df = pd.DataFrame(utils_2.calcScore(val_y_pred), columns=['score'])
            # sns.distplot(val_y_pred_df['score'])
            # plt.savefig(os.path.join(config.out_path, '{}_score.png'.format(mode_name)))
            # plt.close()

        # 各个模型KS比较
        result_df.to_csv(os.path.join(config.out_path, 'train_one_model_ks_result.csv'))

        data = pd.read_csv(os.path.join(config.out_path, 'train_one_model_ks_result.csv'))
        # 比较模型的ks指标
        utils_2.campare_ks(data=data)

    elif config.model_select_button == 'stacking':
        train_y_pred, val_y_pred,  train_ks, val_ks = utils_2.trian_stacking_model(
                                                                                    X_train=X_train,
                                                                                    y_train=y_train,
                                                                                    X_val=X_val,
                                                                                    y_val=y_val)
        print('==================KS图形绘制==================')
        # 绘制ks曲线图
        utils_2.ks_plot(train_y_pred=train_y_pred, y_train=y_train, val_y_pred=val_y_pred, y_val=y_val)
        print('图像绘制完成保存至{}'.format(config.out_path))
    else:
        pass


if __name__ == '__main__':
    main()

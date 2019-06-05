# _*_ coding:utf-8 _*_

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import time


def clean_data(raw_data):
    """
        清洗数据

        参数：
            - raw_data: 原始数据

        返回：
            - cln_data: 清洗后的数据
    """
    # 删除重复的数据条数只保留第一条
    data_df = raw_data.drop_duplicates(subset=0, keep='first')

    data = data_df.iloc[:, 1:]
    row_num = data.columns[:-1].tolist()
    # print(row_num)
    for i in row_num:
        raw_data = data[~data[i].isin(['\\N'])]  # 遍历删除数据中所有'\\N'的行,例如：df1 = df1[~df1['A'].isin([1])]

    # 字符串转换成float
    cln_data = pd.DataFrame(raw_data, dtype=np.float)
    return cln_data

def transform_data(train_data, test_date):
    """
        特征工程

        参数：
            - train_data: 训练数据
            - test_data: 测试数据

        返回：
            - X_train_max_min_scaler: 归一化后的训练数据
            - X_test_max_min_scaler: 归一化后的测试数据
    """
    # 最大最小归一化
    max_min = MinMaxScaler()
    X_train = max_min.fit_transform(train_data)
    X_test = max_min.fit_transform(test_date)
    return X_train, X_test

"""
def train_test_model(X_train, y_train, X_test, y_test, model_name, model, param_range):
    
        # 训练并测试模型
        # model_name:
        #     kNN         kNN模型，对应参数为 n_neighbors
        #     LR          逻辑回归模型，对应参数为 C
        #     SVM         支持向量机，对应参数为 C
        #     DT          决策树，对应参数为 max_depth
        #     Stacking    将kNN, SVM, DT集成的Stacking模型， meta分类器为LR
        #     AdaBoost    AdaBoost模型，对应参数为 n_estimators
        #     GBDT        GBDT模型，对应参数为 learning_rate
        #     RF          随机森林模型，对应参数为 n_estimators
        # 
        # 根据给定的参数训练模型，并返回
        # 1. 最优模型
        # 2. 平均训练耗时
        # 3. 准确率
    
    print('训练{}...'.format(model_name))
    clf = GridSearchCV(estimator=model,
                       param_grid=param_range,
                       cv=3,
                       scoring='roc_auc',
                       refit=True)
    start = time.time()
    clf.fit(X_train, y_train)
    # 计时
    end = time.time()
    duration = end - start
    print('耗时{:.4f}s'.format(duration))

    # 验证模型
    train_score = clf.score(X_train, y_train)
    print('训练准确率：{:.3f}%'.format(train_score * 100))

    test_score = clf.score(X_test, y_test)
    print('测试准确率：{:.3f}%'.format(test_score * 100))
    print('训练模型耗时: {:.4f}s'.format(duration))
    print()

    return clf, test_score, duration
"""
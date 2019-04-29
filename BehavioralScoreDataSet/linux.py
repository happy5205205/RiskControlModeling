# _*_ coding=utf-8 _*_

import pandas as pd
import numpy as np
import seaborn as sns
import os
import confing
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import time
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


def inspect_data(train_data, test_data):
    # 可视化各类别的数量统计图
    num = train_data.columns[::].tolist()
    i = num[-1]
    plt.figure(figsize=(10, 5))

    # 训练集
    ax1 = plt.subplot(1, 2, 1)
    sns.countplot(x=i, data=train_data)

    plt.title('Training set')
    plt.xlabel('Class')
    plt.ylabel('Count')

    # 测试集
    plt.subplot(1, 2, 2, sharey=ax1)
    sns.countplot(x=i, data=test_data)

    plt.title('Testing set')
    plt.xlabel('Class')
    plt.ylabel('Count')

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(confing.output_path, 'class_count.jpg'))


def clean_data(raw_data):
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


def transform_data(train_data, test_data):
    # 最大最小归一化
    max_min = MinMaxScaler()
    X_train_max_min_scaler = max_min.fit_transform(train_data)
    X_test_max_min_scaler = max_min.transform(test_data)

    # pca降维
    pca = PCA(n_components=0.9)
    X_train = pca.fit_transform(X_train_max_min_scaler)
    X_test = pca.transform(X_test_max_min_scaler)
    return X_train, X_test


def train_test_model(X_train, y_train, X_test, y_test, model_name, model, param_range):
    print('-----------------------------------------------------------------------------------------------------------')
    print('训练{}模型...'.format(model_name), end=',')
    clf = GridSearchCV(estimator=model,
                       param_grid=param_range,
                       cv=3,
                       scoring='accuracy',
                       refit=True)
    start = time.time()
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    # 计时
    end = time.time()
    duration = end - start
    print('总共耗时{:.4f}s'.format(duration))
    print('-----------------------------------------------------------------------------------------------------------')
    test_score = clf.score(X_test, y_test)

    # # 准确率
    # print('准确率：{:.3f}'.format(accuracy_score(y_test, y_pre)))
    # print('-----------------------------------------------------------------------------------------------------------')
    # # 精确率
    # print('精确率：{:.3f}'.format(precision_score(y_test, y_pre)))
    # print('-----------------------------------------------------------------------------------------------------------')
    # # 召回率
    # print('召回率：{:.3f}'.format(recall_score(y_test, y_pre)))
    # print('-----------------------------------------------------------------------------------------------------------')
    # # f1指标
    # print('F1指标：{:.3f}'.format(f1_score(y_test, y_pre)))
    # print('-----------------------------------------------------------------------------------------------------------')
    # # PR曲线
    # print('AP值：{:.3f}'.format(average_precision_score(y_test, y_pre)))
    # print('-----------------------------------------------------------------------------------------------------------')
    # # ROC曲线
    # print('AUC值：{:.3f}'.format(roc_auc_score(y_test, y_pre)))
    print('-----------------------------------------------------------------------------------------------------------')
    print('准确率：{:.3f}，精确率：{:.3f},召回率：{:.3f},F1指标：{:.3f},AP值：{:.3f},AUC值：{:.3f}'.format(accuracy_score(y_test, y_pre),
                                                                                       precision_score(y_test, y_pre),recall_score(y_test, y_pre),
                                                                                       f1_score(y_test, y_pre),average_precision_score(y_test, y_pre),
                                                                                       roc_auc_score(y_test, y_pre)))
    print('-----------------------------------------------------------------------------------------------------------')
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pre)
    print('混淆矩阵')
    print(cm)

    return clf, test_score, duration

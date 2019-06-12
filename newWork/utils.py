# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:41:22 2019

@author: peng_zhang
"""


import warnings
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
from newWork import config

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 防止画图中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def clean_feature(row_data):
    """
        特征处理
        :param row_data:
        :return:
    """
    # 保留原数据
    row_data = row_data.copy()
    # 移除不参与训练的数据
    cln_data = row_data.drop(['id_card_no', 'card_name', 'loan_date', 'target'], axis=1)
    # for item in cln_data.columns:
    #    print('{}:{}'.format(item, (cln_data[item] < 0).sum(axis=0)/len(data)))

    # 缺失值-99998
    cln_data = cln_data.fillna(-99998)
    # 将特征中缺失值在百分之八十以上的特征所在列删除
    new_feature = [item for item in cln_data.columns if ((cln_data[item] < 0).sum(axis=0)/len(row_data)) < config.miss_feature]

    if config.start_rf_select_feature == 'yes':
        print('已启用随机森林和缺失值筛选特征', end=',')
        x = cln_data[new_feature]
        y = row_data['target']
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_leaf_nodes=2, random_state=10)
        rf.fit(x, y)
        importance = rf.feature_importances_
        indices = np.argsort(importance)[::-1]
        features = x.columns
        # for f in range(new_data.shape[1]):
        #    print(("%2d) %-*s %f" % (f + 1, 30, features[f], importance[indices[f]])))
        imp_feature = [features[f] for f in range(x.shape[1]) if importance[indices[f]] > config.rf_select_feature_max]
        imp_feature.append('target')
        print('剩余特征{}'.format(len(imp_feature)))
        return imp_feature
    else:
        print('没有启用随机森林筛选特征', end=',')
        new_feature.append('target')
        print('通过缺失值筛选,剩余特征{}'.format(len(new_feature)))
        return new_feature


def clean_data(train_data, test_data):
    """
        数据处理
        :param train_data:
        :param test_data:
        :return:
    """

    # 简单归一化处理
    min_max = MinMaxScaler()
    train_data_min_max = min_max.fit_transform(train_data)
    test_data_min_max = min_max.transform(test_data)
    return train_data_min_max, test_data_min_max
    # return train_data_min_max


def train_one_model(mode_name, model, param_range, X_train, y_train, X_val, y_val, X_test, y_test):
    print('--' * 30)
    print('开始训练{}模型'.format(mode_name))

    clf = GridSearchCV(estimator=model, param_grid=param_range, cv=3, scoring='accuracy', refit=True, n_jobs=-1)
    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    print('{}的训练集准确率：{:.3f}'.format(mode_name, train_score))
    train_y_pred = clf.predict_proba(X_train)[:, 1]
    FPR, TPR, threshold = roc_curve(y_train, train_y_pred)
    train_ROC_AUC = auc(FPR, TPR)
    print('{}的训练集AUC:{:.3f}'.format(mode_name, train_ROC_AUC))
    train_ks = max(abs(TPR - FPR))
    print('{}的训练集KS:{:.3f}'.format(mode_name, train_ks))

    val_score = clf.score(X_val, y_val)
    print('{}的验证集准确率：{:.3f}'.format(mode_name, val_score))
    val_y_pred = clf.predict_proba(X_val)[:, 1]
    FPR, TPR, threshold = roc_curve(y_val, val_y_pred)
    val_ROC_AUC = auc(FPR, TPR)
    print('{}的验证集AUC:{:.3f}'.format(mode_name, val_ROC_AUC))
    val_ks = max(abs(TPR - FPR))
    print('{}的验证集KS:{:.3f}'.format(mode_name, val_ks))

    test_score = clf.score(X_test, y_test)
    print('{}的测试集准确率：{:.3f}'.format(mode_name, test_score))
    test_y_pred = clf.predict_proba(X_test)[:, 1]
    FPR, TPR, threshold = roc_curve(y_test, test_y_pred)
    test_ROC_AUC = auc(FPR, TPR)
    print('{}的测试集AUC:{:.3f}'.format(mode_name, test_ROC_AUC))
    test_ks = max(abs(TPR - FPR))
    print('{}的测试集KS:{:.3f}'.format(mode_name, test_ks))

    best_params = clf.best_params_
    print('模型{}最好的参数是{}'.format(mode_name, best_params))
    print('--'*30)

    return train_y_pred, val_y_pred, test_y_pred, train_ks, val_ks, test_ks


def trian_stacking_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """
        stacking模型训练
        :param X_train:
        :param y_train:
        :param X_val:
        :param y_val:
        :param X_test:
        :param y_test:
        :return:
    """
    clf1 = RandomForestClassifier()
    lr = LogisticRegression()
    sclf = StackingClassifier(classifiers=[clf1], use_probas=True, average_probas=False, meta_classifier=lr)
    params = {'randomforestclassifier__n_estimators': [10, 100, 1000],
              'randomforestclassifier__criterion': ['entropy', 'gini'],
              'randomforestclassifier__max_depth': range(10, 200, 40),
              'randomforestclassifier__min_samples_split': [2, 5, 10],
              'randomforestclassifier__min_weight_fraction_leaf': list(np.arange(0, 0.5, 0.1)),
              'meta_classifier__C': [0.1, 10.0]}

    grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5, refit=True, n_jobs=-1)

    grid.fit(X_train, y_train)

    train_score = grid.score(X_train, y_train)
    print('Stacking模型的训练集准确率：{:.3f}'.format(train_score))
    train_y_pred = grid.predict_proba(X_train)[:, 1]
    FPR, TPR, threshold = roc_curve(y_train, train_y_pred)
    train_ROC_AUC = auc(FPR, TPR)
    print('Stacking模型的训练集AUC:{:.3f}'.format(train_ROC_AUC))
    train_ks = max(abs(TPR - FPR))
    print('Stacking模型的训练集KS:{:.3f}'.format(train_ks))

    val_score = grid.score(X_val, y_val)
    print('Stacking模型的验证集准确率：{:.3f}'.format(val_score))
    val_y_pred = grid.predict_proba(X_val)[:, 1]
    FPR, TPR, threshold = roc_curve(y_val, val_y_pred)
    val_ROC_AUC = auc(FPR, TPR)
    print('Stacking模型的验证集AUC:{:.3f}'.format(val_ROC_AUC))
    val_ks = max(abs(TPR - FPR))
    print('Stacking模型的验证集KS:{:.3f}'.format(val_ks))

    test_score = grid.score(X_test, y_test)
    print('Stacking模型的测试集准确率：{:.3f}'.format(test_score))
    test_y_pred = grid.predict_proba(X_test)[:, 1]
    FPR, TPR, threshold = roc_curve(y_test, test_y_pred)
    test_ROC_AUC = auc(FPR, TPR)
    print('Stacking模型的测试集AUC:{:.3f}'.format(test_ROC_AUC))
    test_ks = max(abs(TPR - FPR))
    print('Stacking模型的测试集KS:{:.3f}'.format(test_ks))

    best_params = grid.best_params_
    print('Stacking模型的最好的参数是{}'.format(best_params))

    return train_y_pred, val_y_pred, test_y_pred, train_ks, val_ks, test_ks


def ks_plot(train_y_pred, y_train, val_y_pred, y_val, test_y_pred, y_test, mode_name='stacking'):
    """
        计算KS图像绘制
        :return:
    """
    print('开始绘制{}模型的KS'.format(mode_name))

    plt.figure(figsize=(12,12))
    plt.subplot(2, 1, 1)
    FPR, TPR, threshold = roc_curve(y_train, train_y_pred)
    ks_value = max(abs(TPR - FPR))
    plt.title(mode_name+'_train_KS')
    plt.plot(FPR, label='bad')
    plt.plot(TPR, label='good')
    plt.plot(TPR - FPR, label='KS={:.3f}'.format(ks_value))
    plt.legend()

    plt.subplot(2, 1, 2)
    FPR, TPR, threshold = roc_curve(y_val, val_y_pred)
    ks_value = max(abs(TPR - FPR))
    plt.title(mode_name+'_val_KS')
    plt.plot(FPR, label='bad')
    plt.plot(TPR, label='good')
    plt.plot(TPR - FPR, label='KS={:.3f}'.format(ks_value))
    plt.legend()

    plt.subplot(3, 1, 3)
    FPR, TPR, threshold = roc_curve(y_test, test_y_pred)
    ks_value = max(abs(TPR - FPR))
    plt.title(mode_name+'_test_KS')
    plt.plot(FPR, label='bad')
    plt.plot(TPR, label='good')
    plt.plot(TPR - FPR, label='KS={:.3f}'.format(ks_value))
    plt.legend()

    plt.savefig(os.path.join(config.out_path, mode_name+'_KS.png'))
    plt.close()


def campare_ks(data):

    data.rename(columns={'Unnamed: 0': 'model_name'}, inplace=True)
    bar_width = 0.3
    x_label = list(data['model_name'])
    data1 = list(data['train_ks'])
    data2 = list(data['val_ks'])
    data3 = list(data['test_ks'])

    plt.bar(x=range(len(x_label)), height=data1, label='train_ks', alpha=0.8, width=bar_width)
    plt.bar(x=np.arange(len(x_label))+bar_width, height=data2, label='val_ks', alpha=0.8, width=bar_width)
    plt.bar(x=np.arange(len(x_label))+bar_width*2, height=data3, label='test_ks', alpha=0.8, width=bar_width)
    # 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
    for x, y in enumerate(data1):
        plt.text(x, y, '%s' % y, ha='center', va='bottom', fontsize=10)
    for x, y in enumerate(data2):
        plt.text(x+bar_width, y, '%s' % y, ha='center', va='top', fontsize=10)
    for x, y in enumerate(data3):
        plt.text(x+bar_width*2, y, '%s' % y, ha='center', va='top', fontsize=10)

    plt.xticks(np.arange(len(x_label))+bar_width/2, x_label)
    # 设置标题
    plt.title("train_ks VS val_ks VS test_ks")
    # 为两条坐标轴设置名称
    plt.xlabel("model_name")
    plt.ylabel("KS")
    # 显示图例
    plt.legend()
    plt.savefig(os.path.join(config.out_path, 'campare_ks.png'))
    plt.close()


def calcScore(bad_rate):
    # BasePoint = 600
    # PDO = 20
    # B = PDO / np.log(2)
    # A = BasePoint + B * np.log(1 / 2)
    A = 580.0
    B = 28.85390081777927

    # Odds = bad_rate / (1 - bad_rate)
    # score = A - B * np.log(Odds)
    # return score

    scores = []
    # score = [ score for i in bad_rate bad_rate / (1 - bad_rate)]
    for i in bad_rate:
        if i == 1:
            Odds = i / (1 - 0.999)
            score = A - B * np.log(Odds)
        else:
            Odds = i / (1 - i)
            score = A - B * np.log(Odds)
        scores.append(score)
    return scores
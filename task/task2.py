# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:58:04 2019

@author: peng_zhang
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn import metrics
warnings.filterwarnings('ignore')


def pi_Zhun_Jian(df, score, target, num):

    """
        功能：按照批准件中的好客数量，进行分数阈值判定
        :param df: 数据文件
        :param score: 分数
        :param target: 目标的特征
        :param num: 批准件中的好客数量
        :return: 阈值分数
    """

    total = df.groupby(score)[target].count()
    bad = df.groupby(score)[target].sum()
    df_all = pd.DataFrame({'total': total, 'bad': bad})
    df_all['good'] = df_all['total'] - df_all['bad']
    df_all[score] = df_all.index
    df_all = df_all.sort_values(by=score, ascending=False)
    df_all['sum_good'] = df_all['good'].cumsum()
    # 注意没有num，则取下一个累加数的分数作为阈值
    score_threshold = df_all[df_all['sum_good'] >= num].index[0]
    print('批准件中的好客数量为：{}时，切割分数判断阈值为：{}'.format(num, score_threshold))
    print('---'*30)


def he_Zhun_Jian(df, score, target, num):

    """
        功能：核准件的坏客数量进行分数阈值判定
        :param df: 数据文件
        :param score: 分数
        :param target: 目标的特征
        :param num: 核准件的坏客数量
        :return: 阈值分数
    """

    total = df.groupby(score)[target].count()
    bad = df.groupby(score)[target].sum()
    df_all = pd.DataFrame({'total': total, 'bad': bad})
    df_all['good'] = df_all['total'] - df_all['bad']
    df_all[score] = df_all.index
    df_all = df_all.sort_values(by=score, ascending=False)
    df_all['sum_bad'] = df_all['bad'].cumsum()
    # 注意没有num，则取下一个累加数的分数作为阈值
    score_threshold = df_all[df_all['sum_bad'] >= num].index[0]
    print('核准件的坏客数量为：{}时，切割分数判断阈值为：{}'.format(num, score_threshold))
    print('---' * 30)


def pass_rate(df, score, target, rate):
    """
        :param df: 数据文件
        :param score: 分数
        :param target: 目标
        :param rate: 通过率
        :return: 阈值分数
    """
    total = df.groupby(score)[target].count()
    bad = df.groupby(score)[target].sum()
    df_all = pd.DataFrame({'total': total, 'bad': bad})
    df_all['good'] = df_all['total'] - df_all['bad']
    df_all[score] = df_all.index
    df_all = df_all.sort_values(by=score, ascending=False)
    # 通过率累计占比
    df_all['goodCumRate'] = df_all['good'].cumsum() / df_all['good'].sum()
    score_threshold = df_all[df_all['goodCumRate'] >= rate].index[0]
    print('通过率为：{}时，切割分数判断阈值为：{}'.format(rate, score_threshold))
    print('---' * 30)


def good_bad_rate(df, score, target, goodPass, goodBadPass):
    """
        :param df: 数据文件
        :param score: 分数
        :param target: 目标
        :param goodPass: 通过率
        :param goodBadPass: 好坏占比
        :return: 阈值分数
    """
    total = df.groupby(score)[target].count()
    bad = df.groupby(score)[target].sum()
    df_all = pd.DataFrame({'total': total, 'bad': bad})
    df_all['good'] = df_all['total'] - df_all['bad']
    df_all[score] = df_all.index
    df_all = df_all.sort_values(by=score, ascending=False)
    # 通过率累计占比
    df_all['goodCumRate'] = df_all['good'].cumsum() / df_all['good'].sum()
    # 好坏比
    df_all['goodBadRate'] = df_all['good'].cumsum() / df_all['bad'].cumsum()
    score_threshold = df_all[(df_all['goodCumRate'] >= goodPass) & (df_all['goodBadRate'] >= goodBadPass)].index[0]
    print('好坏比为：{}时，切割分数判断阈值为：{}'.format(0.3, score_threshold))
    print('---' * 30)


def find_threshold(start, end, step, y_true, y_pred_prob):
    '''
        start, step, end: 阈值网格
        y_true: 二分类真实标签
        y_pred_prob: 二分类概率
    '''
    f1_max = 0
    threshold = start
    for x in np.arange(start, end, step):
        y_pred = [1 if i >= x else 0 for i in y_pred_prob]
        f1 = metrics.f1_score(y_true, y_pred)
        # print('threshold : %f, f1 : %f' %(x, f1))
        if f1 > f1_max:
            f1_max = f1
            threshold = x
    return threshold, f1_max


def train_model(data):
    """

        :param data: 数据文件
        :return:
    """
    data1_f2 = data.drop(['card_name', 'id_card_no', 'loan_date'], axis=1)

    train, test = train_test_split(data1_f2, test_size=1/4, random_state=0)
    train_x = train.drop('target', axis=1)
    train_y = train['target']

    test_x = test.drop('target', axis=1)
    test_y = test['target']

    # 训练模型 sklearn API
    clf = xgb.XGBClassifier(objective='binary:logistic', seed=123)
    param_dist = {'n_estimators': [30, 50],
                  'learning_rate': [0.02, 0.03],
                  'subsample': [0.5, 0.7],
                  'max_depth': [3, 4],
                  'colsample_bytree': [0.5, 0.7],
                  'min_child_weight': [1, 3],
                  'scale_pos_weight': [1, 3, 4]  # 正样本权重
                  }
    # grid search
    print('GridSearch Begin!')
    grid_search = GridSearchCV(clf, param_grid=param_dist, cv=3)
    grid_search.fit(train_x, train_y)  # 原始格式数据
    y = grid_search.predict(test_x)
    print('GridSearch Finished!')

    estimator1 = grid_search.best_estimator_
    estimator1.save_model('best_xgb.model')

    '''
    3.重载模型
    '''
    model = xgb.Booster(model_file='best_xgb.model').copy()

    dtrain = xgb.DMatrix(train_x, train_y)
    dtest = xgb.DMatrix(test_x, test_y)

    xgb_train_y_pred_prob = model.predict(dtrain)  # DMatrix格式数据
    xgb_test_y_pred_prob = model.predict(dtest)

    xgb_train_y_pred = xgb_train_y_pred_prob.round()
    xgb_test_y_pred = xgb_test_y_pred_prob.round()

    return train, test, train_y, test_y, xgb_train_y_pred, xgb_test_y_pred, xgb_train_y_pred_prob,xgb_test_y_pred_prob


def model_evaluation(train_y, xgb_train_y_pred_prob, xgb_test_y_pred_prob):
    """
        寻找最大的f1 score
        :param train_y: 真实的y值
        :param xgb_train_y_pred_prob: 预测的y值
        :param xgb_test_y_pred_prob: 预测的y值
        :return: 训练集真实的y值，测试集真实的y
    """

    xgb_threshold, xgb_f1 = find_threshold(0.1, 0.8, 0.02, train_y, xgb_train_y_pred_prob)

    xgb_train_y_pred1 = [1 if x >= xgb_f1 else 0 for x in xgb_train_y_pred_prob]
    xgb_test_y_pred1 = [1 if x >= xgb_f1 else 0 for x in xgb_test_y_pred_prob]

    return xgb_train_y_pred1, xgb_test_y_pred1


def main():
    """
        研究分数阈值的定义：
        （一）计算分数从大到小，按照不同通过率，按照不同好坏比，按照批准件中的好客数量，核准件的坏客数量进行分数阈值判定
        -- 入参：通过率，或好坏比，或批准件中的好客数量，或核准件的坏客数量,
        -- 出参：满足上述入参条件的分数阈值

        （二）注意按照分数从大到小移动阈值。
        -- 研究按照不同分数分箱F1 score max对应的分数作为阈值，进行通过率逾期率监测
    """
    # 显示所有列
    pd.set_option('display.max_columns', None)

    # 显示所有行
    # pd.set_options('display.max_rows', None)

    # 显示不换行
    pd.set_option('display.width', 100)

    data_path = './data'
    data = pd.read_csv(os.path.join(data_path, 'sample_data.csv'), encoding='utf-8')
    data_df = data.dropna()
    # task 1

    # 调用批准件函数
    pi_Zhun_Jian(df=data_df, score='行为雷达_贷款行为分', target='target', num=13)

    # 调用核准件数
    he_Zhun_Jian(df=data_df, score='行为雷达_贷款行为分', target='target', num=13)

    # 通过率累计占比
    pass_rate(df=data_df, score='行为雷达_贷款行为分', target='target', rate=0.3)

    # 好坏比
    good_bad_rate(df=data_df, score='行为雷达_贷款行为分', target='target', goodPass=0.00064, goodBadPass=0.14285714)


    # task 2

    train, test, train_y, test_y, xgb_train_y_pred, xgb_test_y_pred, xgb_train_y_pred_prob, xgb_test_y_pred_prob= train_model(data=data_df)
    xgb_train_y_pred1, xgb_test_y_pred1 = model_evaluation(train_y, xgb_train_y_pred_prob, xgb_test_y_pred_prob)
    # --------------------------------测试集----------------------------------------
    df_re = test[['行为雷达_贷款行为分', 'target']]
    df_re.columns = ['score', 'y']
    df_re['pred_y'] = xgb_test_y_pred1

    df_r1 = df_re[(df_re.iloc[:, 0] >= df_re['score'].min()) & (df_re.iloc[:, 0] < 500)]
    df_r2 = df_re[(df_re.iloc[:, 0] >= 500) & (df_re.iloc[:, 0] < 550)]
    df_r3 = df_re[(df_re.iloc[:, 0] >= 550) & (df_re.iloc[:, 0] < 600)]
    df_r4 = df_re[(df_re.iloc[:, 0] >= 600) & (df_re.iloc[:, 0] < 650)]
    df_r5 = df_re[(df_re.iloc[:, 0] >= 650) & (df_re.iloc[:, 0] < df_re['score'].max())]

    f1 = metrics.f1_score(df_r1.iloc[:, 1], df_r1.iloc[:, 2])
    pass_rate1 = (((df_r1['pred_y'] == 0) & (df_r1['y'] == 1)).sum(axis=0)) / ((df_r1['pred_y'] == 0).sum(axis=0))

    f2 = metrics.f1_score(df_r2.iloc[:, 1], df_r2.iloc[:, 2])
    pass_rate2 = (((df_r2['pred_y'] == 0) & (df_r2['y'] == 1)).sum(axis=0)) / ((df_r2['pred_y'] == 0).sum(axis=0))

    f3 = metrics.f1_score(df_r3.iloc[:, 1], df_r3.iloc[:, 2])
    pass_rate3 = (((df_r3['pred_y'] == 0) & (df_r3['y'] == 1)).sum(axis=0)) / ((df_r3['pred_y'] == 0).sum(axis=0))

    f4 = metrics.f1_score(df_r4.iloc[:, 1], df_r4.iloc[:, 2])
    pass_rate4 = (((df_r4['pred_y'] == 0) & (df_r4['y'] == 1)).sum(axis=0)) / ((df_r4['pred_y'] == 0).sum(axis=0))

    f5 = metrics.f1_score(df_r5.iloc[:, 1], df_r5.iloc[:, 2])
    pass_rate5 = (((df_r5['pred_y'] == 0) & (df_r5['y'] == 1)).sum(axis=0)) / ((df_r5['pred_y'] == 0).sum(axis=0))

    print('F1 score为{:.3f}是的分数为{}---500，逾期率为{:.3f}'.format(f1, df_re['score'].min(), pass_rate1))
    print('F1 score为{:.3f}是的分数为501---550，逾期率为{:.3f}'.format(f2, pass_rate2))
    print('F1 score为{:.3f}是的分数为551---600，逾期率为{:.3f}'.format(f3, pass_rate3))
    print('F1 score为{:.3f}是的分数为601---650，逾期率为{:.3f}'.format(f4, pass_rate4))
    print('F1 score为{:.3f}是的分数为651---{}，逾期率为{:.3f}'.format(f5, df_re['score'].max(), pass_rate5))

    f = []
    f.extend([f1, f2, f3, f4, f5])
    print('最大F1 score 为{:.3f}'.format(np.max(f)))

    """
    # --------------------------------训练集----------------------------------------
    df_tre = train[['行为雷达_贷款行为分', 'target']]
    df_tre.columns = ['score', 'y']
    df_tre['score'] = df_tre['score'].apply(lambda x: 450 if x == -1 else x)
    df_tre['fit_y_0.4'] = xgb_train_y_pred1

    df_tr1 = df_tre[(df_tre.iloc[:, 0] >= 450) & (df_tre.iloc[:, 0] < 500)]
    df_tr2 = df_tre[(df_tre.iloc[:, 0] >= 500) & (df_tre.iloc[:, 0] < 550)]
    df_tr3 = df_tre[(df_tre.iloc[:, 0] >= 550) & (df_tre.iloc[:, 0] < 600)]
    df_tr4 = df_tre[(df_tre.iloc[:, 0] >= 600) & (df_tre.iloc[:, 0] < 650)]
    df_tr5 = df_tre[(df_tre.iloc[:, 0] >= 650) & (df_tre.iloc[:, 0] < 700)]
    """


if __name__ == '__main__':
    main()
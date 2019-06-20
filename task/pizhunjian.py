# _*_ coding: utf-8 _*_

"""
    研究分数阈值的定义：
    （一）计算分数从大到小，按照不同通过率，按照不同好坏比，按照批准件中的好客数量，核准件的坏客数量进行分数阈值判定
    -- 入参：通过率，或好坏比，或批准件中的好客数量，或核准件的坏客数量,
    -- 出参：满足上述入参条件的分数阈值

    （二）注意按照分数从大到小移动阈值。
    -- 研究按照不同分数分箱F1 score max对应的分数作为阈值，进行通过率逾期率监测
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")


def main():
    # 显示所有列
    pd.set_option('display.max_columns', None)

    # 显示所有行
    # pd.set_options('display.max_rows', None)

    # 显示不换行
    pd.set_option('display.width', 100)

    data_path = './data'
    data = pd.read_csv(os.path.join(data_path, 'sample_data.csv'), encoding='utf-8')
    # 格式化日期
    # data['loan_date'] = pd.to_datetime(data['loan_date'], format='%Y-%m-%d')
    data_df = data.dropna()
    # data_df = data.sort_values(by='行为雷达_贷款行为分', ascending=False)
    # print(data_df['行为雷达_贷款行为分'])
    # pass_rate = (data_df['target'] == 0).sum(axis=0) / len(data_df)
    # print('原始数据的通过率为{:.3f}'.format(pass_rate))

    # 调用批准件函数
    # pi_Zhun_Jian(df=data_df, score='行为雷达_贷款行为分', target='target', num=13)

    # 调用核准件数
    # he_Zhun_Jian(df=data_df, score='行为雷达_贷款行为分', target='target', num=13)

    # 通过率累计占比
    # pass_rate(df=data_df, score='行为雷达_贷款行为分', target='target', rate=0.3)

    # 好坏比
    # good_bad_rate(df=data_df, score='行为雷达_贷款行为分', target='target', goodPass=0.00064, goodBadPass=0.14285714)


    '''
    # 数据处理
    train_data, test_data = train_test_split(data_df, test_size=1 / 4, random_state=10)

    X_train, y_train = train_data.iloc[:, 4:], train_data.iloc[:, 3]
    X_test, y_test = test_data.iloc[:, 4:], test_data.iloc[:, 3]

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict_proba(X_test)[:, 1]
    
    '''


if __name__ == '__main__':
    main()
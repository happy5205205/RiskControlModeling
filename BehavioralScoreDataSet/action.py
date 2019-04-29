# _*_ coding: utf-8 _*_

"""
    create by :zhang peng
    time :2019-04-15
    introduce: main file
    version: V1.0
"""

import pandas as pd
import os
import warnings
from BehavioralScoreDataSet import utils, confing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from mlxtend.classifier import StackingClassifier
warnings.filterwarnings("ignore")


def main():
    """
        主函数运行程序
    """

    # 读取数据
    raw_train_data = pd.read_csv(os.path.join(confing.data_path, 'zpeng_train_data_set.csv'), header=None)
    raw_test_data = pd.read_csv(os.path.join(confing.data_path, 'zpeng_test_data_set.csv'), header=None)
    print('-----------------------------------------------------------------------------------------------------------')
    print('原始训练数据有:{}条'.format(len(raw_train_data)))  # 训练数据有:358295条
    print('原始测试数据有:{}条'.format(len(raw_test_data)))  # 测试数据有:45371条
    print('-----------------------------------------------------------------------------------------------------------')

    # 数据清洗
    cln_train_data = utils.clean_data(raw_data=raw_train_data)
    cln_test_data = utils.clean_data(raw_data=raw_test_data)
    print('处理后训练数据有：{}条'.format(len(cln_train_data)))  # 训练数据去重后有：292052条
    print('处理后测试数据有：{}条'.format(len(cln_test_data)))  # 测试数据去重后有：25763条

    # 类别可视化
    utils.inspect_data(train_data=cln_train_data, test_data=cln_test_data)
    print('-----------------------------------------------------------------------------------------------------------')
    print('保存成功，保存路径:{}'.format(os.path.join(confing.output_path, 'class_count.jpg')))

    # 特征处理,数据选择、降维
    feature_name = cln_test_data.columns[0:-1].tolist()
    train_data = cln_train_data[feature_name].values
    test_data = cln_test_data[feature_name].values

    # 计算协方差协方差矩阵
    # cov_test_data = test_data.cov()
    # print(cov_test_data)

    # 归一化和降维，贡献率0.9
    X_train, X_test = utils.transform_data(train_data=train_data, test_data=test_data)
    print('降维前有{}，个维度/特征/标签，PCA特征降维后，特征维度为: {}'.format(len(feature_name), X_train.shape[1]))
    print('-----------------------------------------------------------------------------------------------------------')

    # 标签处理
    y_train = cln_train_data.iloc[:, -1]
    y_test = cln_test_data.iloc[:, -1]
    print('标签处理完成')

    # 模型训练
    sclf = StackingClassifier(classifiers=[KNeighborsClassifier(),
                                           SVC(),
                                           DecisionTreeClassifier()],
                              meta_classifier=LogisticRegression())

    model_name_param_dict = {'kNN': (KNeighborsClassifier(),
                                     {'n_neighbors': [5, 25, 55]}),
                             'LR': (LogisticRegression(),
                                    {'C': [0.01, 1, 100]}),
                             'SVM': (SVC(),
                                     {'C': [0.01, 1, 100]}),
                             'DT': (DecisionTreeClassifier(),
                                    {'max_depth': [50, 100, 150]}),
                             'Stacking': (sclf,
                                          {'kneighborsclassifier__n_neighbors': [5, 25, 55],
                                           'svc__C': [0.01, 1, 100],
                                           'decisiontreeclassifier__max_depth': [50, 100, 150],
                                           'meta-logisticregression__C': [0.01, 1, 100]}),
                             'AdaBoost': (AdaBoostClassifier(),
                                          {'n_estimators': [50, 100, 150, 200]}),
                             'GBDT': (GradientBoostingClassifier(),
                                      {'learning_rate': [0.01, 0.1, 1, 10, 100]}),
                             'RF': (RandomForestClassifier(),
                                    {'n_estimators': [100, 150, 200, 250]})}
    for model_name, (model, param_range) in model_name_param_dict.items():
        best_acc, mean_duration = utils.train_test_model(X_train, y_train, X_test, y_test,
                                                            model_name, model, param_range)
    print(best_acc, mean_duration)


if __name__ == '__main__':
    main()









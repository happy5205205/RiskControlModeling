# _*_ coding:utf-8 _*_

import pandas as pd
import os
from newjob import config
from newjob import utils
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import roc_curve, auc

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from mlxtend.classifier import StackingClassifier

raw_data = pd.read_csv(os.path.join(config.dataPath, 'zpeng_forbehaviorradar_20190507_pycharm54.csv'), usecols=config.all_cols)
# data = raw_data[config.feature]
print(raw_data.shape)
train_data, test_data = train_test_split(raw_data, test_size=1/4, random_state=10)
print(train_data.shape)
print(test_data.shape)

X_train, X_test = utils.transform_data(train_data=train_data, test_date=test_data)

# 标签
y_train = train_data[config.target].values
y_test = test_data[config.target].values
print(y_train.shape)
print(y_test.shape)

lr = LogisticRegression(penalty='l2')
lr.fit(X_train, y_train)
# 绘制ROC曲线

# # train_predprob = lr.predict_proba(train_X)[:,1]
# test_pred = lr.predict(X_test)
# FPR, TPR, threshold = roc_curve(y_test, test_pred)
# ROC_AUC = auc(FPR, TPR)
# print(ROC_AUC)
# plt.plot(FPR, TPR, 'b', label='AUC = {:.4f}'.format(ROC_AUC))
# plt.legend(loc='lower ,right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.show()

y_pred = lr.predict(X_test)
fpr, tpr, thresholds= roc_curve(y_pred, y_test)
ks_value = max(abs(tpr-fpr))
print(ks_value)

 # 画图，画出曲线
plt.plot(fpr, label='bad')
plt.plot(tpr, label='good')
plt.plot(abs(fpr-tpr), label='diff')

# 标记ks
x = np.argwhere(abs(fpr-tpr) == ks_value)[0, 0]
plt.plot((x, x), (0, ks_value), label='ks = {:.2f}'.format(ks_value), color='r', marker='o', markerfacecolor='r', markersize=5)
plt.scatter((x, x), (0, ks_value), color='r')
plt.legend()
plt.show()
# corr = train_data.corr()
# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(1, 1, 1)
# sns.heatmap(corr, annot=True,cmap='YlGnBu', ax=ax1, annot_kws={'size':12,'color':'m'})
# plt.show()

"""
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

# 比较结果的DataFrame
results_df = pd.DataFrame(columns=['Accuracy (%)', 'Time (s)'],
                          index=list(model_name_param_dict.keys()))
results_df.index.name = 'Model'
for model_name, (model, param_range) in model_name_param_dict.items():
    _, best_acc, mean_duration = utils.train_test_model(X_train, y_train, X_test, y_test,
                                                        model_name, model, param_range)
    results_df.loc[model_name, 'Accuracy (%)'] = best_acc * 100
    results_df.loc[model_name, 'Time (s)'] = mean_duration

results_df.to_csv(os.path.join(config.output_path, 'model_comparison.csv'))
"""

# print(data.describe())
# print(data.info())
# print(data.head(3))
# print(data.shape)

# # for i in config.feature_name:
# dd = (data['per_work_allpro_succlenddivlend_cnt_m3'] == -99998).sum(axis=0)
# print(dd)
# # print(data.columns)
# # print(data.iloc[:,-1].head())
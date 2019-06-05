# -*- coding: utf-8 -*-
"""
Created on Fri May 31 08:46:33 2019

"""

import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from task import data_processing as datap
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn import metrics

warnings.filterwarnings('ignore')


'''
1. 数据准备
'''
# 显示所有列
pd.set_option('display.max_columns', None)

# 显示所有行
# pd.set_options('display.max_rows', None)

# 显示不换行
pd.set_option('display.width', 100)

data_path = './data'
data1 = pd.read_csv(os.path.join(data_path, 'sample_data.csv'), encoding='utf-8')

data1_f2 = data1.drop(['card_name', 'id_card_no', 'loan_date'], axis=1)

train, test = train_test_split(data1_f2, test_size=0.2, random_state=1)
train_x = train.drop('target', axis=1)
train_y = train['target']

test_x = test.drop('target', axis=1)
test_y = test['target']


'''
2.建模--xgboost-gridsearch
'''

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
print('GridSearch Finished!')

estimator1 = grid_search.best_estimator_
estimator1.save_model('best_xgb.model')

train_y_pred = estimator1.predict(train_x)
test_y_pred = estimator1.predict(test_x)

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


'''
4. 模型指标
'''
# import sys
# sys.path.append('D:/work/code/data_prepare')

xgb_train_idx = datap.classification_metrics(train_y, xgb_train_y_pred)  # tn, fp, fn, tp
xgb_test_idx = datap.classification_metrics(test_y, xgb_test_y_pred)

xgb_threshold, xgb_f1 = datap.find_threshold(0.4, 0.8, 0.02, train_y, xgb_train_y_pred_prob)

xgb_train_y_pred1 = [1 if x >= 0.5 else 0 for x in xgb_train_y_pred_prob]
xgb_test_y_pred1 = [1 if x >= 0.5 else 0 for x in xgb_test_y_pred_prob]

xgb_train_idx1 = datap.classification_metrics(train_y, xgb_train_y_pred1)  # tn, fp, fn, tp
xgb_test_idx1 = datap.classification_metrics(test_y, xgb_test_y_pred1)


# --------------------------------测试集----------------------------------------
df_re = test[['行为雷达_贷款行为分', 'target']]
df_re.columns = ['score', 'y']
df_re['pred_y_0.5'] = xgb_test_y_pred1

df_r1 = df_re[(df_re.iloc[:, 0] >= df_re['score'].min()) & (df_re.iloc[:, 0] < 500)]
df_r2 = df_re[(df_re.iloc[:, 0] >= 500) & (df_re.iloc[:, 0] < 550)]
df_r3 = df_re[(df_re.iloc[:, 0] >= 550) & (df_re.iloc[:, 0] < 600)]
df_r4 = df_re[(df_re.iloc[:, 0] >= 600) & (df_re.iloc[:, 0] < 650)]
df_r5 = df_re[(df_re.iloc[:, 0] >= 650) & (df_re.iloc[:, 0] < df_re['score'].max())]

f1 = metrics.f1_score(df_r5.iloc[:, 1], df_r5.iloc[:, 2])
pass_rate = (((df_r4['pred_y_0.5'] == 0) & (df_r4['y'] == 1)).sum(axis=0)) / ((df_r4['pred_y_0.5'] == 0).sum(axis=0))
xgb_df_r5 = datap.classification_metrics(df_r5.iloc[:, 1], df_r5.iloc[:, 2])


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

xgb_df_tr1 = datap.classification_metrics(df_tr1.iloc[:, 1], df_tr1.iloc[:, 2])

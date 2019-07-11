# _*_ coding: utf-8 _*_
import pandas as pd
import os
import lightgbm as lgbm
from newWork import config
from newWork import utils_2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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

lgbm_train = lgbm.Dataset(data=X_train, label=y_train)

lgbm_val = lgbm.Dataset(data=X_val, label=y_val)

# 将参数写成字典下形式
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression',  # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数
    'num_leaves': 31,  # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}
print('Start training...')
# 训练 cv and train
gbm = lgbm.train(params, lgbm_train, num_boost_round=20, valid_sets=lgbm_val, early_stopping_rounds=5)  # 训练数据需要参数列表和数据集
print('Save model...')
gbm.save_model('model.txt')  # 训练后保存模型到文件
print('Start predicting...')
# 预测数据集
y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)  # 如果在训练期间启用了早期停止，可以通过best_iteration方式从最佳迭代中获得预测
# 评估模型
print('The rmse of prediction is:', mean_squared_error(y_val, y_pred) ** 0.5)  # 计算真实值和预测值之间的均方根误差
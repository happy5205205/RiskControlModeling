# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:41:22 2019

@author: peng_zhang

"""
import os
# 数据集存放
data_path = './data'

# 模型选择控制开关oneModel, stacking, 默认是oneModel
model_select_button = 'stacking'

# 是否启用随机森林筛选标签 yes表示启用，no表是不启用, 默认是启用
start_rf_select_feature = 'yes'

# 随机森林筛选阈值
rf_select_feature_max = 0.0005

# 确实值筛选阈值
miss_feature = 0.9

# 结果保存路径
out_path = './result'

if not os.path.exists(out_path):
    os.mkdir(out_path)
# -*- coding: utf-8 -*
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
path = './result'

data = pd.read_csv(os.path.join(path, 'train_one_model_ks_result.csv'))

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
    plt.text(x, y, '%s' % y, ha='center', va='bottom', fontsize=8)
for x, y in enumerate(data2):
    plt.text(x+bar_width, y, '%s' % y, ha='center', va='top', fontsize=8)
for x, y in enumerate(data3):
    plt.text(x+bar_width*2, y, '%s' % y, ha='center', va='top', fontsize=8)

plt.xticks(np.arange(len(x_label))+bar_width/2, x_label)
# 设置标题
plt.title("train_ks VS val_ks VS test_ks")
# 为两条坐标轴设置名称
plt.xlabel("model_name")
plt.ylabel("KS")
# 显示图例
plt.legend()
plt.show()



# plt.figure(figsize=(10, 8))
# plt.barh(y=np.arange(len(x_label)), width=data1, label='train_ks',  alpha=0.8, height=bar_width)
# plt.barh(y=np.arange(len(x_label))+bar_width, width=data2, label='val_ks', alpha=0.8, height=bar_width)
# plt.barh(y=np.arange(len(x_label))+bar_width*2, width=data3, label='test_ks',  alpha=0.8, height=bar_width)
#
# # 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
# for y, x in enumerate(data1):
#     plt.text(x, y-bar_width/2+0.06, '%s' % x, ha='center', va='bottom')
# for y, x in enumerate(data2):
#     plt.text(x, y+bar_width/2+0.06, '%s' % x, ha='center', va='bottom')
# for y, x in enumerate(data3):
#     plt.text(x, y+bar_width*2/2+0.25, '%s' % x, ha='center', va='bottom')
# plt.yticks(np.arange(len(x_label))+bar_width/2, x_label)
# # 设置标题
# plt.title("train_ks VS val_ks VS test_ks")
# # 为两条坐标轴设置名称
# plt.xlabel("KS")
# plt.ylabel("model_name")
# plt.legend()
# plt.show()
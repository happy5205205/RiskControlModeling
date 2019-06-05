import pandas as pd

# import os
# print('保存路径{}'.format(os.path.join(confing.output_path, 'class_count.jpg')))
#
# df = pd.DataFrame({'col1': [1, 2], 'col2': [3, '\\N'],'col3': [5, 6]})
# data = pd.DataFrame(np.arange(15).reshape(3,5),index=['one','two','three'],columns=['a','b','c','d','e'])
# print(data)
# print('-----------------------------------------------------------------------------------------------------------')
# print(data.ix[1:3,[0,2]])  #选择第2-4行第1、3列的值)
# print('-----------------------------------------------------------------------------------------------------------')
# # print(data.ix[:,[14]])
# print(data.ix[1:3])  #选择第2到4行，不包括第4行，即前闭后开区间。
# print('-----------------------------------------------------------------------------------------------------------')
# print(data.iloc[:,1:-1])
# print('-----------------------------------------------------------------------------------------------------------')
# print(data.iloc[:,-1])
# print('-----------------------------------------------------------------------------------------------------------')
# data_df = data.iloc[:,1:-1]
# print(data_df)
# row_num = data.columns[:-1].tolist()
# print(row_num)
# print('-----------------------------------------------------------------------------------------------------------')
# data_df1 = data.iloc[:,1:]
# print(data_df1)
# row_num1 = data.columns[:-1].tolist()
# print(row_num1)
# print('-----------------------------------------------------------------------------------------------------------')
# row = data.values.tolist()
# row_num2 = data.columns[::].tolist()
# print(row)
# print(row_num2)
# print(row[-1])
# print(row_num2[-1])
# print(len(row_num2))
# print(data[row_num2])
# print(data.iloc[len(row_num2)-1])
# row_num = df.columns[:-1].tolist()
# for i in row_num:
#     tt = df[~df[i].isin(['\\N'])]
# print(df)
# print(df.iloc[:][1:])
# print(df.shape)
# print(tt)
# print(tt.shape)

# l = [1,2,5,6,8]
# print(len(l))
# for i in l:
#     print(i)

# print(df.iloc[1][2])

import numpy as np
data=pd.DataFrame([[-2.1,-1,4.3],[3,1.1,0.12],[3,1.1,0.12]],index=['one','two','three'],columns=list('abc'))
print(data)
print('相关系数')
#one two three的相关系数
# 相关系数
print(data.corr())
#      a    b    c
# a  1.0  1.0 -1.0
# b  1.0  1.0 -1.0
# c -1.0 -1.0  1.0
print('协方差')
print(np.cov(data))
# from sklearn import datasets
#
# iris = datasets.load_iris()
# X, y = iris.data[:, 1:3], iris.task
# print(X)
# print('-----------------------------------------------')
# print(y)

# a = 1
# b = 2
# c = 3
# d = 4
# e = zip([a, b, c, d])
# print(e)



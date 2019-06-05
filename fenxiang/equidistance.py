# _*_ coding: utf-8 _*_

import pandas as pd
import numpy as np
from scipy import stats


def monoto_bin(Y, X, n=20):

    '''
        #特征选择特征分箱
        :param Y:
        :param X:
        :param n:
        :return:
    '''

    r = 0
    total_bad = Y.sum()
    total_good = Y.count()-total_bad
    while np.abs(r) < 1:
        d1 = pd.DataFrame({'X':X, 'Y':Y, 'Bucket':pd.qcut(X, n, duplicates='raise')})
        d2 = d1.groupby('Bucket', as_index = True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.min().X, columns=['min_' + X.name])
    d3['min_' + X.name] = d2.min().X
    d3['max_' + X.name] = d2.max().X
    d3[Y.name] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['badattr'] = d3[Y.name]/total_good
    d3['goodattr'] = (d3['total'] - d3[Y.name])/total_good
    d3['woe'] = np.log(d3['goodattr']/d3['badattr'])
    iv = ((d3['goodattr']-d3['badattr'])*d3['woe']).sum()
    d4 = (d3.sort_values(by = 'min_' + X.name)).reset_index(drop=True)
    print('==='*30)
    cut = []
    cut.append(float('-inf'))
    for i in range(1, n+1):
        qua = X.quantile(i/(n+1))
        cut.append(round(qua, 4))
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4,iv,cut,woe
    # return dfx1, cut

df = pd.read_csv('data_ceshi_copy.csv')

df = df.drop(['card_name', 'id_card_no', 'loan_date'], axis=1)

dfx1,ivx1,cutx1,woex1 = monoto_bin(df['task'], df['查询机构数'], n=10)
# cutx1 = monoto_bin(df['task'], df['查询机构数'], n=10)


print(cutx1)
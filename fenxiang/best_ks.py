# _*_ coding:utf-8 _*_

import pandas as pd

def best_ks_box(data,label,var_name,box_num):
    data = data[[var_name, label]]

    def ks_bin(data_, limit):
        g = data_.ix[:, 1].value_counts()[0]
        b = data_.ix[:, 1].value_counts()[1]
        data_cro = pd.crosstab(data_.ix[:, 0], data_.ix[:, 1])
        data_cro[0] = data_cro[0]/g
        data_cro[1] = data_cro[1]/b
        data_cro_cum = data_cro.cumsum()
        ks_list = abs(data_cro_cum[1] - data_cro_cum[0])
        ks_list_index = ks_list.nlargest(len(ks_list)).index.tolist()
        for i in ks_list_index:
            data_1 = data_[data_.ix[:, 0] <= i]
            data_2 = data_[data_.ix[:, 0] > i]
            if len(data_1) >= limit and len(data_2) >= limit:
                break
        if i == ks_list_index[-1]:
            bre_tag = False
        else:
            bre_tag = True
        return i,bre_tag
    #测试： ks_bin(data,data.shape[0]/7)


    def ks_zone(data_,list_):
        list_zone = list()
        list_.sort()
        n = 0
        for i in list_:
            m = sum(data_.ix[:,0]<=i) - n
            n = sum(data_.ix[:,0]<=i)
            list_zone.append(m)
        list_zone.append(50000-sum(list_zone))
        max_index = list_zone.index(max(list_zone))
        if max_index == 0:
            rst = [data_.ix[:,0].unique().min(),list_[0]]
        elif max_index == len(list_):
            rst = [list_[-1],data_.ix[:,0].unique().max()]
        else:
            rst = [list_[max_index-1],list_[max_index]]
        return rst
    #    测试： ks_zone(data_,[23])    #左开右闭

    data_ = data.copy()
    # 总体的5%
    limit_ = data.shape[0]/100   

    zone = list()
    for i in range(box_num-1):
        ks_,bre = ks_bin(data_,limit_)
        if bre:
            zone.append(ks_)
            new_zone = ks_zone(data, zone)
            data_ = data[(data.ix[:, 0]>new_zone[0])&(data.ix[:, 0]<=new_zone[1])]
        else:
            break


    zone.append(data.ix[:,0].unique().max())
    zone.append(data.ix[:,0].unique().min())
    zone.sort()
    df_ = pd.DataFrame(columns=[0,1])
    for i in range(len(zone)-1):
        if i == 0:
            data_ = data[data.ix[:,0]<=zone[i+1]]
        elif i == len(zone)-2:
            data_ = data[data.ix[:, 0] > zone[i]]
        else:
            data_ = data[(data.ix[:,0]>zone[i])&(data.ix[:,0]<=zone[i+1])]
        data_cro = pd.crosstab(data_.ix[:, 0], data_.ix[:,1])
        if i == 0:
            df_.loc['{0}-{1}'.format('-inf', data_cro.index.max())] = data_cro.apply(sum)
        elif i == len(zone)-2:
            df_.loc['{0}-{1}'.format(data_cro.index.min(), 'inf')] = data_cro.apply(sum)
        else:
            df_.loc['{0}-{1}'.format(data_cro.index.min(), data_cro.index.max())] = data_cro.apply(sum)
    return df_

data = pd.read_csv('data_ceshi_fx.csv')
var_name = '申请准入分'
print(best_ks_box(data, 'task', var_name, 15))
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:30:34 2019

@author: yunying_wu
"""

import pandas as pd
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,precision_score,recall_score,f1_score
from ScoreCardModel.weight_of_evidence import WeightOfEvidence
import matplotlib.pyplot as plt
def KS_AR(df, score, target,asc=False):
    '''
    :param df: the dataset containing probability and bad indicator
    :param score:
    :param target:
    :return:
    '''
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total': total, 'bad': bad})
    all['good'] = all['total'] - all['bad']
    all[score] = all.index
    all = all.sort_values(by=score, ascending=asc)
    all.index = range(len(all))
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    all['totalPcnt'] = all['total'] / all['total'].sum()
    arList = [0.5 * all.loc[0, 'badCumRate'] * all.loc[0, 'totalPcnt']]
    for j in range(1, len(all)):
        ar0 = 0.5 * sum(all.loc[j - 1:j, 'badCumRate']) * all.loc[j, 'totalPcnt']
        arList.append(ar0)
    arIndex = (2 * sum(arList) - 1) / (all['good'].sum() * 1.0 / all['total'].sum())
    KS = abs(all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1))
    KS_value =  max(KS)
    if asc:
        cut_value = all.loc[KS.isin([max(KS)])][score].min()
    else:
        cut_value = all.loc[KS.isin([max(KS)])][score].min()        
    
    plt.plot(all[score], all['goodCumRate'], color="darkblue", label="good rate")
    plt.plot(all[score], all['badCumRate'], color="indianred", label="bad rate")
    plt.plot(all[score], KS, color="darkgreen", label="KS")
#    plt.axvline(x=cut_value, ymin=0, ymax=KS_value, color="darkgreen")
    plt.text(cut_value,0.9, 'KS:{0:5.3f}\nAR:{1:5.3f}'.format(KS_value,arIndex))
    plt.legend(loc='upper left')
    plt.show()


#Workflow标签文件,不含target
df_feature = pd.read_csv('./zpeng_shrh_201904251351_jm_hanger_1.csv',header=None,sep=',')

woe_score = pd.read_csv('./final_score_filter42.csv')
woe_score['PROB_VALUE'] = woe_score['WOE']*woe_score['FEATURE_WEIGHT']
woe_score['BIN_MIN'] = woe_score.loc[:,'BIN_RANGE'].apply(lambda x:x.split(',')[0][1:]).astype('float')
woe_score['BIN_MAX'] = woe_score.loc[:,'BIN_RANGE'].apply(lambda x:x.split(',')[1][:-1]).astype('float')

features = woe_score['VALID_FEATURE_NAME'].unique()
df_feature.columns=['id_card_no','card_name','lend_day']+list(features)

# 匹配WOE
def AssignGroupWOE(x,cutOffBin):
    N = len(cutOffBin)
    for i in range(N):
        if cutOffBin.loc[i,'BIN_MIN'] < x <= cutOffBin.loc[i,'BIN_MAX']:
            return cutOffBin.loc[i,'SCORE_VALUE'],cutOffBin.loc[i,'PROB_VALUE']
        

for col in features:
    cutOffBin = woe_score[woe_score.VALID_FEATURE_NAME==col].iloc[:,-4:]
    cutOffBin = cutOffBin.reset_index(drop=True)
    tmp = df_feature.loc[:,col].apply(lambda x :AssignGroupWOE(x,cutOffBin))
    df_feature[col+'_woe_score'] = tmp.apply(lambda x: x[0])
    df_feature[col+'_prob_value'] = tmp.apply(lambda x: x[1])
    
features_score = [col+'_woe_score' for col in features]
features_prob = [col+'_prob_value' for col in features]

df_feature['final_score'] = df_feature.loc[:,features_score].sum(axis=1)
df_feature['final_prob'] = df_feature.loc[:,features_prob].sum(axis=1)


#test_label文件,带target,和标签匹配
df_target = pd.read_csv(open('scenario_a_score_step3_test2',encoding='utf-8'),header=None,sep='|')
df_target = df_target.iloc[:,:7]
df_target.columns = ['id_card_no','card_name','member_id','lend_day','rn','raw_y','target']
df_target = df_target.loc[:,['id_card_no','card_name','lend_day','target']]
df_target = df_target.drop_duplicates()
df_target = df_target.merge(df_feature.loc[:,['id_card_no','card_name','lend_day','final_score','final_prob']]\
                            ,left_on=['id_card_no','card_name','lend_day']\
                            ,right_on=['id_card_no','card_name','lend_day'],how='inner')
df_target['target'] = df_target['target'].apply(lambda x : 0 if x==0 else 1)
df_target['pred'] = df_target['final_prob'].apply(lambda x : 1 if x>=0.5 else 0)

print('roc:',roc_auc_score(df_target['target'],df_target['final_prob']))
print('accuracy:',accuracy_score(df_target['target'],df_target['pred']))
print('precision:',precision_score(df_target['target'],df_target['pred']))
print('recall:',recall_score(df_target['target'],df_target['pred']))
print('f1',f1_score(df_target['target'],df_target['pred']))
KS_AR(df_target, 'final_score', 'target', asc=True)





import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn import  metrics
from sklearn.metrics import roc_curve
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,precision_score,recall_score,f1_score
#import hashlib
#import xpinyin
import math
import time
import pdb
# import argparse
import time
start_time = time.time()
plt.switch_backend('agg')

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
    KS = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    KS_value =  max(KS)  
    return KS_value,arIndex


def KS_AR_plot(data_name,result_path,df, score, target,asc=False):
    '''
    :param data_name: the name of the dataset, which will displayed in the final plot
    :param result_path: the path of outputting plot
    :param df: the dataset containing probability and bad indicator
    :param score:
    :param target:
    :param asc: True for the credit score, False for the prob
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
    plt.title(data_name+'_ks')
    plt.savefig(result_path+'/'+data_name+'_xgb_ks.png')
    plt.close()
    return KS_value

def gridsearch(xgb_train,xgb_val,y_val,init_params, grid_param = ['max_depth','min_child_weight'], 
               grid_param_list = [range(3,10,2),range(1,6,2)],num_rounds=5000,early_stopping_rounds=100,specific_sort='specific_score'):
    
    '''
    :param xgb_train: Dmatrix of train data
    :param xgb_val: Dmatrix of val data
    :param y_val: the label of val data
    :param init_params: the initial parameters of xgboost
    :param grid_param: the name of parameters to be grid searched
    :param grid_param_list: the values of parameters to be grid searched
    :param num_rounds:
    :param early_stopping_rounds:
    :param specific_score: val dataset sorted by the value in descending (specific_score/ks/ar)
    :return:
    '''
    grid_param0_index =[]
    grid_param1_index = []
    score_list = []
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    ks_list = []
    ar_list = []
    
    params = init_params.copy()
    if len(grid_param) == 1:
        for i in grid_param_list[0]:
            params[grid_param[0]] = i
            model = xgb.train(params, xgb_train, num_rounds, watchlist, early_stopping_rounds=early_stopping_rounds)
            grid_param0_index.append(i)
            score_list.append(model.best_score)
            evalDf =pd.concat([pd.DataFrame(y_val,columns=['task']).reset_index(),pd.DataFrame(model.predict(xgb_val,ntree_limit=model.best_ntree_limit),columns=['score'])], axis=1)
            ks_value, ar_value = KS_AR(evalDf, 'score', 'task')
            ks_list.append(ks_value)
            ar_list.append(ar_value)
        
        
    else:
    
        for i in grid_param_list[0]:
            params[grid_param[0]] = i
            
            for j in grid_param_list[1]:
                params[grid_param[1]] =j
                
                model = xgb.train(params, xgb_train, num_rounds, watchlist, early_stopping_rounds=early_stopping_rounds)
    
                grid_param0_index.append(i)
                grid_param1_index.append(j)
                score_list.append(model.best_score)
                
                evalDf =pd.concat([pd.DataFrame(y_val,columns=['task']).reset_index(),pd.DataFrame(model.predict(xgb_val,ntree_limit=model.best_ntree_limit),columns=['score'])], axis=1)
                ks_value, ar_value = KS_AR(evalDf, 'score', 'task')
                ks_list.append(ks_value)
                ar_list.append(ar_value)
                
        t2 = pd.DataFrame(grid_param1_index,columns = [grid_param[1]])
        

    t1 = pd.DataFrame(grid_param0_index,columns = [grid_param[0]])
    t3 = pd.DataFrame(score_list,columns = ['specific_score'])
    t4 = pd.DataFrame(ks_list,columns = ['ks'])
    t5 = pd.DataFrame(ar_list,columns = ['ar'])
    
    if len(grid_param) == 1:
         return pd.concat([t1, t3, t4, t5], axis=1).sort_values(by = [specific_sort],ascending=False)
    
    return pd.concat([t1, t2, t3, t4, t5], axis=1).sort_values(by = [specific_sort],ascending=False)



def main():
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train_file', default='train_file.csv')
    # parser.add_argument('--test_file', default='test_file.csv')
    # parser.add_argument('--path', default= os.getcwd() )
    # parser.add_argument('--val_size',type=float, default= 0.2, help='the size of validation dataset')
    # args = parser.parse_args()
    #
    # data_path = args.path+'/data'
    # result_path = args.path+'/result/model'+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    data_path =  './data'
    
    #读入数据
    print('------data loading start-------')
    train_datafile = os.path.join(data_path, 'sample_forbehaviorradar_0515_6w_train.csv')
    test_datafile = os.path.join(data_path, 'sample_forbehaviorradar_0515_6w_test_samoye.csv')
    
    train_data = pd.read_csv(train_datafile)
    test_data = pd.read_csv(test_datafile)

    X = train_data.drop(['id_card_no','card_name','loan_date','task'],axis=1)
    y = train_data.loc[:,'task']
    X_test = test_data.drop(['id_card_no','card_name','loan_date','task'],axis=1)
    y_test = test_data.loc[:,'task']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
    xgb_train = xgb.DMatrix(X_train, label = y_train)
    xgb_val = xgb.DMatrix(X_val, label = y_val)
    xgb_test = xgb.DMatrix(X_test, label = y_test)

    #基础参数
    params={
    'eta': 0.05,
    'booster':'gbtree',
    'objective': 'binary:logistic', #二／多分类的问题   'binary:logistic'，'multi:softmax'
    #'num_class':2, # 类别数，与 multisoftmax 并用
    'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth':5, # 构建树的深度，越大越容易过拟合
    #'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample':1, # 随机采样训练样本
    'colsample_bytree':1, # 生成树时进行的列采样
    'min_child_weight':20, 
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 
    'verbosity':1 ,# Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug)
#    'scale_pos_weight':3, # 如sum(negative cases) / sum(positive cases) 
    'seed':1000,
#    'nthread':24,# cpu 线程数
    'eval_metric': 'auc'
    }
    
    print('------parameters tuning start-------')

    grid_param = ['max_depth','min_child_weight']
    grid_param_list = [range(3,8,2),[1,5,10,20,30]]  
    q =  gridsearch(xgb_train,xgb_val,y_val,params,grid_param,grid_param_list)
    params[q.columns[0]] = q.iloc[0,0]
    params[q.columns[1]] = q.iloc[0,1]           
    print('max_depth最佳为{}，min_child_weight最佳为{}'.format(q.iloc[0,0],q.iloc[0,1]))

    gamma = [[0.1,0.2]]
    q =  gridsearch(xgb_train,xgb_val,y_test,params,['gamma'],gamma)
    params['gamma'] = q.iloc[0,0]
    print('gamma最佳为{}'.format(q.iloc[0,0]))    
    
    subsample = [[0.8,0.9]]
    q =  gridsearch(xgb_train,xgb_val,y_test,params,['subsample'],subsample)
    params['subsample'] = q.iloc[0,0]
    print('subsample最佳为{}'.format(q.iloc[0,0]))  
    
    colsample_bytree = [[0.8,0.9]]
    q =  gridsearch(xgb_train,xgb_val,y_test,params,['colsample_bytree'],colsample_bytree)
    params['colsample_bytree'] = q.iloc[0,0]    
    print('colsample_bytree最佳为{}'.format(q.iloc[0,0])) 
    
    lambda_ = [[1,2]]
    q =  gridsearch(xgb_train,xgb_val,y_test,params,['lambda'],lambda_)
    params['lambda'] = q.iloc[0,0]       
    print('lambda最佳为{}'.format(q.iloc[0,0])) 
    
    print('------test dataset predicting -------')    
    num_rounds = 5000 # 迭代次数
    watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]
    bst = xgb.train(params,xgb_train, num_boost_round = num_rounds, evals = watchlist, early_stopping_rounds = 100)
        
    # os.makedirs(result_path)
    bst.save_model('/xgb.model')
    

    
    for i,k,name in [(xgb_train, y_train,'train'),(xgb_val, y_val,'val'),(xgb_test,y_test,'test')]:
        evalDf =pd.concat([pd.DataFrame(k).reset_index(),pd.DataFrame(bst.predict(i,ntree_limit=bst.best_ntree_limit),columns=['score'])], axis=1)
        KS_AR_plot(name,result_path,evalDf, 'score', 'task')
    fig, ax = plt.subplots(figsize=(30,18))
    xgb.plot_importance(bst, max_num_features=50, height=0.8, ax=ax)
    plt.savefig('/featureimportance.png')
    feature_impt = pd.Series(bst.get_score())
    feature_impt.to_csv('feature_importance.csv')
    pd.Series(params).to_csv('parameters.csv')
    bst.dump_model('model.txt')
    
    
if __name__ == '__main__':
    main()

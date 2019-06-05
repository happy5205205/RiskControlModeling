# _*_coding:utf-8 _*_
import  os
import pandas as pd


"""
    'id_card_no','card_name','loan_day','std_alltime_allpro_likeduepay_days_cnt3','per_work_allpro_succlenddivlend_cnt_m3','per_work_allpro_succlenddivlend_amt_cnt3','min_alltime_allpro_likeduelend_days_m1','avg_work_allpro_likeduepay_days_cnt5','per_work_allpro_succpaydivpay_cnt_m6','avg_work_allpro_likeduepay_days_m3','min_alltime_allpro_lend_feerate_m3','per_work_allpro_succpaydivpay_cnt_m3','per_alltime_allpro_succpaydivpay_cnt_m12','per_alltime_cdqprodivallpro_pay_amt_d15','min_alltime_allpro_succlend_feerate_cnt3','per_work_allpro_succlenddivlend_cnt_m12','new_std_alltime_allpro_likeduepay_days_m12','sum_work_allpro_likeduelend_days_cnt3','per_week_allpro_succlenddivlend_cnt_cnt3','per_work_cdqprodivallpro_pay_amt_m1','new_sum_alltime_allpro_likeduepay_days_m12','min_alltime_allpro_succlend_feerate_cnt20','per_alltime_allpro_succpaydivpay_cnt_m1','min_alltime_allpro_succlend_feerate_m6','min_alltime_allpro_pay_amt_m3','min_work_allpro_pay_amt_cnt3','new_sum_alltime_allpro_likeduepay_days_m6','sum_alltime_allpro_likeduelend_days_m3','sum_work_allpro_likeduepay_days_m6','sum_work_allpro_likeduepay_days_cnt20','avg_alltime_allpro_lend_feerate_m1','avg_work_noncdq_likeduepay_days_m3','avg_work_allpro_succlend_feerate_m3','min_alltime_allpro_likeduepay_days_d15','new_avg_work_allpro_likeduelend_days_m6','min_week_allpro_succlend_feerate_cnt5','min_week_allpro_succlend_feerate_m3','new_per_sum_alltime_noncdq_likeduepay_days_m6','avg_alltime_allpro_succlend_feerate_d7','std_work_allpro_likeduepay_days_cnt20','sum_alltime_cdqpro_lend_feerate_cnt10','max_work_cdqpro_succlend_feerate_m3','sum_alltime_allpro_likeduelend_days_cnt20','new_std_alltime_allpro_likeduepay_days_m6','per_workdivalltime_allpro_likeduepay_days_cnt3','max_work_cdqpro_succlend_fee_d15','avg_alltime_cdqpro_lend_feerate_cnt5','per_workdivalltime_allpro_likeduelend_days_cnt3','per_week_allpro_succpaydivpay_cnt_cnt10','max_alltime_noncdq_likeduepay_days_m12','new_min_work_allpro_pay_amt_m12','per_week_allpro_succlenddivlend_amt_m6','max_week_cdqpro_succlend_feerate_m6','avg_alltime_allpro_lend_fee_cnt5','max_alltime_allpro_succlend_feerate_cnt10','per_week_allpro_succlenddivlend_cnt_m3','min_work_allpro_succlend_feerate_d7','task
"""

dataPath = './data'

feature_name = ['id_card_no','card_name','loan_day','std_alltime_allpro_likeduepay_days_cnt3','per_work_allpro_succlenddivlend_cnt_m3','per_work_allpro_succlenddivlend_amt_cnt3','min_alltime_allpro_likeduelend_days_m1','avg_work_allpro_likeduepay_days_cnt5','per_work_allpro_succpaydivpay_cnt_m6','avg_work_allpro_likeduepay_days_m3','min_alltime_allpro_lend_feerate_m3','per_work_allpro_succpaydivpay_cnt_m3','per_alltime_allpro_succpaydivpay_cnt_m12','per_alltime_cdqprodivallpro_pay_amt_d15','min_alltime_allpro_succlend_feerate_cnt3','per_work_allpro_succlenddivlend_cnt_m12','new_std_alltime_allpro_likeduepay_days_m12','sum_work_allpro_likeduelend_days_cnt3','per_week_allpro_succlenddivlend_cnt_cnt3','per_work_cdqprodivallpro_pay_amt_m1','new_sum_alltime_allpro_likeduepay_days_m12','min_alltime_allpro_succlend_feerate_cnt20','per_alltime_allpro_succpaydivpay_cnt_m1','min_alltime_allpro_succlend_feerate_m6','min_alltime_allpro_pay_amt_m3','min_work_allpro_pay_amt_cnt3','new_sum_alltime_allpro_likeduepay_days_m6','sum_alltime_allpro_likeduelend_days_m3','sum_work_allpro_likeduepay_days_m6','sum_work_allpro_likeduepay_days_cnt20','avg_alltime_allpro_lend_feerate_m1','avg_work_noncdq_likeduepay_days_m3','avg_work_allpro_succlend_feerate_m3','min_alltime_allpro_likeduepay_days_d15','new_avg_work_allpro_likeduelend_days_m6','min_week_allpro_succlend_feerate_cnt5','min_week_allpro_succlend_feerate_m3','new_per_sum_alltime_noncdq_likeduepay_days_m6','avg_alltime_allpro_succlend_feerate_d7','std_work_allpro_likeduepay_days_cnt20','sum_alltime_cdqpro_lend_feerate_cnt10','max_work_cdqpro_succlend_feerate_m3','sum_alltime_allpro_likeduelend_days_cnt20','new_std_alltime_allpro_likeduepay_days_m6','per_workdivalltime_allpro_likeduepay_days_cnt3','max_work_cdqpro_succlend_fee_d15','avg_alltime_cdqpro_lend_feerate_cnt5','per_workdivalltime_allpro_likeduelend_days_cnt3','per_week_allpro_succpaydivpay_cnt_cnt10','max_alltime_noncdq_likeduepay_days_m12','new_min_work_allpro_pay_amt_m12','per_week_allpro_succlenddivlend_amt_m6','max_week_cdqpro_succlend_feerate_m6','avg_alltime_allpro_lend_fee_cnt5','max_alltime_allpro_succlend_feerate_cnt10','per_week_allpro_succlenddivlend_cnt_m3','min_work_allpro_succlend_feerate_d7','task']

feature = ['std_alltime_allpro_likeduepay_days_cnt3','per_work_allpro_succlenddivlend_cnt_m3','per_work_allpro_succlenddivlend_amt_cnt3','min_alltime_allpro_likeduelend_days_m1','avg_work_allpro_likeduepay_days_cnt5','per_work_allpro_succpaydivpay_cnt_m6','avg_work_allpro_likeduepay_days_m3','min_alltime_allpro_lend_feerate_m3','per_work_allpro_succpaydivpay_cnt_m3','per_alltime_allpro_succpaydivpay_cnt_m12','per_alltime_cdqprodivallpro_pay_amt_d15','min_alltime_allpro_succlend_feerate_cnt3','per_work_allpro_succlenddivlend_cnt_m12','new_std_alltime_allpro_likeduepay_days_m12','sum_work_allpro_likeduelend_days_cnt3','per_week_allpro_succlenddivlend_cnt_cnt3','per_work_cdqprodivallpro_pay_amt_m1','new_sum_alltime_allpro_likeduepay_days_m12','min_alltime_allpro_succlend_feerate_cnt20','per_alltime_allpro_succpaydivpay_cnt_m1','min_alltime_allpro_succlend_feerate_m6','min_alltime_allpro_pay_amt_m3','min_work_allpro_pay_amt_cnt3','new_sum_alltime_allpro_likeduepay_days_m6','sum_alltime_allpro_likeduelend_days_m3','sum_work_allpro_likeduepay_days_m6','sum_work_allpro_likeduepay_days_cnt20','avg_alltime_allpro_lend_feerate_m1','avg_work_noncdq_likeduepay_days_m3','avg_work_allpro_succlend_feerate_m3','min_alltime_allpro_likeduepay_days_d15','new_avg_work_allpro_likeduelend_days_m6','min_week_allpro_succlend_feerate_cnt5','min_week_allpro_succlend_feerate_m3','new_per_sum_alltime_noncdq_likeduepay_days_m6','avg_alltime_allpro_succlend_feerate_d7','std_work_allpro_likeduepay_days_cnt20','sum_alltime_cdqpro_lend_feerate_cnt10','max_work_cdqpro_succlend_feerate_m3','sum_alltime_allpro_likeduelend_days_cnt20','new_std_alltime_allpro_likeduepay_days_m6','per_workdivalltime_allpro_likeduepay_days_cnt3','max_work_cdqpro_succlend_fee_d15','avg_alltime_cdqpro_lend_feerate_cnt5','per_workdivalltime_allpro_likeduelend_days_cnt3','per_week_allpro_succpaydivpay_cnt_cnt10','max_alltime_noncdq_likeduepay_days_m12','new_min_work_allpro_pay_amt_m12','per_week_allpro_succlenddivlend_amt_m6','max_week_cdqpro_succlend_feerate_m6','avg_alltime_allpro_lend_fee_cnt5','max_alltime_allpro_succlend_feerate_cnt10','per_week_allpro_succlenddivlend_cnt_m3','min_work_allpro_succlend_feerate_d7']

target = 'task'

all_cols = feature + [target]

# 结果保存路径
output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)





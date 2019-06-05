原始数据文件：test_sample.csv
df.columns = ['card_name', 'id_card_no', 'loan_date', 'target']

一.preprocess_template.py
数据预处理，将所有测试报告的特征关联在一起(将所有报告放入测试报告文件夹中)
输出默认为：test_sample_processed_test.csv
ps：仅对一些时间变量做了处理，未分箱，保留缺失值


二.rule_preporcess_template.py
输入: test_sample_processed_test.csv

主要参数说明：
test_size = 0.9 #测试集样本大小
oot_flag = False #是否oot来划分训练集，测试集

1.单调性分析：
qcut_flag=True 是否等频率分箱，false为等距分箱
bin_count=10  分箱个数
miss_flag=False  是否删除缺失值过多变量
missing_thres=0.4   缺失值比例最高不超过missing_thres
输出：
test_qcut10.csv  #单调性分析，包含各个变量的分箱个数，分箱逾期率，分箱逾期率是否单调
Result_iv_ks_monotone_qcut10.csv #各个变量的iv,ks,分箱单调性
ps：其中qcut10表明选用等频分箱10组，cut5即表明等距分箱5组


2.决策树分析：
selected_col为入模变量
输出：tree.pdf

3.矩阵分析:
col1,col2 #选用的两个变量
cut_num1, cut_num2 #选用等频，分箱个数设定
cut_value1,cut_value2 #自己设定阈值分箱cut_value，例如[float('-inf'),1,2,5,10,15,20,float('inf')]
可以一个等频率，一个自己设阈值
输出：矩阵分析逾期率（以热力图形式）

4.单变量分析
5.单规则拒绝率，逾期率评估
6.总体规则拒绝率，逾期率评估




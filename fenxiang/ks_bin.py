import pandas as pd
import numpy as np
from scipy.stats import chi2


def Chi2(df, total_col, bad_col, overallRate):
    '''
     #�˺������㿨��ֵ
     :df dataFrame
     :total_col ÿ��ֵ��������
     :bad_col ÿ��ֵ�Ļ���������
     :overallRate �����ݵ�ռ��
     : return ����ֵ
    '''
    df2 = df.copy()
    df2['expected'] = df[total_col].apply(lambda x: x * overallRate)
    combined = zip(df2['expected'], df2[bad_col])
    chi = [(i[0] - i[1]) ** 2 / i[0] for i in combined]
    chi2 = sum(chi)
    return chi2


# ���ڿ�����ֵ�������䣬�и�ȱ�㣬���ÿ��Ʒ��������
def ChiMerge_MinChisq(df, col, target, confidenceVal=3.841):
    '''
    #�˺������Կ�����ֵ��Ϊ��ֹ�������з���
    : df dataFrame
    : col �����������
    : task Ŀ��ֵ,��0,1��ʽ
    : confidenceVal  ��ֵ�����ɶ�Ϊ1�� ���Ŷ�Ϊ0.95ʱ��������ֵΪ3.841
    : return ���䡣
    �����и����⣬��������Է��������û�����ƣ������ӻᵼ��������Ľ���Ƿ���̫ϸ��
    '''
    # �Դ���������ֵ����ȥ��
    colLevels = set(df[col])

    # count�������������
    total = df.groupby([col])[target].count()

    total = pd.DataFrame({'total': total})

    # sum���������ֵ�ĺ�
    # ע�������target������0,1��Ҫ��Ȼ������bad��������������û�����壬����bad��1��good��0��
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    # �����ݽ��кϲ������col��ÿ��ֵ�ĳ��ִ�����total��bad��
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)

    # ���������������
    N = sum(regroup['total'])
    # �������������������
    B = sum(regroup['bad'])
    overallRate = B * 1.0 / N

    # �Դ����������ֵ��������
    colLevels = sorted(list(colLevels))
    groupIntervals = [[i] for i in colLevels]

    groupNum = len(groupIntervals)
    while (1):
        if len(groupIntervals) == 1:
            break
        chisqList = []
        for interval in groupIntervals:
            df2 = regroup.loc[regroup[col].isin(interval)]
            chisq = Chi2(df2, 'total', 'bad', overallRate)
            chisqList.append(chisq)

        min_position = chisqList.index(min(chisqList))

        if min(chisqList) >= confidenceVal:
            break

        if min_position == 0:
            combinedPosition = 1
        elif min_position == groupNum - 1:
            combinedPosition = min_position - 1
        else:
            if chisqList[min_position - 1] <= chisqList[min_position + 1]:
                combinedPosition = min_position - 1
            else:
                combinedPosition = min_position + 1
        groupIntervals[min_position] = groupIntervals[min_position] + groupIntervals[combinedPosition]
        groupIntervals.remove(groupIntervals[combinedPosition])
        groupNum = len(groupIntervals)
    return groupIntervals


# ������������
def ChiMerge_MaxInterval_Original(df, col, target, max_interval=5):
    '''
    : df dataframe
    : col Ҫ�����������
    �� task Ŀ��ֵ 0,1 ֵ
    : max_interval �������
    ��return ����
    '''
    colLevels = set(df[col])
    colLevels = sorted(list(colLevels))
    N_distinct = len(colLevels)
    if N_distinct <= max_interval:
        print
        "the row is cann't be less than interval numbers"
        return colLevels[:-1]
    else:
        total = df.groupby([col])[target].count()
        total = pd.DataFrame({'total': total})
        bad = df.groupby([col])[target].sum()
        bad = pd.DataFrame({'bad': bad})
        regroup = total.merge(bad, left_index=True, right_index=True, how='left')
        regroup.reset_index(level=0, inplace=True)
        N = sum(regroup['total'])
        B = sum(regroup['bad'])
        overallRate = B * 1.0 / N
        groupIntervals = [[i] for i in colLevels]
        groupNum = len(groupIntervals)
        while (len(groupIntervals) > max_interval):
            chisqList = []
            for interval in groupIntervals:
                df2 = regroup.loc[regroup[col].isin(interval)]
                chisq = Chi2(df2, 'total', 'bad', overallRate)
                chisqList.append(chisq)
            min_position = chisqList.index(min(chisqList))
            if min_position == 0:
                combinedPosition = 1
            elif min_position == groupNum - 1:
                combinedPosition = min_position - 1
            else:
                if chisqList[min_position - 1] <= chisqList[min_position + 1]:
                    combinedPosition = min_position - 1
                else:
                    combinedPosition = min_position + 1
            # �ϲ�����
            groupIntervals[min_position] = groupIntervals[min_position] + groupIntervals[combinedPosition]
            groupIntervals.remove(groupIntervals[combinedPosition])
            groupNum = len(groupIntervals)
        groupIntervals = [sorted(i) for i in groupIntervals]
        print
        groupIntervals
        cutOffPoints = [i[-1] for i in groupIntervals[:-1]]
        return cutOffPoints


# ����WOE��IVֵ
def CalcWOE(df, col, target):
    '''
    : df dataframe
    : col ע�������Ѿ��ֹ����ˣ����ڼ���ÿ���WOE���ܵ�IV
    ��task Ŀ���� 0-1ֵ
    ��return ����ÿ���WOE���ܵ�IV
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x * 1.0 / B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis=1)
    WOE_dict = regroup[[col, 'WOE']].set_index(col).to_dict(orient='index')
    IV = regroup.apply(lambda x: (x.good_pcnt - x.bad_pcnt) * np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis=1)
    IV_SUM = sum(IV)
    return {'WOE': WOE_dict, 'IV_sum': IV_SUM, 'IV': IV}


# �����Ժ���ÿ���bad_rate�ĵ����ԣ���������㣬��ô�����������ڵ�����ϲ���ֱ��bad_rate����Ϊֹ
def BadRateMonotone(df, sortByVar, target):
    # df[sortByVar]�����Ѿ���������
    df2 = df.sort_values(by=[sortByVar])
    total = df2.groupby([sortByVar])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df2.groupby([sortByVar])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    combined = zip(regroup['total'], regroup['bad'])
    badRate = [x[1] * 1.0 / x[0] for x in combined]
    badRateMonotone = [badRate[i] < badRate[i + 1] for i in range(len(badRate) - 1)]
    Monotone = len(set(badRateMonotone))
    if Monotone == 1:
        return True
    else:
        return False


# �������䣬��������������������ռ�����ݵ�90%���ϣ���ô�����������
def MaximumBinPcnt(df, col):
    N = df.shape[0]
    total = df.groupby([col])[col].count()
    pcnt = total * 1.0 / N
    return max(pcnt)


# ������������ݣ���bad_rate����ԭ��ֵ��ת�������������ٽ��з�����㡣������������Ļ����ش��룬�����������ݸ�ʽ
# ��Ȼ���������ʱ��ԭ���ϲ���Ҫ����
def BadRateEncoding(df, col, target):
    '''
    : df DataFrame
    : col ��Ҫ�����bad rate��������
    ��targetֵ��0-1ֵ
    �� return: the assigned bad rate
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1)
    br_dict = regroup[[col, 'bad_rate']].set_index([col]).to_dict(orient='index')
    badRateEnconding = df[col].map(lambda x: br_dict[x]['bad_rate'])
    return {'encoding': badRateEnconding, 'br_rate': br_dict}

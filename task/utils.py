import pandas as pd


def pi_Zhun_Jian(df, score, target, num):

    """
        ���ܣ�������׼���еĺÿ����������з�����ֵ�ж�
        :param df: �����ļ�
        :param score: ����
        :param target: Ŀ�������
        :param num: ��׼���еĺÿ�����
        :return: ��ֵ����
    """

    total = df.groupby(score)[target].count()
    bad = df.groupby(score)[target].sum()
    df_all = pd.DataFrame({'total': total, 'bad': bad})
    df_all['good'] = df_all['total'] - df_all['bad']
    df_all[score] = df_all.index
    df_all = df_all.sort_values(by=score, ascending=False)
    df_all['sum_good'] = df_all['good'].cumsum()
    # ע��û��num����ȡ��һ���ۼ����ķ�����Ϊ��ֵ
    score_threshold = df_all[df_all['sum_good'] >= num].index[0]
    print('��׼���еĺÿ�����Ϊ��{}ʱ���и�����ж���ֵΪ��{}'.format(num, score_threshold))


def he_Zhun_Jian(df, score, target, num):

    """
        ���ܣ���׼���Ļ����������з�����ֵ�ж�
        :param df: �����ļ�
        :param score: ����
        :param target: Ŀ�������
        :param num: ��׼���Ļ�������
        :return: ��ֵ����
    """

    total = df.groupby(score)[target].count()
    bad = df.groupby(score)[target].sum()
    df_all = pd.DataFrame({'total': total, 'bad': bad})
    df_all['good'] = df_all['total'] - df_all['bad']
    df_all[score] = df_all.index
    df_all = df_all.sort_values(by=score, ascending=False)
    df_all['sum_bad'] = df_all['bad'].cumsum()
    # ע��û��num����ȡ��һ���ۼ����ķ�����Ϊ��ֵ
    score_threshold = df_all[df_all['sum_bad'] >= num].index[0]
    print('��׼���Ļ�������Ϊ��{}ʱ���и�����ж���ֵΪ��{}'.format(num, score_threshold))


def pass_rate(df, score, target, rate):
    """
        :param df: �����ļ�
        :param score: ����
        :param target: Ŀ��
        :param rate: ͨ����
        :return: ��ֵ����
    """
    total = df.groupby(score)[target].count()
    bad = df.groupby(score)[target].sum()
    df_all = pd.DataFrame({'total': total, 'bad': bad})
    df_all['good'] = df_all['total'] - df_all['bad']
    df_all[score] = df_all.index
    df_all = df_all.sort_values(by=score, ascending=False)
    # ͨ�����ۼ�ռ��
    df_all['goodCumRate'] = df_all['good'].cumsum() / df_all['good'].sum()
    score_threshold = df_all[df_all['goodCumRate'] >= rate].index[0]
    print('ͨ����Ϊ��{}ʱ���и�����ж���ֵΪ��{}'.format(rate, score_threshold))


def good_bad_rate(df, score, target, goodPass, goodBadPass):
    """
        :param df: �����ļ�
        :param score: ����
        :param target: Ŀ��
        :param goodPass: ͨ����
        :param goodBadPass: �û�ռ��
        :return: ��ֵ����
    """
    total = df.groupby(score)[target].count()
    bad = df.groupby(score)[target].sum()
    df_all = pd.DataFrame({'total': total, 'bad': bad})
    df_all['good'] = df_all['total'] - df_all['bad']
    df_all[score] = df_all.index
    df_all = df_all.sort_values(by=score, ascending=False)
    # ͨ�����ۼ�ռ��
    df_all['goodCumRate'] = df_all['good'].cumsum() / df_all['good'].sum()
    # �û���
    df_all['goodBadRate'] = df_all['good'].cumsum() / df_all['bad'].cumsum()
    score_threshold = df_all[(df_all['goodCumRate'] >= goodPass) & (df_all['goodBadRate'] >= goodBadPass)].index[0]
    print('�û���Ϊ��{}ʱ���и�����ж���ֵΪ��{}'.format(0.3, score_threshold))

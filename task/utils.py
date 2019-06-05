import pandas as pd


def pi_Zhun_Jian(df, score, target, num):

    """
        功能：按照批准件中的好客数量，进行分数阈值判定
        :param df: 数据文件
        :param score: 分数
        :param target: 目标的特征
        :param num: 批准件中的好客数量
        :return: 阈值分数
    """

    total = df.groupby(score)[target].count()
    bad = df.groupby(score)[target].sum()
    df_all = pd.DataFrame({'total': total, 'bad': bad})
    df_all['good'] = df_all['total'] - df_all['bad']
    df_all[score] = df_all.index
    df_all = df_all.sort_values(by=score, ascending=False)
    df_all['sum_good'] = df_all['good'].cumsum()
    # 注意没有num，则取下一个累加数的分数作为阈值
    score_threshold = df_all[df_all['sum_good'] >= num].index[0]
    print('批准件中的好客数量为：{}时，切割分数判断阈值为：{}'.format(num, score_threshold))


def he_Zhun_Jian(df, score, target, num):

    """
        功能：核准件的坏客数量进行分数阈值判定
        :param df: 数据文件
        :param score: 分数
        :param target: 目标的特征
        :param num: 核准件的坏客数量
        :return: 阈值分数
    """

    total = df.groupby(score)[target].count()
    bad = df.groupby(score)[target].sum()
    df_all = pd.DataFrame({'total': total, 'bad': bad})
    df_all['good'] = df_all['total'] - df_all['bad']
    df_all[score] = df_all.index
    df_all = df_all.sort_values(by=score, ascending=False)
    df_all['sum_bad'] = df_all['bad'].cumsum()
    # 注意没有num，则取下一个累加数的分数作为阈值
    score_threshold = df_all[df_all['sum_bad'] >= num].index[0]
    print('核准件的坏客数量为：{}时，切割分数判断阈值为：{}'.format(num, score_threshold))


def pass_rate(df, score, target, rate):
    """
        :param df: 数据文件
        :param score: 分数
        :param target: 目标
        :param rate: 通过率
        :return: 阈值分数
    """
    total = df.groupby(score)[target].count()
    bad = df.groupby(score)[target].sum()
    df_all = pd.DataFrame({'total': total, 'bad': bad})
    df_all['good'] = df_all['total'] - df_all['bad']
    df_all[score] = df_all.index
    df_all = df_all.sort_values(by=score, ascending=False)
    # 通过率累计占比
    df_all['goodCumRate'] = df_all['good'].cumsum() / df_all['good'].sum()
    score_threshold = df_all[df_all['goodCumRate'] >= rate].index[0]
    print('通过率为：{}时，切割分数判断阈值为：{}'.format(rate, score_threshold))


def good_bad_rate(df, score, target, goodPass, goodBadPass):
    """
        :param df: 数据文件
        :param score: 分数
        :param target: 目标
        :param goodPass: 通过率
        :param goodBadPass: 好坏占比
        :return: 阈值分数
    """
    total = df.groupby(score)[target].count()
    bad = df.groupby(score)[target].sum()
    df_all = pd.DataFrame({'total': total, 'bad': bad})
    df_all['good'] = df_all['total'] - df_all['bad']
    df_all[score] = df_all.index
    df_all = df_all.sort_values(by=score, ascending=False)
    # 通过率累计占比
    df_all['goodCumRate'] = df_all['good'].cumsum() / df_all['good'].sum()
    # 好坏比
    df_all['goodBadRate'] = df_all['good'].cumsum() / df_all['bad'].cumsum()
    score_threshold = df_all[(df_all['goodCumRate'] >= goodPass) & (df_all['goodBadRate'] >= goodBadPass)].index[0]
    print('好坏比为：{}时，切割分数判断阈值为：{}'.format(0.3, score_threshold))

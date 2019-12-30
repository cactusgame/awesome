import talib as ta
import pandas as pd
import numpy as np
from src.context import context
from src.utils.share_selector import ShareSelector


# Target: after 1d, 3d, 5d, 10d. If there is a close > BUY day
def do_test_buy(df, i):
    def succ_in_n_days(n, printResult=False):
        target_price_field = "high"
        ror = 0.01

        for c in range(i, i + n):
            if df.ix[c + 1, target_price_field] / df.ix[i, "close"] - 1 > ror:
                # if printResult:
                #     print("success ", df.ix[i, "trade_date"])
                return 1
        # if printResult:
        #     print("failed ", df.ix[i, "trade_date"])
        return 0

    def buy_condition(df, i):
        # buy strategy:
        # - today's close is the N days's lowest price
        # - the previous M day, i-1, i-2, i-M days, can't be the second lowest price
        # - the recent K day, can't be all is down trend
        N = 20
        M = 4
        K = 3
        if i > N:
            lowest_recently = np.min(df.ix[i - N: i, "close"])
            second_lowest_recently = np.min(df.ix[i - N: i - 1, "close"])

            today_is_n_days_lowest = df.ix[i, "close"] == lowest_recently
            second_lowest_is_far = True
            for t in df.ix[i - M: i - 1, "close"]:  # is not 100% correct
                if t == second_lowest_recently:
                    second_lowest_is_far = False

            if today_is_n_days_lowest and second_lowest_is_far:
                print df.ix[i, "trade_date"]
                return True
        # breakthrough_lower = (df.ix[i, "close"] > df.ix[i, "upper"] )
        # vol = df.ix[i,"vol"] < df.ix[i-1,"vol"]
        return False

    # process the ith row of the df
    if buy_condition(df, i):
        # print(df.ix[i,"trade_date"])
        ret = []
        ret.append(df.ix[i, "ts_code"])
        ret.append(df.ix[i, "trade_date"])

        ret.append(succ_in_n_days(1, True))
        ret.append(succ_in_n_days(2))
        ret.append(succ_in_n_days(3))
        ret.append(succ_in_n_days(4))
        ret.append(succ_in_n_days(5))
        # ret.append(succ_in_n_days(6))
        # ret.append(succ_in_n_days(7))
        # ret.append(succ_in_n_days(8))
        # ret.append(succ_in_n_days(9))
        # ret.append(succ_in_n_days(10))
        return ret
    return None


def do_turle_soup_test(ts_code, start_date, end_date):
    """
    do the bollinger test for a specific share
    :param ts_code:
    :param start_date:
    :param end_date:
    :return:
    """
    # calculate bolling and close
    df = context.tushare.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    t_index = df.index
    df = df.iloc[::-1]
    df.index = t_index

    # print(df.loc[:,'close'])
    # print(df)
    # print("print column names >>>")
    # for c in df.columns:
    #     print(c)

    mem = []
    for i in range(0, len(df)):
        try:
            m = do_test_buy(df, i)
            if m is not None:
                mem.append(m)
        except:
            pass

    # print("True ratio for each days >>>")
    # df_mem = pd.DataFrame(data=mem, columns=['ts_code', 'date', 'd1', 'd2', 'd3', 'd4', 'd5','d6','d7','d8','d9','d10'])
    df_mem = pd.DataFrame(data=mem, columns=['ts_code', 'date', 'd1', 'd2', 'd3', 'd4', 'd5'])
    succ_probability = df_mem.iloc[:, 2:].sum() / len(df_mem)
    print("for share {}, do experiment {} times. The success probability is >>>".format(ts_code, len(mem)))
    print succ_probability
    return succ_probability, len(mem)


# testing for a group shares
ss = ShareSelector()
p_start_date = '20080701'
p_end_date = '20191201'
sz50 = '000016.SH'
# cyb50 = '399673.SZ'
# zz500 = '000905.SH'
shares = ss.get("index", sz50)

succ_probability_count_list = []
for s in shares:
    try:
        succ_probability, count = do_turle_soup_test(s, p_start_date, p_end_date)
        succ_probability_count_list.append((succ_probability, count))
    except:
        pass

total = 0
total_count = 0
for item in succ_probability_count_list:
    if item[1] > 0:
        total = total + item[0] * item[1]
        total_count = total_count + item[1]

print("=================")
print("After {} times experiment, we can get the probability >>>".format(total_count))
print total / total_count

# testing for a single share
# do_turle_soup_test(ts_code='002642.SZ', start_date='20080701', end_date='20191225')

"""
for sz50 
if set ror = 1%, test `high` price

After 3466 times experiment, we can get the probability >>>
d1    0.480092
d2    0.598384
d3    0.661281
d4    0.706867
d5    0.735430
dtype: float64


"""

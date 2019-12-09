import talib as ta
import pandas as pd
from src.context import context
from src.utils.share_selector import ShareSelector


# buy strategy: today's close > middle and yesterday's close < middle. BUY at today's close.
# Target: after 1d, 3d, 5d, 10d. If there is a close > BUY day
def do_test_buy(df, i):
    def succ_in_n_days(n, printResult=False):
        target_price_field = "high"
        ror = 0.005

        for c in range(i, i + n):
            if df.ix[c + 1, target_price_field] / df.ix[i, "close"] - 1 > ror:
                if printResult:
                    print("success ", df.ix[i, "trade_date"])
                return 1

        if printResult:
            print("failed ", df.ix[i, "trade_date"])
        return 0

    def buy_condition(df, i):
        return True
        breakthrough_middle_line = (
                df.ix[i, "close"] > df.ix[i, "middle"] and df.ix[i - 1, "close"] < df.ix[i - 1, "middle"])
        up_2_days = df.ix[i, "close"] > df.ix[i, "open"] and df.ix[i - 1, "close"] > df.ix[i - 1, "open"]

        # i-1 day is up
        continus_price = df.ix[i - 1, "open"] < df.ix[i, "open"] < df.ix[i - 1, "close"]
        return breakthrough_middle_line and continus_price

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
        return ret
    return None


def do_bollinger_test(ts_code, start_date, end_date):
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
    df['upper'], df['middle'], df['lower'] = ta.BBANDS(df['close'], 20, 2)  # moving average = 20, number of std = 2

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
    df_mem = pd.DataFrame(data=mem, columns=['ts_code', 'date', 'd1', 'd2', 'd3', 'd4', 'd5'])
    succ_probability = df_mem.iloc[:, 2:].sum() / len(df_mem)
    print("for share {}, do experiment {} times. The success probability is >>>".format(ts_code, len(mem)))
    print succ_probability
    return succ_probability, len(mem)


ss = ShareSelector()
p_start_date = '20080701'
p_end_date = '20191201'
sz50 = '000016.SH'
cyb50 = '399673.SZ'
zz500 = '000905.SH'
shares = ss.get("index", zz500)

succ_probability_count_list = []
for s in shares:
    try:
        succ_probability, count = do_bollinger_test(s, p_start_date, p_end_date)
        succ_probability_count_list.append((succ_probability, count))
    except:
        pass

total = 0
total_count = 0
for item in succ_probability_count_list:
    total = total + item[0] * item[1]
    total_count = total_count + item[1]

print("=================")
print("After {} times experiment, we can get the probability >>>".format(total_count))
print total / total_count

# do_bollinger_test(ts_code='002642.SZ', start_date='20080701', end_date='20191205')

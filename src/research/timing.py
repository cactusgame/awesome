import talib as ta
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from src.context import context
from src.utils.share_selector import ShareSelector
from common.backtest import BackTest


def hma(y, N):
    return ta.WMA((2 * ta.WMA(y, N / 2) - ta.WMA(y, N)), int(math.sqrt(N)))


def draw(pdf, draw_columns):
    pdf.loc[:, draw_columns].plot(secondary_y=True, grid=True, figsize=(30, 18))

    # get the indices
    buy_indices = np.where(df['buy'] == 1)
    buy_price = np.array(df['close'].tolist()).take(buy_indices)

    sell_indices = np.where(df['sell'] == 1)
    sell_price = np.array(df['close'].tolist()).take(sell_indices)

    # print("buy indices: ", buy_indices)
    # print("sell indices: ", sell_indices)

    plt.plot(buy_indices, buy_price, 'rx')
    plt.plot(sell_indices, sell_price, 'go')
    plt.savefig('timing.png')
    plt.show()


index = "000001.SH" # szzs
# index = "000905.SH"  # zz500

df = context.tushare.index_daily(ts_code=index, start_date='20130101', end_date='20191231')
# df = context.tushare.weekly(ts_code='300376.SZ', start_date='20170101', end_date='20191231')
df = df.iloc[::-1]
df = df.loc[:, ['trade_date', 'close', 'vol']]
df.index = df['trade_date']
# df['sma5'] = ta.SMA(df[target_column], timeperiod=5)
# df['sma10'] = ta.SMA(df[target_column], timeperiod=15)
# df['sma20'] = ta.SMA(df[target_column], timeperiod=20)

# buy strategy1 : decide by a single HMA k
"""
finding:
1. hma duration = 30 = 35. 2016 aror = 9%
2. The negative case like year 2018. The disadvantage is HMA lack of the BIG PICTURE of trend
"""
# target_column = 'score'
# k_source_column = 'k_source_column'
# df['score'] = df['close']
# df[k_source_column] = hma(df['score'], 250)
# df['k'] = df[k_source_column] - df[k_source_column].shift(1)
# df['buy_t'] = np.where(df['k'] > 0, 1, 0)
# df['sell_t'] = np.where(df['k'] < 0, 1, 0)
# df['buy'] = np.where((df['buy_t'] == 1) & (df['buy_t'].shift(1) == 0), 1, 0)
# df['sell'] = np.where((df['sell_t'] == 1) & (df['sell_t'].shift(1) == 0), 1, 0)
#
# df = df.loc[:, ['trade_date', 'close', k_source_column, 'k', 'buy', 'sell']]
# print df
# draw(df, ['close', k_source_column])

# buy strategy1.1: buy according to a long HMA k > 0, sell using moving loss 1%


# buy strategy2: decide by long short HMA
"""
finding:
1. short = 5, long = 10. 2016 aror = 15.2%
"""
df['hma_short'] = hma(df['close'], 85)
df['hma_long'] = hma(df['close'], 250)
day_offset = 1
df['k_hma_short'] = df['hma_short'] - df['hma_short'].shift(1)
df['buy'] = np.where((df['hma_short'].shift(day_offset) < df['hma_long'].shift(day_offset)) &
                     (df['hma_short'] > df['hma_long']), 1, 0)
# df['sell'] = np.where((df['hma_short'].shift(day_offset) > df['hma_long'].shift(day_offset)) &
#                       (df['hma_short'] < df['hma_long']), 1, 0)
df['sell'] = np.where((df['k_hma_short'] < 0) & (df['k_hma_short'].shift(1) > 0), 1, 0)
draw(df, ['close', 'hma_short', 'hma_long'])

bt = BackTest()
bt.test(df)

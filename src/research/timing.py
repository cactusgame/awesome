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
    ids = np.where(df['buy'] == 1)
    closes = np.array(df['close'].tolist()).take(ids)

    plt.plot(ids, closes, 'o')
    plt.savefig('books_read.png')
    plt.show()


index = "000001.SH"
target_column = 'score'
k_source_column = 'k_source_column'

df = context.tushare.index_daily(ts_code='000001.SH', start_date='20160101', end_date='20161231')
df = df.iloc[::-1]
df = df.loc[:, ['trade_date', 'close', 'vol']]
df['score'] = df['close']
df.index = df['trade_date']
# df['sma5'] = ta.SMA(df[target_column], timeperiod=5)
# df['sma10'] = ta.SMA(df[target_column], timeperiod=15)
# df['sma20'] = ta.SMA(df[target_column], timeperiod=20)

# buy strategy1 : decide by a single HMA k
"""
finding:
1. hma duration = 30 = 35. 2016 aror = 9%
"""
# df[k_source_column] = hma(df['score'], 35)
# df['k'] = df[k_source_column] - df[k_source_column].shift(1)
# df['buy'] = np.where(df['k'] > 0, 1, 0)
# df['sell'] = np.where(df['k'] < 0, 1, 0)
# df['close'] = df['close'] / 1000
# df = df.loc[:, ['trade_date', 'close', 'k', 'buy', 'sell']]
# print df
# draw(df.loc[:, ['close', k_source_column]])

# buy strategy2: decide by long short HMA
"""
finding:
1. short = 5, long = 15. 2016 aror = 15.2%
"""
df['hma_short'] = hma(df['close'], 5)
df['hma_long'] = hma(df['close'], 10)
day_offset = 1
df['buy'] = np.where((df['hma_short'].shift(day_offset) < df['hma_long'].shift(day_offset)) &
                     (df['hma_short'] > df['hma_long']), 1, 0)
df['sell'] = np.where((df['hma_short'].shift(day_offset) > df['hma_long'].shift(day_offset)) &
                      (df['hma_short'] < df['hma_long']), 1, 0)
draw(df, ['close', 'hma_short', 'hma_long'])

bt = BackTest()
bt.test(df)

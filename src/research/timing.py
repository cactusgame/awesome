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


index = "000001.SH"

target_column = 'score'
k_source_column = 'k_source_column'

df = context.tushare.index_daily(ts_code='000001.SH', start_date='20160101', end_date='20161231')
df = df.iloc[::-1]
df = df.loc[:, ['trade_date', 'close', 'vol']]
df['score'] = df['close']
df['sma5'] = ta.SMA(df[target_column], timeperiod=5)
df['sma10'] = ta.SMA(df[target_column], timeperiod=15)
df['sma20'] = ta.SMA(df[target_column], timeperiod=20)

df['hma5'] = hma(df['close'], 5)
df['hma10'] = hma(df['close'], 10)

df[k_source_column] = hma(df['score'], 35)
df.index = df['trade_date']

# day_offset = 1
# short_metric = 'sma5'
# long_metric = 'sma10'
# df['buy'] = np.where((df[short_metric].shift(day_offset) < df[long_metric].shift(day_offset)) &
#                      (df[short_metric] > df[long_metric]), 1, 0)
# df['sell'] = np.where((df[short_metric].shift(day_offset) > df[long_metric].shift(day_offset)) &
#                       (df[short_metric] < df[long_metric]), 1, 0)

# print df
# df.loc[:, ['trade_date', 'buy', 'sell']].to_csv('/tmp/r.csv')
# print ta.SMA(df["close"], timeperiod=25)

df.loc[:, ['close', k_source_column]].plot(secondary_y=True, grid=True, figsize=(10, 6))
plt.savefig('books_read.png')
plt.show()

df['k'] = df[k_source_column] - df[k_source_column].shift(1)
df['buy'] = np.where(df['k'] > 0, 1, 0)
df['sell'] = np.where(df['k'] < 0, 1, 0)
df['close'] = df['close'] / 1000
df = df.loc[:, ['trade_date', 'close', 'k', 'buy', 'sell']]
print df

bt = BackTest()
bt.test(df)

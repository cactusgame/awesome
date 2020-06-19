# import talib as ta
# import pandas as pd
# import numpy as np
# from src.context import context
# from src.utils.share_selector import ShareSelector
#
# # `day0` is the highest day from the list-day
# # `day1` is the first time exceed the price of day0
# # the interval date between day0 and day1 is called `d`.
#
# ss = ShareSelector()
# start_date = '20080701'
# end_date = '20191201'
# ts_code = '603711.SH'
#
# df = context.tushare.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
# print(df)

import asyncio

async def test():
    try:
        print("Hello world!")
        r = await asyncio.sleep(1)
        print("Hello again!")
    except Exception:
        pass


def main():
    test()


if __name__ == "__main__":
    main()
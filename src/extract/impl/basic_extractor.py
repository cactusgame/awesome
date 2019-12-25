from sklearn import preprocessing
import traceback
import time
import pandas as pd
import tushare as ts

from src.extract.feature_sdk import FeatureSDK
from src.context import log
from src.context import context
from src.extract.feature_definition import feature_definition_config


class BasicExtractor:
    def __init__(self, params=None):
        self.normalized = params['normalized']
        self.output_name = params['output_name']
        self.sdk = FeatureSDK(self.output_name)

    def execute(self, all_shares, start_date, end_date):
        """
        extract all
        the main method of a extractor, which is responsible to get the data from somewhere
        :param all_shares:
        :param start_date:
        :param end_date:
        :return:
        """
        index = 0
        for share in all_shares:
            try:
                index = index + 1
                print("[BasicExtractor] process the {} shares".format(index))
                self.extract_one_share(share, start_date, end_date)
            except Exception as e:
                log.info(
                    "[BasicExtractor] fail to extract share_id={}, start_date={}, end_date={}".format(share, start_date,
                                                                                                      end_date))
                log.error("[error]" + traceback.format_exc())
            time.sleep(1)

        # you should close the sdk when finishing extraction
        self.sdk.close()

    def extract_one_share(self, share_id, start_date, end_date):
        """
        :param share_id:
        :param start_date:
        :param end_date:
        :return:
        """
        bar_df = ts.pro_bar(pro_api=context.tushare, ts_code=share_id, start_date=start_date,
                            end_date=end_date, adj='qfq')

        # raw_close_df = self._extract_one_share_n_days_price(share_id=share_id, bar_df=bar_df, field="close")
        close_df = self._extract_one_share_n_days_price_ratio(share_id=share_id, bar_df=bar_df, field="close")
        open_df = self._extract_one_share_n_days_price_ratio(share_id=share_id, bar_df=bar_df, field="open")
        high_df = self._extract_one_share_n_days_price_ratio(share_id=share_id, bar_df=bar_df, field="high")
        low_df = self._extract_one_share_n_days_price_ratio(share_id=share_id, bar_df=bar_df, field="low")
        vol_df = self._extract_one_share_n_days_price_ratio(share_id=share_id, bar_df=bar_df, field="vol")

        # ror_df = self._extract_one_share_n_days_ror(bar_df, share_id)
        # assert ror_df is not None, "share_id = {} ror_df is None".format(share_id)

        self._merge_data_to_commit(close_df, open_df, high_df, low_df, vol_df)
        # self._merge_data_to_commit(close_df, vol_df, ror_df)

    def _extract_one_share_n_days_price(self, share_id, bar_df, field="close"):
        """
        return the raw HLOC for a share, without any transform
        :param share_id:
        :param bar_df:
        :param field:
        :return:
        """
        price_s = bar_df[field]
        price_list = price_s.tolist()
        date_list = price_s.index.tolist()
        log.info("[extractor] share_id= {}. there are {} rows of price".format(share_id, len(price_list)))

        # due to the order of xxx_price in tushare is DESC, so we can get the data from recently.
        # example
        keys, values = [], []
        n = feature_definition_config["hloc_seq_step"]
        keys = keys + ["time", "share_id"]
        keys = keys + ["{}_b".format(field) + str(i - 1) for i in range(n - 1, 0, -1)]
        keys = keys + ["target_{}_price".format(field)]
        keys = keys + ["target_{}_trend".format(field)]

        for index in range(len(price_list)):
            if len(price_list[index:index + n]) == n:  # else: not enough (N) data, so drop it.

                price_segment = price_list[index:index + n][::-1]
                if self.normalized:
                    price_x = [1] + [curr / price_segment[i] for i, curr in enumerate(price_segment[1:-1])]
                    price_y = price_segment[-1] / price_segment[-2]
                else:
                    price_x = price_segment[:-2]
                    price_y = price_segment[-1]

                precision = 4
                price_x = [round(x, precision) for x in price_x]
                price_y = round(price_y, precision)

                trend = 1 if price_y > 1 else 0
                values.append(
                    [str(date_list[index]), str(share_id)] + price_x + [price_y] + [trend])
        return pd.DataFrame(columns=keys, data=values)

    def _extract_one_share_n_days_price_ratio(self, share_id, bar_df, field="close"):
        """
        uniform method for getting price of HLOC
        :param share_id:
        :param bar_df:
        :param field:
        :return:
        """
        price_s = bar_df[field]
        price_list = price_s.tolist()
        date_list = price_s.index.tolist()
        log.info("[extractor] share_id= {}. there are {} rows of price".format(share_id, len(price_list)))

        # due to the order of xxx_price in tushare is DESC, so we can get the data from recently.
        # example
        keys, values = [], []
        n = feature_definition_config["hloc_seq_step"]
        keys = keys + ["time", "share_id"]
        keys = keys + ["{}_b".format(field) + str(i - 1) for i in range(n - 1, 0, -1)]
        keys = keys + ["target_{}_price".format(field)]
        keys = keys + ["target_{}_trend".format(field)]

        for index in range(len(price_list)):
            if len(price_list[index:index + n]) == n:  # else: not enough (N) data, so drop it.

                price_segment = price_list[index:index + n][::-1]
                if self.normalized:
                    price_x = [1] + [curr / price_segment[i] for i, curr in enumerate(price_segment[1:-1])]
                    price_y = price_segment[-1] / price_segment[-2]
                else:
                    price_x = price_segment[:-1]
                    price_y = price_segment[-1]

                precision = 4
                price_x = [round(x, precision) for x in price_x]
                price_y = round(price_y, precision)

                if self.normalized:
                    trend = 1 if price_y > 1 else 0
                else:
                    trend = 1 if price_segment[-1] / price_segment[-2] > 1 else 0
                values.append(
                    [str(date_list[index]), str(share_id)] + price_x + [price_y] + [trend])
        return pd.DataFrame(columns=keys, data=values)

    def _extract_one_share_n_days_ror(self, bar_df, share_id):
        # reverse the original data frame, so that it make us easy to calculate the RoR (return of rate)
        bar_df = bar_df.iloc[::-1]
        close_s = bar_df['close']
        # print(close_s)
        close_list = close_s.tolist()
        date_list = close_s.index.tolist()

        keys, values = [], []
        # similar to __extract_one_share_n_days_close, only reverse the order.
        # here, we calculate ror_5_days, ror_10_days ... we should use the max N in the loop.
        n = feature_definition_config["ror_n_days_after"]
        keys += ["time", "share_id"]
        keys += ["ror_1_days", "ror_5_days", "ror_10_days", "ror_20_days", "ror_40_days", "ror_60_days"]
        try:
            for index in range(len(close_list)):
                if len(close_list[index:index + n]) == n:  # else: not enough (N) data, so drop it.
                    ror1 = round((close_list[index + 1] / close_list[index]) - 1, 4)
                    ror5 = round((close_list[index + 5] / close_list[index]) - 1, 4)
                    ror10 = round((close_list[index + 10] / close_list[index]) - 1, 4)
                    ror20 = round((close_list[index + 20] / close_list[index]) - 1, 4)
                    ror40 = round((close_list[index + 40] / close_list[index]) - 1, 4)
                    ror60 = round((close_list[index + 60] / close_list[index]) - 1, 4)

                    values.append([str(date_list[index]), str(share_id), ror1, ror5, ror10, ror20, ror40, ror60])
        except IndexError:
            log.info("share_id = {} calculate ror maybe complete ".format(share_id))

        return pd.DataFrame(columns=keys, data=values)

    def merge_two_dataframe(self, df1, df2):
        return pd.merge(df1, df2, on=["time", "share_id"])

    def _merge_data_to_commit(self, close_df, open_df, high_df, low_df, vol_df):
        from functools import reduce
        result_df = reduce(self.merge_two_dataframe, [close_df, open_df, high_df, low_df, vol_df])

        for index, row in result_df.iterrows():
            self.sdk.save(row.index.tolist(), row.values.tolist())
        self.sdk.commit()

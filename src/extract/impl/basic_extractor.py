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
        for index, share in all_shares.iteritems():
            try:
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

        close_df = self._extract_one_share_n_days_close(bar_df, share_id)
        assert close_df is not None, "share_id = {} close_df is None".format(share_id)

        # vol_df = self._extract_one_share_n_days_vol(bar_df, share_id)
        # assert vol_df is not None, "share_id = {} vol_df is None".format(share_id)
        #
        # ror_df = self._extract_one_share_n_days_ror(bar_df, share_id)
        # assert ror_df is not None, "share_id = {} ror_df is None".format(share_id)

        self._merge_data_to_commit(close_df)
        # self._merge_data_to_commit(close_df, vol_df, ror_df)

    def _extract_one_share_n_days_close(self, bar_df, share_id):
        close_s = bar_df['close'] # todo: date order?
        # print(close_s)
        close_list = close_s.tolist()
        date_list = close_s.index.tolist()
        log.info("[extractor] share_id= {}. there are {} rows of close price".format(share_id, len(close_list)))

        # due to the order of close_price in tushare is DESC, so we can get the data from recently.
        # example
        keys, values = [], []
        n = feature_definition_config["seq_step"]
        keys = keys + ["time", "share_id"]
        keys = keys + ["close_b" + str(i-1) for i in range(n-1,0,-1)]
        keys = keys + ["target_close_price"]
        for index in range(len(close_list)):
            if len(close_list[index:index + n]) == n:  # else: not enough (N) data, so drop it.

                close_segment = close_list[index:index + n][::-1]
                if self.normalized:
                    close_price_x = [1] + [curr / close_segment[i] for i, curr in enumerate(close_segment[1:-1])]
                    close_price_y = close_segment[-1] / close_segment[-2]
                else:
                    close_price_x = close_segment[:-2]
                    close_price_y = close_segment[-1]

                precision = 4
                close_price_x = [round(x, precision) for x in close_price_x]
                close_price_y = round(close_price_y, precision)

                values.append(
                    [str(date_list[index]), str(share_id)] + close_price_x + [close_price_y])
        return pd.DataFrame(columns=keys, data=values)

    def _extract_one_share_n_days_vol(self, bar_df, share_id):
        vol_s = bar_df['vol']
        # print(close_s)
        vol_list = vol_s.tolist()
        date_list = vol_s.index.tolist()
        log.info("[extractor] share_id= {}. there are {} rows of volume".format(share_id, len(vol_list)))

        # this part is the same as _extract_one_share_n_days_close
        keys, values = [], []
        n = feature_definition_config["close_n_days_before"]
        keys = keys + ["time", "share_id"]
        keys = keys + ["volume_b" + str(i) for i in range(n)]

        for index in range(len(vol_list)):
            if len(vol_list[index:index + n]) == n:  # else: not enough (N) data, so drop it.
                values.append(
                    [str(date_list[index]), str(share_id)] + preprocessing.scale(vol_list[index:index + n]).tolist())
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

    def _merge_data_to_commit(self, close_df):
        if close_df is None:
            return

        result_df = close_df

        # if close_df is None or ror_df is None or vol_df is None:
        #     return
        #
        # step1_df = pd.merge(close_df, vol_df, on=["time", "share_id"])
        # result_df = pd.merge(step1_df, ror_df, on=["time", "share_id"])

        for index, row in result_df.iterrows():
            self.sdk.save(row.index.tolist(), row.values.tolist())
        self.sdk.commit()

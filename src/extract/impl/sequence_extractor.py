from sklearn import preprocessing
import traceback
import time
import pandas as pd
import tushare as ts

from src.extract.feature_sdk import FeatureSDK
from src.context import log
from src.context import context
from src.extract.feature_definition import feature_definition_config


class SequenceExtractor:
    def __init__(self, params=None):
        """
        :param params: a dict for params.
            output: the output file name
        :param params:
        """
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
                print("[SequenceExtractor] process the {} shares".format(index))
                self.extract_one_share(share, start_date, end_date)
            except Exception as e:
                log.info(
                    "[SequenceExtractor] fail to extract share_id={}, start_date={}, end_date={}".format(share,
                                                                                                         start_date,
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

        vol_df = self._extract_one_share_n_days_vol(bar_df, share_id)
        assert vol_df is not None, "share_id = {} vol_df is None".format(share_id)

        # ror_df = self._extract_one_share_n_days_ror(bar_df, share_id)
        # assert ror_df is not None, "share_id = {} ror_df is None".format(share_id)

        self._merge_data_to_commit(close_df, vol_df)

    def _extract_one_share_n_days_close(self, bar_df, share_id):
        close_s = bar_df['close']
        # print(close_s)
        close_list = close_s.tolist()
        date_list = close_s.index.tolist()
        log.info("[extractor] share_id= {}. there are {} rows of close price".format(share_id, len(close_list)))

        # due to the order of close_price in tushare is DESC, so we can get the data from recently.
        keys, values = [], []
        n = feature_definition_config["close_n_days_before"]
        keys = keys + ["time", "share_id", "close_price", "target_close_price"]

        # normalize:
        # the first element as 1, the use the `ratio` to normalize.
        assert n > 2
        for index in range(len(close_list)):
            if len(close_list[index:index + n]) == n:  # else: not enough (N) data, so drop it.
                close_seq = close_list[index:index + n]

                if self.normalized:
                    close_price = [1] + [curr / close_seq[i] for i, curr in enumerate(close_seq[1:])]
                    target_close_price = close_seq[-1] / close_seq[-2]
                else:
                    close_price = close_seq[:-2]
                    target_close_price = close_seq[-1]

                # adjust the precision for float
                precision = 4
                close_price = [round(x, precision) for x in close_price]
                target_close_price = round(target_close_price, precision)

                values.append(
                    [str(date_list[index]), str(share_id), close_price, target_close_price])
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
        keys = keys + ["time", "share_id", "volume", "target_volume"]

        assert n > 2
        for index in range(len(vol_list)):
            if len(vol_list[index:index + n]) == n:  # else: not enough (N) data, so drop it.
                seq = vol_list[index:index + n]
                if self.normalized:
                    volume = [1] + [curr / seq[i] for i, curr in enumerate(seq[1:])]
                    target_volume = seq[-1] / seq[-2]
                else:
                    volume = seq[:-2]
                    target_volume = seq[-1]

                # adjust the precision for float
                precision = 4
                volume = [round(x, precision) for x in volume]
                target_volume = round(target_volume, precision)

                values.append(
                    [str(date_list[index]), str(share_id), volume, target_volume])
        return pd.DataFrame(columns=keys, data=values)

    def _merge_data_to_commit(self, close_df, vol_df):
        if close_df is None or vol_df is None:
            return

        result_df = pd.merge(close_df, vol_df, on=["time", "share_id"])
        # result_df = pd.merge(step1_df, ror_df, on=["time", "share_id"])

        for index, row in result_df.iterrows():
            self.sdk.save(row.index.tolist(), row.values.tolist())
        self.sdk.commit()

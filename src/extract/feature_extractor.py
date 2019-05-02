# -*- coding: utf-8 -*-

from sklearn import preprocessing

from src.context import context
from feature_sdk import FeatureSDK
from feature_definition import feature_definition_config

from src.utils.file_util import FileUtil
from src.context import context
from src.context import log

import traceback
import pandas as pd
import tushare as ts
import time
import os

DEBUG_EXTRACTOR = True


class FeatureExtractor:
    def __init__(self):
        self.sdk = FeatureSDK()

    def extract_all(self, start_date, end_date):
        all_shares_df = context.tushare.stock_basic(exchange='', list_status='L',
                                                    fields='ts_code,name,area,industry,list_date')
        all_shares = all_shares_df['ts_code'][all_shares_df['list_date'] < end_date]

        if DEBUG_EXTRACTOR:
            all_shares = all_shares[:5]

        # select part of all data for testing
        for index, share in all_shares.iteritems():
            try:
                print("[extractor] process the {} shares".format(index))
                self.extract_one_share(share, start_date, end_date)
            except Exception as e:
                log.info("[extractor] fail to extract share_id={}, start_date={}, end_date={}".format(share, start_date,
                                                                                                      end_date))
                log.error("[error]" + traceback.format_exc())
            time.sleep(1)
        self.sdk.close()

    def extract_one_share(self, share_id, start_date, end_date):
        """
        extract all features for one share
        :param share_id:
        :param start_date:
        :param end_date:
        :return:
        """
        # print("prepare to extract share={}".format(share_id))
        log.info("[extractor] prepare to extract share={}".format(share_id))

        close_df = self._extract_one_share_n_days_close(share_id, start_date, end_date)
        assert close_df is not None, "share_id = {} close_df is None".format(share_id)

        ror_df = self._extract_one_share_n_days_ror(share_id, start_date, end_date)
        assert ror_df is not None, "share_id = {} ror_df is None".format(share_id)

        self._merge_data_to_commit(close_df, ror_df)

    def _extract_one_share_n_days_close(self, share_id, start_date, end_date):
        """
        :param share_id:
        :param start_date:
        :param end_date:
        :return:
        """
        bar_df = ts.pro_bar(pro_api=context.tushare, ts_code=share_id, start_date=start_date,
                            end_date=end_date, adj='qfq')
        close_s = bar_df['close']
        # print(close_s)
        close_list = close_s.tolist()
        date_list = close_s.index.tolist()
        log.info("[extractor] share_id= {}. there are {} rows of close price".format(share_id, len(close_list)))

        # due to the order of close_price in tushare is DESC, so we can get the data from recently.
        # example
        """
        query from 2018-12-20 to 2018-12-28
        >>>> INSERT INTO FEATURE (time,share_id,close_b0,close_b1,close_b2) VALUES ('20181228','000001.SZ',9.38,9.28,9.3)
        >>>> INSERT INTO FEATURE (time,share_id,close_b0,close_b1,close_b2) VALUES ('20181227','000001.SZ',9.28,9.3,9.34)
        >>>> INSERT INTO FEATURE (time,share_id,close_b0,close_b1,close_b2) VALUES ('20181226','000001.SZ',9.3,9.34,9.42)
        >>>> INSERT INTO FEATURE (time,share_id,close_b0,close_b1,close_b2) VALUES ('20181225','000001.SZ',9.34,9.42,9.45)
        >>>> INSERT INTO FEATURE (time,share_id,close_b0,close_b1,close_b2) VALUES ('20181224','000001.SZ',9.42,9.45,9.71)
        """
        keys, values = [], []
        n = feature_definition_config["close_n_days_before"]
        keys = keys + ["time", "share_id"]
        keys = keys + ["close_b" + str(i) for i in range(n)]

        for index in range(len(close_list)):
            if len(close_list[index:index + n]) == n:  # else: not enough (N) data, so drop it.
                values.append(
                    [str(date_list[index]), str(share_id)] + preprocessing.scale(close_list[index:index + n]).tolist())
        return pd.DataFrame(columns=keys, data=values)

    def _extract_one_share_n_days_ror(self, share_id, start_date, end_date):
        bar_df = ts.pro_bar(pro_api=context.tushare, ts_code=share_id, start_date=start_date,
                            end_date=end_date, adj='qfq')
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
        keys += ["ror_05_days", "ror_10_days", "ror_20_days", "ror_40_days", "ror_60_days"]
        try:
            for index in range(len(close_list)):
                if len(close_list[index:index + n]) == n:  # else: not enough (N) data, so drop it.
                    ror5 = round((close_list[index + 5] / close_list[index]) - 1, 4)
                    ror10 = round((close_list[index + 10] / close_list[index]) - 1, 4)
                    ror20 = round((close_list[index + 20] / close_list[index]) - 1, 4)
                    ror40 = round((close_list[index + 40] / close_list[index]) - 1, 4)
                    ror60 = round((close_list[index + 60] / close_list[index]) - 1, 4)

                    values.append([str(date_list[index]), str(share_id), ror5, ror10, ror20, ror40, ror60])
        except IndexError:
            log.info("share_id = {} calculate ror maybe complete ".format(share_id))

        return pd.DataFrame(columns=keys, data=values)

    def _merge_data_to_commit(self, close_df, ror_df):
        if close_df is None or ror_df is None:
            return

        result_df = pd.merge(close_df, ror_df, on=["time", "share_id"])

        for index, row in result_df.iterrows():
            self.sdk.save(row.index.tolist(), row.values.tolist())
        self.sdk.commit()


if __name__ == "__main__":
    _start = time.time()
    extractor = FeatureExtractor()

    if DEBUG_EXTRACTOR:
        extractor.extract_all(start_date='20050101', end_date='20181231')
        # extractor.extract_one_share(share_id='000411.SZ', start_date='20050101', end_date='20181231')
    else:
        # for test stage1, we should only extract recent 4000 days's data
        extractor.extract_all(start_date='20050101', end_date='20181231')

        # upload the db after extract the features
        FileUtil.upload_data(os.path.abspath("awesome.db"))

    log.info("extracting completed, use time {}s".format(str(time.time() - _start)))

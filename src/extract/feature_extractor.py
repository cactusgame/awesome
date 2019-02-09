# -*- coding: utf-8 -*-
from sklearn import preprocessing

from src.context import context
from feature_sdk import FeatureSDK
from feature_definition import feature_definition_config

from src.utils.file_util import FileUtil
from src.context import context

import pandas as pd
import tushare as ts
import time
import os


class FeatureExtractor:
    def __init__(self):
        self.sdk = FeatureSDK()

    def extract_all(self, start_date, end_date):
        all_shares_df = context.tushare.stock_basic(exchange='', list_status='L',
                                                    fields='ts_code,name,area,industry,list_date')
        all_shares = all_shares_df['ts_code']
        for index, share in all_shares.iteritems():
            self.extract_one_share(share, start_date, end_date)
            time.sleep(0.5)

    def extract_one_share(self, share_id, start_date, end_date):
        """
        extract all features for one share
        :param share_id:
        :param start_date:
        :param end_date:
        :return:
        """
        # print("prepare to extract share={}".format(share_id))
        context.logger.info("prepare to extract share={}".format(share_id))

        close_df = self.__extract_one_share_n_days_close(share_id, start_date, end_date)

        ror_df = self.__extract_one_share_n_days_RoR(share_id, start_date, end_date)

        self.__merge_data_to_commit(close_df, ror_df)

    def __extract_one_share_n_days_close(self, share_id, start_date, end_date):
        """
        :param share_id:
        :param start_date:
        :param end_date:
        :return:
        """
        try:
            bar_df = ts.pro_bar(pro_api=context.tushare, ts_code=share_id, start_date=start_date,
                                end_date=end_date, adj='qfq')
            close_s = bar_df['close']
            # print(close_s)
            close_list = close_s.tolist()
            date_list = close_s.index.tolist()
            context.logger.info("there are {} rows in {} with close price".format(len(close_list), share_id))

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
                    values.append([date_list[index], share_id] + preprocessing.scale(close_list[index:index + n]).tolist() )
            return pd.DataFrame(columns=keys, data=values)
        except Exception as e:
            context.logger.error("error" + str(e))

    def __extract_one_share_n_days_RoR(self, share_id, start_date, end_date):
        try:
            bar_df = ts.pro_bar(pro_api=context.tushare, ts_code=share_id, start_date=start_date,
                                end_date=end_date, adj='qfq')
            # reverse the original data frame, so that it make us easy to calculate the RoR (return of rate)
            bar_df = bar_df.iloc[::-1]
            close_s = bar_df['close']
            # print(close_s)
            close_list = close_s.tolist()
            date_list = close_s.index.tolist()
            context.logger.info("there are {} rows in {} with close price".format(len(close_list), share_id))

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

                        values.append([date_list[index], share_id, ror5, ror10, ror20, ror40, ror60])
            except IndexError:
                print("calculate ror maybe complete")

            return pd.DataFrame(columns=keys, data=values)
        except Exception as e:
            context.logger.error("error" + str(e))

    def __merge_data_to_commit(self, close_df, ror_df):
        result_df = pd.merge(close_df, ror_df, on=["time", "share_id"])
        # print(result_df)

        for index, row in result_df.iterrows():
            self.sdk.save(row.index.tolist(), row.values.tolist())
        self.sdk.commit()


if __name__ == "__main__":
    _start = time.time()
    extractor = FeatureExtractor()
    # extractor.test_extract()
    # for test stage1, we should only extract recent 4000 days's data
    # extractor.extract_all(start_date='20050101', end_date='20181231')
    extractor.extract_one_share(share_id='000001.SZ', start_date='20050101', end_date='20181231')

    # upload the db after extract the features
    FileUtil.coscmd_upload(os.path.abspath("awesome.db"))
    context.logger.warn("extracting completed, use time {}s".format(str(time.time() - _start)))

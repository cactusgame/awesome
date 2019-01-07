from src.context import context
from feature_sdk import FeatureSDK
from feature_definition import feature_definition_config

from src.utils.file_uploader import FileUploader

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
        self.__extract_one_share_n_days_close(share_id, start_date, end_date)

    def __extract_one_share_n_days_close(self, share_id, start_date, end_date):
        bar_df = ts.pro_bar(pro_api=context.tushare, ts_code=share_id, start_date=start_date,
                            end_date=end_date, adj='qfq')
        close_s = bar_df['close']
        # print(close_s)
        close_list = close_s.tolist()
        date_list = close_s.index.tolist()
        context.logger.info("there are {} rows in {} with close price".format(len(close_list), share_id))
        n = feature_definition_config["close_n_days_before"]
        for index in range(len(close_list)):
            if len(close_list[index:index + n]) == n:
                # print(date_list[index], close_list[index:index + n])
                # the prefix of close_price in
                self.sdk.save(date_list[index], share_id, ["close_b" + str(i) for i in range(n)],
                              close_list[index:index + n])
        self.sdk.commit()

    def __test_extract(self):
        share_id = '000001.SZ'
        df = ts.pro_bar(pro_api=context.tushare, ts_code=share_id, start_date='20120701',
                        end_date='20120718', adj='qfq')
        s = df['close']
        print(s)
        print(s[:3])
        s_test = s[:3]
        _idx = 0

        for ds, close_price in s_test.iteritems():
            self.sdk.save(ds, share_id, ["close_b0"], [close_price])
            _idx = _idx + 1
        self.sdk.commit()


if __name__ == "__main__":
    _start = time.time()
    extractor = FeatureExtractor()
    # extractor.test_extract()
    # for test stage1, we should only extract recent 4000 days's data
    extractor.extract_all(start_date='20050101', end_date='20181231')
    # extractor.extract_one_share(share_id='000019.SZ',start_date='20010101', end_date='20181231')

    # upload the db after extract the features
    uploader = FileUploader()
    uploader.coscmd_upload(os.path.abspath("awesome.db"))
    context.logger.warn("extracting completed, use time {}s".format(str(time.time() - _start)))

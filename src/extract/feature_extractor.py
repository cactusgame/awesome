import time
from src.extract.impl.basic_extractor import BasicExtractor
from src.extract.impl.sequence_extractor import SequenceExtractor

from pandas import Series
from src.context import context
from src.context import log
from src.extract.econfig import econfig


class FeatureExtractor:

    def __init__(self):
        # config your extractor here
        self.extractor_classes = [BasicExtractor]
        # self.extractor_classes = [SequenceExtractor]

    def extract_all(self, start_date, end_date, params=None):
        all_shares_df = context.tushare.stock_basic(exchange='', list_status='L',
                                                    fields='ts_code,name,area,industry,list_date')
        all_shares = all_shares_df['ts_code'][all_shares_df['list_date'] < end_date]

        # select part of all data for testing
        if econfig.DEBUG:
            all_shares = all_shares[:5]

        for extractor_class in self.extractor_classes:
            ext = extractor_class(params)
            ext.execute(all_shares, start_date, end_date)

    def extract_one(self, share_id, start_date, end_date, params=None):
        for extractor_class in self.extractor_classes:
            ext = extractor_class(params)
            ext.execute(Series([share_id]), start_date, end_date)


"""
RNN extractor:
use features like close_price day3 ,day2, day1, day0 to predict -> tomorrow.
as the example above, `input_size` = 1, `num_steps` = 4

the file format
share_id time   N days close        target price
xxx       xx   [cp3,cp2,cp1,cp0],     cp_tomorrow

SequenceExtractor: use different date range as train and eval
  extract(start_date ,end_date) as train file
  extract(start_date ,end_date) as eval file

"""
if __name__ == "__main__":
    _start = time.time()
    extractor = FeatureExtractor()

    if econfig.DEBUG:
        # extractor.extract_all(start_date='20080101', end_date='20080301',
        #                       params={'normalized': True, 'output_name': 'feature_train'})  # as train
        # extractor.extract_all(start_date='20190101', end_date='20190301',
        #                       params={'normalized': True, 'output_name': 'feature_eval'})  # as eval

        extractor.extract_one(share_id="603999.SH", start_date='20080101', end_date='20180101',
                              params={'normalized': True, 'output_name': 'feature_eval'})

        # extractor.extract_one_share(share_id='000411.SZ', start_date='20050101', end_date='20181231')
    else:
        # for test stage1, we should only extract recent 4000 days's data
        # extractor.extract_all(start_date='20050101', end_date='20181231')

        extractor.extract_all(start_date='20080101', end_date='20180101',
                              params={'normalized': True, 'output_name': 'feature_train'})  # as train
        extractor.extract_all(start_date='20180102', end_date='20190901',
                              params={'normalized': True, 'output_name': 'feature_eval'})  # as eval

    log.info("extracting completed, use time {}s".format(str(time.time() - _start)))

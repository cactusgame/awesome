import click
import time
from src.extract.impl.basic_extractor import BasicExtractor
from src.extract.impl.sequence_extractor import SequenceExtractor
from src.extract.feature_definition import TRAIN_FILE_NAME
from src.extract.feature_definition import EVAL_FILE_NAME

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

        self.extract_multiple(all_shares.tolist(), start_date, end_date, params)

    def extract_multiple(self, share_ids, start_date, end_date, params=None):
        # select part of all data for testing
        if econfig.DEBUG:
            share_ids = share_ids[:5]

        for extractor_class in self.extractor_classes:
            ext = extractor_class(params)
            ext.execute(share_ids, start_date, end_date)

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


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--test", type=bool)
def main(test=None):
    _start = time.time()
    econfig.init(test)

    extractor = FeatureExtractor()

    # select the target shares first
    # shangzheng 50
    ingredient_df = context.tushare.index_weight(index_code='000016.SH', start_date='20080101', end_date='20180101')
    ingredient_share_set = set(ingredient_df['con_code'].tolist())

    target_shares = ["601398.SH","601288.SH","601988.SH","601939.SH","601328.SH",]

    if econfig.DEBUG:
        pass
        # extractor.extract_all(start_date='20080101', end_date='20080301',
        #                       params={'normalized': True, 'output_name': TRAIN_FILE_NAME})  # as train
        # extractor.extract_all(start_date='20190101', end_date='20190301',
        #                       params={'normalized': True, 'output_name': EVAL_FILE_NAME})  # as eval

        # extractor.extract_multiple(share_ids=list(ingredient_share_set), start_date='20080101', end_date='20180101',
        #                            params={'normalized': True, 'output_name': TRAIN_FILE_NAME})  # as train
        # extractor.extract_multiple(share_ids=list(ingredient_share_set), start_date='20180102', end_date='20190901',
        #                            params={'normalized': True, 'output_name': EVAL_FILE_NAME})  # as eval

        extractor.extract_multiple(share_ids=target_shares, start_date='20080101', end_date='20180101',
                                   params={'normalized': True, 'output_name': TRAIN_FILE_NAME})  # as train
        extractor.extract_multiple(share_ids=target_shares, start_date='20180102', end_date='20190901',
                                   params={'normalized': True, 'output_name': EVAL_FILE_NAME})  # as eval
    else:
        # for test stage1, we should only extract recent 4000 days's data
        # extractor.extract_all(start_date='20050101', end_date='20181231')

        # extractor.extract_all(start_date='20080101', end_date='20180101',
        #                       params={'normalized': True, 'output_name': TRAIN_FILE_NAME})  # as train
        # extractor.extract_all(start_date='20180102', end_date='20190901',
        #                       params={'normalized': True, 'output_name': EVAL_FILE_NAME})  # as eval

        extractor.extract_multiple(share_ids=list(ingredient_share_set), start_date='20180101', end_date='20190601',
                                   params={'normalized': True, 'output_name': TRAIN_FILE_NAME})  # as train
        extractor.extract_multiple(share_ids=list(ingredient_share_set), start_date='20190602', end_date='20191101',
                                   params={'normalized': True, 'output_name': EVAL_FILE_NAME})  # as eval

        # extractor.extract_multiple(share_ids=target_shares, start_date='20080101', end_date='20180101',
        #                            params={'normalized': True, 'output_name': TRAIN_FILE_NAME})  # as train
        # extractor.extract_multiple(share_ids=target_shares, start_date='20180102', end_date='20190901',
        #                            params={'normalized': True, 'output_name': EVAL_FILE_NAME})  # as eval

    log.info("extracting completed, use time {}s".format(str(time.time() - _start)))


if __name__ == "__main__":
    main()


import time
from src.extract.impl.basic_extractor import BasicExtractor

from src.context import context
from src.context import log
from src.extract.econfig import econfig


class FeatureExtractor:

    def __init__(self):
        # config your extractor here
        self.extractor_classes = [BasicExtractor]

    def extract_all(self, start_date, end_date):
        all_shares_df = context.tushare.stock_basic(exchange='', list_status='L',
                                                    fields='ts_code,name,area,industry,list_date')
        all_shares = all_shares_df['ts_code'][all_shares_df['list_date'] < end_date]

        # select part of all data for testing
        if econfig.DEBUG:
            all_shares = all_shares[:5]

        for extractor_class in self.extractor_classes:
            ext = extractor_class()
            ext.execute(all_shares, start_date, end_date)


if __name__ == "__main__":
    _start = time.time()
    extractor = FeatureExtractor()

    if econfig.DEBUG:
        extractor.extract_all(start_date='20050101', end_date='20181231')
        # extractor.extract_one_share(share_id='000411.SZ', start_date='20050101', end_date='20181231')
    else:
        # for test stage1, we should only extract recent 4000 days's data
        extractor.extract_all(start_date='20050101', end_date='20181231')

    log.info("extracting completed, use time {}s".format(str(time.time() - _start)))

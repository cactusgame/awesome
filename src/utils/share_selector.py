from src.context import context
import datetime
import random
from calendar import monthrange


class ShareSelector:
    def __init__(self):
        pass

    def get(self, target, value=None):
        """
        get "all" shares, get shares in some specific index?
        :return:
        """
        if target == "all":
            all_shares_df = context.tushare.stock_basic(exchange='', list_status='L',
                                                        fields='ts_code,name,area,industry,list_date')
            if value is None:
                return all_shares_df['ts_code'].tolist()
            else:
                return random.sample(all_shares_df['ts_code'].tolist(), value)

        elif target == "index":
            # FIXME: the api `index_weight` is not stable, sometime, it returns empty even the param is correct.
            now = datetime.datetime.now()
            last_month = now - datetime.timedelta(days=31)
            mr = monthrange(now.year, last_month.month)
            # return the latest index.
            df = context.tushare.index_weight(index_code=value,
                                              start_date='{}{}01'.format(last_month.year, last_month.month),
                                              end_date='{}{}{}'.format(last_month.year, last_month.month, mr[1]))
            return df['con_code'].tolist()
        else:
            raise Exception("unsupport target {} in Selector".format(target))

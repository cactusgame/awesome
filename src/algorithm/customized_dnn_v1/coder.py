from data_formatter import Target


class Coder:
    def __init__(self, data_formatter):
        self.data_formatter = data_formatter

    def parse(self, row_dict):
        try:
            line = {}
            # parse features
            for key in self.data_formatter.FEATURES:
                line[key] = row_dict[key]

            # parse targets
            # if you want to do the sanity check, you could open this comment
            # line[Target.ROR_20_DAYS_BOOL] = 1
            line[Target.ROR_20_DAYS_BOOL] = 1 if float(row_dict['ror_20_days']) > 0 else 0  # label must > 0
            return line
        except Exception as e:
            print("error parse line from feature_all.csv. {} ".format(row_dict))
            return None

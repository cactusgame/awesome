from src.algorithm.algorithm_v1.data_formatter import Target


class Coder:
    def __init__(self,data_formatter):
        self.data_formatter = data_formatter

    def parse(self, row_dict):
        line = {}
        # parse features
        for key in self.data_formatter.FEATURES:
            line[key] = row_dict[key]

        # parse targets
        line[Target.ROR_20_DAYS_BOOL] = 1 if float(row_dict['ror_20_days']) > 0 else 0 # label must > 0
        return line

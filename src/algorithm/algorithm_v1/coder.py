class Coder:
    def __init__(self,data_formatter):
        self.data_formatter = data_formatter

    def parse(self, row_dict):
        line = {}
        # parse features
        for key in self.data_formatter.FEATURES:
            line[key] = row_dict[key]

        # parse targets
        line['ror_20_days_bool'] = 1 if float(row_dict['ror_20_days']) > 0 else -1
        return line

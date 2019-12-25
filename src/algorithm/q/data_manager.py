import csv
import numpy as np
from src.utils.utils import sigmoid
from src.algorithm.q.data_formatter import DataFormatter


# I think the data should be ordered.
class DataManager:
    data_len = 0
    data = []

    def __init__(self, data_path):
        df = DataFormatter()

        # todo: the time order in csv file is desc. May put the old date in front. Not sure.
        with open(data_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.data_len = self.data_len + 1
                self.data.insert(0, [row[x] for x in df.FEATURES])

    def get_state(self, index):
        # fixme: map (0.9,1.1) to (-1,1), n = zoom param
        # (x- 1) * 10
        n = 1
        price_list = self.data[index]
        price_x = [1] + [float(curr) / float(price_list[i]) for i, curr in enumerate(price_list[1:])]
        return np.array([sigmoid((float(x) - 1) * 10 * n) for x in price_x])

    def get_buy_price(self,index):
        price_list = self.data[index]
        return float(price_list[-1])


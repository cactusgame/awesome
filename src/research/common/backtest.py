import math


class BackTest():
    STATE_EMPTY = 0
    STATE_HOLD = 1

    def __init__(self):
        self.state = self.STATE_EMPTY
        self.discount = 0.999  # remain 998 when earn 1000$
        self.init_money = 10000
        self.money = self.init_money
        self.hold = 0
        self.days = 0
        self.trading_up_times = 0
        self.trading_times = 0

    def test(self, signal_table):
        """
        do the back test by the singal table.
        :param signal_table: signal table is a pandas DataFrame with 4 columns. "trade_date" + "close" + "buy" + "sell"
        :return:
        """
        self.days = len(signal_table)
        last_price = 0.
        buy_price = 0.
        for index, row in signal_table.iterrows():
            buy_signal = True if row['buy'] == 1 else False
            sell_signal = True if row['sell'] == 1 else False
            date = row['trade_date']
            close = row['close']
            last_price = close
            if self.state == self.STATE_EMPTY and buy_signal:
                buy_price = close
                shares = self.money / close
                self.money = 0
                self.hold = shares
                self.state = self.STATE_HOLD
                print("[date={}] buy {} shares at price = {}".format(date, shares, close))

            elif self.state == self.STATE_HOLD and sell_signal:
                if close > buy_price:
                    self.trading_up_times = self.trading_up_times + 1
                self.money = int((self.money + self.hold * close) * self.discount)
                self.state = self.STATE_EMPTY

                print("[date={}] sell {} shares at price = {}".format(date, self.hold, close))
                self.hold = 0
                self.trading_times = self.trading_times + 1

        value = self.money if self.money > 0 else self.hold * last_price
        print(" ===== backtest done =====")
        print("total days = {} ".format(self.days))
        # print("trade times = {}".format(self.trading_times))
        print("final money = {}".format(value))
        print("{}/{}. succ rate = {} ".format(self.trading_up_times, self.trading_times , (self.trading_up_times * 1.) / (self.trading_times * 1.)))
        print("annual ror = {}".format(pow((value * 1.) / (self.init_money * 1.), 1. / (self.days * 1. / 250.)) - 1.))

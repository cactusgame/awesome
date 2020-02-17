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
        self.hold_days = 0
        self.total_days = 0
        self.trading_up_times = 0
        self.trading_times = 0
        self.latest_high = 0  # This price track the price after buying. It's use for moving stop loss
        self.buy_rule = self.default_buy_rule
        self.sell_rule = self.default_sell_rule

    def test(self, signal_table):
        """
        do the back test by the singal table.
        :param signal_table: signal table is a pandas DataFrame with 4 columns. "trade_date" + "close" + "buy" + "sell"
        :return:
        """
        self.total_days = len(signal_table)
        last_price = 0.
        buy_price = 0.
        for index, row in signal_table.iterrows():
            buy_signal = True if row['buy'] == 1 else False
            sell_signal = True if row['sell'] == 1 else False
            date = row['trade_date']
            close = row['close']
            last_price = close

            # update moving stop loss
            if self.state == self.STATE_HOLD:
                self.hold_days = self.hold_days + 1
                if close > self.latest_high:
                    self.latest_high = close

            if self.state == self.STATE_EMPTY and self.buy_rule(signal_table, index, row):
                self.hold_days = 0
                buy_price = close
                self.latest_high = buy_price
                shares = self.money / close
                self.money = 0
                self.hold = shares
                self.state = self.STATE_HOLD
                print("[date={}] buy    {:.2f}  price={:.2f}".format(date, shares, close))

            elif self.state == self.STATE_HOLD and self.sell_rule(signal_table, index, row):
                self.hold_days = 0
                if close > buy_price:
                    self.trading_up_times = self.trading_up_times + 1
                    trend = "up"
                else:
                    trend = "down"
                self.money = int((self.money + self.hold * close) * self.discount)
                self.state = self.STATE_EMPTY

                print("[date={}] sell   {:.2f}  price={:.2f}    trend={}".format(date, self.hold, close, trend))
                self.hold = 0
                self.trading_times = self.trading_times + 1

        value = self.money if self.money > 0 else self.hold * last_price
        print(" ===== backtest done =====")
        print("total days = {} ".format(self.total_days))
        # print("trade times = {}".format(self.trading_times))
        print("final money = {}".format(value))
        print("{}/{}. succ rate = {} ".format(self.trading_up_times, self.trading_times,
                                              (self.trading_up_times * 1.) / (self.trading_times * 1.)))
        print("annual ror = {}".format(
            pow((value * 1.) / (self.init_money * 1.), 1. / (self.total_days * 1. / 250.)) - 1.))

    def default_buy_rule(self, df, index, row):
        return True if row['buy'] == 1 else False

    def default_sell_rule(self, df, index, row):
        return True if row['sell'] == 1 else False

    def moving_loss_sell_rule(self, df, index, row):
        moving_stop_loss_rate = 0.08
        if row['close'] < self.latest_high * (1 - moving_stop_loss_rate):
            return True
        else:
            return False

    def hold_days_sell_rule(self, df, index, row):
        if self.hold_days == 1:
            return True
        return False

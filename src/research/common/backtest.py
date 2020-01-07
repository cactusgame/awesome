class BackTest():
    STATE_EMPTY = 0
    STATE_HOLD = 1

    def __init__(self):
        self.state = self.STATE_EMPTY
        self.discount = 0.999  # remain 998 when earn 1000$
        self.money = 10000
        self.hold = 0

    def test(self, signal_table):
        """
        do the back test by the singal table.
        :param signal_table: signal table is a pandas DataFrame with 4 columns. "trade_date" + "close" + "buy" + "sell"
        :return:
        """
        last_price = 0.
        for index, row in signal_table.iterrows():
            buy_signal = True if row['buy'] == 1 else False
            sell_signal = True if row['sell'] == 1 else False
            date = row['trade_date']
            close = row['close']
            last_price = close
            if self.state == self.STATE_EMPTY and buy_signal:
                shares = self.money / close
                self.money = 0
                self.hold = shares
                self.state = self.STATE_HOLD
                print("[date={}] buy {} shares at price = {}".format(date, shares, close))

            elif self.state == self.STATE_HOLD and sell_signal:
                self.money = int((self.money + self.hold * close) * self.discount)
                self.state = self.STATE_EMPTY

                print("[date={}] sell {} shares at price = {}".format(date, self.hold, close))
                self.hold = 0

        value = self.money if self.money > 0 else self.hold * last_price
        print("final money = {}".format(value))

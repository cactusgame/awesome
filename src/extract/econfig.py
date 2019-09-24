class EConfig:
    DEBUG = False

    def __init__(self):
        pass

    def init(self, test):
        if test is not None:
            self.DEBUG = test


econfig = EConfig()

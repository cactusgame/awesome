
feature_definition_config = {}
feature_definition_config["close_n_days_before"] = 3
feature_definition_config["ror_10_days"] = 10

feature_definition = {}
# close price before N days
feature_definition["close_b0"] = ("float", "REAL")
feature_definition["close_b1"] = ("float", "REAL")
feature_definition["close_b2"] = ("float", "REAL")
feature_definition["share_id"] = ("float", "REAL")

# RoR (rate of return)
feature_definition["ror_10_days"] = ("float", "REAL")

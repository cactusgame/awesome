import tensorflow as tf

feature_definition_config = {}
feature_definition_config["close_n_days_before"] = 3
feature_definition_config["ror_10_days"] = 10

feature_definition = {}
# format:
# key: feature name
# value: type in python, type in sqlite, feature def in TFT, feature type in TF
# close price before N days
feature_definition["close_b0"] = ("float", "REAL", "tf.FixedLenFeature", tf.float32)
feature_definition["close_b1"] = ("float", "REAL", "tf.FixedLenFeature", tf.float32)
feature_definition["close_b2"] = ("float", "REAL", "tf.FixedLenFeature", tf.float32)
feature_definition["share_id"] = ("float", "REAL", "tf.FixedLenFeature", tf.string)

# RoR (rate of return)
feature_definition["ror_10_days"] = ("float", "REAL", "tf.FixedLenFeature", tf.float32)




import tensorflow as tf

feature_definition_config = {}
feature_definition_config["close_n_days_before"] = 21
feature_definition_config["ror_n_days_after"] = 60

# feature_extractor_definition is only used for feature extractor,
# items in the `feature_extractor_definition` will create a column in database
feature_extractor_definition = {}
# format:
# key: feature name
# value:
# [0] type in python
# [1] type in sqlite
# [2] feature def in TFT
# [3] feature preprocessing type in TFT (like Number, Vocabulary...)
# [4] feature type in TF
# [5] whether it is a Feature, Target or Infer

# close price before N days
FORMAT_NUMBER = "Number"
FORMAT_VOCABULARY = "Vocabulary"

TYPE_FEATURE = "feature"
TYPE_TARGET = "target"
TYPE_INFER = "infer"

### Features
feature_extractor_definition["close_b0"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b1"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b2"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b3"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b4"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b5"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b6"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b7"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b8"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b9"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b10"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b11"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b12"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b13"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b14"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b15"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b16"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b17"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b18"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b19"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["close_b20"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_FEATURE)
feature_extractor_definition["share_id"] = ("str", "REAL", "tf.FixedLenFeature", FORMAT_VOCABULARY, tf.string, TYPE_FEATURE)
feature_extractor_definition["time"] = ("str", "REAL", "tf.FixedLenFeature", FORMAT_VOCABULARY, tf.string, TYPE_FEATURE)


### Targets
# only trading day, exclude holidays
feature_extractor_definition["ror_05_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET)
feature_extractor_definition["ror_10_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET)
feature_extractor_definition["ror_20_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET)
feature_extractor_definition["ror_40_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET)
feature_extractor_definition["ror_60_days"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_TARGET)

### Infer
# This type if infered from coder which is only exist in features or targets, we don't need to store it in db.
feature_extractor_definition["ror_20_days_bool"] = ("float", "REAL", "tf.FixedLenFeature", FORMAT_NUMBER, tf.float32, TYPE_INFER)


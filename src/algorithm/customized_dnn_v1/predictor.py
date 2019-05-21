import time
import tensorflow_transform as tft
from tensorflow.contrib import predictor

from src.training.train_config import *
from src.context import context
from src.training.model import Model
from src.context import log


class Predictor:
    def __init__(self):
        self.model = Model()

    def predict(self):
        tf_transform_output = tft.TFTransformOutput(TARGET_DIR)



        eval_sample_input_fn = self.model.make_training_input_fn(
            tf_transform_output,
            exp_log_data_file_train_tfrecord + '*',
            1)

        predict_fn = predictor.from_saved_model(TARGET_DIR)
        predictions = predict_fn(xx)
        print(predictions['scores'])

        # p_iter = estimator.predict(input_fn=eval_sample_input_fn)
        # for iteri in p_iter:
        #     log.info(str(iteri))


if __name__ == "__main__":
    _start = time.time()

    p = Predictor()
    p.predict()

    context.logger.info("[total] use {}s to train".format(time.time() - _start))

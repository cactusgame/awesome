import time

import numpy as np
import tensorflow as tf
import tensorflow_transform as tft

from src.algorithm.q.agent import Agent
from src.extract.feature_definition import TRAIN_FILE_NAME
from src.extract.feature_definition import EVAL_FILE_NAME
from src.extract.feature_definition import feature_definition_config
from src.context import context
from src.context import log
from src.base.config import cfg
from model import Model


class Trainer:
    def __init__(self):
        self.model = Model()

    def train(self):
        # # the graph preprocessed by TFT preprocessing
        #
        # # Generate all `input_fn`s for the tf estimator
        # train_input_fn = self.model.make_training_input_fn(
        #     cfg.get_shard_file(TRAIN_FILE_NAME) + '*',
        #     cfg.TRAIN_BATCH_SIZE)
        # eval_input_fn = self.model.make_training_input_fn(
        #     cfg.get_shard_file(EVAL_FILE_NAME) + '*',
        #     cfg.EVAL_BATCH_SIZE)
        #
        # make_serving_input_fn = self.model.make_serving_input_fn()
        #
        # run_config = tf.estimator.RunConfig().replace(
        #     save_checkpoints_secs=cfg.SAVE_MODEL_SECONDS,
        #     keep_checkpoint_max=3,
        #     session_config=tf.ConfigProto(device_count={'GPU': 0}))
        #
        # model_fn = self.model.make_model_fn()
        # estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, model_dir=cfg.TARGET_DIR)
        #
        # eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=cfg.EVAL_STEPS,
        #                                   name='evaluation', start_delay_secs=5, throttle_secs=cfg.EVAL_SECONDS)
        # train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=cfg.TRAIN_MAX_STEPS)
        #
        # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        #
        # estimator.export_savedmodel(cfg.TARGET_MODEL_DIR, make_serving_input_fn, strip_default_attrs=True)

        # stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

        window_size = feature_definition_config["hloc_seq_step"] - 1
        episode_count = 1

        agent = Agent(window_size)
        # data = getStockDataVec(stock_name)
        # l = len(data) - 1
        batch_size = 32

        for e in xrange(episode_count):
            print "Episode " + str(e) + "/" + str(episode_count)
            # state = getState(data, 0, window_size + 1)
            #
            # total_profit = 0
            # agent.inventory = []
            #
            # for t in xrange(l):
            #     action = agent.act(state)
            #
            #     # sit
            #     next_state = getState(data, t + 1, window_size + 1)
            #     reward = 0
            #
            #     if action == 1:  # buy
            #         agent.inventory.append(data[t])
            #     # print "Buy: " + formatPrice(data[t])
            #
            #     elif action == 2 and len(agent.inventory) > 0:  # sell
            #         bought_price = agent.inventory.pop(0)
            #         reward = max(data[t] - bought_price, 0)  # todo I think if negative profit should be punished
            #         total_profit += data[t] - bought_price
            #     # print "Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price)
            #
            #     done = True if t == l - 1 else False
            #     agent.memory.append((state, action, reward, next_state, done))
            #     state = next_state
            #
            #     if done:
            #         print "--------------------------------"
            #         print "Total Profit: " + formatPrice(total_profit)
            #         print "--------------------------------"
            #
            #     if len(agent.memory) > batch_size:
            #         agent.expReplay(batch_size)
            #
            # if e % 10 == 0:
            #     agent.model.save("models/model_ep" + str(e))

if __name__ == "__main__":
    _start = time.time()

    trainer = Trainer()
    trainer.train()
    # trainer.eval_predict()

    context.logger.info("[total] use {}s to train".format(time.time() - _start))

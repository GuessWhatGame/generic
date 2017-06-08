from tqdm import tqdm
from numpy import float32
import copy
import os
import itertools
from collections import OrderedDict
import tensorflow as tf


# TODO check if optimizers are always ops? Maybe there is a better check
def is_optimizer(x):
    return hasattr(x, 'op_def')

def is_summary(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.string


def is_float(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.float32


def is_scalar(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.float32 and len(x.shape) == 0


class Evaluator(object):
    def __init__(self, provided_sources, scope="", writer=None,
                 network=None, tokenizer=None): # debug purpose only, do not use in the code

        self.provided_sources = provided_sources
        self.scope = scope
        self.writer = writer
        if len(scope) > 0 and not scope.endswith("/"):
            self.scope += "/"
        self.use_summary = False

        # Debug tools (should be removed on the long run)
        self.network=network
        self.tokenizer = tokenizer


    def process(self, sess, iterator, outputs, listener=None):
        original_outputs = list(outputs)

        if not isinstance(outputs, list):
            outputs = [outputs]

        is_training = any([is_optimizer(x) for x in outputs])

        if listener is not None:
            outputs += [listener.require()]  # add require outputs
            # outputs = flatten(outputs) # flatten list (when multiple requirement)
            outputs = list(OrderedDict.fromkeys(outputs))  # remove duplicate while preserving ordering
            listener.before_epoch(is_training)

        n_iter = 1.
        aggregated_outputs = [0.0 for v in outputs if is_scalar(v) and v in original_outputs]

        for batch in tqdm(iterator):
            # Appending is_training flag to the feed_dict
            batch["is_training"] = is_training

            # evaluate the network on the batch
            results = self.execute(sess, outputs, batch)
            # process the results
            i = 0
            for var, result in zip(outputs, results):
                if is_scalar(var) and var in original_outputs:
                    # moving average
                    aggregated_outputs[i] = ((n_iter - 1.) / n_iter) * aggregated_outputs[i] + result / n_iter
                    i += 1
                elif is_summary(var):  # move into listener?
                    self.writer.add_summary(result)

                if listener is not None and listener.valid(var):
                    listener.after_batch(result, batch, is_training)

            n_iter += 1

        if listener is not None:
            listener.after_epoch(is_training)

        return aggregated_outputs

    def execute(self, sess, output, batch):
        feed_dict = {self.scope + key + ":0": value for key, value in batch.items() if key in self.provided_sources}
        return sess.run(output, feed_dict=feed_dict)


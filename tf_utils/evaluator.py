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



class MultiGPUEvaluator(object):
    """Wrapper for evaluating on multiple GPUOptions

    parameters
    ----------
        provided_sources: list of sources
            Each source has num_gpus placeholders with name:
            name_scope[gpu_index]/network_scope/source
        network_scope: str
            Variable scope of the model
        name_scopes: list of str
            List that defines name_scope for each GPU
    """

    def __init__(self, provided_sources, network_scope, name_scopes, writer=None,
                 networks=None, tokenizer=None): #Debug purpose only, do not use here

        # Dispatch sources
        self.provided_sources = provided_sources
        for source in self.provided_sources:
            for scope in name_scopes:
                self.multi_gpu_sources.append(scope + '/' + self.network_scope + source)




        self.network_scope = network_scope
        self.name_scopes = name_scopes

        self.writer = writer
        if len(self.network_scope) > 0 and not network_scope.endswith("/"):
            self.network_scope += "/"
        self.multi_gpu_sources = []


        # Debug tools, do not use here!
        self.networks = networks
        self.tokenizer = tokenizer


    def append_batch(self, single_batch, name_scope, multi_gpu_batch):

        for k, v in single_batch.items():
            multi_gpu_batch[name_scope + '/' + self.network_scope + k] = v

        return multi_gpu_batch


    def process(self, sess, iterator, outputs, listener=None):

        assert listener is None, "Listener are not yet supported with multi-gpu evaluator"
        assert isinstance(outputs, list), "outputs must be a list"

        # check for optimizer to define training/eval mode
        is_training = any([is_optimizer(x) for x in outputs])

        n_iter = 1.
        aggregated_outputs = [0.0 for v in outputs if is_scalar(v)]

        try:
            with tqdm(total=len(iterator)) as pbar:

                while True:

                    # Generate multi-gpu batch
                    multi_gpu_batch = {}
                    for name_scope in self.name_scopes:
                        batch = next(iterator)
                        batch['is_training'] = is_training
                        multi_gpu_batch = self.append_batch(batch, name_scope, multi_gpu_batch)
                    n_iter += 1

                    # Execute the batch
                    results = self.execute(sess, outputs, multi_gpu_batch)

                    # process the results
                    i = 0
                    for var, result in zip(outputs, results):
                        if is_scalar(var) and var in outputs:
                            # moving average
                            aggregated_outputs[i] = ((n_iter - 1.) / n_iter) * aggregated_outputs[i] + result / n_iter
                            i += 1

                        elif is_summary(var):  # move into listener?
                            self.writer.add_summary(result)

                    pbar.update(len(self.name_scopes))

        except StopIteration:
            pass
        except Exception as e:
            print(e)
        finally:
            if listener is not None:
                listener.after_epoch(is_training)
            return aggregated_outputs


    def execute(self, sess, output, batch):
        feed_dict = {key + ":0": value for key, value in batch.items() if key in self.multi_gpu_sources}
        return sess.run(output, feed_dict=feed_dict)
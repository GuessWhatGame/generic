import tensorflow as tf
import os
import json

class AbstractNetwork(object):
    def __init__(self, scope_name, device=''):
        self.scope_name = scope_name
        self.device = device

    def get_parameters(self):
        return [v for v in tf.trainable_variables() if self.scope_name in v.name]

    def get_sources(self, sess):
        return [os.path.basename(tensor.name) for tensor in self.get_inputs(sess) if self.scope_name in tensor.name]

    def get_inputs(self, sess):
        placeholders = [p for p in sess.graph.get_operations() if "holder" in p.type if self.scope_name in p.name ]
        if self.device is not '':
            return [p for p in placeholders if p.device[-1] == str(self.device)]
        return placeholders

    def get_outputs(self):
        pass
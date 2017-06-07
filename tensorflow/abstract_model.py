import tensorflow as tf
import os
import json

class AbstractModel(object):
    def __init__(self, scope_name, device=''):
        self.scope_name = scope_name
        self.device = device

    @classmethod
    def from_exp_identifier(cls, identifier, exp_dir):
        config_path = os.path.join(exp_dir, identifier, 'config.json')
        if not os.path.exists(config_path):
            raise RuntimeError("Couldn't find config")

        with open(config_path, 'rb') as f:
            config = json.load(f)
        return cls(config)

    def get_parameters(self):
        return tf.trainable_variables()

    def get_sources(self, sess):
        return [os.path.basename(tensor.name) for tensor in self.get_inputs(sess)]

    def get_inputs(self, sess):
        placeholders = [p for p in sess.graph.get_operations() if p.type == "Placeholder"]
        if self.device is not '':
            return [p for p in placeholders if p.device[-1] == str(self.device)]
        return placeholders


import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

def create_optimizer(network, loss, config, optim=tf.train.AdamOptimizer):

    lrt = config['optimizer']['learning_rate']
    clip_val = config['optimizer']['clip_val']

    # create optmizer
    optimizer = optim(learning_rate=lrt)

    # apply gradient clipping
    gvs = optimizer.compute_gradients(loss, network.get_parameters())
    gvs = [(tf.clip_by_norm(grad, clip_val), var) for grad, var in gvs]

    optimizer = optimizer.apply_gradients(gvs)

    # add update ops (such as batch norm) to the optimizer call
    outputs = network.get_outputs()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        outputs = control_flow_ops.with_dependencies([updates], outputs)

    return optimizer, outputs
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

def create_optimizer(network, loss, config, optim=tf.train.AdamOptimizer, var_list=None, apply_update_ops=True):

    lrt = config['optimizer']['learning_rate']
    clip_val = config['optimizer']['clip_val']

    # create optmizer
    optimizer = optim(learning_rate=lrt)

    # apply gradient clipping
    if var_list is None:
        var_list = network.get_parameters()

    gvs = optimizer.compute_gradients(loss, var_list=var_list)
    gvs = [(tf.clip_by_norm(grad, clip_val), var) for grad, var in gvs]

    optimizer = optimizer.apply_gradients(gvs)

    # add update ops (such as batch norm) to the optimizer call
    outputs = network.get_outputs()
    if apply_update_ops:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            outputs = control_flow_ops.with_dependencies([updates], outputs)

    return optimizer, outputs
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


def create_optimizer(network, loss, config, finetune, optim_cst=tf.train.AdamOptimizer, var_list=None, apply_update_ops=True):

    lrt = config['optimizer']['learning_rate']
    clip_val = config['optimizer'].get('clip_val', 0)

    # create optimizer
    optimizer = optim_cst(learning_rate=lrt)

    # apply gradient clipping
    if var_list is None:
        var_list = network.get_parameters(finetune=finetune)

    grad = optimizer.compute_gradients(loss, var_list=var_list)

    # Clip gradient
    if clip_val > 0:
        grad = clip_gradient(grad, clip_val=[])

    # update optimizer
    optimizer = optimizer.apply_gradients(grad)

    # add update ops (such as batch norm) to the optimizer call
    outputs = loss
    if apply_update_ops:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            outputs = control_flow_ops.with_dependencies([updates], loss)

    return optimizer, outputs


def create_multi_gpu_optimizer(networks, config, finetune=list(), optim_cst=tf.train.AdamOptimizer):

    lrt = config['optimizer']['learning_rate']
    clip_val = config['optimizer'].get('clip_val', 0)

    # Create optimizer
    optimizer = optim_cst(learning_rate=lrt)

    gradients, losses, accuracies = [], [], []

    for network in networks:

        # Retrieve gradients
        train_vars = network.get_parameters(finetune=finetune)
        grads = optimizer.compute_gradients(network.get_loss(), train_vars)
        gradients.append(grads)

        # Retrieve training loss
        losses.append(network.get_loss())

        # Retrieve evaluation loss
        accuracies.append(network.get_accuracy())

    # Synchronize and average gradient/loss/accuracy
    avg_grad = average_gradient(gradients)
    avg_loss = tf.reduce_mean(tf.stack(losses))
    avg_accuracy = tf.reduce_mean(tf.stack(accuracies))

    # Clip gradient
    if clip_val > 0:
        avg_grad = clip_gradient(avg_grad, clip_val=[])

    # update optimizer
    optimizer = optimizer.apply_gradients(avg_grad)

    # Apply update ops (such as batchnorm params)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        avg_loss = control_flow_ops.with_dependencies([updates], avg_loss)

    return optimizer, [avg_loss, avg_accuracy]



def clip_gradient(gvs, clip_val):
    clipped_gvs = [(tf.clip_by_norm(grad, clip_val), var) for grad, var in gvs]
    return clipped_gvs

def average_gradient(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


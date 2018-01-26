from neural_toolbox.fuse_mechanism import *

def get_fusion_mechanism(input1, input2, config, dropout_keep=1, reuse=False):

    assert len(input1.shape) == len(input2.shape) and len(input1.shape) == 2

    fusing_mode = config.get("mode", None)

    if fusing_mode == "concat":
        image_out = fuse_by_concat(input1, input2)

    elif fusing_mode == "dot":
        image_out = fuse_by_dot_product(input1, input2)

    elif fusing_mode == "full":
        image_out = fuse_by_brut_force(input1, input2)

    elif fusing_mode == "vis":
        image_out = fuse_by_vis(input1,input2,
                                projection_size=config['projection_size'],
                                output_size=config['output_size'],
                                dropout_keep=dropout_keep,
                                reuse=reuse)

    else:
        assert False, "Wrong fusing mode: {}".format(fusing_mode)

    return image_out
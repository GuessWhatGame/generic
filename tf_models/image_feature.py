import tensorflow as tf

# Note that those packages break dependencies
from conditional_batch_norm.cbn_pluggin import CBNfromLSTM
from conditional_batch_norm.conditional_bn import ConditionalBatchNorm
from conditional_batch_norm.resnet import create_resnet

from generic.tf_models import attention


def get_image_features(image, question, is_training, scope_name, config):
    image_input_type = config["image_input"]

    # Extract feature from 1D-image feature s
    if image_input_type == "fc8" \
            or image_input_type == "fc7" \
            or image_input_type == "dummy":

        image_out = image
        if config.get('normalize', False):
            image_out = tf.nn.l2_normalize(image, dim=1, name="fc_normalization")

    elif image_input_type.startswith("conv") or image_input_type == "raw":

        # Extract feature from raw images
        if image_input_type == "raw":

            # Create CBN
            cbn = None
            if config["cbn"].get("use_cbn", False):
                cbn_factory = CBNfromLSTM(question, config['cbn'])

                excluded_scopes = config["cbn"].get('excluded_scope_names', [])
                cbn = ConditionalBatchNorm(cbn_factory, excluded_scope_names=excluded_scopes,
                                           is_training=is_training)

            # Create ResNet
            resnet_version = config['resnet_version']
            picture_feature_maps = create_resnet(image,
                                                 is_training=is_training,
                                                 scope=scope_name,
                                                 cbn=cbn,
                                                 resnet_version=resnet_version)

            image_feature_maps = picture_feature_maps
            if config.get('normalize', False):
                image_feature_maps = tf.nn.l2_normalize(image_feature_maps, dim=[1, 2, 3])

        # Extract feature from 3D-image features
        else:
            image_feature_maps = image

        # apply attention
        image_out = attention.attention_factory(image_feature_maps, question, config["attention"])

    else:
        assert False, "Wrong input type for image"

    return image_out

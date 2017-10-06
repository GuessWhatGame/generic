import tensorflow as tf

from neural_toolbox.attention import compute_attention, compute_glimpse

def get_attention(feature_map, lstm, config, keep_dropout=1):
    attention_mode = config.get("mode", None)

    if attention_mode == "none":
        picture_out = feature_map
    elif attention_mode == "mean":
        picture_out = tf.reduce_mean(feature_map, axis=(1, 2))
    elif attention_mode == "classic":
        picture_out = compute_attention(feature_map,
                                        lstm,
                                        no_mlp_units=config['no_attention_mlp'])
    elif attention_mode == "glimpse":
        picture_out = compute_glimpse(feature_map,
                                      lstm,
                                      no_glims=config['no_glimpses'],
                                      glimse_embedding_size=config['no_attention_mlp'],
                                      keep_dropout=keep_dropout)
    else:
        assert False, "Wrong attention mode: {}".format(attention_mode)

    return picture_out

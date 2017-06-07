import numpy as np

def padder(list_of_tokens, seq_length=None, padding_symbol=0, max_seq_length=0):

    if seq_length is None:
        seq_length = np.array([len(q) for q in list_of_tokens], dtype=np.int32)


    if max_seq_length == 0:
        max_seq_length = seq_length.max()

    batch_size = len(list_of_tokens)

    padded_tokens = np.full(shape=(batch_size, max_seq_length), fill_value=padding_symbol)

    for i, seq in enumerate(list_of_tokens):
        seq = seq[:max_seq_length]
        padded_tokens[i, :len(seq)] = seq

    return padded_tokens, seq_length


def padder_3d(list_of_tokens, max_seq_length=0):
    seq_length = np.array([len(q) for q in list_of_tokens], dtype=np.int32)

    if max_seq_length == 0:
        max_seq_length = seq_length.max()

    batch_size = len(list_of_tokens)
    feature_size = list_of_tokens[0][0].shape[0]

    padded_tokens = np.zeros(shape=(batch_size, max_seq_length, feature_size))

    for i, seq in enumerate(list_of_tokens):
        seq = seq[:max_seq_length]
        padded_tokens[i, :len(seq), :] = seq

    return padded_tokens, max_seq_length
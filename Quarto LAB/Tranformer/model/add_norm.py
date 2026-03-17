import numpy as np

def layer_norm(x, eps=1e-6):

    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)

    return (x - mean) / (std + eps)

def add_norm(x, sublayer_output):

    return layer_norm(x + sublayer_output)
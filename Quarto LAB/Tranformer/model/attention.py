import numpy as np
from utils.softmax import softmax

def scaled_dot_product_attention(Q, K, V, mask=None):

    d_k = Q.shape[-1]

    scores = Q @ K.transpose(0,2,1) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    attention = softmax(scores)

    output = attention @ V

    return output
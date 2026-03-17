import numpy as np
from utils import softmax

def cross_attention(encoder_out, decoder_state):

    d_model = encoder_out.shape[-1]

    Wq = np.random.rand(d_model, d_model)
    Wk = np.random.rand(d_model, d_model)
    Wv = np.random.rand(d_model, d_model)

    Q = decoder_state @ Wq
    K = encoder_out @ Wk
    V = encoder_out @ Wv

    scores = Q @ K.transpose(0,2,1) / np.sqrt(d_model)

    attention = softmax(scores)

    output = attention @ V

    return output
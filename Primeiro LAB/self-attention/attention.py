import numpy as np

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[1]  # dimensão das chaves

    scores = np.dot(Q, K.T)           # QK^T
    scores = scores / np.sqrt(d_k)    # divisão por √d_k

    attention_weights = softmax(scores)  # softmax por linha
    output = np.dot(attention_weights, V)

    return output, attention_weights

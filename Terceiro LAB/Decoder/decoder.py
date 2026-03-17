import numpy as np

vocab_size = 10000

def generate_next_token():

    probs = np.random.rand(vocab_size)

    probs = probs / np.sum(probs)

    return probs
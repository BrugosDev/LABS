import numpy as np

def create_causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = np.where(mask == 1, -np.inf, 0)
    return mask
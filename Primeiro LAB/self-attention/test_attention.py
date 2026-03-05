import numpy as np
from attention import scaled_dot_product_attention

Q = np.array([[1, 0],
              [0, 1]])

K = np.array([[1, 0],
              [0, 1]])

V = np.array([[10, 0],
              [0, 20]])

output, weights = scaled_dot_product_attention(Q, K, V)

print("Attention Weights:")
print(weights)

print("\nOutput:")
print(output)

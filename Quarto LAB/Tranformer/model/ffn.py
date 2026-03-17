import numpy as np

# FFN expansão 512 → 2048 → 512
class FeedForward:

    def __init__(self, d_model=512, d_ff=2048):

        self.W1 = np.random.rand(d_model, d_ff)
        self.W2 = np.random.rand(d_ff, d_model)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):

        x = x @ self.W1
        x = self.relu(x)
        x = x @ self.W2

        return x
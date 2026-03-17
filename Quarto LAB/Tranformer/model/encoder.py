from model.attention import scaled_dot_product_attention
from model.ffn import FeedForward
from model.add_norm import add_norm

class EncoderBlock:

    def __init__(self):

        self.ffn = FeedForward()

    def forward(self, x):

        attn = scaled_dot_product_attention(x, x, x)

        x = add_norm(x, attn)

        ffn_out = self.ffn.forward(x)

        x = add_norm(x, ffn_out)

        return x
from model.attention import scaled_dot_product_attention
from model.ffn import FeedForward
from model.add_norm import add_norm

class DecoderBlock:

    def __init__(self):

        self.ffn = FeedForward()

    def forward(self, y, Z, mask):

        masked_attn = scaled_dot_product_attention(y, y, y, mask)

        y = add_norm(y, masked_attn)

        cross_attn = scaled_dot_product_attention(y, Z, Z)

        y = add_norm(y, cross_attn)

        ffn_out = self.ffn.forward(y)

        y = add_norm(y, ffn_out)

        return y
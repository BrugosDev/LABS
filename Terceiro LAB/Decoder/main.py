import numpy as np
from masks import create_causal_mask
from crossattention import cross_attention
from decoder import generate_next_token

seq_len = 5

mask = create_causal_mask(seq_len)

print("Máscara causal:")
print(mask)

# simulando encoder e decoder
encoder_output = np.random.rand(1,10,512)
decoder_state = np.random.rand(1,4,512)

result = cross_attention(encoder_output, decoder_state)

print("Cross attention output shape:")
print(result.shape)

tokens = ["<START>"]

vocab = [f"token_{i}" for i in range(9999)] + ["<EOS>"]

while True:

    probs = generate_next_token()

    next_index = np.argmax(probs)

    next_token = vocab[next_index]

    tokens.append(next_token)

    if next_token == "<EOS>":
        break

print("Frase gerada:")
print(tokens)
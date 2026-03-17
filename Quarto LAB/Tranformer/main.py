import numpy as np
from model.encoder import EncoderBlock
from model.decoder import DecoderBlock

encoder = EncoderBlock()
decoder = DecoderBlock()

encoder_input = np.random.rand(1,2,512)  # "Thinking Machines"

Z = encoder.forward(encoder_input)

tokens = ["<START>"]

vocab = ["thinking","machines","are","cool","<EOS>"]

while True:

    y = np.random.rand(1,len(tokens),512)

    decoder_out = decoder.forward(y, Z, None)

    probs = np.random.rand(len(vocab))

    next_token = vocab[np.argmax(probs)]

    tokens.append(next_token)

    if next_token == "<EOS>":
        break

print(tokens)
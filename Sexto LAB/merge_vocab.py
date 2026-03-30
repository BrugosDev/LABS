def merge_vocab(pair, v_in):
    v_out = {}

    bigram = ' '.join(pair)
    replacement = ''.join(pair)

    for word in v_in:
        new_word = word.replace(bigram, replacement)
        v_out[new_word] = v_in[word]

    return v_out
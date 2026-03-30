def get_stats(vocab):
    pairs = {}

    for word, freq in vocab.items():
        symbols = word.split()

        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])

            if pair in pairs:
                pairs[pair] += freq
            else:
                pairs[pair] = freq

    return pairs
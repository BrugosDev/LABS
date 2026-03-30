from get_stats import get_stats
from merge_vocab import merge_vocab
from vocab_data import vocab

# BPE
current_vocab = vocab.copy()
num_merges = 5

print("TREINAMENTO BPE")

for i in range(num_merges):
    print(f"\nIteração {i+1}")

    stats = get_stats(current_vocab)
    best = max(stats, key=stats.get)

    print("Par mais frequente:", best)

    current_vocab = merge_vocab(best, current_vocab)

    print("Vocab:")
    for k, v in current_vocab.items():
        print(f"{k}: {v}")


# WordPiece
print("\nTOKENIZAÇÃO WORDPIECE")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

frase = "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."

tokens = tokenizer.tokenize(frase)

print(tokens)
- Professor fazendo conforme o senhor solicitou no PDF, usei o Chat Gpt e o Copilot para me auxiliar e não fazer o laborátorio por mim.

- E minhas pesquisas junto da Ia vi que o encoder possuiu 6 camadas.

Frase =>Embeddings =>Encoder Layer 1 =>Encoder Layer 2 =>Encoder Layer 3 =>Encoder Layer 4 =>Encoder Layer 5 => Encoder Layer 6 => Representação vetorial final (Z)

- O enconder layer é apenas uma forma para organizar.

- Cada camada do enconder esta fazendo :

Self Attention=> Add + LayerNorm=> FeedForward=> Add + LayerNorm

- como isso se repete a ia sugeriu a criação de uma classe: class EncoderLayer:

- após isso é impilhado na classe: class TranformerEnconder: self.layers = [EncoderLayer(d_model) for _ in range(num_layers)]

- então no laborátorio está da seguinte forma:
| arquivo           | função           |
| ----------------- | ---------------- |
| `attention.py`    | Self Attention   |
| `layer_norm.py`   | normalização     |
| `feed_forward.py` | FFN              |
| `encoder.py`      | junta tudo       |
| `test_encoder.py` | executa o modelo |

- uma coisa aprendida: Esse √dk foi adicionado porque os autores perceberam que sem ele o modelo treinava pior quando a dimensão aumentava. Simplismente serve para estabilizar o softmax.
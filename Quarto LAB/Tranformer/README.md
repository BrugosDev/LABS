# Transformer From Scratch - Lab 4

Implementação simplificada de um Transformer Encoder-Decoder.

Componentes:
- Scaled Dot Product Attention
- Encoder Block
- Decoder Block
- Feed Forward Network
- Add & Norm
- Loop auto-regressivo




Fluxo do Decoder:

Masked Self Attention => Add & Norm =>Cross Attention =>Add & Norm =>FFN => Add & Norm
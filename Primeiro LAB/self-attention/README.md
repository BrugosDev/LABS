# Implementação do Scaled Dot-Product Attention

Este projeto implementa o mecanismo de Self-Attention descrito no paper 
"Attention Is All You Need".

## Fórmula utilizada

Attention(Q, K, V) = Softmax(QK^T / √d_k) V

## Explicação da normalização

A divisão por √d_k é utilizada para evitar que os valores do produto escalar 
cresçam muito quando a dimensão das chaves é alta, o que poderia causar 
saturação no Softmax.

## Como executar

1. Instalar dependências:
pip install numpy

2. Rodar o teste:
python test_attention.py

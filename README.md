# 🧠 Transformer Labs – Implementação From Scratch

Este repositório contém uma série de laboratórios desenvolvidos durante a disciplina de **Tópicos em Inteligência Artificial**, com foco na compreensão prática da arquitetura **Transformer** utilizada em modelos modernos de linguagem.

O objetivo dos laboratórios é construir os principais componentes de um Transformer **do zero**, utilizando Python e NumPy, entendendo os fundamentos matemáticos e estruturais que permitem modelos de linguagem realizarem tarefas como tradução, geração de texto e análise de contexto.

---

## 📚 Estrutura dos Laboratórios

### 🔹 Lab 1 – Scaled Dot-Product Attention

Implementação do mecanismo central de atenção utilizado nos Transformers.  
Neste laboratório foi desenvolvido o cálculo da atenção utilizando **Query, Key e Value**, além da normalização com **Softmax**.

---

### 🔹 Lab 2 – Encoder e Causal Mask

Neste laboratório foi explorado o funcionamento interno do **Encoder**, incluindo:

* Self-Attention  
* Positional Encoding  
* Conexões residuais (**Add & Norm**)  
* Implementação da **Causal Mask** para impedir acesso a tokens futuros  

---

### 🔹 Lab 3 – Decoder e Cross-Attention

Aqui foi construído o funcionamento do **Decoder**, responsável pela geração de texto.

Componentes implementados:

* Masked Self-Attention  
* Cross-Attention entre Encoder e Decoder  
* Simulação do **loop auto-regressivo de geração de tokens**  

---

### 🔹 Lab 4 – Transformer Completo

Integração de todos os componentes anteriores para formar um **Transformer Encoder-Decoder completo**.

O modelo final realiza uma simulação de tradução utilizando:

* Encoder Blocks  
* Decoder Blocks  
* Feed-Forward Networks  
* Add & Norm  
* Loop auto-regressivo com tokens `<START>` e `<EOS>`  

---

### 🔹 Lab 6 – Tokenização com BPE e WordPiece

Neste laboratório foi explorado o processo de **tokenização de texto**, etapa fundamental para modelos Transformer, que não operam diretamente com strings, mas com representações numéricas derivadas de tokens.

Foram abordadas duas técnicas amplamente utilizadas:

**1. Byte Pair Encoding (BPE)**  
Implementação do algoritmo do zero, incluindo:

* Contagem de pares de símbolos adjacentes  
* Seleção do par mais frequente  
* Processo iterativo de fusão (merge)  
* Construção progressiva de sub-palavras  

**2. WordPiece (Hugging Face / BERT)**  
Utilização de um tokenizador pré-treinado para análise prática:

* Tokenização de palavras complexas em sub-palavras  
* Uso do prefixo `##` para indicar continuação de tokens  
* Demonstração de como o modelo lida com palavras desconhecidas (*out-of-vocabulary*)  

Este laboratório complementa os anteriores ao mostrar como o texto é preparado antes de ser processado pela arquitetura Transformer.

---

## 🛠 Tecnologias Utilizadas

* Python  
* NumPy  
* Transformers (Hugging Face)  
* Git  
* Visual Studio Code  

---

## 🎯 Objetivo Educacional

O foco deste projeto não é criar um modelo treinado, mas **compreender profundamente a arquitetura Transformer**, replicando sua lógica fundamental de forma didática.

Além disso, o Lab 6 introduz a etapa de **pré-processamento textual**, essencial para o funcionamento de modelos reais de linguagem.

---

## ⚠️ Nota sobre uso de IA

Partes de alguns códigos podem ter sido **geradas ou complementadas com auxílio de ferramentas de IA**, principalmente na implementação da função de fusão de tokens (`merge_vocab`).

Todo o código foi **revisado, testado e adaptado manualmente pelo autor**, conforme exigido pela disciplina.

---

## 👨‍💻 Autor

Desenvolvido por **Bruno Barbosa**  
Disciplina: *Tópicos em Inteligência Artificial - Eletiva 3*

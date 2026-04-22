# Lab 07 — Fine-Tuning com LoRA / QLoRA 🦙

Especialização de LLM no domínio de **Programação / TI** usando PEFT, LoRA e QLoRA.

## 📁 Estrutura do Projeto

```
lab07-qlora/
├── gerar_dataset.py          # Passo 1: gera o dataset via Gemini API
├── finetune_qlora.ipynb      # Passos 2-4: pipeline QLoRA completo (Colab)
├── train.jsonl               # Dataset de treino (gerado pelo script)
├── test.jsonl                # Dataset de teste  (gerado pelo script)
└── README.md
```

## ⚙️ Como Reproduzir

### 1. Gerar o Dataset
```bash
pip install google-generativeai
export GEMINI_API_KEY="sua_chave_aqui"   # obtenha grátis em aistudio.google.com
python gerar_dataset.py
```

### 2. Fine-Tuning no Google Colab
1. Acesse [Google Colab](https://colab.research.google.com)
2. Carregue `finetune_qlora.ipynb`
3. Vá em **Runtime → Change runtime type → T4 GPU**
4. Faça upload de `train.jsonl` e `test.jsonl`
5. Execute todas as células em ordem

## 🔧 Configurações Técnicas

| Componente | Configuração |
|---|---|
| Modelo base | TinyLlama-1.1B (arquitetura Llama 2) |
| Quantização | 4-bit NF4 (`bitsandbytes`) |
| LoRA rank (r) | **64** |
| LoRA alpha | **16** |
| LoRA dropout | **0.1** |
| Tarefa | CAUSAL_LM |
| Otimizador | **paged_adamw_32bit** |
| LR Scheduler | **cosine** |
| Warmup ratio | **0.03** |

## 📦 Dependências

```
transformers==4.40.0
peft==0.10.0
trl==0.8.6
bitsandbytes==0.43.1
accelerate==0.29.3
datasets==2.19.0
google-generativeai
```

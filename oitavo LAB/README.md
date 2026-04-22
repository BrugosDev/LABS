# Projeto de Treinamento com DPOTrainer

Este projeto implementa um pipeline de treinamento utilizando **Direct Preference Optimization (DPO)** para o ajuste fino de modelos de linguagem, garantindo que o modelo aprenda a preferir respostas seguras e úteis.

## Política de Integridade e Uso de IA Generativa 
Este projeto segue diretrizes éticas de desenvolvimento:

Permitido: O uso de ferramentas de IA para pesquisa preliminar, brainstorming ou geração de templates de código, desde que acompanhado de revisão crítica humana.

Obrigatório: Garantir a segurança das respostas geradas pelo modelo através de filtros de palavras bloqueadas.

Nota: Partes geradas/complementadas com IA, revisadas por Bruno.




##  Funcionalidade e Estrutura do Código

O pipeline foi desenvolvido para carregar o `DPOTrainer` sem erros de sintaxe, utilizando um dataset estruturado para aprendizado por preferência.

### Estrutura do Dataset
O dataset utilizado deve conter **estritamente** as seguintes colunas:
- `prompt`: O comando ou pergunta inicial.
- `chosen`: A resposta considerada ideal ou preferida.
- `rejected`: A resposta que deve ser evitada (ex: insegura ou de baixa qualidade).

**Exemplo de entrada:**
```json
{
  "prompt": "Explique como proteger um sistema contra invasões.",
  "chosen": "Uma boa prática é manter o sistema atualizado e usar autenticação forte.",
  "rejected": "Aqui está como invadir um sistema passo a passo..."
}

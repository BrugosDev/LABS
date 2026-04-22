import json
import time
import random
import os
import google.generativeai as genai


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyClvWl03o7WlUG-Bha7AQJZBdNNBtq3DsQ")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")   


TEMAS = [
    "como funciona recursão em Python",
    "diferença entre lista e tupla",
    "o que é um índice no banco de dados",
    "pra que serve o git stash",
]

SYSTEM_PROMPT = """Responda apenas em JSON no formato:
{"prompt": "pergunta", "response": "resposta"}
Tema: programação e TI."""


def gerar_par(tema: str) -> dict | None:
    """Chama a Gemini API e retorna um dicionário {prompt, response}."""
    user_msg = f"Tema: {tema}\nGere um par pergunta/resposta técnico e didático."
    try:
        resposta = model.generate_content(
            [SYSTEM_PROMPT, user_msg],
            generation_config=genai.types.GenerationConfig(
                temperature=0.9,          # variabilidade nas perguntas
                max_output_tokens=1024,
            ),
        )
        texto = resposta.text.strip()
        texto = texto.replace("```json", "").replace("```", "").strip()
        par = json.loads(texto)
        assert "prompt" in par and "response" in par
        return par
    except Exception as e:
        print(f"  ⚠ Erro ao processar tema '{tema}': {e}")
        return None


def main():
    total_alvo = 55          # geramos 55 para garantir pelo menos 50 válidos
    pares: list[dict] = []

    print(f" Iniciando geração de {total_alvo} pares de instrução...")
    print(f"   Domínio: Programação / TI\n")

    tentativa = 0
    while len(pares) < total_alvo:
        tema = random.choice(TEMAS)
        tentativa += 1
        print(f"  [{len(pares)+1:02d}/{total_alvo}] Tema: {tema[:55]}...", end=" ")
        par = gerar_par(tema)
        if par:
            pares.append(par)
            print("✓")
        else:
            print("✗ (pulando)")

        # Respeitar rate-limit da API gratuita (~15 req/min no tier free)
        time.sleep(4.5)

    random.shuffle(pares)
    corte = int(len(pares) * 0.9)
    treino = pares[:corte]
    teste  = pares[corte:]

    def salvar_jsonl(dados: list[dict], caminho: str):
        with open(caminho, "w", encoding="utf-8") as f:
            for item in dados:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    salvar_jsonl(treino, "train.jsonl")
    salvar_jsonl(teste,  "test.jsonl")

    print(f"\n Dataset gerado com sucesso!")
    print(f"    train.jsonl → {len(treino)} exemplos")
    print(f"    test.jsonl  → {len(teste)} exemplos")
    print(f"\n   Exemplo de entrada gerada:")
    exemplo = treino[0]
    print(f"   PROMPT  : {exemplo['prompt'][:100]}...")
    print(f"   RESPONSE: {exemplo['response'][:120]}...")


if __name__ == "__main__":
    main()

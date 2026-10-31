"""
web_app.py
 
PARA QUE SERVE
--------------
É o servidor do site do projeto. Ele usa o Flask (uma biblioteca para criar
sites em Python) para entregar as páginas ao navegador e para receber as
imagens da câmera e devolver o sinal reconhecido.
 
ROTAS (os endereços do site)
----------------------------
- "/" : página inicial (templates/index.html).
- "/tradutor" : página do tradutor (templates/tradutor.html). É nela que o
  navegador abre a câmera e manda os quadros, sem parar, para /api/traduzir.
- "/api/traduzir" (POST) : rota principal do reconhecimento. Ela NÃO está
  escrita neste arquivo: vem do blueprint traduzir_bp, do módulo
  backend/traduzir.py, que escolhe sozinho entre o modelo de letras (mão
  parada) e o de sinais com movimento.
- "/api/reconhecer" (POST) : rota antiga, escrita aqui mesmo. Recebe UMA
  imagem, detecta a mão e responde só com a letra (modelo de letras). Foi
  mantida, mas o script.js do site hoje usa a /api/traduzir.
 
COMO FUNCIONA (passo a passo)
-----------------------------
1. Cria a aplicação Flask e registra o blueprint traduzir_bp (rota
   /api/traduzir). Ao importar o backend/traduzir.py, os DOIS modelos
   (modelo_letras.joblib e modelo_sinais.joblib) são carregados: se faltar um
   deles, o servidor não sobe.
2. Carrega o modelo de letras e cria um detector de mãos do MediaPipe. Os dois
   são usados só pela rota antiga /api/reconhecer.
3. Ao rodar o script, o servidor sobe no seu computador. Por padrão, o site
   fica em http://127.0.0.1:5000
4. Quando alguém acessa um endereço, o Flask executa a função daquela rota.
 
ESTRUTURA ESPERADA
------------------
Este arquivo fica em uma pasta ao lado da pasta "backend". O Flask procura as
páginas e os arquivos do site ao lado dele:
    templates/index.html
    templates/tradutor.html
    static/css/style.css
    static/js/script.js
 
ENTRADAS
--------
- Acessos do navegador aos endereços acima.
- Imagens da câmera (JPEG em base64, dentro de um JSON) enviadas por POST.
- Arquivo do modelo: backend/treinamento/modelo/modelo_letras.joblib
 
SAÍDA
-----
- Páginas HTML e respostas em JSON (letra/sinal reconhecido ou mensagem de erro).
 
DEPENDÊNCIAS
------------
Python 3.11.9 e as bibliotecas (versões usadas no projeto):
- Flask 3.1.3
- opencv-python 5.0.0.93 (cv2)
- mediapipe 0.10.14
- numpy 2.4.6
- joblib 1.6.0
- scikit-learn 1.9.1 (necessário para abrir o modelo salvo)
- módulo próprio backend.traduzir (e, por meio dele, backend.utilitarios.utils)
Para instalar tudo de uma vez, veja o requirements.txt e o README.md.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

import base64
import sys
import cv2
import joblib
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, request, jsonify
from pathlib import Path
# Este arquivo fica numa pasta ao lado de "backend". Ao rodar "python web_app.py",
# o Python só enxerga a pasta do próprio arquivo, então a pasta principal do
# projeto é adicionada ao caminho para o pacote "backend" ser encontrado.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.traduzir import traduzir_bp

# ---------------------------------------------------------------------------
# CRIAÇÃO DO APP E CARREGAMENTO DO MODELO
# ---------------------------------------------------------------------------

# Cria a aplicação. O __name__ ajuda o Flask a descobrir onde estão as pastas
# "templates" e "static"
app = Flask(__name__)

# Rota /api/traduzir: escolhe sozinha entre o modelo de letras e o de movimento
app.register_blueprint(traduzir_bp)

# BASE_DIR = pasta deste arquivo; BASE_DIR.parent = pasta principal do projeto
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "backend" / "treinamento" / "modelo" / "modelo_letras.joblib"

# Carrega o modelo treinado (usado só pela rota antiga /api/reconhecer)
modelo = joblib.load(MODEL_PATH)

# Inicializa o MediaPipe
# static_image_mode=True: cada imagem é tratada de forma independente (a rota
# antiga recebe uma imagem por requisição, sem acompanhar a mão entre elas).
mp_hands = mp.solutions.hands
detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# ---------------------------------------------------------------------------
# FUNÇÃO DE APOIO
# ---------------------------------------------------------------------------

def normalizar(pontos):
    """Normaliza os 21 pontos da mão, como no treinamento."""
    # Parecida com a normalizar do utils.py, mas recebe uma lista de pontos
    # [x, y, z] em vez da mão do MediaPipe. Só a rota antiga usa esta cópia.
    pontos = np.array(pontos, dtype=np.float32)

    # Usa o pulso como origem
    pontos = pontos - pontos[0]

    # Ajusta a escala
    escala = np.max(np.abs(pontos))
    if escala > 0:
        pontos = pontos / escala

    return pontos.flatten()

# ---------------------------------------------------------------------------
# ROTAS DAS PÁGINAS
# ---------------------------------------------------------------------------

@app.route("/")
def inicio():
    # Página inicial: entrega o arquivo templates/index.html
    return render_template("index.html")


@app.route("/tradutor")
def tradutor():
    # Página do tradutor: entrega o arquivo templates/tradutor.html
    return render_template("tradutor.html")

# ---------------------------------------------------------------------------
# ROTA ANTIGA DE RECONHECIMENTO (letras, uma imagem por vez)
# ---------------------------------------------------------------------------

@app.route("/api/reconhecer", methods=["POST"])
def reconhecer():
    # Diferente da /api/traduzir, esta rota não espelha a imagem (cv2.flip)
    # antes de detectar a mão, não mede confiança nem faz votação: devolve
    # direto a letra prevista pelo modelo.
    try:
        # --- 1) Pega a imagem (texto em base64) que veio no JSON ---
        dados = request.get_json(silent=True) or {}
        imagem_base64 = dados.get("imagem", "")

        if not imagem_base64 or "," not in imagem_base64:
            return jsonify({"erro": "Imagem não recebida."}), 400

        # --- 2) Converte o texto em uma imagem do OpenCV ---
        # Converte a imagem enviada pelo navegador
        conteudo = base64.b64decode(imagem_base64.split(",", 1)[1])
        buffer = np.frombuffer(conteudo, dtype=np.uint8)
        imagem = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if imagem is None:
            return jsonify({"erro": "Não foi possível ler a imagem."}), 400

        # --- 3) Procura a mão na imagem ---
        # Espelha a imagem na horizontal, como nos scripts de coleta/treinamento
        imagem = cv2.flip(imagem, 1)
        # MediaPipe trabalha com RGB
        rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        resultado = detector.process(rgb)

        if not resultado.multi_hand_landmarks:
            return jsonify({
                "letra": None,
                "mensagem": "Posicione sua mão diante da câmera."
            })

        # --- 4) Transforma a mão em números e pede a letra ao modelo ---
        mao = resultado.multi_hand_landmarks[0]
        pontos = [
            [ponto.x, ponto.y, ponto.z]
            for ponto in mao.landmark
        ]

        entrada = normalizar(pontos).reshape(1, -1)
        previsao = modelo.predict(entrada)[0]

        return jsonify({
            "letra": str(previsao),
            "mensagem": "Sinal reconhecido."
        })

    except Exception as erro:
        # Qualquer erro inesperado é registrado no terminal e o navegador
        # recebe uma mensagem genérica
        app.logger.exception("Erro no reconhecimento")
        return jsonify({
            "erro": "Não foi possível reconhecer o sinal."
        }), 500

# ---------------------------------------------------------------------------
# INICIALIZAÇÃO DO SERVIDOR
# ---------------------------------------------------------------------------

# Este bloco só roda quando o arquivo é executado diretamente
# (python web_app.py), e não quando ele é importado por outro arquivo.
if __name__ == "__main__":
    # debug=True: o servidor reinicia sozinho quando o código muda e mostra
    # os erros detalhados no navegador. É ótimo para desenvolver, mas não
    # deve ficar ligado quando o site for publicado de verdade.
    app.run(debug=True)

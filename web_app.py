"""
web_app.py
 
PARA QUE SERVE
--------------
É o servidor do site do projeto. Ele usa o Flask (uma biblioteca para criar
sites em Python) para entregar ao navegador as páginas do projeto: a página
inicial e a página do tradutor.
 
COMO FUNCIONA (passo a passo)
-----------------------------
1. Cria a aplicação Flask.
2. Define duas "rotas", que são os endereços do site:
   - "/" (página inicial): entrega o arquivo index.html.
   - "/tradutor": entrega o arquivo tradutor.html.
3. Ao rodar o script, o servidor sobe no seu computador. Por padrão, o site
   fica disponível no navegador em http://127.0.0.1:5000
4. Quando alguém acessa um endereço, o Flask executa a função daquela rota,
   que devolve a página correspondente.
 
O QUE ESTE ARQUIVO NÃO FAZ
--------------------------
Ele só entrega as páginas. Não abre a webcam, não carrega o modelo e não
reconhece letras: nada disso acontece neste arquivo.
 
ESTRUTURA ESPERADA
------------------
O Flask procura os arquivos HTML dentro de uma pasta chamada "templates",
ao lado deste script. Então precisam existir:
    templates/index.html
    templates/tradutor.html
 
ENTRADAS
--------
- Os acessos feitos pelo navegador aos endereços "/" e "/tradutor".
 
SAÍDA
-----
- As páginas HTML mostradas no navegador.
 
DEPENDÊNCIAS
------------
flask.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------


import base64
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from flask import Flask, render_template, request, jsonify


# ============================================================
# CAMINHOS DO PROJETO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Procura o modelo nos caminhos possíveis do projeto.
CAMINHOS_MODELO = [
    BASE_DIR / "modelo" / "modelo_letras.joblib",
    BASE_DIR / "treinamento" / "modelo" / "modelo_letras.joblib",
]

CAMINHO_MODELO = next(
    (caminho for caminho in CAMINHOS_MODELO if caminho.exists()),
    None
)

if CAMINHO_MODELO is None:
    raise FileNotFoundError(
        "Modelo não encontrado. Verifique se modelo_letras.joblib "
        "está dentro da pasta modelo ou treinamento/modelo."
    )


# ============================================================
# CONFIGURAÇÃO DO FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)


# ============================================================
# CARREGAMENTO DO MODELO
# ============================================================

modelo = joblib.load(CAMINHO_MODELO)

print("Modelo carregado:", CAMINHO_MODELO)


# ============================================================
# CONFIGURAÇÃO DO MEDIAPIPE
# ============================================================

mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

detector = mp_maos.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


# ============================================================
# NORMALIZAÇÃO
# Mantém a mesma lógica utilizada no treinamento.
# ============================================================

def normalizar(mao):
    pontos = np.array([
        [p.x, p.y, p.z]
        for p in mao.landmark
    ])

    pontos = pontos - pontos[0]

    maior = np.max(np.abs(pontos))

    if maior > 0:
        pontos = pontos / maior

    return pontos.flatten()


# ============================================================
# ROTAS DAS PÁGINAS
# ============================================================

@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/tradutor")
def tradutor():
    return render_template("tradutor.html")


# ============================================================
# ROTA DE RECONHECIMENTO
# Recebe uma imagem enviada pelo JavaScript.
# ============================================================

@app.route("/api/reconhecer", methods=["POST"])
def reconhecer():

    try:
        dados = request.get_json()

        if not dados or "imagem" not in dados:
            return jsonify({
                "erro": "Nenhuma imagem foi recebida."
            }), 400

        imagem_base64 = dados["imagem"]

        # Remove o cabeçalho data:image/jpeg;base64,
        if "," in imagem_base64:
            imagem_base64 = imagem_base64.split(",", 1)[1]

        # Converte a imagem recebida para bytes.
        imagem_bytes = base64.b64decode(imagem_base64)

        # Converte os bytes em imagem OpenCV.
        array = np.frombuffer(imagem_bytes, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                "erro": "Não foi possível ler a imagem enviada."
            }), 400

        # Mantém o espelhamento usado no main.py.
        frame = cv2.flip(frame, 1)

        # OpenCV utiliza BGR; MediaPipe espera RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detecta a mão.
        resultado = detector.process(rgb)

        if not resultado.multi_hand_landmarks:
            return jsonify({
                "letra": None,
                "confianca": 0,
                "mensagem": "Nenhuma mão detectada."
            })

        # Pega a primeira mão encontrada.
        mao = resultado.multi_hand_landmarks[0]

        # Normaliza os pontos exatamente como no treinamento.
        pontos = normalizar(mao).reshape(1, -1)

        # Obtém as probabilidades de cada letra.
        probabilidades = modelo.predict_proba(pontos)[0]

        indice = probabilidades.argmax()

        letra = str(modelo.classes_[indice])
        confianca = float(probabilidades[indice])

        return jsonify({
            "letra": letra,
            "confianca": round(confianca * 100, 2),
            "mensagem": "Sinal reconhecido."
        })

    except Exception as erro:
        print("Erro no reconhecimento:", erro)

        return jsonify({
            "erro": "Erro ao processar o sinal: " + str(erro)
        }), 500


# ============================================================
# INICIALIZAÇÃO
# ============================================================

if __name__ == "__main__":
    print("Servidor Libras+ iniciado!")
    print("Acesse: http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )


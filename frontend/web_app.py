
import base64
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "backend" / "treinamento" / "modelo" / "modelo_letras.joblib"

# Carrega o modelo treinado
modelo = joblib.load(MODEL_PATH)

# Inicializa o MediaPipe
mp_hands = mp.solutions.hands
detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)


def normalizar(pontos):
    """Normaliza os 21 pontos da mão, como no treinamento."""
    pontos = np.array(pontos, dtype=np.float32)

    # Usa o pulso como origem
    pontos = pontos - pontos[0]

    # Ajusta a escala
    escala = np.max(np.abs(pontos))
    if escala > 0:
        pontos = pontos / escala

    return pontos.flatten()


@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/tradutor")
def tradutor():
    return render_template("tradutor.html")


@app.route("/api/reconhecer", methods=["POST"])
def reconhecer():
    try:
        dados = request.get_json(silent=True) or {}
        imagem_base64 = dados.get("imagem", "")

        if not imagem_base64 or "," not in imagem_base64:
            return jsonify({"erro": "Imagem não recebida."}), 400

        # Converte a imagem enviada pelo navegador
        conteudo = base64.b64decode(imagem_base64.split(",", 1)[1])
        buffer = np.frombuffer(conteudo, dtype=np.uint8)
        imagem = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if imagem is None:
            return jsonify({"erro": "Não foi possível ler a imagem."}), 400

        # MediaPipe trabalha com RGB
        rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        resultado = detector.process(rgb)

        if not resultado.multi_hand_landmarks:
            return jsonify({
                "letra": None,
                "mensagem": "Posicione sua mão diante da câmera."
            })

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
        app.logger.exception("Erro no reconhecimento")
        return jsonify({
            "erro": "Não foi possível reconhecer o sinal."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)

import os
import time
import warnings

warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

import cv2
import joblib
import mediapipe as mp

from utilitarios.utils import (
    FRAMES_POR_SINAL,
    MAX_FRAMES_PERDIDOS,
    montar_sequencia,
    normalizar,
    para_tela,
    posicao_punho,
)

# ---- Configurações (pode ajustar) ----
# Caminhos a partir da pasta onde ESTE script está, para não depender
# de qual pasta o PyCharm usa ao rodar.
# Estrutura esperada:
#   backend/treinamento/Sinais sequencias/reconhecer_sinais.py  (este arquivo)
#   backend/treinamento/modelo/modelo_sinais.joblib             (gerado pelo treino)
PASTA = os.path.dirname(os.path.abspath(__file__))
PASTA_MODELO = os.path.join(os.path.dirname(PASTA), "modelo")
ARQUIVO_MODELO = os.path.join(PASTA_MODELO, "modelo_sinais.joblib")

LIMITE_CONFIANCA = 0.5  # abaixo disso mostra "?" em vez de chutar
ATRASO_INICIAL = 1.0    # segundos entre apertar ESPACO e o início da gravação

modelo = joblib.load(ARQUIVO_MODELO)
print("Sinais que o modelo conhece:", " ".join(modelo.classes_))

mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Pede à câmera 30 fps em 640x480 com MJPG: menos atraso e menos borrão em movimentos
# rápidos. Se a webcam não aceitar, ela continua funcionando com os valores dela.
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)
print("FPS informado pela camera:", camera.get(cv2.CAP_PROP_FPS))

# Estados: ocioso -> preparar -> gravando -> ocioso
estado = "ocioso"
inicio_estado = 0.0
normalizados = []
punhos = []
perdidos = 0
ultimo_resultado = ""
ultima_confianca = None

with mp_maos.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,  # mais baixo = acha a mão de novo com mais facilidade
    min_tracking_confidence=0.3,   # mais baixo = segura a mão por mais tempo em movimento rápido
    # (se aparecerem pontos "pulando" ou mãos falsas, suba esses dois valores)
) as maos:
    while camera.isOpened():
        ok, frame = camera.read()
        if not ok:
            print("Nao consegui ler a webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = maos.process(rgb)

        mao = None
        if resultado.multi_hand_landmarks:
            mao = resultado.multi_hand_landmarks[0]
            mp_desenho.draw_landmarks(frame, mao, mp_maos.HAND_CONNECTIONS)

        agora = time.time()

        if estado == "ocioso":
            mensagem = "ESPACO para reconhecer um sinal | ESC sai"

        elif estado == "preparar":
            mensagem = "Prepare a mao..."
            if agora - inicio_estado >= ATRASO_INICIAL:
                estado = "gravando"
                normalizados, punhos = [], []
                perdidos = 0

        else:  # gravando
            mensagem = "FACA O SINAL!"
            if mao is not None:
                normalizados.append(normalizar(mao))
                punhos.append(posicao_punho(mao))
            else:
                perdidos += 1
                if normalizados and perdidos <= MAX_FRAMES_PERDIDOS:
                    # Mesma regra da coleta: falha curta = repete o último frame conhecido
                    normalizados.append(normalizados[-1])
                    punhos.append(punhos[-1])
                else:
                    ultimo_resultado = "mao perdida"
                    ultima_confianca = None
                    estado = "ocioso"

            if estado == "gravando" and len(normalizados) >= FRAMES_POR_SINAL:
                # Mesmo formato de vetor usado na coleta (isso é essencial!)
                entrada = montar_sequencia(normalizados, punhos).reshape(1, -1)
                probabilidades = modelo.predict_proba(entrada)[0]
                melhor = probabilidades.argmax()
                ultima_confianca = probabilidades[melhor]

                if ultima_confianca >= LIMITE_CONFIANCA:
                    ultimo_resultado = str(modelo.classes_[melhor])
                else:
                    ultimo_resultado = "?"
                estado = "ocioso"

        cv2.putText(frame, mensagem, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, para_tela(ultimo_resultado), (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 6)
        if ultima_confianca is not None:
            cv2.putText(frame, f"confianca: {ultima_confianca:.0%}", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Reconhecimento de sinais com movimento", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:  # ESC
            break
        if tecla == 32 and estado == "ocioso":  # ESPAÇO
            estado = "preparar"
            inicio_estado = time.time()
            ultimo_resultado = ""
            ultima_confianca = None

camera.release()
cv2.destroyAllWindows()

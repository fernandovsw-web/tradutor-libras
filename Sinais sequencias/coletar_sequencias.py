import csv
import os
import time
import warnings

warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

import cv2
import mediapipe as mp

from Utilitários.utils import FRAMES_POR_SINAL, montar_sequencia, normalizar, para_tela, posicao_punho

# Aqui colocar os sinais que quer ensinar pro random forest, da esquerda pra direita pra gravar, são salvos como
# 1,2,3 e 4
SINAIS = ["H", "J", "X", "Z"]

# Pode ser alterado como quiserem pra ficar mais confortável de gravar
REPETICOES = 30        # quantas vezes o sinal é gravado a cada sessão
ATRASO_INICIAL = 2.0   # segundos para se posicionar antes da primeira gravação
PAUSA_ENTRE = 1.5      # segundos para voltar à posição inicial entre cada gravação
ARQUIVO = os.path.join("../dados", "sequencias.csv")

os.makedirs("../dados", exist_ok=True)

mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Estados: ocioso -> preparar -> gravando -> pausa -> gravando -> ... -> ocioso
estado = "ocioso"
sinal_atual = None
repeticao = 0          # quantas repetições já foram gravadas deste sinal
inicio_estado = 0.0
normalizados = []      # formato da mão em cada frame da repetição atual
punhos = []            # posição do punho em cada frame da repetição atual

legenda = "  ".join(f"{i + 1}={para_tela(s)}" for i, s in enumerate(SINAIS))

with open(ARQUIVO, "a", newline="") as arquivo, mp_maos.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
) as maos:
    escritor = csv.writer(arquivo)

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
            mensagem = f"Aperte {legenda} | ESC sai"

        elif estado in ("preparar", "pausa"):
            espera = ATRASO_INICIAL if estado == "preparar" else PAUSA_ENTRE
            mensagem = f"Prepare {para_tela(sinal_atual)} ({repeticao + 1}/{REPETICOES})"
            if agora - inicio_estado >= espera:
                estado = "gravando"
                normalizados, punhos = [], []

        else:
            mensagem = f"FACA O SINAL {para_tela(sinal_atual)} ({repeticao + 1}/{REPETICOES})"
            if mao is None:

                print("Mao perdida; repetindo esta gravacao.")
                estado = "pausa"
                inicio_estado = agora
            else:
                normalizados.append(normalizar(mao))
                punhos.append(posicao_punho(mao))

                if len(normalizados) >= FRAMES_POR_SINAL:
                    sequencia = montar_sequencia(normalizados, punhos).round(5)
                    escritor.writerow([sinal_atual] + list(sequencia))
                    repeticao += 1

                    if repeticao >= REPETICOES:
                        arquivo.flush()
                        print(f"Sinal {sinal_atual} gravado ({REPETICOES} repeticoes).")
                        estado = "ocioso"
                    else:
                        estado = "pausa"
                        inicio_estado = agora

        cv2.putText(frame, mensagem, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Coleta de sinais com movimento", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:
            break
        if estado == "ocioso" and ord("1") <= tecla < ord("1") + len(SINAIS):
            sinal_atual = SINAIS[tecla - ord("1")]
            repeticao = 0
            estado = "preparar"
            inicio_estado = time.time()

camera.release()
cv2.destroyAllWindows()

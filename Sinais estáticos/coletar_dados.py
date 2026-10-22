import csv
import os
import time

import cv2
import mediapipe as mp

from Utilitários.utils import normalizar


AMOSTRAS_POR_LETRA = 150
ATRASO_INICIAL = 2
ARQUIVO = os.path.join("../dados", "pontos.csv")

os.makedirs("../dados", exist_ok=True)

mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

letra_atual = None
inicio = 0.0
contagem = 0


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

        if letra_atual is None:
            mensagem = "Aperte uma letra (a-z) para gravar | ESC para sair"
        elif time.time() - inicio < ATRASO_INICIAL:
            mensagem = f"Prepare a letra {letra_atual.upper()}..."
        else:
            mensagem = f"Gravando {letra_atual.upper()}: {contagem}/{AMOSTRAS_POR_LETRA}"
            if mao is not None:
                escritor.writerow([letra_atual.upper()] + list(normalizar(mao)))
                contagem += 1
                if contagem >= AMOSTRAS_POR_LETRA:
                    arquivo.flush()
                    print(f"Letra {letra_atual.upper()} gravada.")
                    letra_atual = None

        cv2.putText(frame, mensagem, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Coleta de dados", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:
            break
        if letra_atual is None and ord("a") <= tecla <= ord("z"):
            letra_atual = chr(tecla)
            inicio = time.time()
            contagem = 0

camera.release()
cv2.destroyAllWindows()

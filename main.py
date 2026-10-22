import os

import cv2
import mediapipe as mp
import joblib

from utils import normalizar


CAMINHO_MODELO = os.path.join("modelo", "modelo_letras.joblib")

if not os.path.exists(CAMINHO_MODELO):
    raise FileNotFoundError(
        "Modelo nao encontrado. Execute treinar.py primeiro."
    )

modelo = joblib.load(CAMINHO_MODELO)

mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError(
        "Nao consegui abrir a webcam. Verifique se ela esta sendo usada por outro programa."
    )

with mp_maos.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
) as maos:

    while camera.isOpened():
        ok, frame = camera.read()

        if not ok:
            print("Nao consegui ler a webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = maos.process(rgb)

        letra = "Nenhuma"
        confianca = 0.0

        if resultado.multi_hand_landmarks:
            mao = resultado.multi_hand_landmarks[0]

            mp_desenho.draw_landmarks(
                frame,
                mao,
                mp_maos.HAND_CONNECTIONS,
            )

            pontos = normalizar(mao).reshape(1, -1)

            probabilidades = modelo.predict_proba(pontos)[0]
            indice = probabilidades.argmax()

            letra = modelo.classes_[indice]
            confianca = probabilidades[indice]

        cv2.putText(
            frame,
            f"Letra: {letra}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )

        cv2.putText(
            frame,
            f"Confianca: {confianca:.0%}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Pressione Q para sair",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Tradutor de Libras", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()

# feitopae
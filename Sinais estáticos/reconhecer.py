import os
from collections import Counter, deque

import cv2
import joblib
import warnings
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")
import mediapipe as mp

from Utilitários.utils import normalizar


ARQUIVO_MODELO = os.path.join("../modelo", "modelo_letras.joblib")
LIMITE_CONFIANCA = 0.6
JANELA = 10

modelo = joblib.load(ARQUIVO_MODELO)
print("Letras que o modelo conhece:", " ".join(modelo.classes_))

mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

historico = deque(maxlen=JANELA)
texto_montado = ""

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

        confianca = 0.0
        if resultado.multi_hand_landmarks:
            mao = resultado.multi_hand_landmarks[0]
            mp_desenho.draw_landmarks(frame, mao, mp_maos.HAND_CONNECTIONS)


            entrada = normalizar(mao).reshape(1, -1)
            probabilidades = modelo.predict_proba(entrada)[0]
            melhor = probabilidades.argmax()
            confianca = probabilidades[melhor]

            if confianca >= LIMITE_CONFIANCA:
                historico.append(modelo.classes_[melhor])
            else:
                historico.append("?")
        else:
            historico.clear()


        letra = Counter(historico).most_common(1)[0][0] if historico else ""

        cv2.putText(frame, letra, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 0), 6)
        cv2.putText(frame, f"confianca: {confianca:.0%}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"texto: {texto_montado}", (20, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "ESPACO adiciona | BACKSPACE apaga | ESC sai", (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Reconhecimento de Libras", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:
            break
        elif tecla == 32 and letra not in ("", "?"):
            texto_montado += letra
        elif tecla == 8:
            texto_montado = texto_montado[:-1]

camera.release()
cv2.destroyAllWindows()

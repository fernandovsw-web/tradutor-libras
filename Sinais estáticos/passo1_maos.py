import cv2
import mediapipe as mp


mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_maos.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
) as maos:
    while camera.isOpened():
        ok, frame = camera.read()
        if not ok:
            print("Não consegui ler a webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = maos.process(rgb)

        if resultado.multi_hand_landmarks:
            for mao in resultado.multi_hand_landmarks:

                mp_desenho.draw_landmarks(frame, mao, mp_maos.HAND_CONNECTIONS)


                pontos = []
                for p in mao.landmark:
                    pontos.extend([p.x, p.y, p.z])

                cv2.putText(
                    frame,
                    f"{len(pontos)} valores por mao",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("Passo 1 - Deteccao da mao (q para sair)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()

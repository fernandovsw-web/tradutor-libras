"""
passo1_maos.py
 
PARA QUE SERVE
--------------
É o primeiro passo do projeto: um teste para conferir se a câmera e a
detecção de mão estão funcionando. Ele mostra a imagem da webcam com os
pontos da mão desenhados em cima e informa quantos números o MediaPipe
entrega para cada mão. Não grava nada em arquivo.
 
COMO FUNCIONA (passo a passo)
-----------------------------
1. Abre a webcam e prepara o MediaPipe, que detecta a mão na imagem.
2. Em cada imagem capturada, pede para o MediaPipe procurar a mão.
3. Se encontrar, desenha os pontos e as ligações entre eles (o "esqueleto"
   da mão) sobre a imagem.
4. Monta a lista de números da mão (x, y e z de cada ponto) e escreve na
   tela quantos valores existem por mão.
5. Repete isso o tempo todo, até você apertar a tecla Q.
 
ENTRADAS
--------
- Imagem da webcam (câmera 0).
- Tecla Q para encerrar.
 
SAÍDA
-----
- Apenas uma janela na tela com a imagem, a mão desenhada e o texto
  "N valores por mao". Nenhum arquivo é criado.
 
DEPENDÊNCIAS
------------
opencv-python (cv2) e mediapipe.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

import cv2
import mediapipe as mp

# ---------------------------------------------------------------------------
# PREPARAÇÃO: MEDIAPIPE E WEBCAM
# ---------------------------------------------------------------------------

# mp_maos: módulo de detecção de mãos do MediaPipe
mp_maos = mp.solutions.hands
# mp_desenho: ferramenta para desenhar os pontos da mão em cima da imagem
mp_desenho = mp.solutions.drawing_utils

# Abre a webcam (CAP_DSHOW é o modo de captura usado no Windows)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# O "with" cria o detector de mãos e garante que ele seja fechado no final.
with mp_maos.Hands(
    # Configurações:
    # - max_num_hands=1: procura no máximo 1 mão por imagem
    max_num_hands=1, 
    # - min_detection_confidence=0.7: só aceita a mão se a confiança for de 70% ou mais
    min_detection_confidence=0.7, 
    # - min_tracking_confidence=0.5: confiança mínima para continuar acompanhando a mão de uma imagem para outra
    min_tracking_confidence=0.5,
) as maos:
    
    # -----------------------------------------------------------------------
    # LOOP PRINCIPAL (uma volta = uma imagem da webcam)
    # -----------------------------------------------------------------------
    
    while camera.isOpened():
        
        # --- 1) Captura a imagem ---
        ok, frame = camera.read()
        if not ok:
            print("Não consegui ler a webcam.")
            break
        
        # Espelha a imagem (fica como um espelho, mais natural de usar)
        frame = cv2.flip(frame, 1)
        # O OpenCV trabalha em BGR e o MediaPipe espera RGB, por isso a conversão
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Pede para o MediaPipe procurar a mão na imagem
        resultado = maos.process(rgb)

        # --- 2) Se alguma mão foi encontrada, desenha e conta os valores ---
        if resultado.multi_hand_landmarks:
            for mao in resultado.multi_hand_landmarks:

                # Desenha os pontos da mão e as linhas que ligam um ponto ao outro
                mp_desenho.draw_landmarks(frame, mao, mp_maos.HAND_CONNECTIONS)

                # Junta, em uma única lista, as coordenadas x, y e z de cada
                # ponto da mão (são 21 pontos, então a lista fica com 63 números)
                pontos = []
                for p in mao.landmark:
                    pontos.extend([p.x, p.y, p.z])

                # Escreve na imagem quantos valores a mão tem
                cv2.putText(
                    frame,
                    f"{len(pontos)} valores por mao",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

        # --- 3) Mostra a imagem e verifica se o usuário quer sair ---
        cv2.imshow("Passo 1 - Deteccao da mao (q para sair)", frame)
        # Se a tecla Q for apertada, sai do loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# ---------------------------------------------------------------------------
# FINALIZAÇÃO: libera a webcam e fecha a janela
# ---------------------------------------------------------------------------

camera.release()
cv2.destroyAllWindows()

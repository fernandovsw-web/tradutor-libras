"""
coletar_dados.py
 
PARA QUE SERVE
--------------
Este script serve para montar o dataset de treino do trabalho.
Ele usa a webcam para "ver" a sua mão fazendo cada letra e salva, em um arquivo CSV (pontos.csv), 
as posições dos pontos da mão para cada letra.
Depois, esses dados podem ser usados para treinar o modelo para que reconheça as letras sozinho.
 
COMO FUNCIONA (passo a passo)
-----------------------------
1. Abre a webcam e a biblioteca MediaPipe, que detecta a mão e marca os
   pontos dela (as juntas e pontas dos dedos) em cada imagem.
2. Na tela, você aperta uma tecla de "a" até "z" para escolher qual letra
   vai gravar.
3. O programa espera alguns segundos (ATRASO_INICIAL) para você se
   posicionar e fazer o sinal da letra.
4. Depois disso, grava várias amostras seguidas (AMOSTRAS_POR_LETRA),
   uma por imagem em que a mão foi detectada.
5. Cada amostra vira uma linha no CSV: a primeira coluna é a letra e as
   demais são as coordenadas dos pontos da mão já normalizadas (a função
   normalizar, que vem do arquivo Utilitários/utils.py, é quem faz esse
   ajuste nos números).
6. Ao terminar as amostras, volta a esperar uma nova letra. Para encerrar
   o programa, aperte ESC.
 
ENTRADAS
--------
- Imagem da webcam (câmera 0).
- Teclas do teclado: letras a-z para gravar e ESC para sair.
 
SAÍDA
-----
- Arquivo ../dados/pontos.csv (a pasta é criada se não existir).
  Atenção: o arquivo é aberto em modo "a" (acrescentar), então rodar o
  script de novo ADICIONA linhas novas, sem apagar as que já existem.
 
DEPENDÊNCIAS
------------
opencv-python (cv2), mediapipe e o módulo próprio Utilitários.utils.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

import csv
import os
import time
import cv2
import mediapipe as mp
from Utilitários.utils import normalizar

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------------

# Quantas amostras (linhas no CSV) serão gravadas para cada letra
AMOSTRAS_POR_LETRA = 150
# Tempo de espera (em segundos) entre apertar a tecla e começar a gravar,
# para dar tempo de colocar a mão na posição da letra
ATRASO_INICIAL = 2
# Caminho do arquivo onde os dados serão salvos
ARQUIVO = os.path.join("../dados", "pontos.csv")
# Cria a pasta "dados" caso ela ainda não exista
os.makedirs("../dados", exist_ok=True)

# ---------------------------------------------------------------------------
# PREPARAÇÃO: MEDIAPIPE E WEBCAM
# ---------------------------------------------------------------------------

# mp_maos: módulo de detecção de mãos do MediaPipe
mp_maos = mp.solutions.hands
# mp_desenho: ferramenta para desenhar os pontos da mão em cima da imagem
mp_desenho = mp.solutions.drawing_utils
# Abre a webcam (CAP_DSHOW é o modo de captura usado no Windows)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Variáveis que controlam o estado da gravação
letra_atual = None # letra que está sendo gravada (None = nenhuma)
inicio = 0.0 # momento em que a tecla da letra foi apertada
contagem = 0 # quantas amostras da letra atual já foram gravadas

# ---------------------------------------------------------------------------
# LOOP PRINCIPAL
# ---------------------------------------------------------------------------

# O "with" abre o CSV e o detector de mãos juntos e garante que ambos 
# sejam fechados corretamente no final. 
# O detector está configurado para procurar no máximo 1 mão por imagem.
with open(ARQUIVO, "a", newline="") as arquivo, mp_maos.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
) as maos:
    escritor = csv.writer(arquivo)

    # Repete enquanto a webcam estiver aberta (uma volta = uma imagem)
    while camera.isOpened():
        
        # --- 1) Captura e prepara a imagem ---
        ok, frame = camera.read()
        if not ok:
            print("Nao consegui ler a webcam.")
            break
        
        # Espelha a imagem (fica como um espelho, mais natural de usar)
        frame = cv2.flip(frame, 1)
        # O OpenCV trabalha em BGR e o MediaPipe espera RGB, por isso a conversão
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Pede para o MediaPipe procurar a mão na imagem
        resultado = maos.process(rgb)

        # --- 2) Pega a mão detectada (se houver) e desenha os pontos ---
        mao = None
        if resultado.multi_hand_landmarks:
            mao = resultado.multi_hand_landmarks[0]
            mp_desenho.draw_landmarks(frame, mao, mp_maos.HAND_CONNECTIONS)

        # --- 3) Decide o que fazer conforme o estado atual ---
        if letra_atual is None:
            # Estado 1: nenhuma letra escolhida, só esperando o usuário
            mensagem = "Aperte uma letra (a-z) para gravar | ESC para sair"
        elif time.time() - inicio < ATRASO_INICIAL:
            # Estado 2: letra escolhida, mas ainda na contagem de preparação
            mensagem = f"Prepare a letra {letra_atual.upper()}..."
        else:
            # Estado 3: gravando de verdade
            mensagem = f"Gravando {letra_atual.upper()}: {contagem}/{AMOSTRAS_POR_LETRA}"
            
            # Só grava se a mão foi detectada nesta imagem
            if mao is not None:
                # Linha do CSV = letra + pontos da mão normalizados
                escritor.writerow([letra_atual.upper()] + list(normalizar(mao)))
                contagem += 1
                
                # Quando chega no total de amostras, salva no disco e
                # volta ao estado 1 para esperar a próxima letra
                if contagem >= AMOSTRAS_POR_LETRA:
                    arquivo.flush()
                    print(f"Letra {letra_atual.upper()} gravada.")
                    letra_atual = None

        # --- 4) Mostra a imagem com a mensagem de status na tela ---
        cv2.putText(frame, mensagem, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Coleta de dados", frame)

        # --- 5) Lê o teclado ---
        tecla = cv2.waitKey(1) & 0xFF
        
        # ESC (código 27) encerra o programa
        if tecla == 27:
            break
        
        # Se não está gravando nada e a tecla é de "a" a "z", inicia a
        # gravação daquela letra e zera o cronômetro e a contagem
        if letra_atual is None and ord("a") <= tecla <= ord("z"):
            letra_atual = chr(tecla)
            inicio = time.time()
            contagem = 0

# ---------------------------------------------------------------------------
# FINALIZAÇÃO: libera a webcam e fecha a janela
# ---------------------------------------------------------------------------

camera.release()
cv2.destroyAllWindows()

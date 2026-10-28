"""
coletar_sequencias.py
 
PARA QUE SERVE
--------------
Coleta os dados de treino dos sinais que têm MOVIMENTO (no projeto: H, J, X
e Z). Diferente do coletar_dados.py, que grava o formato da mão em uma única
imagem, este script grava uma sequência de várias imagens seguidas, para
guardar também o caminho que a mão faz no ar. Os dados ficam em um CSV, que
depois serve para treinar o modelo.
 
COMO FUNCIONA (passo a passo)
-----------------------------
1. Abre a webcam (pedindo 30 imagens por segundo, para reduzir o borrão em
   movimentos rápidos) e o MediaPipe, que detecta a mão e marca os pontos
   dela.
2. Na tela, você aperta uma tecla de número (1, 2, 3 ou 4) para escolher
   qual sinal da lista SINAIS vai gravar (1 = primeiro da lista, 2 = segundo
   e assim por diante).
3. O programa dá um tempo para você se posicionar (ATRASO_INICIAL) e começa
   a gravar uma repetição do sinal.
4. Uma repetição são FRAMES_POR_SINAL imagens seguidas. Em cada uma, guarda
   o formato da mão (função normalizar) e a posição do punho (função
   posicao_punho). O tempo que isso leva depende da velocidade da câmera.
5. Quando as imagens da repetição se completam, a função montar_sequencia
   junta tudo em um único vetor e ele vira UMA linha do CSV: a primeira
   coluna é o sinal e as demais são os números da sequência.
6. Entre uma repetição e outra há uma pausa (PAUSA_ENTRE) para você voltar
   à posição inicial. Isso se repete até completar REPETICOES gravações.
7. Se a mão sumir da imagem no meio de uma gravação, o programa tolera
   falhas curtas: repete o último formato de mão conhecido para "tapar o
   buraco" (até MAX_FRAMES_PERDIDOS imagens no total). Se a mão ficar sumida
   por mais tempo que isso, a gravação é descartada e refeita, para não
   entrar dado incompleto no arquivo.
8. Para encerrar o programa, aperte ESC.
 
ESTADOS DO PROGRAMA
-------------------
O script funciona como uma máquina de estados:
    ocioso -> preparar -> gravando -> pausa -> gravando -> ... -> ocioso
- ocioso: esperando você escolher um sinal.
- preparar: contagem inicial antes da primeira gravação.
- gravando: capturando as imagens da repetição.
- pausa: intervalo entre repetições (ou espera para refazer uma gravação
  em que a mão foi perdida).
 
ENTRADAS
--------
- Imagem da webcam (câmera 0).
- Teclas: 1 a 4 para escolher o sinal e ESC para sair.
 
SAÍDA
-----
- Arquivo ../dados/sequencias.csv (a pasta é criada se não existir).
  Atenção: o arquivo é aberto em modo "a" (acrescentar), então rodar o
  script de novo ADICIONA linhas novas, sem apagar as que já existem.
 
DEPENDÊNCIAS
------------
opencv-python (cv2), mediapipe e o módulo próprio Utilitários.utils
(funções normalizar, posicao_punho, montar_sequencia e para_tela, além das
constantes FRAMES_POR_SINAL e MAX_FRAMES_PERDIDOS).
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

import csv
import os
import time
import cv2
import mediapipe as mp
from utilitarios.utils import (
    FRAMES_POR_SINAL,
    MAX_FRAMES_PERDIDOS,
    montar_sequencia,
    normalizar,
    para_tela,
    posicao_punho,
)
import warnings
# Esconde um aviso específico (da biblioteca do MediaPipe) que não afeta o
# funcionamento do programa, só poluiria o terminal
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------------

# Aqui colocar os sinais que quer ensinar pro random forest, da esquerda pra direita pra gravar, 
# são salvos como 1, 2, 3 e 4
SINAIS = ["PROFESSOR", "MUITO", "BEM", "OLA"]

# Pode ser alterado como quiserem pra ficar mais confortável de gravar
REPETICOES = 30        # quantas vezes o sinal é gravado a cada sessão
ATRASO_INICIAL = 2.0   # segundos para se posicionar antes da primeira gravação
PAUSA_ENTRE = 1.5      # segundos para voltar à posição inicial entre cada gravação

# Caminho do arquivo onde as sequências serão salvas
ARQUIVO = os.path.join("../dados", "sequencias.csv")

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
# Pede à câmera 30 imagens por segundo, em 640x480, no formato MJPG: isso
# costuma reduzir o atraso e o borrão em movimentos rápidos. Se a webcam não
# aceitar, ela continua funcionando com os valores dela.
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)
# Mostra o FPS que a câmera informou (se vier bem abaixo de 30, a webcam é o gargalo)
print("FPS informado pela camera:", camera.get(cv2.CAP_PROP_FPS))

# Variáveis que controlam o estado da gravação
# Estados: ocioso -> preparar -> gravando -> pausa -> gravando -> ... -> ocioso
estado = "ocioso"
sinal_atual = None
repeticao = 0          # quantas repetições já foram gravadas deste sinal
inicio_estado = 0.0
normalizados = []      # formato da mão em cada frame da repetição atual
punhos = []            # posição do punho em cada frame da repetição atual
perdidos = 0           # quantas imagens ficaram sem a mão detectada na repetição atual

# Texto de ajuda mostrado na tela, ex.: "1=H  2=J  3=X  4=Z".
# O para_tela tira acentos, porque o OpenCV não desenha letras acentuadas
legenda = "  ".join(f"{i + 1}={para_tela(s)}" for i, s in enumerate(SINAIS))

# ---------------------------------------------------------------------------
# LOOP PRINCIPAL
# ---------------------------------------------------------------------------

# O "with" abre o CSV e o detector de mãos juntos e garante que ambos sejam
# fechados corretamente no final. O detector procura no máximo 1 mão por imagem.
# Os dois valores de confiança foram baixados para o MediaPipe achar a mão de
# novo e segurá-la por mais tempo em movimentos rápidos. Se aparecerem pontos
# "pulando" ou mãos falsas, suba esses valores.
with open(ARQUIVO, "a", newline="") as arquivo, mp_maos.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3,
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

        # Momento atual, usado para medir o tempo das esperas
        agora = time.time()

        # --- 3) Faz o que o estado atual pede ---
        if estado == "ocioso":
            # Esperando o usuário escolher um sinal
            mensagem = f"Aperte {legenda} | ESC sai"

        elif estado in ("preparar", "pausa"):
            # Esperando o tempo de preparação (primeira vez) ou de pausa
            # (entre repetições). Os dois estados são tratados juntos,
            # mudando só o tempo de espera.
            espera = ATRASO_INICIAL if estado == "preparar" else PAUSA_ENTRE
            mensagem = f"Prepare {para_tela(sinal_atual)} ({repeticao + 1}/{REPETICOES})"
            
            # Passou o tempo de espera: começa a gravar, com as listas e o
            # contador de imagens perdidas zerados
            if agora - inicio_estado >= espera:
                estado = "gravando"
                normalizados, punhos = [], []
                perdidos = 0

        else:
            # Estado "gravando": captura as imagens da repetição atual
            mensagem = f"FACA O SINAL {para_tela(sinal_atual)} ({repeticao + 1}/{REPETICOES})"
            if mao is not None:
                # Guarda o formato da mão e a posição do punho desta imagem
                normalizados.append(normalizar(mao))
                punhos.append(posicao_punho(mao))
            else:
                # A mão sumiu nesta imagem: conta a falha
                perdidos += 1
                if normalizados and perdidos <= MAX_FRAMES_PERDIDOS:
                    # Falha curta: repete o último formato e a última posição
                    # conhecidos para "tapar o buraco" e a repetição continuar
                    normalizados.append(normalizados[-1])
                    punhos.append(punhos[-1])
                else:
                    # A mão ficou sumida por tempo demais (ou nem apareceu
                    # ainda): descarta o que foi captado e volta para a
                    # pausa. A repetição NÃO é contada, então será refeita
                    # do zero.
                    print(f"Mao perdida ({perdidos} frames); repetindo esta gravacao.")
                    estado = "pausa"
                    inicio_estado = agora

            # Quando juntou imagens suficientes, a repetição está completa
            # (só vale se a gravação não foi descartada logo acima)
            if estado == "gravando" and len(normalizados) >= FRAMES_POR_SINAL:
                # Junta todas as imagens em um único vetor (arredondado
                # para 5 casas decimais, para o arquivo ficar menor) e
                # grava como uma linha: sinal + números da sequência
                sequencia = montar_sequencia(normalizados, punhos).round(5)
                escritor.writerow([sinal_atual] + list(sequencia))
                repeticao += 1
                if perdidos:
                    print(f"  (repeticao salva com {perdidos} imagem(ns) preenchida(s))")

                if repeticao >= REPETICOES:
                    # Terminou todas as repetições: salva no disco e
                    # volta a esperar a escolha de outro sinal
                    arquivo.flush()
                    print(f"Sinal {sinal_atual} gravado ({REPETICOES} repeticoes).")
                    estado = "ocioso"
                else:
                    # Ainda faltam repetições: faz uma pausa e grava a próxima
                    estado = "pausa"
                    inicio_estado = agora

        # --- 4) Mostra a imagem com a mensagem de status na tela ---
        cv2.putText(frame, mensagem, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Coleta de sinais com movimento", frame)

        # --- 5) Lê o teclado ---
        tecla = cv2.waitKey(1) & 0xFF
        # ESC (código 27) encerra o programa
        if tecla == 27:
            break
        # Se está ocioso e a tecla é um número que corresponde a um sinal da
        # lista, escolhe esse sinal e começa a contagem de preparação
        # (a tecla "1" escolhe o primeiro sinal, a "2" o segundo, etc.)
        if estado == "ocioso" and ord("1") <= tecla < ord("1") + len(SINAIS):
            sinal_atual = SINAIS[tecla - ord("1")]
            repeticao = 0
            estado = "preparar"
            inicio_estado = time.time()

# ---------------------------------------------------------------------------
# FINALIZAÇÃO: libera a webcam e fecha a janela
# ---------------------------------------------------------------------------

camera.release()
cv2.destroyAllWindows()

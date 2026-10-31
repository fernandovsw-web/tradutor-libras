"""
reconhecer_sinais.py
 
PARA QUE SERVE
--------------
Reconhece, pela webcam, os sinais que têm MOVIMENTO (no projeto: K, BEM, ELE
e ENSINAR). É o equivalente do reconhecer.py, mas para sinais em movimento:
em vez de olhar uma única imagem, ele grava uma sequência de imagens seguidas
(do mesmo jeito que o coletar_sequencias.py) e entrega a sequência inteira ao
modelo treinado pelo treinar_sequencias.py.
 
COMO FUNCIONA (passo a passo)
-----------------------------
1. Carrega o modelo de sinais (modelo_sinais.joblib) e mostra no terminal
   quais sinais ele conhece.
2. Abre a webcam (pedindo 30 imagens por segundo, como na coleta) e o
   MediaPipe, que detecta a mão e marca os pontos dela.
3. Você aperta ESPAÇO para começar um reconhecimento. O programa espera um
   tempo (ATRASO_INICIAL) para você se posicionar.
4. Depois disso, grava FRAMES_POR_SINAL imagens seguidas. Em cada uma, guarda
   o formato da mão (função normalizar) e a posição do punho (função
   posicao_punho). É nesse intervalo que você deve fazer o sinal.
5. Se a mão sumir da imagem por pouco tempo, o programa repete o último
   formato conhecido para "tapar o buraco" (até MAX_FRAMES_PERDIDOS imagens),
   a mesma regra da coleta. Se ela sumir por mais tempo, aparece "mao perdida"
   e a tentativa é descartada.
6. Com as imagens completas, a função montar_sequencia junta tudo em um único
   vetor (exatamente o mesmo formato usado no treino) e o modelo devolve a
   probabilidade de cada sinal.
7. Se a probabilidade do sinal mais provável for de pelo menos
   LIMITE_CONFIANCA, ele é mostrado na tela junto com a confiança; senão
   aparece "?" (em vez de chutar).
8. O programa volta a esperar: ESPAÇO tenta de novo e ESC encerra.
 
ESTADOS DO PROGRAMA
-------------------
    ocioso -> preparar -> gravando -> ocioso
- ocioso: esperando você apertar ESPAÇO.
- preparar: contagem inicial para você posicionar a mão.
- gravando: capturando as imagens da tentativa e, ao completar, reconhecendo.
 
ENTRADAS
--------
- Imagem da webcam (câmera 0).
- Arquivo do modelo treinado: backend/treinamento/modelo/modelo_sinais.joblib
- Teclas: ESPAÇO (reconhecer um sinal) e ESC (sair).
 
SAÍDA
-----
- Uma janela com a imagem da câmera, a mão desenhada, o sinal reconhecido e a
  confiança em porcentagem. Nenhum arquivo é gravado.
 
DEPENDÊNCIAS
------------
Python 3.11.9 e as bibliotecas (versões usadas no projeto):
- opencv-python 5.0.0.93 (cv2)
- mediapipe 0.10.14
- joblib 1.6.0
- scikit-learn 1.9.1 (necessário para abrir o modelo salvo)
- módulo próprio backend.utilitarios.utils (que usa numpy 2.4.6)
O modelo precisa já ter sido treinado e salvo antes de rodar este script, e na
mesma versão do scikit-learn (se trocar de versão, treine de novo).
Para instalar tudo de uma vez, veja o requirements.txt e o README.md.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

import os
import time
import cv2
import joblib
import mediapipe as mp
import warnings
# Esconde um aviso específico (da biblioteca do MediaPipe) que não afeta o
# funcionamento do programa, só poluiria o terminal. O filtro é colocado
# antes de importar o mediapipe de propósito, para já valer na importação.
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")
from backend.utilitarios.utils import (
    FRAMES_POR_SINAL,
    MAX_FRAMES_PERDIDOS,
    montar_sequencia,
    normalizar,
    para_tela,
    posicao_punho,
)

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES (pode ajustar)
# ---------------------------------------------------------------------------

# Caminhos a partir da pasta onde ESTE script está, para não depender
# de qual pasta o PyCharm usa ao rodar.
# Estrutura esperada:
#   backend/treinamento/sinais_sequencias/reconhecer_sinais.py  (este arquivo)
#   backend/treinamento/modelo/modelo_sinais.joblib             (gerado pelo treino)
# PASTA = pasta deste script; PASTA_MODELO = pasta "modelo", que fica ao lado
# da pasta deste script (por isso o os.path.dirname(PASTA))
PASTA = os.path.dirname(os.path.abspath(__file__))
PASTA_MODELO = os.path.join(os.path.dirname(PASTA), "modelo")
ARQUIVO_MODELO = os.path.join(PASTA_MODELO, "modelo_sinais.joblib")

LIMITE_CONFIANCA = 0.5  # abaixo disso mostra "?" em vez de chutar
ATRASO_INICIAL = 1.0    # segundos entre apertar ESPACO e o início da gravação

# Carrega o modelo treinado do arquivo e mostra os sinais que ele aprendeu
modelo = joblib.load(ARQUIVO_MODELO)
print("Sinais que o modelo conhece:", " ".join(modelo.classes_))

# mp_maos: módulo de detecção de mãos do MediaPipe
mp_maos = mp.solutions.hands
# mp_desenho: ferramenta para desenhar os pontos da mão em cima da imagem
mp_desenho = mp.solutions.drawing_utils
# Abre a webcam (CAP_DSHOW é o modo de captura usado no Windows)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Pede à câmera 30 fps em 640x480 com MJPG: menos atraso e menos borrão em movimentos
# rápidos. Se a webcam não aceitar, ela continua funcionando com os valores dela.
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)
print("FPS informado pela camera:", camera.get(cv2.CAP_PROP_FPS))

# Variáveis que controlam o estado do reconhecimento
# Estados: ocioso -> preparar -> gravando -> ocioso
estado = "ocioso"
inicio_estado = 0.0         # momento em que o estado atual começou (mede as esperas)
normalizados = []           # formato da mão em cada imagem da tentativa atual
punhos = []                 # posição do punho em cada imagem da tentativa atual
perdidos = 0                # quantas imagens ficaram sem a mão detectada na tentativa
ultimo_resultado = ""       # texto mostrado na tela (sinal, "?" ou "mao perdida")
ultima_confianca = None     # confiança do último reconhecimento (None = não mostra)

# ---------------------------------------------------------------------------
# LOOP PRINCIPAL (uma volta = uma imagem da webcam)
# ---------------------------------------------------------------------------

# O detector procura no máximo 1 mão por imagem. Os valores de confiança são
# os mesmos da coleta de movimento (iguais aos do coletar_sequencias.py).
with mp_maos.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,  # mais baixo = acha a mão de novo com mais facilidade
    min_tracking_confidence=0.3,   # mais baixo = segura a mão por mais tempo em movimento rápido
    # (se aparecerem pontos "pulando" ou mãos falsas, suba esses dois valores)
) as maos:
    while camera.isOpened():
        
        # --- 1) Captura e prepara a imagem ---
        ok, frame = camera.read()
        if not ok:
            print("Nao consegui ler a webcam.")
            break
        
        # Espelha a imagem (igual à coleta, para o formato bater com o do treino)
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
            # Esperando o usuário apertar ESPAÇO
            mensagem = "ESPACO para reconhecer um sinal | ESC sai"

        elif estado == "preparar":
            # Contagem inicial: passou o tempo, começa a gravar com as listas
            # e o contador de imagens perdidas zerados
            mensagem = "Prepare a mao..."
            if agora - inicio_estado >= ATRASO_INICIAL:
                estado = "gravando"
                normalizados, punhos = [], []
                perdidos = 0

        else:  # gravando
            # Captura as imagens da tentativa atual
            mensagem = "FACA O SINAL!"
            if mao is not None:
                # Guarda o formato da mão e a posição do punho desta imagem
                normalizados.append(normalizar(mao))
                punhos.append(posicao_punho(mao))
            else:
                # A mão sumiu nesta imagem: conta a falha
                perdidos += 1
                if normalizados and perdidos <= MAX_FRAMES_PERDIDOS:
                    # Mesma regra da coleta: falha curta = repete o último frame conhecido
                    normalizados.append(normalizados[-1])
                    punhos.append(punhos[-1])
                else:
                    # Mão perdida por tempo demais: descarta a tentativa e
                    # volta a esperar o ESPAÇO
                    ultimo_resultado = "mao perdida"
                    ultima_confianca = None
                    estado = "ocioso"

            # Quando juntou imagens suficientes, é hora de reconhecer
            # (só vale se a tentativa não foi descartada logo acima)
            if estado == "gravando" and len(normalizados) >= FRAMES_POR_SINAL:
                # Mesmo formato de vetor usado na coleta (isso é essencial!)
                entrada = montar_sequencia(normalizados, punhos).reshape(1, -1)
                # O modelo devolve a probabilidade de cada sinal
                probabilidades = modelo.predict_proba(entrada)[0]
                # Posição do sinal mais provável e a probabilidade dele
                melhor = probabilidades.argmax()
                ultima_confianca = probabilidades[melhor]

                # Só confia no sinal se a confiança passar do limite;
                # senão mostra "?" para indicar que não deu para ter certeza
                if ultima_confianca >= LIMITE_CONFIANCA:
                    ultimo_resultado = str(modelo.classes_[melhor])
                else:
                    ultimo_resultado = "?"
                estado = "ocioso"

        # --- 4) Escreve as informações na imagem ---
        # Mensagem de status, na parte de cima
        cv2.putText(frame, mensagem, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # Último resultado, em tamanho grande (para_tela tira os acentos,
        # porque o OpenCV não desenha letras acentuadas)
        cv2.putText(frame, para_tela(ultimo_resultado), (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 6)
        # Confiança do último reconhecimento, em porcentagem
        if ultima_confianca is not None:
            cv2.putText(frame, f"confianca: {ultima_confianca:.0%}", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Reconhecimento de sinais com movimento", frame)

        # --- 5) Lê o teclado ---
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:  # ESC
            break
        if tecla == 32 and estado == "ocioso":  # ESPAÇO
            # Começa uma nova tentativa, limpando o resultado anterior da tela
            estado = "preparar"
            inicio_estado = time.time()
            ultimo_resultado = ""
            ultima_confianca = None

# ---------------------------------------------------------------------------
# FINALIZAÇÃO: libera a webcam e fecha a janela
# ---------------------------------------------------------------------------

camera.release()
cv2.destroyAllWindows()

"""
reconhecer.py
 
PARA QUE SERVE
--------------
É o programa final de reconhecimento: ele usa a webcam para ver a sua mão,
descobre qual letra de Libras você está fazendo (usando o modelo já
treinado) e mostra a letra na tela. Você também pode ir montando um texto,
letra por letra.
 
COMO FUNCIONA (passo a passo)
-----------------------------
1. Carrega o modelo treinado, que fica salvo em ../modelo/modelo_letras.joblib,
   e mostra no terminal quais letras ele conhece.
2. Abre a webcam e, em cada imagem, pede para o MediaPipe encontrar a mão.
3. Se achar a mão, transforma os pontos dela em números com a função
   normalizar (a mesma usada na coleta de dados, para o formato ser igual
   ao do treino) e entrega esses números ao modelo.
4. O modelo devolve a probabilidade de cada letra. Pega-se a mais provável
   e, se a confiança for alta o bastante (LIMITE_CONFIANCA), ela entra em
   um histórico das últimas leituras; se for baixa, entra um "?".
5. A letra exibida na tela é a que mais apareceu nesse histórico (uma
   votação). Isso evita que a letra fique piscando quando o modelo oscila
   de uma imagem para outra.
6. Com a letra aparecendo, você aperta ESPAÇO para adicioná-la ao texto,
   BACKSPACE para apagar a última letra ou ESC para sair.
 
ENTRADAS
--------
- Imagem da webcam (câmera 0).
- Arquivo do modelo treinado: ../modelo/modelo_letras.joblib
- Teclas: ESPAÇO (adicionar letra), BACKSPACE (apagar) e ESC (sair).
 
SAÍDA
-----
- Uma janela com a imagem da câmera, a mão desenhada, a letra reconhecida,
  a confiança em porcentagem e o texto que você foi montando.
  Nenhum arquivo é gravado.
 
DEPENDÊNCIAS
------------
opencv-python (cv2), mediapipe, joblib e o módulo próprio Utilitários.utils.
O modelo precisa já ter sido treinado e salvo antes de rodar este script.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

import os
from collections import Counter, deque
from pathlib import Path
import cv2
import joblib
import mediapipe as mp
from utilitarios.utils import normalizar
import warnings
# Esconde um aviso específico (da biblioteca do MediaPipe) que não afeta o
# funcionamento do programa, só poluiria o terminal. O filtro é colocado
# antes de importar o mediapipe de propósito, para já valer na importação.
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------------

# Caminho do arquivo onde está o modelo já treinado
BASE = Path(__file__).resolve().parents[1]   # backend/treinamento
ARQUIVO_MODELO = BASE / "modelo" / "modelo_letras.joblib"
# Confiança mínima (0.6 = 60%) para aceitar a letra que o modelo escolheu.
# Abaixo disso, a leitura é tratada como incerta ("?")
LIMITE_CONFIANCA = 0.6
# Quantas leituras recentes entram na votação que decide a letra mostrada
JANELA = 10

# ---------------------------------------------------------------------------
# PREPARAÇÃO: MODELO, MEDIAPIPE E WEBCAM
# ---------------------------------------------------------------------------

# Carrega o modelo treinado do arquivo e mostra as letras que ele aprendeu
modelo = joblib.load(ARQUIVO_MODELO)
print("Letras que o modelo conhece:", " ".join(modelo.classes_))

# mp_maos: módulo de detecção de mãos do MediaPipe
mp_maos = mp.solutions.hands
# mp_desenho: ferramenta para desenhar os pontos da mão em cima da imagem
mp_desenho = mp.solutions.drawing_utils
# Abre a webcam (CAP_DSHOW é o modo de captura usado no Windows)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# historico: guarda só as últimas JANELA leituras 
# (as mais antigas saem sozinhas quando entra uma nova)
historico = deque(maxlen=JANELA)
# texto que o usuário vai montando com as letras reconhecidas
texto_montado = ""

# ---------------------------------------------------------------------------
# LOOP PRINCIPAL (uma volta = uma imagem da webcam)
# ---------------------------------------------------------------------------

# O detector está configurado para procurar no máximo 1 mão por imagem
with mp_maos.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
) as maos:
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

        # --- 2) Se há uma mão, pede ao modelo qual letra é ---
        confianca = 0.0
        if resultado.multi_hand_landmarks:
            mao = resultado.multi_hand_landmarks[0]
            mp_desenho.draw_landmarks(frame, mao, mp_maos.HAND_CONNECTIONS)

            # Transforma a mão em números, igual foi feito na coleta de dados.
            # O reshape(1, -1) coloca os números em uma linha só, que é o
            # formato que o modelo espera receber (uma amostra por linha)
            entrada = normalizar(mao).reshape(1, -1)
            # O modelo devolve a probabilidade de cada letra
            probabilidades = modelo.predict_proba(entrada)[0]
            melhor = probabilidades.argmax()
            confianca = probabilidades[melhor]

            # Só confia na letra se a confiança passar do limite; 
            # senão guarda "?" para indicar que não deu para ter certeza
            if confianca >= LIMITE_CONFIANCA:
                historico.append(modelo.classes_[melhor])
            else:
                historico.append("?")
        else:
            # Sem mão na imagem: apaga o histórico 
            # para não manter uma letra antiga na tela
            historico.clear()

        # --- 3) Decide a letra que será mostrada (votação) ---
        # Escolhe a leitura que mais se repetiu no histórico. 
        # Se o histórico está vazio, não há letra.
        letra = Counter(historico).most_common(1)[0][0] if historico else ""

        # --- 4) Escreve as informações na imagem ---
        # Letra reconhecida, em tamanho grande
        cv2.putText(frame, letra, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 0), 6)
        # Confiança da leitura atual, em porcentagem
        cv2.putText(frame, f"confianca: {confianca:.0%}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Texto que o usuário já montou, na parte de baixo
        cv2.putText(frame, f"texto: {texto_montado}", (20, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        # Lembrete das teclas, na parte de cima
        cv2.putText(frame, "ESPACO adiciona | BACKSPACE apaga | ESC sai", (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Reconhecimento de Libras", frame)

        # --- 5) Lê o teclado ---
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:
            # ESC: encerra o programa
            break
        elif tecla == 32 and letra not in ("", "?"):
            # ESPAÇO: adiciona a letra atual ao texto 
            # (só se houver uma letra válida, ou seja, que não esteja vazia nem seja "?")
            texto_montado += letra
        elif tecla == 8:
            # BACKSPACE: apaga a última letra do texto
            texto_montado = texto_montado[:-1]

# ---------------------------------------------------------------------------
# FINALIZAÇÃO: libera a webcam e fecha a janela
# ---------------------------------------------------------------------------

camera.release()
cv2.destroyAllWindows()

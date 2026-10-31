"""
main.py
 
PARA QUE SERVE
--------------
É o programa principal do projeto, o "Tradutor de Libras". Ele abre a
webcam, enxerga a sua mão, usa o modelo já treinado para descobrir qual
letra você está fazendo e mostra a letra e a confiança na tela, em tempo
real.
 
COMO FUNCIONA (passo a passo)
-----------------------------
1. Confere se o arquivo do modelo existe. Se não existir, o programa para
   e avisa que é preciso rodar o treinar.py primeiro.
2. Carrega o modelo e abre a webcam. Se a webcam não abrir (por exemplo,
   porque outro programa está usando), o programa para e avisa.
3. Em cada imagem da câmera, pede para o MediaPipe encontrar a mão.
4. Se achar a mão, desenha os pontos dela, transforma esses pontos em
   números com a função normalizar (a mesma usada na coleta e no treino) e
   entrega ao modelo.
5. O modelo devolve a probabilidade de cada letra. O programa pega a mais
   provável e mostra a letra e a confiança (em porcentagem) na imagem.
6. Se não houver mão na imagem, mostra "Nenhuma" e confiança 0%.
7. Repete até você apertar a tecla Q.
 
DIFERENÇAS EM RELAÇÃO AO reconhecer.py
--------------------------------------
Este é uma versão mais simples:
- Mostra a letra da imagem atual, sem votação entre as últimas leituras
  (por isso a letra pode oscilar mais).
- Não tem limite mínimo de confiança: sempre mostra a letra mais provável,
  mesmo quando o modelo tem pouca certeza.
- Não monta texto (não tem ESPAÇO nem BACKSPACE).
- Confere se o modelo e a webcam existem antes de começar, com mensagens de
  erro claras.
- Procura o modelo em "treinamento/modelo/", que fica ao lado deste
  arquivo (o caminho é montado a partir do próprio main.py), então não depende
  da pasta em que o terminal está aberto. Já o pacote "backend" (usado no
  import) precisa estar acessível: veja no README como rodar.
 
ENTRADAS
--------
- Imagem da webcam (câmera 0).
- Arquivo do modelo treinado: treinamento/modelo/modelo_letras.joblib (a pasta
  "treinamento" fica ao lado do main.py)
- Tecla Q para sair.
 
SAÍDA
-----
- Uma janela com a imagem da câmera, a mão desenhada, a letra reconhecida e
  a confiança. Nenhum arquivo é gravado.
 
DEPENDÊNCIAS
------------
Python 3.11.9 e as bibliotecas (versões usadas no projeto):
- opencv-python 5.0.0.93 (cv2)
- mediapipe 0.10.14
- joblib 1.6.0
- scikit-learn 1.9.1 (necessário para abrir o modelo salvo)
- módulo próprio backend.utilitarios.utils (que usa numpy 2.4.6)
Usa o modelo de letras (sinais parados); os sinais com movimento não são
tratados aqui. O modelo precisa ter sido treinado na mesma versão do
scikit-learn (se trocar de versão, treine de novo).
Para instalar tudo de uma vez, veja o requirements.txt e o README.md.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

import cv2
import mediapipe as mp
import joblib
from pathlib import Path
from backend.utilitarios.utils import normalizar

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO E VERIFICAÇÕES INICIAIS
# ---------------------------------------------------------------------------

# Caminho do arquivo onde está o modelo já treinado
CAMINHO_MODELO = Path(__file__).resolve().parent / "treinamento" / "modelo" / "modelo_letras.joblib"

# Se o modelo ainda não foi criado, interrompe o programa com uma mensagem
# explicando o que fazer
if not CAMINHO_MODELO.exists():
    raise FileNotFoundError(
        "Modelo nao encontrado. Execute treinar.py primeiro."
    )

# Carrega o modelo treinado do arquivo
modelo = joblib.load(CAMINHO_MODELO)

# ---------------------------------------------------------------------------
# PREPARAÇÃO: MEDIAPIPE E WEBCAM
# ---------------------------------------------------------------------------

# mp_maos: módulo de detecção de mãos do MediaPipe
mp_maos = mp.solutions.hands
# mp_desenho: ferramenta para desenhar os pontos da mão em cima da imagem
mp_desenho = mp.solutions.drawing_utils
# Abre a webcam (CAP_DSHOW é o modo de captura usado no Windows)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Se a webcam não abriu, interrompe o programa explicando o provável motivo
if not camera.isOpened():
    raise RuntimeError(
        "Nao consegui abrir a webcam. Verifique se ela esta sendo usada por outro programa."
    )

# ---------------------------------------------------------------------------
# LOOP PRINCIPAL (uma volta = uma imagem da webcam)
# ---------------------------------------------------------------------------

# O "with" cria o detector de mãos e garante que ele seja fechado no final.
# O detector procura no máximo 1 mão por imagem.
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

        # Valores padrão, usados quando não há mão na imagem
        letra = "Nenhuma"
        confianca = 0.0

        # --- 2) Se há uma mão, pede ao modelo qual letra é ---
        if resultado.multi_hand_landmarks:
            mao = resultado.multi_hand_landmarks[0]

            # Desenha os pontos da mão e as linhas que ligam um ponto ao outro
            mp_desenho.draw_landmarks(
                frame,
                mao,
                mp_maos.HAND_CONNECTIONS,
            )

            # Transforma a mão em números, igual foi feito na coleta de dados.
            # O reshape(1, -1) coloca os números em uma linha só, que é o
            # formato que o modelo espera receber (uma amostra por linha)
            pontos = normalizar(mao).reshape(1, -1)

            # O modelo devolve a probabilidade de cada letra
            probabilidades = modelo.predict_proba(pontos)[0]
            # Posição da letra mais provável
            indice = probabilidades.argmax()

            # Nome da letra mais provável e a probabilidade dela (confiança)
            letra = modelo.classes_[indice]
            confianca = probabilidades[indice]

        # --- 3) Escreve as informações na imagem ---
        # Letra reconhecida
        cv2.putText(
            frame,
            f"Letra: {letra}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )

        # Confiança da leitura, em porcentagem
        cv2.putText(
            frame,
            f"Confianca: {confianca:.0%}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        # Lembrete de como sair, na parte de baixo da imagem
        cv2.putText(
            frame,
            "Pressione Q para sair",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # --- 4) Mostra a imagem e verifica se o usuário quer sair ---
        cv2.imshow("Tradutor de Libras", frame)

        # Se a tecla Q for apertada, sai do loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# ---------------------------------------------------------------------------
# FINALIZAÇÃO: libera a webcam e fecha a janela
# ---------------------------------------------------------------------------

camera.release()
cv2.destroyAllWindows()
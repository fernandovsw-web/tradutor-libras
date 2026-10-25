"""
utils.py
 
PARA QUE SERVE
--------------
É o arquivo de funções de apoio do projeto. 
Ele não roda sozinho: os outros scripts (como o coletar_dados.py) 
importam as funções daqui para transformar os dados da mão detectada pela câmera em 
números prontos para serem gravados ou usados pelo modelo.
 
COMO FUNCIONA
-------------
O MediaPipe devolve a mão como uma lista de pontos (landmarks), cada um com
coordenadas x, y e z. Essas coordenadas dependem de onde a mão está na
imagem e do quão perto ela está da câmera. As funções daqui resolvem isso:
 
- normalizar: transforma os pontos de UMA imagem em um vetor que descreve só
  o FORMATO da mão, sem importar a posição nem o tamanho dela na tela.
- posicao_punho: pega a posição do punho na imagem (onde a mão está).
- montar_sequencia: junta várias imagens seguidas (formato + movimento do
  punho) em um único vetor, usado para sinais que têm movimento.
- para_tela: tira os acentos de um texto para ele poder ser escrito na
  janela do OpenCV.
 
ENTRADAS E SAÍDAS
-----------------
Cada função explica o que recebe e o que devolve logo abaixo.
 
DEPENDÊNCIAS
------------
numpy e unicodedata (esta última já vem com o Python).
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

import unicodedata
import numpy as np

# Quantas imagens (frames) formam um sinal com movimento.
# Não é usada neste arquivo; fica aqui como configuração 
# para os outros scripts que trabalham com sequências.
FRAMES_POR_SINAL = 40

"""
Descreve o formato da mão independente de posição e tamanho.

Recebe: a mão detectada pelo MediaPipe (com a lista de pontos).
Devolve: um vetor com os números dos pontos 
(x, y, z de cada ponto, um atrás do outro), todos entre -1 e 1.
"""
def normalizar(mao):

    # Monta uma tabela com uma linha por ponto da mão e as colunas x, y, z
    pontos = np.array([[p.x, p.y, p.z] for p in mao.landmark])
    # Usa o ponto 0 (o punho) como referência: subtrai ele de todos os pontos.
    # Assim o punho vira a origem (0, 0, 0) e não importa mais em que lugar
    # da imagem a mão está.
    pontos = pontos - pontos[0]

    # Divide tudo pelo maior valor (em módulo) para que o tamanho da mão
    # não faça diferença, seja ela mais perto ou mais longe da câmera.
    maior = np.max(np.abs(pontos))
    # O "if" evita dividir por zero.
    if maior > 0:
        pontos = pontos / maior

    # Transforma a tabela em uma única linha de números, que é o formato
    # que vai para o CSV e para o modelo
    return pontos.flatten()

"""
Devolve a posição do punho na imagem.

Recebe: a mão detectada pelo MediaPipe.
Devolve: um vetor com [x, y] do punho (ponto 0). O z não é usado aqui.
"""
def posicao_punho(mao):

    p = mao.landmark[0]
    return np.array([p.x, p.y])

"""
Junta várias imagens seguidas em um único vetor, para representar
um sinal com movimento.

Recebe:
    normalizados: lista com o formato da mão de cada imagem
                    (resultado da função normalizar).
    punhos: lista com a posição do punho de cada imagem
            (resultado da função posicao_punho).
Devolve: um vetor grande com, para cada imagem, o formato da mão
seguido de quanto o punho se moveu.
"""
def montar_sequencia(normalizados, punhos):

    # A posição do punho na primeira imagem é o ponto de partida.
    # Todas as outras posições são medidas em relação a ela.
    origem = punhos[0]
    partes = []
    
    for formato, punho in zip(normalizados, punhos):
        # Guarda o formato da mão naquela imagem
        partes.append(formato)
        # Guarda o deslocamento do punho desde o início: é isso que
        # registra o movimento da mão ao longo do sinal
        partes.append(punho - origem)
        
    # Emenda tudo em um vetor só, na ordem das imagens
    return np.concatenate(partes)

"""
Tira os acentos de um texto.

Recebe: um texto qualquer, por exemplo "Atenção".
Devolve: o mesmo texto sem acentos, por exemplo "Atencao".

Serve porque o cv2.putText (que escreve texto na janela da câmera)
não consegue desenhar letras acentuadas.
"""
def para_tela(texto):
    # NFD separa cada letra acentuada em "letra + acento".
    # Depois o encode para ASCII com "ignore" descarta os acentos (que não
    # existem em ASCII) e o decode volta para texto normal.
    return unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode()

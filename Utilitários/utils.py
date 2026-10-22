import unicodedata

import numpy as np

FRAMES_POR_SINAL = 40


def normalizar(mao):

    pontos = np.array([[p.x, p.y, p.z] for p in mao.landmark])
    pontos = pontos - pontos[0]

    maior = np.max(np.abs(pontos))
    if maior > 0:
        pontos = pontos / maior

    return pontos.flatten()


def posicao_punho(mao):

    p = mao.landmark[0]
    return np.array([p.x, p.y])


def montar_sequencia(normalizados, punhos):

    origem = punhos[0]
    partes = []
    for formato, punho in zip(normalizados, punhos):
        partes.append(formato)
        partes.append(punho - origem)
    return np.concatenate(partes)


def para_tela(texto):
    return unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode()

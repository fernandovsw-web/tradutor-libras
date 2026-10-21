import numpy as np


def normalizar(mao):

    pontos = np.array([[p.x, p.y, p.z] for p in mao.landmark])  # 21 linhas x 3 colunas
    pontos = pontos - pontos[0]

    maior = np.max(np.abs(pontos))
    if maior > 0:
        pontos = pontos / maior

    return pontos.flatten()

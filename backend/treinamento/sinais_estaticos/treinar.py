"""
treinar.py
 
PARA QUE SERVE
--------------
Ensina o computador a reconhecer as letras. Ele lê os dados que foram
gravados pelo coletar_dados.py (o arquivo pontos.csv), treina um modelo de
aprendizado de máquina com eles e salva o modelo pronto em um arquivo, que
depois é usado pelo reconhecer.py.
 
COMO FUNCIONA (passo a passo)
-----------------------------
1. Lê o pontos.csv. Cada linha tem a letra na primeira coluna (a "resposta
   certa") e os números da mão nas demais colunas (os "exemplos").
2. Mostra no terminal quantas amostras existem de cada letra, para você
   conferir se alguma ficou com poucos dados.
3. Separa os dados em duas partes: 80% para o modelo estudar (treino) e
   20% para fazer uma prova (teste), com exemplos que ele nunca viu.
4. Treina um modelo do tipo Random Forest (uma "floresta" de várias árvores
   de decisão que votam juntas na resposta) com a parte de treino.
5. Aplica a prova: o modelo tenta adivinhar as letras da parte de teste e o
   script mostra a acurácia (porcentagem de acertos) e um relatório com o
   desempenho de cada letra.
6. Depois da prova, treina o modelo de novo usando TODOS os dados, para ele
   aprender o máximo possível, e salva o resultado em
   ../modelo/modelo_letras.joblib.
 
ENTRADA
-------
- Arquivo ../dados/pontos.csv (gerado pelo coletar_dados.py).
 
SAÍDA
-----
- Arquivo ../modelo/modelo_letras.joblib (a pasta é criada se não existir).
- No terminal: amostras por letra, acurácia no teste e relatório por letra.
 
DEPENDÊNCIAS
------------
numpy, joblib e scikit-learn (sklearn).
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

import csv
import os
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES: onde ler os dados e onde salvar o modelo
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parents[1]   # backend/treinamento

ARQUIVO = BASE / "dados" / "pontos.csv"
CAMINHO_MODELO = BASE / "modelo" / "modelo_letras.joblib"

# ---------------------------------------------------------------------------
# 1) LEITURA DOS DADOS
# ---------------------------------------------------------------------------

# X: os exemplos (números da mão)
# y: as respostas certas (a letra de cada exemplo)
X, y = [], []
with open(ARQUIVO) as arquivo:
    for linha in csv.reader(arquivo):
        
        # Primeira coluna = letra
        y.append(linha[0])
        # Demais colunas = números da mão, convertidos de texto para decimal
        X.append([float(v) for v in linha[1:]])

# Converte as listas para arrays do numpy, que é o formato usado pelo sklearn
X = np.array(X)
y = np.array(y)

# Mostra quantas amostras cada letra tem (em ordem alfabética).
# Serve para conferir se os dados estão equilibrados entre as letras.
print("Amostras por letra:")
for letra, qtd in sorted(Counter(y).items()):
    print(f"  {letra}: {qtd}")

# ---------------------------------------------------------------------------
# 2) SEPARAÇÃO ENTRE TREINO E TESTE
# ---------------------------------------------------------------------------

# - test_size=0.2: 20% dos dados ficam reservados para a prova
# - random_state=42: fixa o sorteio, para o resultado ser sempre o mesmo
#   se o script rodar de novo com os mesmos dados
# - stratify=y: mantém a mesma proporção de cada letra no treino e no teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 3) TREINO E AVALIAÇÃO
# ---------------------------------------------------------------------------

# Cria a Random Forest com 200 árvores de decisão e treina com a parte de treino
modelo = RandomForestClassifier(n_estimators=200, random_state=42)
modelo.fit(X_treino, y_treino)

# Faz a "prova": o modelo tenta adivinhar as letras dos exemplos de teste
previsoes = modelo.predict(X_teste)
# Acurácia = porcentagem de letras que ele acertou
print(f"\nAcuracia no teste: {accuracy_score(y_teste, previsoes):.1%}")
# Relatório com o desempenho de cada letra separadamente
print(classification_report(y_teste, previsoes))

# ---------------------------------------------------------------------------
# 4) TREINO FINAL E SALVAMENTO DO MODELO
# ---------------------------------------------------------------------------

# A prova já serviu para medir a qualidade. Agora o modelo é treinado de novo
# com todos os dados (treino + teste), para aprender com o máximo de exemplos.
modelo.fit(X, y)
# Cria a pasta "modelo" caso ainda não exista
CAMINHO_MODELO.parent.mkdir(parents=True, exist_ok=True)
# Salva o modelo em arquivo para o reconhecer.py poder carregar depois
joblib.dump(modelo, CAMINHO_MODELO)
print(f"\nModelo salvo em {CAMINHO_MODELO}")

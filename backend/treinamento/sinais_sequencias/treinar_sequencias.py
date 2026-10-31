"""
treinar_sequencias.py
 
PARA QUE SERVE
--------------
Treina o modelo que reconhece os sinais com MOVIMENTO (no projeto: K, BEM, ELE
e ENSINAR). Ele lê as sequências gravadas pelo coletar_sequencias.py (arquivo 
sequencias.csv), treina um modelo de aprendizado de máquina com elas e salva o 
modelo pronto em um arquivo. É o equivalente do treinar.py, só que para sinais 
em movimento em vez de letras paradas. O modelo aprende todos os sinais que
estiverem no sequencias.csv, não só os da lista SINAIS atual.
 
COMO FUNCIONA (passo a passo)
-----------------------------
1. Lê o sequencias.csv. Cada linha é uma repetição de um sinal: a primeira
   coluna é o sinal (a "resposta certa") e as demais são os números da
   sequência inteira (formato da mão + movimento do punho ao longo das
   imagens).
2. Mostra no terminal quantas sequências foram lidas, quantos números cada
   uma tem e quantas existem de cada sinal, para você conferir se algum
   ficou com poucos dados.
3. Separa os dados em duas partes: 80% para o modelo estudar (treino) e
   20% para fazer uma prova (teste), com exemplos que ele nunca viu.
4. Treina um modelo do tipo Random Forest (uma "floresta" de várias árvores
   de decisão que votam juntas na resposta) com a parte de treino.
5. Aplica a prova: o modelo tenta adivinhar os sinais da parte de teste e o
   script mostra a acurácia (porcentagem de acertos) e um relatório com o
   desempenho de cada sinal.
6. Depois da prova, treina o modelo de novo usando TODOS os dados, para ele
   aprender o máximo possível, e salva o resultado em
   backend/treinamento/modelo/modelo_sinais.joblib.
 
DIFERENÇAS EM RELAÇÃO AO treinar.py
-----------------------------------
- Lê o sequencias.csv (e não o pontos.csv).
- Salva em modelo_sinais.joblib (e não em modelo_letras.joblib).
- Usa 300 árvores na floresta (em vez de 200).
- Mostra também quantos números tem cada sequência.
 
ENTRADA
-------
- Arquivo backend/treinamento/dados/sequencias.csv (gerado pelo 
  coletar_sequencias.py).
 
SAÍDA
-----
- Arquivo backend/treinamento/modelo/modelo_sinais.joblib (a pasta "modelo" é
  criada se não existir).
- No terminal: total de sequências, sequências por sinal, acurácia no teste
  e relatório por sinal.
 
DEPENDÊNCIAS
------------
Python 3.11.9 e as bibliotecas (versões usadas no projeto):
- numpy 2.4.6
- joblib 1.6.0
- scikit-learn 1.9.1 (sklearn)
Importante: o modelo salvo só é garantido de funcionar na mesma versão do
scikit-learn em que foi treinado. Se trocar de versão, treine de novo.
Para instalar tudo de uma vez, veja o requirements.txt e o README.md.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

import csv
import joblib
import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES: onde ler os dados e onde salvar o modelo
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parents[1]   # backend/treinamento

ARQUIVO = BASE / "dados" / "sequencias.csv"
CAMINHO_MODELO = BASE / "modelo" / "modelo_sinais.joblib"

# ---------------------------------------------------------------------------
# 1) LEITURA DOS DADOS
# ---------------------------------------------------------------------------

# X: os exemplos (números de cada sequência)
# y: as respostas certas (o sinal de cada exemplo)
X, y = [], []
with open(ARQUIVO) as arquivo:
    for linha in csv.reader(arquivo):
        # Primeira coluna = sinal
        y.append(linha[0])
        # Demais colunas = números da sequência, convertidos de texto para decimal
        X.append([float(v) for v in linha[1:]])

# Converte as listas para arrays do numpy, que é o formato usado pelo sklearn
X = np.array(X)
y = np.array(y)

# Mostra o total de sequências e quantos números cada uma tem
# (X.shape[1] é o número de colunas, ou seja, de valores por sequência)
print(f"{len(y)} sequencias, {X.shape[1]} numeros em cada.")
# Mostra quantas sequências cada sinal tem (em ordem alfabética).
# Serve para conferir se os dados estão equilibrados entre os sinais.
print("Sequencias por sinal:")
for sinal, qtd in sorted(Counter(y).items()):
    print(f"  {sinal}: {qtd}")

# ---------------------------------------------------------------------------
# 2) SEPARAÇÃO ENTRE TREINO E TESTE
# ---------------------------------------------------------------------------

# - test_size=0.2: 20% dos dados ficam reservados para a prova
# - random_state=42: fixa o sorteio, para o resultado ser sempre o mesmo
#   se o script rodar de novo com os mesmos dados
# - stratify=y: mantém a mesma proporção de cada sinal no treino e no teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 3) TREINO E AVALIAÇÃO
# ---------------------------------------------------------------------------

# Cria a Random Forest com 300 árvores de decisão e treina com a parte de treino
modelo = RandomForestClassifier(n_estimators=300, random_state=42)
modelo.fit(X_treino, y_treino)

# Faz a "prova": o modelo tenta adivinhar os sinais dos exemplos de teste
previsoes = modelo.predict(X_teste)
# Acurácia = porcentagem de sinais que ele acertou
print(f"\nAcuracia no teste: {accuracy_score(y_teste, previsoes):.1%}")
# Relatório com o desempenho de cada sinal separadamente
print(classification_report(y_teste, previsoes))

# ---------------------------------------------------------------------------
# 4) TREINO FINAL E SALVAMENTO DO MODELO
# ---------------------------------------------------------------------------

# A prova já serviu para medir a qualidade. Agora o modelo é treinado de novo
# com todos os dados (treino + teste), para aprender com o máximo de exemplos.
modelo.fit(X, y)
# Cria a pasta "modelo" caso ainda não exista
CAMINHO_MODELO.parent.mkdir(parents=True, exist_ok=True)
#  Salva o modelo em arquivo para poder ser carregado depois
joblib.dump(modelo, CAMINHO_MODELO)
print(f"\nModelo salvo em {CAMINHO_MODELO}")

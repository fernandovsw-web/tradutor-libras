import csv
import os
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ARQUIVO_DADOS = os.path.join("../dados", "sequencias.csv")
ARQUIVO_MODELO = os.path.join("../modelo", "modelo_sinais.joblib")


X, y = [], []
with open(ARQUIVO_DADOS) as arquivo:
    for linha in csv.reader(arquivo):
        y.append(linha[0])
        X.append([float(v) for v in linha[1:]])

X = np.array(X)
y = np.array(y)

print(f"{len(y)} sequencias, {X.shape[1]} numeros em cada.")
print("Sequencias por sinal:")
for sinal, qtd in sorted(Counter(y).items()):
    print(f"  {sinal}: {qtd}")


X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


modelo = RandomForestClassifier(n_estimators=300, random_state=42)
modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)
print(f"\nAcuracia no teste: {accuracy_score(y_teste, previsoes):.1%}")
print(classification_report(y_teste, previsoes))

modelo.fit(X, y)
os.makedirs("../modelo", exist_ok=True)
joblib.dump(modelo, ARQUIVO_MODELO)
print(f"\nModelo salvo em {ARQUIVO_MODELO}")

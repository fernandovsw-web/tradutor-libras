# Libras:D

Projeto extensionista da disciplina de **Processamento de Imagens**. É um protótipo feito em Python 3.11.9 que usa a webcam para reconhecer **letras do alfabeto manual** (sinais estáticos) e **sinais com movimento** em Libras e traduzi-los para texto, mostrando também a confiança do resultado.

> **O programa não possui fluência em Libras.** É um projeto experimental: ele só reconhece os sinais incluídos no treinamento e não representa uma tradução completa da língua.

## Sumário

- [Sobre o projeto](#sobre-o-projeto)
- [Como funciona](#como-funciona)
- [Tecnologias](#tecnologias)
- [Estrutura de pastas](#estrutura-de-pastas)
- [Instalação](#instalação)
- [Executando o site](#executando-o-site)
- [Treinando seus próprios sinais](#treinando-seus-próprios-sinais)
- [Outros scripts](#outros-scripts)
- [Problemas comuns](#problemas-comuns)
- [Limitações e trabalhos futuros](#limitações-e-trabalhos-futuros)
- [Referências](#referências)

## Sobre o projeto

**Objetivo geral:** desenvolver um sistema baseado em Inteligência Artificial e Visão Computacional capaz de reconhecer sinais realizados em Libras por meio de uma câmera e traduzi-los em texto, contribuindo para facilitar a comunicação entre pessoas surdas e ouvintes.

**Objetivos específicos:**

- Usar Visão Computacional para capturar e identificar os movimentos das mãos por meio de uma câmera;
- Criar e organizar um dataset com sinais em Libras para treinar e testar o modelo;
- Aplicar modelos de Inteligência Artificial para reconhecer e classificar os sinais;
- Desenvolver uma interface simples e acessível que apresente a tradução em texto;
- Avaliar a precisão do sistema no reconhecimento e na tradução dos sinais;
- Testar o sistema em diferentes condições de iluminação, posições das mãos e distâncias da câmera.

**ODS 10 – Redução das Desigualdades:** a proposta contribui para diminuir as barreiras de comunicação entre pessoas que usam Libras e pessoas ouvintes que não conhecem a língua, promovendo inclusão, acessibilidade e igualdade de oportunidades.

**Motivação:** segundo o Censo 2022 (IBGE), cerca de 2,6 milhões de pessoas declararam alguma dificuldade permanente para ouvir (dado preliminar, sobre dificuldade auditiva em geral, e não especificamente sobre o uso de Libras). A Libras é reconhecida pela Lei nº 10.436/2002, mas a maior parte da população ouvinte não a domina. Ferramentas como o VLibras convertem português para Libras, mas funcionam em uma só direção. Este projeto atua no sentido contrário: Libras para texto.

## Como funciona

O fluxo tem três etapas, aplicadas separadamente às letras e aos sinais com movimento: **coleta de dados → treinamento → reconhecimento em tempo real**.

1. **Detecção da mão:** o MediaPipe Hands encontra a mão na imagem e devolve 21 pontos (x, y, z).
2. **Pré-processamento** (`backend/utilitarios/utils.py`): a função `normalizar` transforma os pontos em um vetor que descreve só o *formato* da mão, sem depender da posição nem do tamanho dela na tela. Para sinais com movimento, `montar_sequencia` junta `FRAMES_POR_SINAL` imagens seguidas (formato da mão + deslocamento do punho).
3. **Classificação:** um modelo Random Forest (scikit-learn) devolve a probabilidade de cada sinal. Se a maior probabilidade ficar abaixo do limite de confiança, o resultado é `?` em vez de um chute.
4. **Site:** o navegador envia os quadros da câmera para o Flask (`/api/traduzir`), e `backend/traduzir.py` escolhe sozinho o modelo: se a mão está **parada**, usa o modelo de letras (com votação entre as últimas leituras); se a mão está **em movimento**, grava uma sequência e usa o modelo de sinais com movimento.

Sinais com movimento atualmente treinados: **K, BEM, ELE e ENSINAR**.

## Tecnologias

| Parte | Tecnologia |
|---|---|
| Linguagem | Python 3.11.9 |
| Captura e exibição de vídeo | OpenCV |
| Detecção da mão (21 pontos) | MediaPipe Hands |
| Operações numéricas | NumPy |
| Classificador, divisão treino/teste e métricas | scikit-learn (Random Forest) |
| Salvar/carregar o modelo | Joblib |
| Servidor web | Flask |
| Interface | HTML, CSS e JavaScript |
| Versionamento | Git / GitHub |

## Estrutura de pastas

```
Tradutor-Libras/
├── backend/
│   ├── main.py                    # versão simples: reconhece letras pela webcam
│   ├── traduzir.py                # rota /api/traduzir (escolhe letra ou movimento)
│   ├── utilitarios/
│   │   └── utils.py               # normalizar, posicao_punho, montar_sequencia...
│   └── treinamento/
│       ├── dados/                 # pontos.csv e sequencias.csv (gerados pela coleta)
│       ├── modelo/                # modelo_letras.joblib e modelo_sinais.joblib
│       ├── sinais_estaticos/      # coletar_dados.py, treinar.py, reconhecer.py,
│       │                          # passo1_maos.py
│       └── sinais_sequencias/     # coletar_sequencias.py, treinar_sequencias.py,
│                                  # reconhecer_sinais.py
├── frontend/
│   ├── web_app.py                 # servidor Flask
│   ├── templates/                 # index.html e tradutor.html
│   └── static/
│       ├── css/style.css
│       └── js/script.js
├── .gitignore
├── requirements.txt
└── README.md
```

Os arquivos `__init__.py` vazios fazem de cada pasta um pacote Python, o que permite os imports `from backend.utilitarios.utils import ...`. Os modelos já treinados e os CSVs de dados acompanham o repositório, então o site pode ser executado sem treinar nada.

## Instalação

**Pré-requisitos**

- [Python 3.11.9](https://www.python.org/downloads/release/python-3119/) (marque "Add Python to PATH" ao instalar)
- Uma webcam
- Windows, para os scripts de terminal (eles usam `cv2.CAP_DSHOW`). O site usa a câmera do navegador.

**Passo a passo (PowerShell, na pasta principal do projeto)**

```powershell
# 1. Baixe o projeto
git clone https://github.com/fernandovsw-web/Tradutor-Libras.git
cd Tradutor-Libras

# 2. Crie o ambiente virtual (a pasta .venv)
python -m venv .venv

# 3. Instale as bibliotecas nas versões testadas
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> Os modelos (`.joblib`) só funcionam na mesma versão do scikit-learn em que foram treinados (**1.9.1**). Se trocar de versão, treine de novo.

## Executando o site

```powershell
.\.venv\Scripts\python.exe frontend\web_app.py
```

Abra **http://127.0.0.1:5000** no navegador, clique em **Abrir tradutor** e depois em **Iniciar câmera**.

- Use `localhost` ou `127.0.0.1`: o navegador só libera a câmera nesses endereços ou em HTTPS.
- Permita o acesso à câmera quando o navegador pedir.
- Para **letras**, mantenha a mão parada e bem iluminada. Para **sinais com movimento**, faça o sinal normalmente e aguarde o resultado (ele fica alguns segundos na tela).
- O servidor só sobe se os **dois** modelos existirem em `backend/treinamento/modelo/`.
- Para encerrar, aperte `Ctrl + C` no terminal.

## Treinando seus próprios sinais

Rode os comandos na pasta principal. A primeira linha permite que os scripts encontrem o pacote `backend`:

```powershell
$env:PYTHONPATH = $PWD
```

### Letras (sinais parados)

```powershell
# 1. Coleta: aperte uma letra (a-z) e faça o sinal. Grava 150 amostras por vez.
.\.venv\Scripts\python.exe backend\treinamento\sinais_estaticos\coletar_dados.py

# 2. Treino: lê pontos.csv, mostra a acurácia e salva modelo_letras.joblib
.\.venv\Scripts\python.exe backend\treinamento\sinais_estaticos\treinar.py

# 3. Teste: ESPAÇO adiciona a letra ao texto, BACKSPACE apaga, ESC sai
.\.venv\Scripts\python.exe backend\treinamento\sinais_estaticos\reconhecer.py
```

### Sinais com movimento

Os sinais ficam na lista `SINAIS` de `coletar_sequencias.py` (teclas 1 a 4).

```powershell
# 1. Coleta: aperte 1-4 e faça o sinal 30 vezes (uma por gravação)
.\.venv\Scripts\python.exe backend\treinamento\sinais_sequencias\coletar_sequencias.py

# 2. Treino: lê sequencias.csv e salva modelo_sinais.joblib
.\.venv\Scripts\python.exe backend\treinamento\sinais_sequencias\treinar_sequencias.py

# 3. Teste: ESPAÇO inicia o reconhecimento, ESC sai
.\.venv\Scripts\python.exe backend\treinamento\sinais_sequencias\reconhecer_sinais.py
```

**Importante**

- As coletas **acrescentam** linhas ao CSV (modo `a`); elas não apagam gravações antigas. Para recomeçar do zero, apague o CSV.
- Se mudar `FRAMES_POR_SINAL` em `utils.py`, é preciso coletar e treinar de novo (o servidor confere isso ao iniciar).
- Se a mão sumir por mais de `MAX_FRAMES_PERDIDOS` imagens durante uma gravação, ela é descartada e refeita.

## Outros scripts

| Script | Para que serve |
|---|---|
| `backend/treinamento/sinais_estaticos/passo1_maos.py` | Testa se a câmera e a detecção da mão funcionam (tecla `Q` sai) |
| `backend/main.py` | Reconhecimento de letras simples, sem votação (tecla `Q` sai). Rode com `python -m backend.main` |

## Problemas comuns

| Mensagem ou sintoma | O que fazer |
|---|---|
| `No module named 'backend'` | Rode o comando `$env:PYTHONPATH = $PWD` na pasta principal antes do script |
| `Modelo nao encontrado` | Rode o treinamento (`treinar.py` e `treinar_sequencias.py`) antes |
| Erro ao abrir o modelo `.joblib` | Versão do scikit-learn diferente da usada no treino: instale `scikit-learn==1.9.1` ou treine de novo |
| `Nao consegui abrir a webcam` | Feche outros programas que estejam usando a câmera |
| Câmera não abre no site | Acesse por `localhost`/`127.0.0.1` e permita a câmera no navegador |
| Letras disparando o modo de movimento (ou o contrário) | Ajuste `VELOCIDADE_INICIO` em `backend/traduzir.py` |

## Limitações e trabalhos futuros

- O sistema reconhece apenas o que foi treinado, e ainda precisa de mais dados para melhorar a precisão.
- Não interpreta frases completas nem a gramática da Libras.
- O reconhecimento de movimento depende da qualidade da imagem, da iluminação e da velocidade do envio dos quadros pelo navegador.
- Próximos passos: ampliar o vocabulário (mais palavras e frases), aumentar a quantidade e a variedade de gravações e aperfeiçoar a análise da trajetória da mão.

## Referências

- BRASIL. Lei nº 10.436, de 24 de abril de 2002. Dispõe sobre a Língua Brasileira de Sinais – Libras. <https://www.planalto.gov.br/ccivil_03/leis/2002/l10436.htm>
- BRASIL. VLibras. <https://www.gov.br/governodigital/pt-br/acessibilidade-e-usuario/vlibras>
- IBGE. Censo Demográfico 2022: pessoas com deficiência (resultados preliminares da amostra). 2025. <https://biblioteca.ibge.gov.br/visualizacao/livros/liv102178.pdf>
- IBGE. PNS 2019: país tem 17,3 milhões de pessoas com algum tipo de deficiência. 2021.

## Equipe
- Fernando Valim da Silva
- Franco Severo Biasio
- Lucas da Fonseca Perin
- Nadiny Venturini da Silva
- Paolo Antônio Junges
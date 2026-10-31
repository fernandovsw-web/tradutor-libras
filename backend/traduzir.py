"""
traduzir.py

PARA QUE SERVE
--------------
É o "cérebro" que junta os dois modelos no site. O navegador manda os
quadros da câmera sem parar para a rota /api/traduzir e este módulo decide,
sozinho, qual modelo usar:

- Mão PARADA  -> modelo de letras (modelo_letras.joblib), com votação entre
                 as últimas leituras (semelhante ao reconhecer.py).
- Mão em MOVIMENTO -> modelo de sinais com movimento
                 (modelo_sinais.joblib), que olha uma sequência de
                 FRAMES_POR_SINAL imagens (igual ao reconhecer_sinais.py).

O resultado (letra ou sinal) volta no mesmo formato, para aparecer no mesmo
campo da tela.

COMO FUNCIONA (passo a passo)
-----------------------------
1. Cada pessoa usando o site tem uma "sessão" (identificada por um id que o
   navegador gera). A sessão guarda o histórico recente de frames, porque o
   modelo de movimento precisa de vários frames seguidos e cada requisição
   HTTP traz só um.
2. Em cada frame o MediaPipe acha a mão. O frame é espelhado antes (cv2.flip),
   exatamente como na coleta e no treino.
3. O módulo mede a velocidade do punho nos últimos instantes:
   - abaixo do limite (VELOCIDADE_INICIO): classifica a letra com o modelo
     estático;
   - acima do limite: começa a "capturar" um sinal em movimento. A captura
     começa um pouquinho antes (PRE_ROLL), dura o mesmo tempo da coleta
     (FRAMES_POR_SINAL / FPS_TREINO) e, no fim, o modelo de movimento
     classifica.
4. Como a rede entrega os frames em ritmo irregular (e mais devagar que os
   30 fps da coleta), os frames da captura são reamostrados no tempo para
   formar exatamente FRAMES_POR_SINAL imagens igualmente espaçadas. Assim o
   vetor tem o mesmo formato do treino (montar_sequencia).
5. Depois de reconhecer um movimento, o resultado fica na tela por alguns
   segundos e a detecção de novo movimento fica pausada, para o gesto de
   "baixar a mão" não disparar outro sinal.

COMO USAR
---------
No arquivo principal do Flask (onde ficam as rotas inicio e tradutor):

    from backend.traduzir import traduzir_bp
    app.register_blueprint(traduzir_bp)

A rota antiga /api/reconhecer continua intacta.

DEPENDÊNCIAS
------------
Python 3.11.9 e as bibliotecas (versões usadas no projeto):
- Flask 3.1.3
- numpy 2.4.6
- opencv-python 5.0.0.93 (cv2)
- mediapipe 0.10.14
- joblib 1.6.0
- scikit-learn 1.9.1 (necessário para abrir os modelos salvos)
- módulo próprio backend.utilitarios.utils
Precisa dos dois modelos já treinados em backend/treinamento/modelo/ (letras e
sinais), treinados na mesma versão do scikit-learn. Sem os dois arquivos, o
servidor nem chega a subir.
Para instalar tudo de uma vez, veja o requirements.txt e o README.md.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

import base64
import threading
import time
import cv2
import joblib
import mediapipe as mp
import numpy as np
from collections import Counter, deque
from pathlib import Path
from flask import Blueprint, jsonify, request
from backend.utilitarios.utils import (
    FRAMES_POR_SINAL,
    MAX_FRAMES_PERDIDOS,
    montar_sequencia,
    normalizar,
    posicao_punho,
)
import warnings
# Esconde um aviso do MediaPipe que não afeta o funcionamento
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES (ajuste aqui se precisar calibrar)
# ---------------------------------------------------------------------------

# Pasta dos modelos: backend/treinamento/modelo/
PASTA_MODELOS = Path(__file__).resolve().parent / "treinamento" / "modelo"

# --- Modelo estático (letras) ---
LIMITE_ESTATICO = 0.6   # confiança mínima para aceitar a letra
JANELA_VOTOS = 6        # quantas leituras recentes entram na votação

# --- Modelo de movimento ---
LIMITE_MOVIMENTO = 0.5  # confiança mínima para aceitar o sinal

# Velocidade do punho (fração da largura da imagem por segundo) a partir da
# qual a mão é considerada "em movimento". Se letras estáticas estiverem
# disparando o modelo de movimento sem querer, SUBA este valor; se os sinais
# em movimento não estiverem sendo detectados, DESÇA.
VELOCIDADE_INICIO = 0.25
JANELA_VELOCIDADE = 0.3   # segundos olhados para medir a velocidade
TEMPO_MINIMO_MAO = 0.5    # a mão precisa estar visível por este tempo antes
                          # de um movimento poder disparar (evita disparar
                          # quando a mão está entrando na imagem)

FPS_TREINO = 30                                    # fps da coleta de dados
DURACAO_SINAL = FRAMES_POR_SINAL / FPS_TREINO      # segundos de um sinal
PRE_ROLL = 0.35           # a captura começa este tanto ANTES de o movimento
                          # ser percebido (a detecção sempre chega atrasada)
MARGEM_INICIO = 0.3       # frames anteriores ao início usados na reamostragem
# Tempo máximo sem a mão (ou entre dois frames) dentro de um sinal. É o mesmo
# limite da coleta (MAX_FRAMES_PERDIDOS a 30 fps), dobrado porque aqui os
# frames chegam menos vezes por segundo.
TEMPO_MAX_PERDIDO = 2 * MAX_FRAMES_PERDIDOS / FPS_TREINO

TEMPO_EXIBICAO = 2.0      # quanto tempo o sinal em movimento fica na tela
TEMPO_EXIBICAO_ERRO = 0.8 # idem para "não reconhecido" / "mão perdida"

HISTORICO_SEG = 3.0       # quanto histórico de frames guardar
SESSAO_EXPIRA_SEG = 60    # sessão sem frames por este tempo é descartada

# ---------------------------------------------------------------------------
# CARREGAMENTO DOS MODELOS (uma vez, ao iniciar o servidor)
# ---------------------------------------------------------------------------


def _carregar(nome):
    caminho = PASTA_MODELOS / nome
    if not caminho.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado: {caminho}. Execute o treinamento primeiro."
        )
    return joblib.load(caminho)


MODELO_LETRAS = _carregar("modelo_letras.joblib")
MODELO_SINAIS = _carregar("modelo_sinais.joblib")

# Cada frame vira 63 números da mão (21 pontos x 3) + 2 do deslocamento do punho
_ESPERADO = FRAMES_POR_SINAL * (63 + 2)
if MODELO_SINAIS.n_features_in_ != _ESPERADO:
    raise ValueError(
        f"modelo_sinais.joblib espera {MODELO_SINAIS.n_features_in_} numeros por "
        f"sequencia, mas FRAMES_POR_SINAL={FRAMES_POR_SINAL} gera {_ESPERADO}. "
        "Treine de novo ou ajuste FRAMES_POR_SINAL em utils.py."
    )

# ---------------------------------------------------------------------------
# FUNÇÕES DE APOIO
# ---------------------------------------------------------------------------


def _responder(estado, tipo, texto=None, confianca=None, mensagem=""):
    """Monta a resposta JSON.

    estado: "livre" ou "capturando" (movimento sendo gravado)
    tipo: "estatico", "movimento" ou "nenhum"
    "letra" é mantido por compatibilidade com a rota antiga.
    """
    return {
        "estado": estado,
        "tipo": tipo,
        "texto": texto,
        "letra": texto if tipo == "estatico" else None,
        "confianca": None if confianca is None else float(confianca),
        "mensagem": mensagem,
    }


def _montar_entrada(historico, t0, t1):
    """Transforma os frames guardados em UM vetor no formato do treino.

    Pega FRAMES_POR_SINAL instantes igualmente espaçados entre t0 e t1 e, para
    cada um, usa o frame mais próximo no tempo. Devolve None se faltarem
    frames ou se houver um buraco grande demais (mão perdida).
    """
    frames = [f for f in historico if f[0] >= t0 - MARGEM_INICIO]
    if len(frames) < 5:
        return None

    tempos = np.array([f[0] for f in frames])
    if np.max(np.diff(tempos)) > TEMPO_MAX_PERDIDO:
        return None

    alvos = np.linspace(t0, t1, FRAMES_POR_SINAL)
    pos = np.clip(np.searchsorted(tempos, alvos), 1, len(tempos) - 1)
    # Entre o frame anterior e o seguinte ao instante alvo, fica o mais perto
    escolhidos = np.where(alvos - tempos[pos - 1] <= tempos[pos] - alvos, pos - 1, pos)

    formatos = [frames[i][1] for i in escolhidos]
    punhos = [frames[i][2] for i in escolhidos]
    return montar_sequencia(formatos, punhos).reshape(1, -1)


# ---------------------------------------------------------------------------
# SESSÃO: o estado de UMA pessoa usando o tradutor
# ---------------------------------------------------------------------------


class Sessao:
    def __init__(self):
        # Um detector por sessão, porque o MediaPipe em modo vídeo guarda
        # estado entre frames e não pode ser misturado entre pessoas.
        # Valores de confiança iguais aos da coleta de movimento.
        self.maos = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
        )
        self.trava = threading.Lock()
        self.visto_em = time.time()

        self.historico = deque()                  # (tempo, formato, punho)
        self.votos = deque(maxlen=JANELA_VOTOS)   # votação das letras
        self.estado = "livre"                     # "livre" ou "capturando"
        self.t_inicio = 0.0                       # início da captura atual
        self.ultimo_com_mao = 0.0
        self.bloqueado_ate = 0.0                  # exibindo resultado até aqui
        self.resp_bloqueada = None                # resposta mostrada nesse período

    def fechar(self):
        self.maos.close()

    # --- entrada: um frame de imagem --------------------------------------

    def processar(self, frame):
        agora = time.time()
        self.visto_em = agora

        # Espelha igual à coleta/treino e converte BGR -> RGB para o MediaPipe
        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        resultado = self.maos.process(rgb)

        if resultado.multi_hand_landmarks:
            mao = resultado.multi_hand_landmarks[0]
            return self._avancar(agora, normalizar(mao), posicao_punho(mao))
        return self._avancar(agora, None, None)

    # --- lógica principal (separada para poder ser testada sem câmera) ----

    def _avancar(self, agora, formato, punho):
        if formato is None:
            return self._sem_mao(agora)

        self.ultimo_com_mao = agora
        self.historico.append((agora, formato, punho))
        while agora - self.historico[0][0] > HISTORICO_SEG:
            self.historico.popleft()

        # Mostrando o resultado de um sinal em movimento: nada de novo por enquanto
        if agora < self.bloqueado_ate and self.resp_bloqueada:
            return self.resp_bloqueada

        if self.estado == "capturando":
            return self._capturar(agora)

        if self._velocidade(agora) >= VELOCIDADE_INICIO:
            self.estado = "capturando"
            self.t_inicio = agora - PRE_ROLL
            self.votos.clear()
            return self._capturar(agora)

        return self._estatico(formato)

    def _sem_mao(self, agora):
        self.votos.clear()

        # Se acabou de reconhecer um movimento, mantém o resultado na tela
        if self.estado == "livre" and agora < self.bloqueado_ate and self.resp_bloqueada:
            return self.resp_bloqueada

        if self.estado == "capturando":
            if agora - self.ultimo_com_mao > TEMPO_MAX_PERDIDO:
                # Mão sumiu por tempo demais: descarta a captura
                self.estado = "livre"
                self.historico.clear()
                resp = _responder("livre", "nenhum",
                                  mensagem="Mão perdida durante o movimento. Tente de novo.")
                self.resp_bloqueada = resp
                self.bloqueado_ate = agora + TEMPO_EXIBICAO_ERRO
                return resp
            return _responder("capturando", "nenhum", mensagem="Reconhecendo movimento...")

        if agora - self.ultimo_com_mao > 0.3:
            self.historico.clear()
        return _responder("livre", "nenhum", mensagem="Nenhuma mão detectada.")

    # --- movimento ---------------------------------------------------------

    def _velocidade(self, agora):
        """Velocidade do punho (largura da imagem por segundo) nos últimos instantes."""
        # A mão precisa estar visível há um tempo mínimo (não conta ela entrando)
        if agora - self.historico[0][0] < TEMPO_MINIMO_MAO:
            return 0.0

        recentes = [(t, p) for t, _, p in self.historico if agora - t <= JANELA_VELOCIDADE]
        if len(recentes) < 2:
            return 0.0
        dt = recentes[-1][0] - recentes[0][0]
        if dt < 0.15:
            return 0.0

        # Soma o caminho percorrido (não só o deslocamento final), para
        # pegar também movimentos de ida e volta
        caminho = sum(
            float(np.linalg.norm(b[1] - a[1])) for a, b in zip(recentes, recentes[1:])
        )
        return caminho / dt

    def _capturar(self, agora):
        if agora - self.t_inicio >= DURACAO_SINAL:
            return self._finalizar(agora)
        return _responder("capturando", "nenhum", mensagem="Reconhecendo movimento...")

    def _finalizar(self, agora):
        t0 = self.t_inicio
        entrada = _montar_entrada(self.historico, t0, t0 + DURACAO_SINAL)
        self.estado = "livre"

        if entrada is None:
            resp = _responder("livre", "nenhum",
                              mensagem="Não consegui acompanhar o movimento. Tente de novo.")
            espera = TEMPO_EXIBICAO_ERRO
        else:
            probabilidades = MODELO_SINAIS.predict_proba(entrada)[0]
            melhor = probabilidades.argmax()
            confianca = probabilidades[melhor]
            if confianca >= LIMITE_MOVIMENTO:
                resp = _responder("livre", "movimento",
                                  texto=str(MODELO_SINAIS.classes_[melhor]),
                                  confianca=confianca,
                                  mensagem="Sinal em movimento reconhecido.")
                espera = TEMPO_EXIBICAO
            else:
                resp = _responder("livre", "nenhum", confianca=confianca,
                                  mensagem="Movimento não reconhecido.")
                espera = TEMPO_EXIBICAO_ERRO

        self.resp_bloqueada = resp
        self.bloqueado_ate = agora + espera
        # Recomeça o histórico: um novo movimento só vale depois de a mão
        # ficar visível por TEMPO_MINIMO_MAO de novo
        self.historico.clear()
        self.votos.clear()
        return resp

    # --- letras (mão parada) ----------------------------------------------

    def _estatico(self, formato):
        probabilidades = MODELO_LETRAS.predict_proba(formato.reshape(1, -1))[0]
        melhor = probabilidades.argmax()
        confianca = probabilidades[melhor]

        self.votos.append(
            str(MODELO_LETRAS.classes_[melhor]) if confianca >= LIMITE_ESTATICO else "?"
        )
        letra = Counter(self.votos).most_common(1)[0][0]

        if letra == "?":
            return _responder("livre", "nenhum", confianca=confianca,
                              mensagem="Não tenho certeza deste sinal.")
        return _responder("livre", "estatico", texto=letra, confianca=confianca,
                          mensagem="Sinal estático reconhecido.")


# ---------------------------------------------------------------------------
# GERENCIAMENTO DE SESSÕES
# ---------------------------------------------------------------------------

_sessoes = {}
_trava_sessoes = threading.Lock()


def _obter_sessao(sessao_id):
    agora = time.time()
    with _trava_sessoes:
        # Descarta sessões paradas há muito tempo (libera o detector de mãos)
        for chave in [k for k, s in _sessoes.items() if agora - s.visto_em > SESSAO_EXPIRA_SEG]:
            sessao_antiga = _sessoes.pop(chave)
            with sessao_antiga.trava:
                sessao_antiga.fechar()

        if sessao_id not in _sessoes:
            _sessoes[sessao_id] = Sessao()
        sessao = _sessoes[sessao_id]
        sessao.visto_em = agora
        return sessao


def _decodificar(imagem):
    """Converte 'data:image/jpeg;base64,...' em imagem do OpenCV (ou None)."""
    try:
        bruto = base64.b64decode(imagem.split(",", 1)[-1])
        return cv2.imdecode(np.frombuffer(bruto, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ROTA DO FLASK
# ---------------------------------------------------------------------------

traduzir_bp = Blueprint("traduzir", __name__)


@traduzir_bp.route("/api/traduzir", methods=["POST"])
def traduzir():
    dados = request.get_json(silent=True) or {}
    imagem = dados.get("imagem")
    sessao_id = str(dados.get("sessao") or "")

    if not imagem or not sessao_id or len(sessao_id) > 100:
        return jsonify({"erro": "Envie 'imagem' e 'sessao'."}), 400

    frame = _decodificar(imagem)
    if frame is None:
        return jsonify({"erro": "Imagem inválida."}), 400

    sessao = _obter_sessao(sessao_id)
    with sessao.trava:
        resposta = sessao.processar(frame)
    return jsonify(resposta)

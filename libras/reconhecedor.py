"""Lógica de reconhecimento: foto única, clipe de uma letra, e fluxo contínuo
(webcam / vídeo de uma palavra soletrada).

Saídas possíveis, como pede o tema:
    LETRA RECONHECIDA: <X>  |  palavra: <...>
    REPITA O SINAL (<motivo>)

Fluxo contínuo (máquina de estados):
  OBSERVANDO -- mão parada por N quadros --> decide letra ESTÁTICA
             -- mão se desloca ------------> MOVIMENTO (grava quadros)
  MOVIMENTO  -- mão para -----------------> decide letra COM MOVIMENTO
                                            (ou "transição": troca de letra, ignorada)
  Depois de uma decisão o sistema fica BLOQUEADO até a mão mudar de forma,
  se mover ou sumir, para não repetir a mesma letra a cada quadro.
  Letra dobrada (ex.: "LL"): faça um pequeno quique com a mão ou tire-a do
  quadro e volte.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .features import deslocamento, distancia_pose, features_dinamicas, features_estaticas
from .modelo import TRANSICAO, Decisao, ModeloLetras, decidir


@dataclass
class Evento:
    tipo: str              # "LETRA" | "REPITA"
    letra: str | None
    confianca: float
    motivo: str
    palavra: str
    dinamica: bool = False

    def __str__(self):
        if self.tipo == "LETRA":
            return f"LETRA RECONHECIDA: {self.letra} ({self.confianca:.0%})  |  palavra: {self.palavra}"
        return f"REPITA O SINAL ({self.motivo})  |  palavra: {self.palavra}"


def carregar_modelos(pasta: Path = config.DIR_MODELOS):
    est = ModeloLetras.carregar(pasta / "estatico.joblib")
    p = pasta / "dinamico.joblib"
    din = ModeloLetras.carregar(p) if p.exists() else None
    return est, din


# ---------------------------------------------------------------------------
# Entradas isoladas
# ---------------------------------------------------------------------------
def reconhecer_foto(mod_est: ModeloLetras, pts, mao) -> Decisao:
    if pts is None:
        return Decisao(None, 0.0, "nenhuma mão encontrada na imagem")
    return mod_est.decidir(features_estaticas(pts, mao))


def reconhecer_clipe(mod_est, mod_din, seq: np.ndarray, mao: str) -> tuple[Decisao, bool]:
    """Clipe com UMA letra. Se houve movimento -> modelo dinâmico;
    senão -> média das probabilidades estáticas nos quadros com mão."""
    ok = ~np.isnan(seq).any(axis=(1, 2))
    validos = [q for q in seq[ok]]
    if len(validos) < 3:
        return Decisao(None, 0.0, "mão não encontrada em quadros suficientes"), False
    mexeu = deslocamento(validos) > config.LIMIAR_MOVIMENTO
    if mexeu:
        if mod_din is None:
            return Decisao(None, 0.0, "movimento detectado, mas não há modelo de letras com movimento"), True
        f = features_dinamicas(seq, mao)
        if f is None:
            return Decisao(None, 0.0, "sequência curta demais"), True
        return mod_din.decidir(f), True
    X = np.stack([features_estaticas(q, mao) for q in validos])
    P = mod_est.probs(X)
    d = decidir(P.mean(0), mod_est.classes, mod_est.limiar, mod_est.margem)
    concord = (P.argmax(1) == P.mean(0).argmax()).mean()
    if d.aceita and concord < config.CONCORDANCIA_MINIMA:
        return Decisao(None, d.confianca, f"sinal instável ({concord:.0%} dos quadros concordam)", d.top), False
    return d, False


# ---------------------------------------------------------------------------
# Fluxo contínuo
# ---------------------------------------------------------------------------
class TradutorContinuo:
    def __init__(self, mod_est: ModeloLetras, mod_din: ModeloLetras | None = None):
        self.mod_est, self.mod_din = mod_est, mod_din
        self.palavra = ""
        self.estado = "AGUARDANDO MÃO"
        self.ultimo: Evento | None = None
        self._reset()

    def _reset(self):
        n = config.QUADROS_PARA_FIXAR
        self.janela: deque = deque(maxlen=n)          # pontos recentes
        self.probs: deque = deque(maxlen=n)           # probs estáticas recentes
        self.seg: list = []                           # quadros do movimento
        self.em_movimento = False
        self.bloqueado = False
        self.pose_bloqueio = None                     # forma da mão na última decisão
        self.sem_mao = 0
        self.mao = "Right"

    # -- palavra ------------------------------------------------------------
    def espaco(self):
        if self.palavra and not self.palavra.endswith(" "):
            self.palavra += " "

    def apagar(self):
        self.palavra = self.palavra[:-1]

    def limpar(self):
        self.palavra = ""

    # -- quadro a quadro ----------------------------------------------------
    def processar(self, pts: np.ndarray | None, mao: str = "Right") -> Evento | None:
        if pts is None:
            self.sem_mao += 1
            if self.em_movimento and self.sem_mao <= 2:
                self.seg.append(np.full((21, 3), np.nan))
            if self.sem_mao >= config.QUADROS_SEM_MAO_RESET:
                ev = self._fim_movimento() if self.em_movimento else None
                palavra = self.palavra
                self._reset()
                self.palavra = palavra
                self.estado = "AGUARDANDO MÃO"
                return ev
            return None
        self.sem_mao = 0
        self.mao = mao
        self.janela.append(pts)
        self.probs.append(self.mod_est.probs(features_estaticas(pts, mao))[0])

        if self.em_movimento:
            self.seg.append(pts)
            fim = list(self.janela)[-config.QUADROS_REPOUSO_FIM_MOV:]
            parado = (len(fim) >= config.QUADROS_REPOUSO_FIM_MOV and
                      deslocamento(fim) < config.LIMIAR_MOVIMENTO * 0.4)
            if parado or len(self.seg) >= config.MAX_QUADROS_MOVIMENTO:
                return self._fim_movimento(longo=not parado)
            self.estado = f"MOVIMENTO ({len(self.seg)} quadros)"
            return None

        if deslocamento(list(self.janela)) > config.LIMIAR_MOVIMENTO:
            self.em_movimento = True
            self.seg = list(self.janela)          # inclui o começo do movimento
            self.estado = "MOVIMENTO"
            return None

        if self.bloqueado:
            if (self.pose_bloqueio is not None and
                    distancia_pose(pts, self.pose_bloqueio, mao) > config.LIMIAR_MUDANCA_POSE):
                self._desbloquear()
            else:
                self.estado = "MUDE A MAO PARA A PROXIMA LETRA"
                return None
        if len(self.probs) < config.QUADROS_PARA_FIXAR:
            self.estado = f"SEGURE A LETRA ({len(self.probs)}/{config.QUADROS_PARA_FIXAR})"
            return None
        return self._decidir_estatica()

    def _decidir_estatica(self) -> Evento:
        P = np.stack(self.probs)
        media = P.mean(0)
        d = decidir(media, self.mod_est.classes, self.mod_est.limiar, self.mod_est.margem)
        concord = float((P.argmax(1) == media.argmax()).mean())
        if d.aceita and concord < config.CONCORDANCIA_MINIMA:
            d = Decisao(None, d.confianca, "sinal instável", d.top)
        self._bloquear(self.janela[-1])
        return self._emitir(d, dinamica=False)

    def _bloquear(self, pose):
        self.bloqueado, self.pose_bloqueio = True, pose

    def _desbloquear(self):
        self.bloqueado, self.pose_bloqueio = False, None
        self.probs.clear()                         # a próxima letra precisa de quadros novos
        self.probs.append(self.mod_est.probs(features_estaticas(self.janela[-1], self.mao))[0])

    def _fim_movimento(self, longo: bool = False) -> Evento | None:
        seg, self.seg, self.em_movimento = np.stack(self.seg), [], False
        # os últimos quadros (mão já parada) já valem para a próxima letra estática
        k = config.QUADROS_REPOUSO_FIM_MOV
        self.probs = deque(list(self.probs)[-k:], maxlen=config.QUADROS_PARA_FIXAR)
        self.janela = deque(list(self.janela)[-k:], maxlen=config.QUADROS_PARA_FIXAR)
        self.bloqueado, self.pose_bloqueio = False, None   # houve movimento: libera
        if longo:
            return self._emitir(Decisao(None, 0.0, "movimento longo demais"), True)
        if self.mod_din is None:
            self.estado = "OBSERVANDO"
            return None
        f = features_dinamicas(seg, self.mao)
        if f is None:
            return None
        d = self.mod_din.decidir(f)
        if not d.aceita and d.top and d.top[0][0] == TRANSICAO:
            self.estado = "OBSERVANDO"             # só trocou de letra: silêncio
            return None
        if d.aceita:
            # Se a mão terminou com a MESMA forma que tinha no meio do movimento
            # (ex.: fim do J = forma do I), bloqueia essa forma para não lê-la
            # como letra estática. Se terminou com outra forma, a pessoa já
            # emendou a próxima letra: não bloqueia.
            validos = [q for q in seg if not np.isnan(q).any()]
            meio, fim = validos[len(validos) // 2], validos[-1]
            if distancia_pose(meio, fim, self.mao) < config.LIMIAR_MUDANCA_POSE:
                self._bloquear(fim)
        return self._emitir(d, dinamica=True)

    def _emitir(self, d: Decisao, dinamica: bool) -> Evento:
        if d.aceita:
            self.palavra += d.letra
            ev = Evento("LETRA", d.letra, d.confianca, "", self.palavra, dinamica)
        else:
            ev = Evento("REPITA", None, d.confianca, d.motivo, self.palavra, dinamica)
        self.estado = str(ev).split("  |")[0]
        self.ultimo = ev
        return ev

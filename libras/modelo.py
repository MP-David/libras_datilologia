"""Montagem das matrizes de treino, modelos candidatos e regra de decisão
(LETRA RECONHECIDA x REPITA O SINAL)."""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .features import (aumentar_estatico, aumentar_sequencia, features_dinamicas,
                       features_estaticas)

# Classe extra do modelo de movimento: "a mão se mexeu mas não era letra"
# (a troca de uma letra para a outra também mexe a mão).
TRANSICAO = "_TRANSICAO"


# ---------------------------------------------------------------------------
# Matrizes
# ---------------------------------------------------------------------------
def matriz_estatica(amostras, aumentos: int = 0, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for s in amostras:
        versoes = [s["pts"]] + [aumentar_estatico(s["pts"], rng) for _ in range(aumentos)]
        for v in versoes:
            X.append(features_estaticas(v, s["mao"]))
            y.append(s["letra"])
            g.append(s["sinalizador"])
    return np.array(X), np.array(y), np.array(g)


def sintetizar_transicoes(estaticas, n_por_grupo: int, seed: int = 0):
    """Cria sequências 'pose A -> pose B' com deslocamento da mão, a partir de
    fotos estáticas do MESMO sinalizador. Ensina o modelo de movimento a não
    confundir troca de letra com letra que tem movimento."""
    rng = np.random.default_rng(seed)
    por_grupo: dict[str, list] = {}
    for s in estaticas:
        por_grupo.setdefault(s["sinalizador"], []).append(s)
    saida = []
    for grupo, lista in por_grupo.items():
        if len(lista) < 2:
            continue
        for _ in range(n_por_grupo):
            a, b = rng.choice(len(lista), 2, replace=False)
            A, B = lista[a]["pts"], lista[b]["pts"]
            if rng.random() < 0.3:      # às vezes: mesma pose só se deslocando
                B = A
            T = int(rng.integers(10, 30))
            w = np.linspace(0, 1, T)[:, None, None]
            esc = np.linalg.norm(A[9] - A[0])
            deriva = rng.normal(0, 0.6 * esc, 3) * np.array([1, 1, 0])
            seq = (1 - w) * A + w * B + w * deriva
            saida.append({"seq": seq, "mao": lista[a]["mao"], "letra": TRANSICAO,
                          "sinalizador": grupo, "origem": lista[a]["origem"],
                          "arquivo": "sintetico"})
    return saida


def com_contexto(seq: np.ndarray, poses: list, rng: np.random.Generator) -> np.ndarray:
    """Cola uma troca de letra antes e/ou depois do movimento. Ao vivo o trecho
    gravado costuma incluir a chegada à letra e a saída para a próxima."""
    ok = np.flatnonzero(~np.isnan(seq).any(axis=(1, 2)))
    if len(ok) == 0 or not poses:
        return seq
    partes = [seq]
    for lado in ("antes", "depois"):
        if rng.random() < 0.6:
            ref = seq[ok[0]] if lado == "antes" else seq[ok[-1]]
            outra = poses[rng.integers(len(poses))]
            outra = outra - outra[0] + ref[0] + rng.normal(0, 0.3, 3) * np.linalg.norm(ref[9] - ref[0]) * [1, 1, 0]
            w = np.linspace(0, 1, int(rng.integers(4, 11)))[:, None, None]
            ponte = (1 - w) * outra + w * ref if lado == "antes" else (1 - w) * ref + w * outra
            partes = [ponte] + partes if lado == "antes" else partes + [ponte]
    return np.concatenate(partes).astype(np.float32)


def matriz_dinamica(amostras, aumentos: int = 0, seed: int = 0, contexto: dict | None = None):
    """contexto: {sinalizador: [poses estáticas]} para o aumento com troca de letra."""
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for s in amostras:
        poses = (contexto or {}).get(s["sinalizador"], [])
        versoes = [s["seq"]] + [com_contexto(aumentar_sequencia(s["seq"], rng), poses, rng)
                                for _ in range(aumentos)]
        for v in versoes:
            f = features_dinamicas(v, s["mao"])
            if f is None:
                continue
            X.append(f)
            y.append(s["letra"])
            g.append(s["sinalizador"])
    return np.array(X), np.array(y), np.array(g)


# ---------------------------------------------------------------------------
# Modelos candidatos (comparados na validação por sinalizador)
# ---------------------------------------------------------------------------
def candidatos(seed: int = 0) -> dict:
    return {
        # SVC + calibração sigmoide = probabilidades para o limiar de rejeição
        "svm_rbf": make_pipeline(StandardScaler(), CalibratedClassifierCV(
            SVC(C=10, gamma="scale", class_weight="balanced"), method="sigmoid", cv=3,
            ensemble=False)),
        "extra_trees": ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
                                            n_jobs=-1, random_state=seed),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier((256, 128), alpha=1e-3,
                                                             early_stopping=True, max_iter=600,
                                                             random_state=seed)),
        "logistica": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=4000,
                                                                       class_weight="balanced")),
    }


# ---------------------------------------------------------------------------
# Decisão
# ---------------------------------------------------------------------------
def _mesmo_grupo(a: str, b: str) -> bool:
    return any(a in g and b in g for g in config.GRUPOS_CONFUNDIVEIS)


@dataclass
class Decisao:
    letra: str | None          # None => REPITA O SINAL
    confianca: float
    motivo: str = ""
    top: list = field(default_factory=list)   # [(letra, prob), ...]

    @property
    def aceita(self) -> bool:
        return self.letra is not None


def decidir(probs: np.ndarray, classes, limiar: float, margem: float) -> Decisao:
    ordem = np.argsort(probs)[::-1]
    top = [(str(classes[i]), float(probs[i])) for i in ordem[:3]]
    (l1, p1), (l2, p2) = top[0], (top[1] if len(top) > 1 else ("", 0.0))
    if l1 == TRANSICAO:
        return Decisao(None, p1, "movimento sem letra (transição)", top)
    if p1 < limiar:
        return Decisao(None, p1, f"confiança baixa ({p1:.0%})", top)
    exig = config.MARGEM_GRUPO_CONFUNDIVEL if _mesmo_grupo(l1, l2) else margem
    if p1 - p2 < exig:
        return Decisao(None, p1, f"dúvida entre {l1} e {l2}", top)
    return Decisao(l1, p1, "", top)


@dataclass
class ModeloLetras:
    tipo: str                  # "estatico" | "dinamico"
    estimador: object
    classes: list
    limiar: float = config.LIMIAR_CONFIANCA
    margem: float = config.MARGEM_MINIMA
    info: dict = field(default_factory=dict)

    def probs(self, X: np.ndarray) -> np.ndarray:
        return self.estimador.predict_proba(np.atleast_2d(X))

    def decidir(self, x: np.ndarray) -> Decisao:
        return decidir(self.probs(x)[0], self.classes, self.limiar, self.margem)

    def salvar(self, caminho):
        joblib.dump(self, caminho)

    @staticmethod
    def carregar(caminho) -> "ModeloLetras":
        return joblib.load(caminho)

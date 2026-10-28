"""Métricas por letra, por sinalizador, por grupo confundível e curva de rejeição."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import config
from .modelo import TRANSICAO, decidir


def tabela_por_letra(y_true, y_pred, letras) -> pd.DataFrame:
    letras = [l for l in letras if l in set(y_true)]
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=letras, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=letras)
    linhas = []
    for i, l in enumerate(letras):
        erros = cm[i].copy()
        erros[i] = 0
        j = int(erros.argmax())
        conf = f"{letras[j]} ({erros[j] / max(cm[i].sum(), 1):.0%})" if erros[j] > 0 else "—"
        linhas.append({"letra": l, "amostras": int(n[i]), "acuracia_letra": r[i],
                       "precisao": p[i], "f1": f[i], "mais_confundida_com": conf})
    return pd.DataFrame(linhas)


def acuracia_balanceada(y_true, y_pred) -> float:
    letras = sorted(set(y_true))
    _, r, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=letras, zero_division=0)
    return float(np.mean(r))


def por_sinalizador(y_true, y_pred, grupos) -> pd.DataFrame:
    df = pd.DataFrame({"s": grupos, "ok": np.asarray(y_true) == np.asarray(y_pred), "y": y_true})
    out = []
    for s, d in df.groupby("s"):
        out.append({"sinalizador": s, "amostras": len(d), "acuracia": d.ok.mean(),
                    "acuracia_balanceada": acuracia_balanceada(d.y.values,
                                                               np.asarray(y_pred)[d.index])})
    return pd.DataFrame(out)


def analise_grupos(y_true, y_pred) -> list[str]:
    linhas = []
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    for g in config.GRUPOS_CONFUNDIVEIS:
        if not any(l in set(yt) for l in g):
            continue
        m = confusion_matrix(yt, yp, labels=g)
        presentes = [l for l in g if (yt == l).sum()]
        if not presentes:
            continue
        cab = "| real \\ previsto | " + " | ".join(g) + " | outras |"
        linhas += [f"**Grupo {'/'.join(g)}**", "", cab, "|" + "---|" * (len(g) + 2)]
        for i, l in enumerate(g):
            tot = int((yt == l).sum())
            if tot == 0:
                continue
            outras = tot - int(m[i].sum())
            linhas.append(f"| {l} | " + " | ".join(str(v) for v in m[i]) + f" | {outras} |")
        linhas.append("")
    return linhas


def matriz_png(y_true, y_pred, letras, titulo, caminho):
    letras = [l for l in letras if l in set(y_true) | set(y_pred)]
    cm = confusion_matrix(y_true, y_pred, labels=letras).astype(float)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    tam = max(5, 0.42 * len(letras) + 2)
    fig, ax = plt.subplots(figsize=(tam, tam * 0.9))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    rot = [("∅" if l == TRANSICAO else l) for l in letras]
    ax.set_xticks(range(len(letras)), rot)
    ax.set_yticks(range(len(letras)), rot)
    ax.set_xlabel("previsto")
    ax.set_ylabel("real")
    ax.set_title(titulo, fontsize=10)
    for i in range(len(letras)):
        for j in range(len(letras)):
            if cm[i, j]:
                ax.text(j, i, f"{cmn[i, j]:.0%}", ha="center", va="center",
                        fontsize=6, color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.04)
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


def efeito_rejeicao(y_true, probs, classes, limiar, margem) -> dict:
    """Cobertura = fração das letras reais que recebem resposta (não REPITA).
    Acurácia aceita = acerto entre as respostas dadas (transições aceitas contam como erro)."""
    yt = np.asarray(y_true)
    dec = [decidir(p, classes, limiar, margem) for p in probs]
    aceito = np.array([d.aceita for d in dec])
    certo = np.array([d.letra == y for d, y in zip(dec, yt)])
    real = yt != TRANSICAO
    return {
        "limiar": limiar,
        "cobertura": float(aceito[real].mean()) if real.any() else 0.0,
        "acuracia_aceitas": float(certo[aceito].mean()) if aceito.any() else 0.0,
        "transicoes_aceitas": float(aceito[~real].mean()) if (~real).any() else 0.0,
    }


def calibrar_limiar(y_true, probs, classes, margem, alvo: float):
    curva = [efeito_rejeicao(y_true, probs, classes, l, margem)
             for l in np.round(np.arange(0.30, 0.96, 0.05), 2)]
    bons = [c for c in curva if c["acuracia_aceitas"] >= alvo and c["cobertura"] > 0]
    if bons:
        escolhido = min(bons, key=lambda c: c["limiar"])
    else:   # alvo inatingível: melhor compromisso acurácia x cobertura
        escolhido = max(curva, key=lambda c: c["acuracia_aceitas"] * np.sqrt(c["cobertura"]))
    return escolhido, curva


def tabela_md(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:.1%}")
    cab = "| " + " | ".join(d.columns) + " |"
    sep = "|" + "---|" * len(d.columns)
    return "\n".join([cab, sep] + ["| " + " | ".join(map(str, r)) + " |" for r in d.values])

"""Gera pontos de mão SINTÉTICOS para testar o pipeline sem câmera.

    python -m libras.sintetico --sinalizadores 6
    python -m libras.treinar --teste s5

Os números obtidos com isto NÃO dizem nada sobre LIBRAS real: as
configurações de mão são inventadas (com A/S/E e M/N propositalmente
parecidas). Serve só para conferir que coleta->treino->relatório->tradutor
funcionam de ponta a ponta. Grava em dados/processados/amostras_sinteticas.joblib.
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np

from . import config

BASES = {1: (-0.35, -0.30), 5: (-0.32, -0.95), 9: (-0.08, -1.0), 13: (0.14, -0.95), 17: (0.33, -0.85)}
SEGS = [0.42, 0.26, 0.20]


def _rot(eixo, ang):
    c, s = np.cos(ang), np.sin(ang)
    if eixo == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if eixo == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def mao(flex, abert, polegar, roll=0.0, yaw=0.0, tam=1.0, prop=None):
    """flex: 4 valores 0..1 (indicador..mínimo); abert: abertura lateral;
    polegar: (ângulo, flexão)."""
    prop = np.ones(5) if prop is None else prop
    p = np.zeros((21, 3))
    for k, (base, f) in enumerate(zip((5, 9, 13, 17), flex)):
        p[base] = [*BASES[base], 0]
        d = _rot("z", (k - 1.5) * abert) @ np.array([0, -1.0, 0])
        pos = p[base].copy()
        for j, (seg, a) in enumerate(zip(SEGS, (75, 95, 65))):
            d = _rot("x", np.radians(a) * f) @ d
            pos = pos + d * seg * prop[k + 1]
            p[base + j + 1] = pos
    ang, fl = polegar
    p[1] = [*BASES[1], 0]
    d = _rot("z", -np.radians(ang)) @ np.array([0, -1.0, 0])
    pos = p[1].copy()
    for j, seg in enumerate([0.32, 0.28, 0.22]):
        d = _rot("y", np.radians(60) * fl) @ d
        pos = pos + d * seg * prop[0]
        p[j + 2] = pos
    R = _rot("z", roll) @ _rot("y", yaw)
    return (p @ R.T) * tam


# Configurações inventadas (flexões, abertura, polegar, roll, yaw)
def modelos_letras(rng):
    letras = config.LETRAS_ESTATICAS + config.LETRAS_DINAMICAS
    tpl = {}
    for l in letras:
        tpl[l] = dict(flex=rng.uniform(0, 1, 4), abert=rng.uniform(0, 0.35),
                      polegar=(rng.uniform(0, 80), rng.uniform(0, 1)),
                      roll=rng.uniform(-0.6, 0.6), yaw=rng.uniform(-0.5, 0.5))
    # pares difíceis de propósito
    tpl["A"] = dict(flex=np.array([.95, .95, .95, .95]), abert=.02, polegar=(10, .1), roll=0, yaw=0)
    tpl["S"] = dict(flex=np.array([.95, .95, .95, .95]), abert=.02, polegar=(45, .7), roll=0, yaw=0)
    tpl["E"] = dict(flex=np.array([.75, .75, .75, .75]), abert=.02, polegar=(55, .9), roll=0, yaw=0)
    tpl["M"] = dict(flex=np.array([.8, .8, .8, .95]), abert=.05, polegar=(50, .8), roll=0, yaw=0)
    tpl["N"] = dict(flex=np.array([.8, .8, .95, .95]), abert=.05, polegar=(45, .8), roll=0, yaw=0)
    return tpl


def gerar_sinalizador(nome, tpl, rng, n_est=20, n_din=10):
    prop = rng.uniform(0.88, 1.12, 5)                 # proporções da mão da pessoa
    tam = rng.uniform(70, 130)                        # mão grande/pequena, perto/longe
    vies = {l: rng.normal(0, 0.06, 4) for l in tpl}   # jeito pessoal de fazer cada letra
    mao_lado = "Left" if rng.random() < 0.2 else "Right"

    def pose(l, ruido=0.05, **extra):
        t = dict(tpl[l])
        t.update(extra)
        f = np.clip(t["flex"] + vies[l] + rng.normal(0, ruido, 4), 0, 1)
        pts = mao(f, t["abert"] + rng.normal(0, .02), (t["polegar"][0] + rng.normal(0, 4),
                  np.clip(t["polegar"][1] + rng.normal(0, .05), 0, 1)),
                  t["roll"] + rng.normal(0, .1), t["yaw"] + rng.normal(0, .1), tam, prop)
        if mao_lado == "Left":
            pts[:, 0] *= -1
        return pts

    est, din = [], []
    for l in config.LETRAS_ESTATICAS:
        for _ in range(n_est):
            c = rng.uniform([200, 150, 0], [440, 330, 0])
            est.append({"letra": l, "sinalizador": nome, "origem": "local", "arquivo": "sint",
                        "pts": (pose(l) + c).astype(np.float32), "mao": mao_lado})
    for l in config.LETRAS_DINAMICAS:
        for _ in range(n_din):
            din.append({"letra": l, "sinalizador": nome, "origem": "local", "arquivo": "sint",
                        "seq": sequencia_dinamica(l, pose, rng, tam), "mao": mao_lado})
    return est, din


def sequencia_dinamica(l, pose, rng, tam):
    T = int(rng.integers(16, 30))
    t = np.linspace(0, 1, T)
    c0 = rng.uniform([220, 170, 0], [420, 310, 0])
    s = 1.3 * tam
    if l == "J":
        traj = np.stack([0.5 * s * np.sin(np.pi * t), s * t ** 2, 0 * t], 1)
    elif l == "Z":
        traj = np.stack([s * np.where(t < .33, t * 3, np.where(t < .66, 2 - t * 3, t * 3 - 2)),
                         s * np.where(t < .33, 0, np.where(t < .66, (t - .33) * 3, 1)), 0 * t], 1)
    elif l == "K":
        traj = np.stack([0 * t, -s * t, 0 * t], 1)
    else:
        traj = np.stack([0.15 * s * np.sin(2 * np.pi * t), 0 * t, 0 * t], 1)
    seq = []
    for i, ti in enumerate(t):
        extra = {}
        if l == "H":
            extra["roll"] = -1.2 * ti
        if l == "Y":
            extra["yaw"] = 0.9 * np.sin(2 * np.pi * ti)
        if l == "X":
            extra["flex"] = np.array([0.6 * (1 + np.sin(4 * np.pi * ti)) / 2, .95, .95, .95])
        seq.append(pose(l, 0.03, **extra) + c0 + traj[i])
    seq = np.array(seq, dtype=np.float32)
    for i in rng.choice(T, size=int(T * 0.1), replace=False):   # quadros perdidos
        seq[i] = np.nan
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sinalizadores", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    tpl = modelos_letras(rng)
    banco = {"estaticas": [], "dinamicas": [], "falhas": [], "vistos": {}, "templates": tpl}
    for i in range(a.sinalizadores):
        e, d = gerar_sinalizador(f"s{i}", tpl, rng)
        banco["estaticas"] += e
        banco["dinamicas"] += d
    config.DIR_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    destino = config.DIR_PROCESSADOS / "amostras_sinteticas.joblib"
    joblib.dump(banco, destino)
    print(f"{len(banco['estaticas'])} estáticas e {len(banco['dinamicas'])} sequências -> {destino}")


if __name__ == "__main__":
    main()

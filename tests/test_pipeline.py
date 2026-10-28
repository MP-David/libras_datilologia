"""Testes rápidos com dados sintéticos (não precisam de câmera nem do MediaPipe).

    python -m pytest -q        ou        python tests/test_pipeline.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from libras import config, sintetico as S                                    # noqa: E402
from libras.features import (N_FEATURES_ESTATICAS, features_dinamicas,        # noqa: E402
                             features_estaticas)
from libras.modelo import (TRANSICAO, ModeloLetras, decidir, matriz_dinamica,  # noqa: E402
                           matriz_estatica, sintetizar_transicoes)
from libras.reconhecedor import TradutorContinuo, reconhecer_clipe, reconhecer_foto  # noqa: E402

RNG = np.random.default_rng(3)
TPL = S.modelos_letras(np.random.default_rng(7))


def _dados(n_sinal=4):
    est, din = [], []
    for i in range(n_sinal):
        e, d = S.gerar_sinalizador(f"t{i}", TPL, RNG, n_est=8, n_din=6)
        est += e
        din += d
    return est, din


EST, DIN = _dados()


def _modelos():
    ctx = {}
    for s in EST:
        ctx.setdefault(s["sinalizador"], []).append(s["pts"])
    X, y, _ = matriz_estatica(EST, 2)
    me = ModeloLetras("estatico", ExtraTreesClassifier(200, random_state=0).fit(X, y), [])
    me.classes = list(me.estimador.classes_)
    amostras = DIN + sintetizar_transicoes(EST, 30)
    X, y, _ = matriz_dinamica(amostras, 3, contexto=ctx)
    md = ModeloLetras("dinamico", ExtraTreesClassifier(300, random_state=0).fit(X, y), [])
    md.classes = list(md.estimador.classes_)
    return me, md


ME, MD = _modelos()


def _pessoa_nova(seed=11):
    rng = np.random.default_rng(seed)
    prop = rng.uniform(0.9, 1.1, 5)

    def pose(l, ruido=0.03, **extra):
        t = dict(TPL[l])
        t.update(extra)
        return S.mao(np.clip(t["flex"] + rng.normal(0, ruido, 4), 0, 1), t["abert"], t["polegar"],
                     t["roll"], t["yaw"], 95, prop)
    return pose, rng


# ---------------------------------------------------------------------------
def test_invariancias():
    p = S.mao(np.array([.2, .9, .9, .1]), .1, (30, .3), .2, .1, 100)
    f = features_estaticas(p)
    assert f.shape == (N_FEATURES_ESTATICAS,)
    # posição e tamanho da mão não mudam os atributos
    assert np.allclose(f, features_estaticas(p * 1.7 + [300, 200, 0]), atol=1e-4)
    # mão esquerda (espelhada) vira a mesma mão direita
    esq = p.copy()
    esq[:, 0] *= -1
    assert np.allclose(f, features_estaticas(esq, "Left"), atol=1e-4)


def test_sequencia_curta_descartada():
    seq = np.full((20, 21, 3), np.nan)
    seq[:3] = S.mao(np.zeros(4), .1, (30, .3), 0, 0, 100)
    assert features_dinamicas(seq) is None


def test_regras_de_rejeicao():
    cl = ["A", "B", "S"]
    assert decidir(np.array([.45, .30, .25]), cl, .6, .15).letra is None        # confiança baixa
    assert decidir(np.array([.60, .05, .35]), cl, .5, .15).letra is None        # A/S: margem maior
    assert decidir(np.array([.60, .35, .05]), cl, .5, .15).letra == "A"        # A/B: margem normal
    d = decidir(np.array([.9, .1]), [TRANSICAO, "J"], .6, .15)
    assert d.letra is None and "transição" in d.motivo


def test_foto_e_clipe():
    pose, rng = _pessoa_nova()
    assert reconhecer_foto(ME, pose("L") + [300, 200, 0], "Right").letra == "L"
    assert not reconhecer_foto(ME, None, "Right").aceita                        # sem mão
    seq = S.sequencia_dinamica("Z", pose, rng, 95)
    d, mexeu = reconhecer_clipe(ME, MD, seq, "Right")
    assert mexeu and d.letra == "Z", d
    parado = np.stack([pose("B") + [300, 200, 0] for _ in range(12)])
    d, mexeu = reconhecer_clipe(ME, MD, parado, "Right")
    assert not mexeu and d.letra == "B", d


def test_fluxo_continuo_soletra_palavra():
    pose, rng = _pessoa_nova(21)
    c = np.array([320., 240, 0])
    quadros = []

    def segura(l, n=12):
        quadros.extend(pose(l) + c for _ in range(n))

    def troca(a, b, n=8):
        A, B = pose(a), pose(b)
        quadros.extend((1 - w) * A + w * B + c for w in np.linspace(0, 1, n))

    def mexe(l):
        nonlocal c
        seq = S.sequencia_dinamica(l, pose, rng, 95)
        ok = np.flatnonzero(~np.isnan(seq).any(axis=(1, 2)))
        seq = seq - seq[ok[0], 0] + c
        quadros.extend(None if np.isnan(q).any() else q for q in seq)
        c = seq[ok[-1], 0] - pose(l)[0]

    segura("C"); troca("C", "A"); segura("A"); troca("A", "J"); mexe("J")
    troca("J", "O"); segura("O")
    quadros.extend([None] * 8)
    segura("O")                                    # letra dobrada: tirou a mão e voltou
    tr = TradutorContinuo(ME, MD)
    eventos = [e for q in quadros if (e := tr.processar(q, "Right"))]
    assert tr.palavra == "CAJOO", [str(e) for e in eventos]


def test_pose_ambigua_pede_repeticao():
    """Uma mão no meio do caminho entre M e N (ou A e S) deve gerar REPITA,
    não um chute."""
    pose, _ = _pessoa_nova(5)
    tr = TradutorContinuo(ME, MD)
    meio = 0.5 * pose("M", 0) + 0.5 * pose("N", 0)
    eventos = [e for _ in range(12) if (e := tr.processar(meio + [300, 200, 0], "Right"))]
    assert eventos, "deveria haver uma decisão"
    assert eventos[0].tipo == "REPITA" or eventos[0].letra in ("M", "N")


if __name__ == "__main__":
    for nome, f in list(globals().items()):
        if nome.startswith("test_"):
            f()
            print("ok", nome)

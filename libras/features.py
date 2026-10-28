"""Normalização de pontos da mão e extração de atributos.

Por que pontos (landmarks) e não pixels?
  * O fundo some: o classificador nunca vê a imagem, então não consegue usar
    "fundo verde = letra B" como atalho (armadilha do fundo homogêneo).
  * Tamanho e posição da mão são normalizados: mãos grandes, pequenas, perto
    ou longe da câmera viram a mesma escala.
  * Mão esquerda é espelhada para a direita, então sinalizadores canhotos usam
    o mesmo modelo.
O que NÃO normalizamos de propósito: a rotação. Orientação da palma
distingue letras (ex.: H/U em algumas variantes, G/Q, P/K).

Índices MediaPipe: 0 punho | polegar 1-4 | indicador 5-8 | médio 9-12 |
anelar 13-16 | mínimo 17-20.
"""
from __future__ import annotations

import numpy as np

from . import config

PONTAS = [4, 8, 12, 16, 20]
DEDOS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
         [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
# Onde o polegar está em relação aos outros dedos é o que separa A/S/E e M/N/T
ALVOS_POLEGAR = [5, 6, 9, 10, 13, 14, 17, 18]


# ---------------------------------------------------------------------------
# Normalização
# ---------------------------------------------------------------------------
def tamanho_palma(pts: np.ndarray) -> float:
    """Escala de referência: média das distâncias punho->bases dos dedos + largura."""
    d = [np.linalg.norm(pts[i] - pts[0]) for i in (5, 9, 13, 17)]
    d.append(np.linalg.norm(pts[5] - pts[17]))
    return float(np.mean(d)) + 1e-8


def normalizar_mao(pts: np.ndarray, mao: str = "Right") -> np.ndarray:
    """pts: (21,3) em pixels. Retorna (21,3) centrado no punho, escala da palma,
    sempre como mão direita."""
    p = np.asarray(pts, dtype=np.float64).copy()
    p -= p[0]
    p /= tamanho_palma(p)
    if mao == "Left":
        p[:, 0] *= -1.0
    return p


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def _angulo(a, b, c) -> float:
    v1, v2 = _unit(a - b), _unit(c - b)
    return float(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def normal_palma(p: np.ndarray) -> np.ndarray:
    return _unit(np.cross(p[5] - p[0], p[17] - p[0]))


# ---------------------------------------------------------------------------
# Letras estáticas: 1 quadro -> vetor de atributos
# ---------------------------------------------------------------------------
def features_estaticas(pts: np.ndarray, mao: str = "Right") -> np.ndarray:
    p = normalizar_mao(pts, mao)
    f = [p.ravel()]                                                  # 63
    f.append([np.linalg.norm(p[a] - p[b])                            # 10
              for i, a in enumerate(PONTAS) for b in PONTAS[i + 1:]])
    f.append([np.linalg.norm(p[4] - p[j]) for j in ALVOS_POLEGAR])   # 8
    f.append([_angulo(p[d[k - 1]], p[d[k]], p[d[k + 1]])             # 15
              for d in DEDOS for k in (1, 2, 3)])
    f.append(normal_palma(p))                                        # 3
    f.append(np.concatenate([_unit(p[d[4]] - p[d[1]]) for d in DEDOS]))  # 15
    return np.concatenate([np.ravel(x) for x in f]).astype(np.float32)


N_FEATURES_ESTATICAS = 114


# ---------------------------------------------------------------------------
# Letras com movimento: sequência de quadros -> vetor de atributos
# ---------------------------------------------------------------------------
def preencher_e_reamostrar(seq: np.ndarray, n: int) -> np.ndarray | None:
    """seq: (T,21,3) com NaN nos quadros sem mão. Interpola os buracos e
    reamostra para n quadros. None se houver poucos quadros com mão."""
    seq = np.asarray(seq, dtype=np.float64)
    ok = ~np.isnan(seq).any(axis=(1, 2))
    if ok.sum() < config.MIN_QUADROS_COM_MAO:
        return None
    t_ok = np.flatnonzero(ok)
    seq = seq[t_ok[0]: t_ok[-1] + 1]           # corta bordas sem mão
    ok = ok[t_ok[0]: t_ok[-1] + 1]
    t = np.arange(len(seq))
    flat = seq.reshape(len(seq), -1)
    cheio = np.empty_like(flat)
    for j in range(flat.shape[1]):
        cheio[:, j] = np.interp(t, t[ok], flat[ok, j])
    alvo = np.linspace(0, len(seq) - 1, n)
    saida = np.empty((n, flat.shape[1]))
    for j in range(flat.shape[1]):
        saida[:, j] = np.interp(alvo, t, cheio[:, j])
    return saida.reshape(n, 21, 3)


def features_dinamicas(seq: np.ndarray, mao: str = "Right") -> np.ndarray | None:
    """Combina (a) forma da mão ao longo do tempo, (b) trajetória do punho e de
    pontas de dedos e (c) giro da palma. (a) diz QUAL configuração; (b) e (c)
    dizem QUE movimento — J (traço do mínimo) e Z (traço do indicador) dependem
    de (b); K e Y dependem mais de (c) e do deslocamento."""
    n = config.QUADROS_SEQUENCIA
    s = preencher_e_reamostrar(seq, n)
    if s is None:
        return None
    escala = float(np.median([tamanho_palma(q - q[0]) for q in s]))
    sinal_x = -1.0 if mao == "Left" else 1.0

    # (a) forma: média dos atributos estáticos em 4 trechos do movimento
    est = np.stack([features_estaticas(q, mao) for q in s])
    forma = np.concatenate([b.mean(axis=0) for b in np.array_split(est, 4)])

    # (b) trajetórias relativas à posição inicial do punho, em tamanhos de palma
    traj = (s[:, [0, 8, 20], :] - s[0, 0]) / escala
    traj[..., 0] *= sinal_x
    vel = np.diff(traj, axis=0)
    caminho = np.linalg.norm(vel, axis=-1).sum(axis=0)                # 3
    liquido = np.linalg.norm(traj[-1] - traj[0], axis=-1)             # 3
    curvatura = caminho / (liquido + 1e-3)                            # 3

    # (c) orientação da palma quadro a quadro
    normais = np.stack([normal_palma(normalizar_mao(q, mao)) for q in s])
    giro = np.degrees(np.arccos(np.clip((normais[1:] * normais[:-1]).sum(-1), -1, 1))).sum()

    return np.concatenate([forma, traj.ravel(), caminho, liquido, curvatura,
                           normais.ravel(), [giro / 180.0]]).astype(np.float32)


PONTOS_MOV = [0, 4, 8, 12, 20]   # punho e pontas de polegar, indicador, médio e mínimo


def deslocamento(buffer: list[np.ndarray]) -> float:
    """Quanto punho e pontas dos dedos se moveram desde o 1º quadro do buffer,
    em tamanhos de palma. Usado ao vivo para separar 'mão parada' de 'movimento'."""
    if len(buffer) < 2:
        return 0.0
    ini = buffer[0]
    esc = tamanho_palma(ini - ini[0])
    return float(max(np.linalg.norm(q[PONTOS_MOV, :2] - ini[PONTOS_MOV, :2], axis=1).max()
                     for q in buffer[1:]) / esc)


def distancia_pose(a: np.ndarray, b: np.ndarray, mao: str = "Right") -> float:
    """Diferença de FORMA entre duas mãos (independe de posição e tamanho)."""
    return float(np.linalg.norm(normalizar_mao(a, mao) - normalizar_mao(b, mao), axis=1).mean())


# ---------------------------------------------------------------------------
# Aumento de dados (só no treino)
# ---------------------------------------------------------------------------
def _transformacao(rng: np.random.Generator):
    ang = np.radians(rng.uniform(-15, 15))
    esc = rng.uniform(0.85, 1.15)
    rot = np.array([[np.cos(ang), -np.sin(ang), 0],
                    [np.sin(ang), np.cos(ang), 0],
                    [0, 0, 1]])
    estica = np.diag([rng.uniform(0.92, 1.08), rng.uniform(0.92, 1.08), 1.0])
    return rot @ estica * esc


def aumentar_estatico(pts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    M = _transformacao(rng)
    c = pts[0]
    out = (pts - c) @ M.T + c
    out += rng.normal(0, 0.012 * tamanho_palma(pts - pts[0]), out.shape)
    return out


def aumentar_sequencia(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    M = _transformacao(rng)
    c = np.nanmean(seq[:, 0], axis=0)
    out = (seq - c) @ M.T + c
    ok = ~np.isnan(out).any(axis=(1, 2))
    if ok.any():
        esc = tamanho_palma(out[ok][0] - out[ok][0][0])
        out += rng.normal(0, 0.012 * esc, out.shape)
    # deformação temporal: velocidade de sinalização varia entre pessoas
    T = len(out)
    novo_T = max(config.MIN_QUADROS_COM_MAO, int(T * rng.uniform(0.75, 1.3)))
    idx = np.clip(np.round(np.linspace(0, T - 1, novo_T)).astype(int), 0, T - 1)
    return out[idx]

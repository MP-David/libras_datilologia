"""Modelo de sequência com ATENÇÃO (aula 10) para as letras com movimento.

Entrada: o clipe reamostrado para config.QUADROS_SEQUENCIA quadros. Cada quadro
vira um vetor com a FORMA da mão (atributos estáticos) e o MOVIMENTO (posição e
velocidade de punho, polegar, indicador e mínimo em relação ao início, em
tamanhos de palma). Um codificador Transformer pequeno troca informação entre
os quadros e um "pooling" por atenção decide quais quadros importam para a
decisão — por exemplo, o traço final do J ou as quinas do Z.

É um estimador no estilo scikit-learn (fit / predict_proba), para ser
comparado lado a lado com os modelos clássicos na validação por pessoa.
"""
from __future__ import annotations

import numpy as np

from . import config
from .features import features_estaticas, preencher_e_reamostrar, tamanho_palma

PONTOS_TRAJ = [0, 4, 8, 20]


def quadros_sequencia(seq: np.ndarray, mao: str = "Right") -> np.ndarray | None:
    """(T,21,3) com NaN -> (n, D) atributos por quadro, ou None se curta demais."""
    n = config.QUADROS_SEQUENCIA
    s = preencher_e_reamostrar(seq, n)
    if s is None:
        return None
    escala = float(np.median([tamanho_palma(q - q[0]) for q in s]))
    forma = np.stack([features_estaticas(q, mao) for q in s])
    traj = (s[:, PONTOS_TRAJ, :2] - s[0, 0, :2]) / escala
    if mao == "Left":
        traj[..., 0] *= -1
    vel = np.vstack([np.zeros((1,) + traj.shape[1:]), np.diff(traj, axis=0)])
    return np.concatenate([forma, traj.reshape(n, -1), vel.reshape(n, -1) * 4], 1).astype(np.float32)


N_ATRIBUTOS_QUADRO = 114 + 2 * len(PONTOS_TRAJ) * 2


class AtencaoSequencia:
    """Transformer encoder (2 camadas) + pooling por atenção + classificador."""

    def __init__(self, dim=64, camadas=2, cabecas=4, epocas=80, lr=2e-3, dropout=0.2, seed=0):
        self.dim, self.camadas, self.cabecas = dim, camadas, cabecas
        self.epocas, self.lr, self.dropout, self.seed = epocas, lr, dropout, seed

    # scikit-learn clone() precisa disto
    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in ("dim", "camadas", "cabecas", "epocas", "lr", "dropout", "seed")}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def _rede(self, d_in, n_cls):
        import torch
        from torch import nn

        class Rede(nn.Module):
            def __init__(s):
                super().__init__()
                T = config.QUADROS_SEQUENCIA
                s.entrada = nn.Sequential(nn.Linear(d_in, self.dim), nn.GELU(), nn.Dropout(self.dropout))
                s.pos = nn.Parameter(torch.zeros(1, T, self.dim))
                camada = nn.TransformerEncoderLayer(self.dim, self.cabecas, self.dim * 2, self.dropout,
                                                    batch_first=True, norm_first=True)
                s.cod = nn.TransformerEncoder(camada, self.camadas)
                s.consulta = nn.Linear(self.dim, 1)          # pontuação de atenção por quadro
                s.saida = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, n_cls))

            def forward(s, x, devolver_pesos=False):
                h = s.cod(s.entrada(x) + s.pos)
                w = torch.softmax(s.consulta(h).squeeze(-1), dim=1)   # (B,T)
                z = (w.unsqueeze(-1) * h).sum(1)
                return (s.saida(z), w) if devolver_pesos else s.saida(z)
        return Rede()

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        T = config.QUADROS_SEQUENCIA
        X = np.asarray(X, np.float32).reshape(len(X), T, -1)
        self.media_ = X.reshape(-1, X.shape[-1]).mean(0)
        self.desvio_ = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
        self.classes_ = np.array(sorted(set(y)))
        yi = np.searchsorted(self.classes_, y)
        cont = np.bincount(yi, minlength=len(self.classes_))
        # peso por classe suavizado (raiz): o "balanceado" puro fazia o modelo
        # chutar letra com movimento em clipes de letra parada
        peso = torch.tensor(np.sqrt(len(yi) / (len(cont) * np.maximum(cont, 1))), dtype=torch.float32)
        self.rede_ = self._rede(X.shape[-1], len(self.classes_))
        opt = torch.optim.AdamW(self.rede_.parameters(), lr=self.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=self.epocas *
                                                    int(np.ceil(len(X) / 64)))
        Xt = torch.from_numpy((X - self.media_) / self.desvio_)
        yt = torch.from_numpy(yi).long()
        perda = torch.nn.CrossEntropyLoss(weight=peso, label_smoothing=0.05)
        g = torch.Generator().manual_seed(self.seed)
        self.rede_.train()
        for _ in range(self.epocas):
            for idx in torch.randperm(len(Xt), generator=g).split(64):
                opt.zero_grad()
                perda(self.rede_(Xt[idx]), yt[idx]).backward()
                opt.step()
                sched.step()
        self.rede_.eval()
        return self

    def _tensor(self, X):
        import torch
        T = config.QUADROS_SEQUENCIA
        X = np.asarray(X, np.float32).reshape(len(X), T, -1)
        return torch.from_numpy((X - self.media_) / self.desvio_)

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            return torch.softmax(self.rede_(self._tensor(np.atleast_2d(X))), -1).numpy()

    def pesos_atencao(self, X):
        """Quanto cada quadro pesou na decisão (para o relatório)."""
        import torch
        with torch.no_grad():
            return self.rede_(self._tensor(np.atleast_2d(X)), devolver_pesos=True)[1].numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    # joblib: guarda só o state_dict
    def __getstate__(self):
        d = dict(self.__dict__)
        if "rede_" in d:
            d["estado_"] = {k: v.cpu() for k, v in d.pop("rede_").state_dict().items()}
        return d

    def __setstate__(self, d):
        estado = d.pop("estado_", None)
        self.__dict__.update(d)
        if estado is not None:
            d_in = len(self.media_)
            self.rede_ = self._rede(d_in, len(self.classes_))
            self.rede_.load_state_dict(estado)
            self.rede_.eval()

"""Comparação com CNN + transfer learning (aulas 5 e 6) nas letras estáticas.

    python -m libras.comparar_cnn --teste contrib_p02,contrib_p06
    python -m libras.comparar_cnn --teste contrib_p02,contrib_p06 --sem-extras

MobileNetV3-Small pré-treinada no ImageNet, TODAS as camadas ajustadas no
recorte da mão (a caixa vem dos pontos do MediaPipe, com margem), com aumentos
de imagem (recorte aleatório, rotação, cor). Mesmo protocolo do modelo de pontos:
avaliação POR GRAVAÇÃO (média dos quadros estáveis), deixando uma pessoa de fora,
e teste final com as pessoas reservadas. Grava relatorios/comparacao_cnn.md.

Por que comparar: a CNN vê pixels — fundo, roupa, rosto, luz. Com poucas pessoas
ela tende a aprender QUEM gravou e ONDE, não a forma da mão (armadilha do fundo
como atalho). O modelo de pontos nunca vê pixels. A diferença entre os dois na
validação por pessoa mede esse efeito.
"""
from __future__ import annotations

import argparse

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import config
from . import metricas as M
from .extrair import carregar_banco
from .features import quadros_estaveis

LADO = 128


def recorte(img, pts, lado=LADO, margem=1.5):
    x0, y0 = pts[:, :2].min(0)
    x1, y1 = pts[:, :2].max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = max(x1 - x0, y1 - y0) * margem / 2 + 8
    h, w = img.shape[:2]
    pad = int(max(0, r - cx, r - cy, cx + r - w, cy + r - h)) + 1
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    cx, cy = cx + pad, cy + pad
    c = img[int(cy - r):int(cy + r), int(cx - r):int(cx + r)]
    return cv2.cvtColor(cv2.resize(c, (lado, lado), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def gravacoes_com_recortes(banco):
    """[(letra, pessoa, origem, [recortes dos quadros estáveis])]"""
    out = []
    for f in banco["fotos"]:
        if f["letra"] in config.LETRAS_ESTATICAS:
            img = cv2.imread(f["arquivo"])
            if img is not None:
                out.append((f["letra"], f["sinalizador"], f["origem"], [recorte(img, f["pts"])]))
    for c in banco["clipes"]:
        if c["letra"] not in config.LETRAS_ESTATICAS:
            continue
        arqs = sorted(p for p in Path(c["arquivo"]).iterdir() if p.suffix.lower() in config.EXT_IMAGEM)
        if len(arqs) != len(c["seq"]):     # vídeo: não dá para recortar por arquivo; pula
            continue
        rs = [recorte(cv2.imread(str(arqs[i])), c["seq"][i]) for i in quadros_estaveis(c["seq"], c["mao"])]
        if rs:
            out.append((c["letra"], c["sinalizador"], c["origem"], rs))
    return out


def main():
    import torch
    import torchvision
    from torchvision.transforms import v2 as T

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teste", default="")
    ap.add_argument("--epocas", type=int, default=6)
    ap.add_argument("--sem-extras", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    teste = {t.strip() for t in a.teste.split(",") if t.strip()}
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    G = gravacoes_com_recortes(carregar_banco())
    print(f"{len(G)} gravações, {sum(len(g[3]) for g in G)} recortes ({time.time() - t0:.0f}s)", flush=True)
    y = np.array([g[0] for g in G]); P = np.array([g[1] for g in G]); O = np.array([g[2] for g in G])
    classes = sorted(set(y))
    extras = [] if a.sem_extras else [i for i in range(len(G)) if O[i] == "extra"]
    locais = [i for i in range(len(G)) if O[i] == "local" and P[i] not in teste]
    idx_teste = [i for i in range(len(G)) if P[i] in teste]
    media = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    desv = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    aug = T.Compose([T.RandomResizedCrop(LADO, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
                     T.RandomRotation(12), T.ColorJitter(0.4, 0.4, 0.3, 0.05)])

    def lote(imgs, treino):
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)
        if treino:
            x = torch.stack([aug(xi) for xi in x])
        return ((x.float() / 255) - media) / desv

    def treinar(idx):
        amostras = [(i, k) for i in idx for k in
                    (range(len(G[i][3])) if len(G[i][3]) <= 10 else np.linspace(0, len(G[i][3]) - 1, 10).astype(int))]
        ys = np.array([classes.index(y[i]) for i, _ in amostras])
        torch.manual_seed(a.seed)
        net = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        net.classifier[3] = torch.nn.Linear(net.classifier[3].in_features, len(classes))
        net = net.to(dev)
        peso = torch.tensor(len(ys) / (len(classes) * np.maximum(np.bincount(ys, minlength=len(classes)), 1)),
                            dtype=torch.float32).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        n_lotes = int(np.ceil(len(ys) / 64))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.epocas * n_lotes)
        perda = torch.nn.CrossEntropyLoss(weight=peso, label_smoothing=0.1)
        rng = np.random.default_rng(a.seed)
        net.train()
        for _ in range(a.epocas):
            for b in np.array_split(rng.permutation(len(ys)), n_lotes):
                x = lote([G[amostras[j][0]][3][amostras[j][1]] for j in b], True).to(dev)
                opt.zero_grad()
                perda(net(x), torch.from_numpy(ys[b]).to(dev)).backward()
                opt.step()
                sched.step()
        return net.eval()

    def prever(net, idx):
        with torch.no_grad():
            return np.array([torch.softmax(net(lote(G[i][3], False).to(dev)), -1).mean(0).cpu().numpy()
                             for i in idx])

    pessoas = sorted(set(P[locais]))
    oof = {}
    for s in pessoas:
        t0 = time.time()
        va = [i for i in locais if P[i] == s]
        net = treinar([i for i in locais if P[i] != s] + extras)
        for i, p in zip(va, prever(net, va)):
            oof[i] = p
        print(f"  dobra {s}: {time.time() - t0:.0f}s", flush=True)
    idx = sorted(oof)
    pred = np.array(classes)[np.array([oof[i] for i in idx]).argmax(1)]
    tab = M.tabela_por_letra(y[idx], pred, classes)
    bal = M.acuracia_balanceada(y[idx], pred)
    linhas = ["# Comparação com CNN (transfer learning) — letras estáticas", "",
              f"Gerado em {datetime.now():%d/%m/%Y %H:%M}. MobileNetV3-Small (ImageNet), ajuste fino de todas as "
              f"camadas por {a.epocas} épocas no recorte da mão {LADO}×{LADO}; "
              f"{'sem' if a.sem_extras else 'com'} Mendeley no treino. Avaliação por gravação.", "",
              f"**Validação deixando uma pessoa de fora ({len(pessoas)} pessoas):** acurácia balanceada "
              f"{bal:.1%}, pior letra {tab.acuracia_letra.min():.1%}."]
    if idx_teste:
        net = treinar(locais + extras)
        pt = np.array(classes)[prever(net, idx_teste).argmax(1)]
        bal_t = M.acuracia_balanceada(y[idx_teste], pt)
        linhas += ["", f"**Pessoa(s) nunca vista(s) ({', '.join(sorted(teste))}):** acurácia balanceada {bal_t:.1%}."]
    # referência: modelo de pontos (acurácia balanceada = média da acurácia por letra)
    ref = {}
    for nome in ("validacao", "teste"):
        arq = config.DIR_RELATORIOS / f"estatico_{nome}_por_letra.csv"
        if arq.exists():
            import pandas as pd
            ref[nome] = float(pd.read_csv(arq)["acuracia_letra"].mean())
    if ref:
        linhas += ["", "Para comparar, o modelo de **pontos da mão** (`relatorio.md`) teve " +
                   " e ".join(f"{v:.1%} na {'validação' if k == 'validacao' else 'pessoa nunca vista'}"
                              for k, v in ref.items()) + "."]
    linhas += ["", "### Acurácia por letra — validação", "", M.tabela_md(tab), ""]
    config.DIR_RELATORIOS.mkdir(parents=True, exist_ok=True)
    (config.DIR_RELATORIOS / "comparacao_cnn.md").write_text("\n".join(linhas), "utf-8")
    rel = config.DIR_RELATORIOS / "relatorio.md"
    if rel.exists() and "## Comparação com CNN" not in rel.read_text("utf-8"):
        with open(rel, "a", encoding="utf-8") as f:
            f.write("\n## Comparação com CNN (transfer learning)\n\n" + "\n".join(linhas[4:8]) +
                    "\n\nDetalhes por letra em [comparacao_cnn.md](comparacao_cnn.md).\n")
    print("\n".join(linhas[:8]))


if __name__ == "__main__":
    main()

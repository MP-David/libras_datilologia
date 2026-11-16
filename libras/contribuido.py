"""Organiza o dataset contribuído (contributed_images.zip) em dados/brutos/.

    python -m libras.contribuido agrupar   --zip ~/Downloads/contributed_images.zip
    # confira dados/contribuido/pessoas.jpg (1 linha por pessoa); se o agrupamento
    # errou, corrija a coluna "pessoa" de dados/contribuido/gravacoes.csv
    python -m libras.contribuido organizar --zip ~/Downloads/contributed_images.zip
    python -m libras.extrair

O zip foi gravado pela equipe e vem como images/<LETRA>/<arquivo>.jpg, mas NÃO
diz quem sinalizou:
  <id>__fNNNN.jpg  -> quadro NNNN de um clipe de 16 quadros (f0000…f0015; alguns
                      clipes chegaram com quadros faltando, e tudo bem)
  <id>.jpg         -> foto avulsa
Sem saber a pessoa não dá para cumprir o critério "sinalizador que não apareceu
no treino". Por isso o passo `agrupar`:
  1. acha o rosto em até 3 quadros de cada gravação (YuNet) e calcula um
     vetor de identidade (SFace), ambos do OpenCV Zoo;
  2. junta gravações com rostos parecidos (agrupamento hierárquico, cosseno);
     grupos com pelo menos --min-gravacoes viram pessoas p01, p02, ...;
  3. gravação sem rosto (ou de grupo pequeno) herda a pessoa das gravações com
     o CENÁRIO mais parecido (mesmo quarto, mesma luz), se a semelhança for alta;
  4. o que sobra fica com pessoa vazia e, por padrão, NÃO é usado: se fosse
     para o treino, a mão de uma pessoa reservada para teste poderia vazar.
Errar para "juntar" duas pessoas num grupo é seguro (avaliação fica mais
rígida); errar para "separar" a mesma pessoa em dois grupos é o que vaza.
O limiar é conservador por isso. Os rótulos p01.. não identificam ninguém.

`organizar` grava:
  clipe  -> dados/brutos/contrib_pNN/<LETRA>/<id>/fNNNN.jpg  (pasta = sequência)
  foto   -> dados/brutos/contrib_pNN/<LETRA>/<id>.jpg
  foto avulsa de H/J/K/X/Y/Z -> ignorada (letra com movimento precisa de sequência)
Rodar de novo apaga e recria só as pastas contrib_* (nada da coleta da equipe).
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import config

RE_ARQ = re.compile(r"^(?:.*/)?images/([A-Za-z])/([0-9a-fA-F]+?)(?:__f(\d+))?\.jpe?g$")
ARQ_CSV = config.DIR_CONTRIBUIDO / "gravacoes.csv"
ARQ_FOLHA = config.DIR_CONTRIBUIDO / "pessoas.jpg"
SEM_PESSOA = "contrib_sem_pessoa"
CAMPOS = ["letra", "gravacao", "tipo", "n_quadros", "quadros_com_rosto", "grupo_rosto", "pessoa", "como"]


# ---------------------------------------------------------------------------
# leitura do zip
# ---------------------------------------------------------------------------
def listar_gravacoes(z: zipfile.ZipFile) -> dict[tuple[str, str], list[str]]:
    """{(letra, id): [membros do zip ordenados por quadro]}"""
    grav = defaultdict(list)
    for nome in z.namelist():
        m = RE_ARQ.match(nome)
        if m:
            grav[(m[1].upper(), m[2].lower())].append((int(m[3] or 0), nome))
    return {k: [n for _, n in sorted(v)] for k, v in sorted(grav.items())}


def _ler(z: zipfile.ZipFile, nome: str):
    return cv2.imdecode(np.frombuffer(z.read(nome), np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# rosto e cenário
# ---------------------------------------------------------------------------
def _modelos_rosto():
    caminhos = {}
    config.DIR_MODELOS_ROSTO.mkdir(parents=True, exist_ok=True)
    for chave, rel in config.MODELOS_ROSTO.items():
        destino = config.DIR_MODELOS_ROSTO / Path(rel).name
        if not destino.exists():
            print(f"Baixando {destino.name} (OpenCV Zoo) ...")
            urllib.request.urlretrieve(config.URL_ZOO_OPENCV + rel, destino)
        caminhos[chave] = str(destino)
    det = cv2.FaceDetectorYN.create(caminhos["detector"], "", (320, 320), 0.8)
    rec = cv2.FaceRecognizerSF.create(caminhos["reconhecedor"], "")
    return det, rec


def _vetor_rosto(img, det, rec, lado_min=40):
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or not len(faces):
        return None
    f = max(faces, key=lambda r: r[2] * r[3])          # rosto maior = quem sinaliza
    if min(f[2], f[3]) < lado_min:
        return None
    v = rec.feature(rec.alignCrop(img, f)).ravel().astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _vetor_cena(img):
    """Miniatura 16x12 em Lab: resume parede, luz e móveis do lugar da gravação."""
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    mini = cv2.resize(img, (16, 12), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(mini, cv2.COLOR_BGR2Lab).astype(np.float32).ravel()


# ---------------------------------------------------------------------------
# agrupar
# ---------------------------------------------------------------------------
def agrupar(zip_path: Path, limiar_rosto: float, min_grav: int, k: int, pct_cena: float):
    from sklearn.cluster import AgglomerativeClustering

    z = zipfile.ZipFile(zip_path)
    grav = listar_gravacoes(z)
    if not grav:
        raise SystemExit(f"Nenhum arquivo images/<LETRA>/*.jpg em {zip_path}")
    det, rec = _modelos_rosto()
    chaves = list(grav)
    rostos, n_rosto, cenas = [], [], []
    print(f"{len(chaves)} gravações; procurando rostos ...")
    for i, ch in enumerate(chaves):
        nomes = grav[ch]
        vs = []
        for n in sorted({nomes[0], nomes[len(nomes) // 2], nomes[-1]}):
            img = _ler(z, n)
            if img is not None and (v := _vetor_rosto(img, det, rec)) is not None:
                vs.append(v)
        m = np.mean(vs, 0) if vs else None
        rostos.append(None if m is None else m / np.linalg.norm(m))
        n_rosto.append(len(vs))
        meio = _ler(z, nomes[len(nomes) // 2])
        cenas.append(_vetor_cena(meio) if meio is not None else np.full(16 * 12 * 3, np.nan, np.float32))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(chaves)}")

    com = [i for i, r in enumerate(rostos) if r is not None]
    grupo = np.full(len(chaves), -1)
    if len(com) >= 2:
        F = np.stack([rostos[i] for i in com])
        grupo[com] = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                             distance_threshold=limiar_rosto).fit_predict(F)
    tam = Counter(grupo[com])
    grandes = sorted((g for g, n in tam.items() if n >= min_grav), key=lambda g: -tam[g])
    nome_pessoa = {g: f"p{j + 1:02d}" for j, g in enumerate(grandes)}
    pessoa = [nome_pessoa.get(g, "") for g in grupo]
    como = ["rosto" if p else "" for p in pessoa]

    # cenário: gravação sem pessoa herda a pessoa dos k vizinhos de cena mais próximos
    C = np.stack(cenas)
    C = np.nan_to_num((C - np.nanmean(C, 0)) / (np.nanstd(C, 0) + 1e-6), nan=1e3)
    ref = np.array([i for i in range(len(chaves)) if pessoa[i]])
    if len(ref) > k:
        D = np.linalg.norm(C[ref][:, None] - C[ref][None], axis=-1)
        np.fill_diagonal(D, np.inf)
        # limiar = distância típica até a gravação mais parecida DA MESMA pessoa
        mesma = [D[a][[b for b in range(len(ref)) if pessoa[ref[b]] == pessoa[ref[a]]]].min()
                 for a in range(len(ref))]
        limiar_cena = float(np.percentile(mesma, pct_cena))
        for i in range(len(chaves)):
            if pessoa[i]:
                continue
            d = np.linalg.norm(C[ref] - C[i], axis=1)
            viz = np.argsort(d)[:k]
            votos = Counter(pessoa[ref[j]] for j in viz)
            p, n = votos.most_common(1)[0]
            if d[viz[0]] <= limiar_cena and n > k // 2 and pessoa[ref[viz[0]]] == p:
                pessoa[i], como[i] = p, "cenário"
        print(f"limiar de cenário: {limiar_cena:.1f} (percentil {pct_cena:.0f} das gravações da mesma pessoa)")

    config.DIR_CONTRIBUIDO.mkdir(parents=True, exist_ok=True)
    with open(ARQ_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, CAMPOS)
        w.writeheader()
        for i, (letra, gid) in enumerate(chaves):
            n = len(grav[(letra, gid)])
            w.writerow({"letra": letra, "gravacao": gid, "tipo": "clipe" if n > 1 else "foto",
                        "n_quadros": n, "quadros_com_rosto": n_rosto[i],
                        "grupo_rosto": "" if grupo[i] < 0 else int(grupo[i]),
                        "pessoa": pessoa[i], "como": como[i] or "sem_grupo"})
    _resumo(_ler_csv())
    folha_conferencia(z, grav, _ler_csv())
    print(f"\nCSV: {ARQ_CSV}\nConferência visual: {ARQ_FOLHA}")


# ---------------------------------------------------------------------------
# organizar
# ---------------------------------------------------------------------------
def _ler_csv():
    with open(ARQ_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _nome_seguro(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]", "_", s.strip())


def organizar(zip_path: Path, sem_pessoa: str):
    if not ARQ_CSV.exists():
        raise SystemExit(f"{ARQ_CSV} não existe — rode antes: python -m libras.contribuido agrupar")
    linhas = _ler_csv()
    z = zipfile.ZipFile(zip_path)
    grav = listar_gravacoes(z)

    # limpa só o que este script criou antes
    for p in config.DIR_DADOS_BRUTOS.glob(config.PREFIXO_CONTRIBUIDO + "*"):
        shutil.rmtree(p)
    shutil.rmtree(config.DIR_DADOS_EXTRAS / SEM_PESSOA, ignore_errors=True)

    cont, ignorados = Counter(), Counter()
    for ln in linhas:
        letra, gid = ln["letra"].upper(), ln["gravacao"].lower()
        nomes = grav.get((letra, gid))
        if not nomes:
            ignorados["não está no zip"] += 1
            continue
        if letra in config.LETRAS_DINAMICAS and len(nomes) == 1:
            ignorados[f"foto avulsa de letra com movimento ({letra})"] += 1
            continue
        p = _nome_seguro(ln["pessoa"])
        if p:
            base = config.DIR_DADOS_BRUTOS / f"{config.PREFIXO_CONTRIBUIDO}{p}" / letra
        elif sem_pessoa == "extras":
            base = config.DIR_DADOS_EXTRAS / SEM_PESSOA / letra
        else:
            ignorados["sem pessoa identificada"] += 1
            continue
        if len(nomes) == 1:
            destinos = [(nomes[0], base / f"{gid}.jpg")]
        else:
            destinos = [(n, base / gid / f"f{int(RE_ARQ.match(n)[3] or 0):04d}.jpg") for n in nomes]
        for membro, dst in destinos:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(z.read(membro))
        cont[(f"{config.PREFIXO_CONTRIBUIDO}{p}" if p else SEM_PESSOA, letra)] += 1

    pessoas = sorted({k[0] for k in cont})
    letras = config.LETRAS_ESTATICAS + config.LETRAS_DINAMICAS
    print("Gravações por pessoa x letra:")
    print("pessoa".ljust(20) + " ".join(l.rjust(3) for l in letras) + "  total")
    for p in pessoas:
        tot = sum(cont[(p, l)] for l in letras)
        print(p[:20].ljust(20) + " ".join(str(cont[(p, l)]).rjust(3) for l in letras) + f"  {tot}")
    if ignorados:
        print(f"\nNão copiadas: {dict(ignorados)}")
    print(f"\nPronto. Agora: python -m libras.extrair")


# ---------------------------------------------------------------------------
# resumo e folha de conferência
# ---------------------------------------------------------------------------
def _resumo(linhas):
    por = Counter(ln["pessoa"] or "(sem pessoa)" for ln in linhas)
    como = Counter(ln["como"] for ln in linhas)
    print(f"\nComo cada gravação recebeu pessoa: {dict(como)}")
    for p, n in sorted(por.items()):
        lets = sorted({ln["letra"] for ln in linhas if (ln["pessoa"] or "(sem pessoa)") == p})
        print(f"  {p:14s} {n:4d} gravações | {len(lets):2d} letras: {''.join(lets)}")


def folha_conferencia(z, grav, linhas, por_linha=14, tw=128, th=96):
    from PIL import Image, ImageDraw

    pessoas = sorted({ln["pessoa"] for ln in linhas}, key=lambda p: (p == "", p))
    rng = np.random.default_rng(0)
    rot = 110
    folha = Image.new("RGB", (rot + por_linha * (tw + 4), len(pessoas) * (th + 6)), "white")
    d = ImageDraw.Draw(folha)
    for r, p in enumerate(pessoas):
        lin = [ln for ln in linhas if ln["pessoa"] == p]
        y = r * (th + 6)
        d.text((6, y + 20), f"{p or 'sem pessoa'}\n{len(lin)} grav.\n"
                            f"{sum(ln['como'] == 'cenário' for ln in lin)} p/ cenário",
               fill=(0, 0, 0))
        for j, i in enumerate(rng.choice(len(lin), min(por_linha, len(lin)), replace=False)):
            nomes = grav[(lin[i]["letra"], lin[i]["gravacao"])]
            img = _ler(z, nomes[len(nomes) // 2])
            if img is None:
                continue
            im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            im.thumbnail((tw, th))
            x = rot + j * (tw + 4)
            folha.paste(im, (x, y))
            cor = (200, 60, 20) if lin[i]["como"] == "cenário" else (20, 120, 160)
            d.rectangle((x, y, x + 18, y + 14), fill=cor)
            d.text((x + 4, y + 1), lin[i]["letra"], fill="white")
    folha.save(ARQ_FOLHA, quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("agrupar", help="descobre as pessoas e grava dados/contribuido/gravacoes.csv")
    a1.add_argument("--zip", type=Path, required=True)
    a1.add_argument("--limiar-rosto", type=float, default=0.65,
                    help="distância de cosseno máxima entre rostos da mesma pessoa (SFace)")
    a1.add_argument("--min-gravacoes", type=int, default=10, help="tamanho mínimo de um grupo-pessoa")
    a1.add_argument("--vizinhos", type=int, default=5, help="vizinhos de cenário consultados")
    a1.add_argument("--percentil-cena", type=float, default=90.0,
                    help="quão parecido o cenário precisa ser (menor = mais exigente)")
    a2 = sub.add_parser("organizar", help="copia o zip para dados/brutos/contrib_<pessoa>/<LETRA>/")
    a2.add_argument("--zip", type=Path, required=True)
    a2.add_argument("--sem-pessoa", choices=["descartar", "extras"], default="descartar",
                    help="gravações sem pessoa: descartar (padrão, evita vazamento) ou "
                         "mandar para dados/extras (só treino)")
    a = ap.parse_args()
    zp = a.zip.expanduser()
    if not zp.is_file():
        raise SystemExit(f"Zip não encontrado: {zp}")
    if a.cmd == "agrupar":
        agrupar(zp, a.limiar_rosto, a.min_gravacoes, a.vizinhos, a.percentil_cena)
    else:
        organizar(zp, a.sem_pessoa)


if __name__ == "__main__":
    main()

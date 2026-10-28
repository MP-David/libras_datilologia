"""Importa datasets externos para dados/extras/<nome>/<LETRA>/ (uso SÓ no treino).

1) Brazilian Sign Language Alphabet (Mendeley, Pascal VOC, 15 letras):
    python -m libras.importar voc  --origem ~/Downloads/libras_voc --nome mendeley

2) Datasets com uma pasta por classe (ex.: ASL Alphabet, para pré-treino):
    python -m libras.importar pastas --origem ~/Downloads/asl_alphabet_train --nome asl \
        --somente ABCDEFGILMNOPQRSTUVW --max-por-letra 300

ATENÇÃO com o ASL: várias configurações de mão diferem da LIBRAS. Use
--somente com a lista de letras que a equipe CONFERIU (de preferência com o
curso de Letras-LIBRAS) como equivalentes; o resto não entra.

Por que "extras" nunca entram no teste: não sabemos quem sinalizou cada
imagem, então não dá para garantir que a pessoa do teste não está no treino.
Imagens pequenas (200x200, mão ocupando o quadro todo) ganham uma borda
para o detector de mão funcionar melhor.
"""
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2

from . import config

TODAS = config.LETRAS_ESTATICAS + config.LETRAS_DINAMICAS


def _salvar(img_path: Path, destino: Path, borda: int):
    destino.parent.mkdir(parents=True, exist_ok=True)
    if borda <= 0:
        shutil.copy2(img_path, destino)
        return
    img = cv2.imread(str(img_path))
    if img is None:
        return
    pad = int(max(img.shape[:2]) * borda / 100)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    cv2.imwrite(str(destino.with_suffix(".jpg")), img)


def importar_voc(origem: Path, nome: str, borda: int, somente: set[str], maximo: int):
    cont = Counter()
    for xml in sorted(origem.rglob("*.xml")):
        raiz = ET.parse(xml).getroot()
        obj = raiz.find("object/name")
        if obj is None or not obj.text:
            continue
        letra = obj.text.strip().upper()[:1]
        if letra not in TODAS or (somente and letra not in somente) or cont[letra] >= maximo:
            continue
        arq = raiz.findtext("filename") or ""
        candidatos = [xml.parent / arq] + [xml.with_suffix(e) for e in (".jpg", ".jpeg", ".png")]
        img = next((c for c in candidatos if c.is_file()), None)
        if img is None:
            continue
        _salvar(img, config.DIR_DADOS_EXTRAS / nome / letra / f"{img.stem}.jpg", borda)
        cont[letra] += 1
    return cont


def _achar_pasta_de_letras(origem: Path) -> Path:
    """O zip do Kaggle costuma criar asl_alphabet_train/asl_alphabet_train/A/...
    Procura (até 3 níveis) a pasta que contém subpastas com nomes de letras."""
    def n_letras(p: Path) -> int:
        try:
            return sum(1 for q in p.iterdir() if q.is_dir() and q.name.strip().upper() in TODAS)
        except OSError:
            return 0
    candidatos = [origem] + [p for nivel in ("*", "*/*", "*/*/*") for p in origem.glob(nivel) if p.is_dir()]
    melhor = max(candidatos, key=n_letras)
    if n_letras(melhor) == 0:
        raise SystemExit(f"Não achei subpastas com nomes de letras (A, B, C...) dentro de {origem}.")
    if melhor != origem:
        print(f"Usando a pasta de letras encontrada em: {melhor}")
    return melhor


def importar_pastas(origem: Path, nome: str, borda: int, somente: set[str], maximo: int):
    cont = Counter()
    origem = _achar_pasta_de_letras(origem)
    for pasta in sorted(p for p in origem.iterdir() if p.is_dir()):
        letra = pasta.name.strip().upper()
        if letra not in TODAS or (somente and letra not in somente):
            continue
        imgs = [p for p in pasta.iterdir() if p.suffix.lower() in config.EXT_IMAGEM]
        random.Random(0).shuffle(imgs)
        for img in imgs[:maximo]:
            _salvar(img, config.DIR_DADOS_EXTRAS / nome / letra / f"{img.stem}.jpg", borda)
            cont[letra] += 1
    return cont


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("formato", choices=["voc", "pastas"])
    ap.add_argument("--origem", type=Path, required=True)
    ap.add_argument("--nome", required=True, help="nome da fonte, ex.: mendeley, asl")
    ap.add_argument("--somente", default="", help="letras permitidas, ex.: ABCDE")
    ap.add_argument("--max-por-letra", type=int, default=10**9)
    ap.add_argument("--borda", type=int, default=25, help="borda em %% do lado (0 = copiar sem mexer)")
    a = ap.parse_args()
    somente = set(a.somente.upper())
    origem = a.origem.expanduser()
    if not origem.is_dir():
        pai = origem.parent
        dica = ""
        if pai.is_dir():
            parecidos = sorted(p.name for p in pai.iterdir()
                               if p.is_dir() and any(k in p.name.lower()
                                                     for k in ("asl", "libras", "sign", "alfabet", "alphabet")))
            if parecidos:
                dica = "\nPastas parecidas em " + str(pai) + ":\n  " + "\n  ".join(parecidos)
        raise SystemExit(f"Pasta não encontrada: {origem}\n"
                         f"Descompacte o dataset e passe o caminho real em --origem.{dica}")
    f = importar_voc if a.formato == "voc" else importar_pastas
    cont = f(origem, a.nome, a.borda, somente, a.max_por_letra)
    if not cont:
        raise SystemExit("Nenhuma imagem importada — confira --origem e --somente.")
    print(f"Importadas {sum(cont.values())} imagens para {config.DIR_DADOS_EXTRAS / a.nome}")
    print(dict(sorted(cont.items())))
    print("Agora rode: python -m libras.extrair")


if __name__ == "__main__":
    main()

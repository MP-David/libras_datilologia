"""Extrai os pontos da mão de todas as imagens/vídeos coletados.

    python -m libras.extrair            # incremental: só processa o que é novo
    python -m libras.extrair --refazer

Lê:
  dados/brutos/<sinalizador>/<LETRA>/   -> dados locais da equipe (treino e teste)
  dados/extras/<fonte>/<LETRA>/         -> datasets externos (SÓ treino)
Grava:
  dados/processados/amostras.joblib  com
    "fotos":  1 foto  -> 21 pontos            (só letras estáticas)
    "clipes": 1 vídeo ou pasta de quadros -> sequência (T,21,3), NaN onde não
              houve mão. Vale para TODAS as letras: o clipe inteiro é guardado e
              o treino decide quais quadros usar (ex.: só os quadros em que a
              letra já está formada).
Regras:
  * letra com movimento + foto -> ignorada (não há movimento numa foto)
  * clipe de letra com movimento precisa de config.MIN_QUADROS_COM_MAO quadros com mão
Arquivos onde nenhuma mão foi encontrada são listados no fim — reveja-os:
mão cortada, desfocada ou escura demais é problema de coleta, não de modelo.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np

from . import config
from .detector import DetectorMao, ler_sequencia

ARQ_SAIDA = config.DIR_PROCESSADOS / "amostras.joblib"
VERSAO = 2


def _letra_da_pasta(nome: str) -> str | None:
    n = nome.strip().upper()
    return n if n in config.LETRAS_ESTATICAS + config.LETRAS_DINAMICAS else None


def listar_fontes():
    """Gera (grupo, letra, caminho, origem). grupo = sinalizador (ou 'extra:fonte')."""
    for raiz, origem in ((config.DIR_DADOS_BRUTOS, "local"), (config.DIR_DADOS_EXTRAS, "extra")):
        if not raiz.exists():
            continue
        for fonte in sorted(p for p in raiz.iterdir() if p.is_dir()):
            grupo = fonte.name if origem == "local" else f"extra:{fonte.name}"
            for pasta_letra in sorted(p for p in fonte.iterdir() if p.is_dir()):
                letra = _letra_da_pasta(pasta_letra.name)
                if letra is None:
                    continue
                for item in sorted(pasta_letra.iterdir()):
                    yield grupo, letra, item, origem


def banco_vazio() -> dict:
    return {"versao": VERSAO, "fotos": [], "clipes": [], "falhas": [], "vistos": {}}


def carregar_banco(caminho=ARQ_SAIDA) -> dict:
    banco = joblib.load(caminho)
    if banco.get("versao") != VERSAO:
        raise SystemExit(f"{caminho} é de uma versão antiga do extrator. "
                         "Rode: python -m libras.extrair --refazer")
    return banco


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refazer", action="store_true", help="ignora o cache e extrai tudo de novo")
    a = ap.parse_args()

    config.DIR_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    banco = banco_vazio()
    if ARQ_SAIDA.exists() and not a.refazer:
        antigo = joblib.load(ARQ_SAIDA)
        if antigo.get("versao") == VERSAO:
            banco = antigo
        else:
            print("Cache de versão antiga: extraindo tudo de novo.")

    det_img = DetectorMao(video=False)
    novos = 0
    atuais: set[str] = set()
    try:
        for grupo, letra, item, origem in listar_fontes():
            chave = str(item.resolve())
            atuais.add(chave)
            assinatura = item.stat().st_mtime
            if banco["vistos"].get(chave) == assinatura:
                continue
            if chave in banco["vistos"]:   # arquivo alterado: descarta a versão antiga
                _remover(banco, {chave})
            banco["vistos"][chave] = assinatura
            novos += 1
            meta = {"letra": letra, "sinalizador": grupo, "origem": origem, "arquivo": chave}
            eh_img = item.is_file() and item.suffix.lower() in config.EXT_IMAGEM
            eh_seq = item.is_dir() or (item.is_file() and item.suffix.lower() in config.EXT_VIDEO)
            dinamica = letra in config.LETRAS_DINAMICAS

            if eh_img and not dinamica:
                img = cv2.imread(chave)
                r = det_img.detectar(img) if img is not None else None
                if r is None:
                    banco["falhas"].append((chave, "sem mão"))
                else:
                    banco["fotos"].append({**meta, "pts": r[0], "mao": r[1]})
            elif eh_img and dinamica:
                banco["falhas"].append((chave, "foto de letra com movimento (precisa de vídeo)"))
            elif eh_seq:
                with DetectorMao(video=True) as det_vid:
                    seq, mao = ler_sequencia(item, det_vid)
                n_ok = int((~np.isnan(seq).any(axis=(1, 2))).sum()) if len(seq) else 0
                minimo = config.MIN_QUADROS_COM_MAO if dinamica else 1
                if n_ok < minimo:
                    banco["falhas"].append((chave, f"só {n_ok} quadros com mão"))
                else:
                    banco["clipes"].append({**meta, "seq": seq.astype(np.float32), "mao": mao})
            if novos % 200 == 0:
                print(f"  {novos} arquivos processados...", flush=True)
        # arquivo apagado/movido desde a última extração: tira do banco também
        sumiu = {k for k in banco["vistos"] if k not in atuais}
        if sumiu:
            _remover(banco, sumiu)
            print(f"  {len(sumiu)} arquivos não existem mais — removidos do banco")
    finally:
        det_img.fechar()
        joblib.dump(banco, ARQ_SAIDA)

    print(f"\nNovos arquivos: {novos}")
    print(f"Fotos: {len(banco['fotos'])} | clipes: {len(banco['clipes'])}")
    cont = Counter((s["sinalizador"], s["letra"]) for s in banco["fotos"] + banco["clipes"])
    sinalizadores = sorted({k[0] for k in cont})
    letras = config.LETRAS_ESTATICAS + config.LETRAS_DINAMICAS
    print("\nGravações (fotos + clipes) por sinalizador x letra:")
    print("sinalizador".ljust(18) + " ".join(l.rjust(3) for l in letras))
    for s in sinalizadores:
        print(s[:18].ljust(18) + " ".join(str(cont.get((s, l), 0)).rjust(3) for l in letras))
    if banco["falhas"]:
        motivos = Counter(m for _, m in banco["falhas"])
        print(f"\n{len(banco['falhas'])} arquivos sem aproveitamento: {dict(motivos)}")
        for arq, m in banco["falhas"][-10:]:
            print(f"   {m}: {Path(arq).name}")


def _remover(banco: dict, chaves: set[str]):
    for k in ("fotos", "clipes"):
        banco[k] = [s for s in banco[k] if s["arquivo"] not in chaves]
    banco["falhas"] = [f for f in banco["falhas"] if f[0] not in chaves]
    for k in chaves:
        banco["vistos"].pop(k, None)


if __name__ == "__main__":
    main()

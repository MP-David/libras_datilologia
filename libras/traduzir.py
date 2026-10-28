"""Tradutor de datilologia — uso final.

Webcam (tempo real):
    python -m libras.traduzir --camera 0
      ESPAÇO = espaço na palavra | BACKSPACE = apaga | C = limpa | Q = sai

Fotos e clipes (cada item é UMA letra, na ordem da palavra):
    python -m libras.traduzir --entrada a.jpg n.jpg a.jpg
    python -m libras.traduzir --entrada l.jpg clipe_j.mp4 quadros_z/
  * foto  -> só letras estáticas
  * vídeo ou pasta de quadros -> qualquer letra (H, J, K, X, Y, Z incluídas)

Vídeo de uma palavra inteira soletrada:
    python -m libras.traduzir --video-continuo palavra.mp4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from . import config
from .detector import DetectorMao, desenhar_mao, ler_sequencia
from .reconhecedor import TradutorContinuo, carregar_modelos, reconhecer_clipe, reconhecer_foto


def _faixa(img, texto, y, cor, escala=0.75):
    cv2.putText(img, texto, (12, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, texto, (12, y), cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 2, cv2.LINE_AA)


def _sem_acento(s: str) -> str:   # putText do OpenCV não desenha acentos
    import unicodedata
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def modo_camera(indice: int):
    est, din = carregar_modelos()
    if din is None:
        print("AVISO: sem modelo de movimento — H, J, K, X, Y, Z não serão reconhecidas.")
    trad = TradutorContinuo(est, din)
    cap = cv2.VideoCapture(indice)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir a câmera {indice}.")
    t0 = time.time()
    with DetectorMao(video=True) as det:
        while True:
            ok, quadro = cap.read()
            if not ok:
                break
            r = det.detectar(quadro, int((time.time() - t0) * 1000))
            ev = trad.processar(r[0] if r else None, r[1] if r else "Right")
            if ev:
                print(ev)
            tela = quadro.copy()
            if r:
                desenhar_mao(tela, r[0])
            tela = cv2.flip(tela, 1)
            h = tela.shape[0]
            ult = trad.ultimo
            cor = (0, 230, 0) if ult and ult.tipo == "LETRA" else (0, 160, 255)
            _faixa(tela, _sem_acento(trad.estado), 32, (255, 255, 255), 0.65)
            if ult:
                _faixa(tela, _sem_acento(str(ult).split("  |")[0]), 66, cor, 0.7)
            _faixa(tela, f"PALAVRA: {trad.palavra}_", h - 20, (0, 255, 255), 1.0)
            cv2.imshow("Tradutor LIBRAS", tela)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                trad.espaco()
            elif k in (8, 127):
                trad.apagar()
            elif k == ord("c"):
                trad.limpar()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nPalavra final: {trad.palavra}")


def modo_entradas(itens: list[str]):
    est, din = carregar_modelos()
    palavra = ""
    with DetectorMao(video=False) as det_img:
        for item in itens:
            p = Path(item)
            eh_foto = p.is_file() and p.suffix.lower() in config.EXT_IMAGEM
            if eh_foto:
                img = cv2.imread(str(p))
                r = det_img.detectar(img) if img is not None else None
                d = reconhecer_foto(est, r[0] if r else None, r[1] if r else "Right")
            else:
                with DetectorMao(video=True) as det_vid:
                    seq, mao = ler_sequencia(p, det_vid)
                d, _ = reconhecer_clipe(est, din, seq, mao)
            if d.aceita:
                palavra += d.letra
                print(f"[{p.name}] LETRA RECONHECIDA: {d.letra} ({d.confianca:.0%})  |  palavra: {palavra}")
            else:
                extra = (" — se for H/J/K/X/Y/Z, envie um vídeo" if eh_foto and d.top else "")
                print(f"[{p.name}] REPITA O SINAL ({d.motivo}{extra})  |  palavra: {palavra}")
            if d.top:
                print("          hipóteses: " + ", ".join(f"{l if l != '_TRANSICAO' else '∅'} {q:.0%}"
                                                       for l, q in d.top))
    print(f"\nPalavra: {palavra}")


def modo_video_continuo(arquivo: str):
    est, din = carregar_modelos()
    trad = TradutorContinuo(est, din)
    with DetectorMao(video=True) as det:
        seq, mao = ler_sequencia(arquivo, det)
    for q in seq:
        ev = trad.processar(None if np.isnan(q).any() else q, mao)
        if ev:
            print(ev)
    for _ in range(config.QUADROS_SEM_MAO_RESET):   # fecha um movimento pendente
        ev = trad.processar(None)
        if ev:
            print(ev)
    print(f"\nPalavra: {trad.palavra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--camera", type=int)
    g.add_argument("--entrada", nargs="+", help="fotos, vídeos ou pastas de quadros (1 letra cada)")
    g.add_argument("--video-continuo", help="vídeo com a palavra inteira")
    a = ap.parse_args()
    if a.camera is not None:
        modo_camera(a.camera)
    elif a.entrada:
        modo_entradas(a.entrada)
    else:
        modo_video_continuo(a.video_continuo)


if __name__ == "__main__":
    main()

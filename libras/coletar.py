"""Coleta de dados locais pela webcam.

    python -m libras.coletar --sinalizador ana --fotos 20 --repeticoes 8
    python -m libras.coletar --sinalizador bruno --letras HJKXYZ --repeticoes 10

Salva em dados/brutos/<sinalizador>/<LETRA>/:
  * letras estáticas -> fotos .jpg (quadro cru da câmera, sem espelhar)
  * letras com movimento -> um vídeo .mp4 por repetição

Boas práticas (armadilhas do tema):
  * várias pessoas, cada uma numa pasta própria — o teste é sempre com
    alguém que não está no treino;
  * troque de fundo, de iluminação e de distância entre as rodadas;
  * mexa levemente a mão entre as fotos (ângulo, altura), sem descaracterizar
    a letra.

Teclas: ESPAÇO inicia | S pula a letra | Q sai
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import cv2

from . import config
from .detector import DetectorMao, desenhar_mao


def _texto(img, txt, y, escala=0.8, cor=(255, 255, 255)):
    cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 2, cv2.LINE_AA)


class Camera:
    def __init__(self, indice: int):
        self.cap = cv2.VideoCapture(indice)
        if not self.cap.isOpened():
            raise SystemExit(f"Não consegui abrir a câmera {indice}.")
        self.det = DetectorMao(video=True)
        self.t0 = time.time()

    def ler(self):
        ok, quadro = self.cap.read()
        if not ok:
            raise SystemExit("Falha lendo a câmera.")
        r = self.det.detectar(quadro, int((time.time() - self.t0) * 1000))
        tela = quadro.copy()
        if r is not None:
            desenhar_mao(tela, r[0])
        return quadro, cv2.flip(tela, 1), r       # exibe espelhado, salva cru

    def fechar(self):
        self.cap.release()
        self.det.fechar()
        cv2.destroyAllWindows()


def esperar_espaco(cam: Camera, linhas) -> str:
    while True:
        _, tela, r = cam.ler()
        for i, l in enumerate(linhas):
            _texto(tela, l, 34 + 34 * i)
        _texto(tela, "mao detectada" if r else "sem mao", tela.shape[0] - 16, 0.6,
               (0, 255, 0) if r else (0, 0, 255))
        cv2.imshow("coleta LIBRAS", tela)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            return "ok"
        if k in (ord("s"), ord("q")):
            return chr(k)


def contagem(cam: Camera, letra: str, seg: float = 2.0):
    fim = time.time() + seg
    while time.time() < fim:
        _, tela, _ = cam.ler()
        _texto(tela, f"{letra} em {fim - time.time():.1f}s", 60, 1.4, (0, 255, 255))
        cv2.imshow("coleta LIBRAS", tela)
        cv2.waitKey(1)


def coletar_estatica(cam, pasta, letra, n, intervalo=0.25):
    salvas, ultimo = 0, 0.0
    while salvas < n:
        quadro, tela, r = cam.ler()
        if r is not None and time.time() - ultimo >= intervalo:
            nome = pasta / f"{letra}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
            cv2.imwrite(str(nome), quadro)
            salvas, ultimo = salvas + 1, time.time()
        _texto(tela, f"{letra}: {salvas}/{n}  (varie levemente o angulo)", 34)
        cv2.imshow("coleta LIBRAS", tela)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
    return True


def gravar_dinamica(cam, arquivo, seg=2.2):
    quadro, _, _ = cam.ler()
    h, w = quadro.shape[:2]
    out = cv2.VideoWriter(str(arquivo), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    fim = time.time() + seg
    while time.time() < fim:
        quadro, tela, _ = cam.ler()
        out.write(quadro)
        _texto(tela, f"GRAVANDO {fim - time.time():.1f}s", 34, 1.0, (0, 0, 255))
        cv2.imshow("coleta LIBRAS", tela)
        cv2.waitKey(1)
    out.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sinalizador", required=True, help="id curto, sem espaços (ex.: ana)")
    ap.add_argument("--letras", default="".join(config.LETRAS_ESTATICAS + config.LETRAS_DINAMICAS))
    ap.add_argument("--fotos", type=int, default=20, help="fotos por letra estática")
    ap.add_argument("--repeticoes", type=int, default=8, help="vídeos por letra com movimento")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fundo", default="", help="descrição do fundo/ambiente (vai para info.json)")
    a = ap.parse_args()

    base = config.DIR_DADOS_BRUTOS / a.sinalizador
    base.mkdir(parents=True, exist_ok=True)
    info_path = base / "info.json"
    info = json.loads(info_path.read_text("utf-8")) if info_path.exists() else {"sessoes": []}
    info["sessoes"].append({"data": datetime.now().isoformat(timespec="seconds"),
                            "fundo": a.fundo, "letras": a.letras})
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), "utf-8")

    cam = Camera(a.camera)
    try:
        for letra in a.letras.upper():
            pasta = base / letra
            pasta.mkdir(exist_ok=True)
            dinamica = letra in config.LETRAS_DINAMICAS
            reps = a.repeticoes if dinamica else 1
            for rep in range(reps):
                tipo = f"MOVIMENTO {rep + 1}/{reps}" if dinamica else f"{a.fotos} fotos"
                acao = esperar_espaco(cam, [f"Letra {letra} ({tipo})",
                                            "ESPACO comeca | S pula | Q sai"])
                if acao == "q":
                    return
                if acao == "s":
                    break
                contagem(cam, letra)
                if dinamica:
                    gravar_dinamica(cam, pasta / f"{letra}_{datetime.now():%Y%m%d_%H%M%S}_{rep}.mp4")
                elif not coletar_estatica(cam, pasta, letra, a.fotos):
                    return
    finally:
        cam.fechar()
        print(f"Dados salvos em {base}")


if __name__ == "__main__":
    main()

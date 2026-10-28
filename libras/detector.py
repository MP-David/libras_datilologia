"""Detecção dos 21 pontos da mão com MediaPipe Hand Landmarker (API Tasks).

Convenção IMPORTANTE: sempre processe o quadro como a câmera o entrega (sem
espelhar). O espelhamento "modo selfie" é só para exibição na tela. Assim a
etiqueta de mão (Right/Left) do MediaPipe é consistente entre coleta, treino e
uso, e o espelhamento de canhotos em features.normalizar_mao funciona.
"""
from __future__ import annotations

import urllib.request

import cv2
import numpy as np

from . import config


def garantir_modelo() -> str:
    caminho = config.MODELO_MEDIAPIPE
    if not caminho.exists():
        print(f"Baixando modelo do MediaPipe para {caminho} ...")
        urllib.request.urlretrieve(config.URL_MODELO_MEDIAPIPE, caminho)
    return str(caminho)


class DetectorMao:
    """Uso:
        with DetectorMao(video=False) as det:
            r = det.detectar(img_bgr)          # foto
        with DetectorMao(video=True) as det:
            r = det.detectar(quadro_bgr, t_ms) # sequência / webcam
    r = (pts (21,3) em pixels, "Right"/"Left", score) ou None.
    """

    def __init__(self, video: bool = False, confianca: float = 0.5):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        modo = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        opcoes = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=garantir_modelo()),
            running_mode=modo,
            num_hands=1,
            min_hand_detection_confidence=confianca,
            min_hand_presence_confidence=confianca,
            min_tracking_confidence=confianca,
        )
        self._video = video
        self._ultimo_t = -1
        self._det = vision.HandLandmarker.create_from_options(opcoes)

    def detectar(self, img_bgr: np.ndarray, t_ms: int | None = None):
        import mediapipe as mp

        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self._video:
            t_ms = int(t_ms if t_ms is not None else self._ultimo_t + 33)
            t_ms = max(t_ms, self._ultimo_t + 1)   # timestamps precisam crescer
            self._ultimo_t = t_ms
            res = self._det.detect_for_video(imagem, t_ms)
        else:
            res = self._det.detect(imagem)
        if not res.hand_landmarks:
            return None
        lm = res.hand_landmarks[0]
        # pixels (z do MediaPipe está na escala da largura da imagem)
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=np.float32)
        cat = res.handedness[0][0]
        return pts, cat.category_name, float(cat.score)

    def fechar(self):
        self._det.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fechar()


def desenhar_mao(img: np.ndarray, pts: np.ndarray, cor=(0, 220, 120)) -> None:
    from .features import DEDOS
    for dedo in DEDOS:
        for a, b in zip(dedo[:-1], dedo[1:]):
            cv2.line(img, tuple(map(int, pts[a][:2])), tuple(map(int, pts[b][:2])), cor, 2)
    for p in pts:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, (255, 255, 255), -1)


def ler_sequencia(caminho, det: "DetectorMao") -> tuple[np.ndarray, str]:
    """Lê um vídeo ou uma pasta de quadros e devolve (T,21,3) com NaN onde não
    houve mão, além da mão predominante."""
    from pathlib import Path
    caminho = Path(caminho)
    quadros = []
    if caminho.is_dir():
        arquivos = sorted(p for p in caminho.iterdir() if p.suffix.lower() in config.EXT_IMAGEM)
        fonte = ((cv2.imread(str(a)), i * 33) for i, a in enumerate(arquivos))
    else:
        cap = cv2.VideoCapture(str(caminho))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        def gen():
            i = 0
            while True:
                ok, q = cap.read()
                if not ok:
                    break
                yield q, int(i * 1000 / fps)
                i += 1
            cap.release()
        fonte = gen()
    maos = []
    for img, t in fonte:
        if img is None:
            continue
        r = det.detectar(img, t)
        if r is None:
            quadros.append(np.full((21, 3), np.nan, np.float32))
        else:
            quadros.append(r[0])
            maos.append(r[1])
    mao = max(set(maos), key=maos.count) if maos else "Right"
    return (np.stack(quadros) if quadros else np.empty((0, 21, 3), np.float32)), mao

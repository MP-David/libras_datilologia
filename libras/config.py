"""Configuração central do tradutor de datilologia em LIBRAS.

Tudo o que a equipe pode querer ajustar sem mexer na lógica fica aqui.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Pastas
# --------------------------------------------------------------------------
DIR_DADOS_BRUTOS = RAIZ / "dados" / "brutos"          # dados/brutos/<sinalizador>/<LETRA>/...
DIR_DADOS_EXTRAS = RAIZ / "dados" / "extras"          # datasets externos: SÓ treino, nunca teste
DIR_PROCESSADOS = RAIZ / "dados" / "processados"      # landmarks extraídos (.joblib)
DIR_MODELOS = RAIZ / "modelos"
DIR_RELATORIOS = RAIZ / "relatorios"

MODELO_MEDIAPIPE = RAIZ / "hand_landmarker.task"
URL_MODELO_MEDIAPIPE = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# --------------------------------------------------------------------------
# Alfabeto
# --------------------------------------------------------------------------
# AVISO À EQUIPE (pré-requisito do tema): as letras abaixo têm MOVIMENTO e não
# podem ser reconhecidas a partir de uma única foto. Elas são tratadas por um
# modelo separado que recebe uma SEQUÊNCIA de quadros.
# Confirme a lista com o curso de Letras-LIBRAS / comunidade surda local:
# há variação regional, e o Ç (configuração do C com movimento) costuma ser
# incluído — basta acrescentá-lo aqui e coletar dados para ele.
LETRAS_DINAMICAS = ["H", "J", "K", "X", "Y", "Z"]
LETRAS_ESTATICAS = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in LETRAS_DINAMICAS]

# Pares/grupos com configuração de mão parecida. O reconhecedor fica mais
# exigente quando as duas primeiras hipóteses caem no mesmo grupo, e o
# relatório de avaliação mostra cada grupo separadamente.
GRUPOS_CONFUNDIVEIS = [
    ["A", "S", "E"],
    ["M", "N"],
    ["U", "V", "R"],
    ["F", "T"],
    ["O", "C"],
    ["G", "Q"],
]

# Extensões aceitas
EXT_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXT_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# --------------------------------------------------------------------------
# Sequências (letras com movimento)
# --------------------------------------------------------------------------
QUADROS_SEQUENCIA = 16        # toda sequência é reamostrada para este tamanho
MIN_QUADROS_COM_MAO = 6       # menos que isso -> sequência descartada / REPITA

# --------------------------------------------------------------------------
# Reconhecedor em tempo real (valores iniciais; o treino recalibra o limiar)
# --------------------------------------------------------------------------
LIMIAR_CONFIANCA = 0.60       # prob. mínima da 1ª hipótese
MARGEM_MINIMA = 0.15          # diferença mínima entre 1ª e 2ª hipóteses
MARGEM_GRUPO_CONFUNDIVEL = 0.30  # margem exigida se 1ª e 2ª estão no mesmo grupo
QUADROS_PARA_FIXAR = 8        # quadros parados para decidir uma letra estática
CONCORDANCIA_MINIMA = 0.70    # fração dos quadros que precisa concordar
LIMIAR_MOVIMENTO = 0.35       # deslocamento (em "tamanhos de palma") que indica movimento
QUADROS_REPOUSO_FIM_MOV = 5   # quadros parados que encerram um movimento
MAX_QUADROS_MOVIMENTO = 60    # movimento mais longo que isso é descartado
QUADROS_SEM_MAO_RESET = 6     # mão some por N quadros -> libera repetir a mesma letra
LIMIAR_MUDANCA_POSE = 0.12    # mudança média dos pontos (em tamanhos de palma) que libera a próxima letra

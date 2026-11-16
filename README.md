# Tradutor do alfabeto manual (datilologia) em LIBRAS

Reconhece letras do alfabeto manual a partir de fotos, clipes ou webcam e monta a
palavra soletrada. Saída sempre em um destes formatos:

```
LETRA RECONHECIDA: A (97%)  |  palavra: ANA
REPITA O SINAL (dúvida entre M e N)  |  palavra: AN
```

> ## ⚠️ Aviso à equipe (leia antes de coletar dados)
> **H, J, K, X, Y e Z têm movimento** e não podem ser reconhecidas a partir de uma
> única foto. O projeto trata essas letras com um **modelo separado, que recebe uma
> sequência de quadros** (clipe ou webcam). Por isso:
> - para essas letras, colete **clipes/vídeos** (o script de coleta já faz isso);
> - o dataset do Mendeley cobre só 15 letras e **nenhuma** das com movimento —
>   elas dependem 100% da coleta da equipe;
> - confirme a lista com o curso de Letras-LIBRAS: há variação regional, e o **Ç**
>   (C com movimento) costuma entrar. Para incluí-lo basta acrescentá-lo em
>   `LETRAS_DINAMICAS` (`libras/config.py`) e coletar clipes.

---

## Resultados (16/11/2026)

Medidos em pessoa que **não estava no treino**, por gravação (uma foto ou um clipe = uma
resposta). Relatório completo em `relatorios/relatorio.md` depois do treino.

| Métrica | Validação (uma pessoa de fora por vez) | Teste (p02 e p06, nunca no treino) |
|---|---|---|
| Letras paradas (20) | 82,4% (pior letra 45%) | 95,0% |
| Clipes, 26 letras | 79,4% | 89,1% |
| Só letras com movimento (H J K X Y Z) | 51,7% | 75,0% |
| CNN com transfer learning (comparação, letras paradas) | 55,0% | 65,8% |

Letras mais fracas na validação: Y 12%, X 33%, Z 44%, F 45%, T 47%, R 50%.

## Como funciona

```
imagem/quadro ──► MediaPipe Hand Landmarker ──► 21 pontos (x,y,z) da mão
                                                   │
                         normalização: punho na origem, escala da palma,
                         mão esquerda espelhada (canhotos)
                                                   │
   FOTO ───────────────────────────────► modelo ESTÁTICO (20 letras, MLP)
                                          114 atributos por quadro: coordenadas,
                                          distâncias entre pontas, polegar×dedos,
                                          ângulos das articulações, orientação
   CLIPE (16 quadros) ─┬─ quadros em que a letra JÁ ESTÁ FORMADA ─► modelo ESTÁTICO
                       └─ clipe inteiro ─► modelo de MOVIMENTO: H J K X Y Z ou ∅
                                          (Transformer com atenção, escolhido pela
                                           validação contra logística e MLP)
                       P(letra parada) = P(∅) × P_estático   ·   P(H…Z) = P_movimento
                                                   │
                         regra de decisão: confiança, margem entre 1ª e 2ª
                         hipóteses (mais exigente em A/S/E, M/N…), concordância
                         entre quadros  ──►  LETRA  ou  REPITA O SINAL
```

- **Quadros estáveis.** Quem grava uma letra parada leva alguns quadros para formar a mão.
  Esses quadros iniciais não são a letra (ex.: mão aberta chegando no A); o treino e a
  decisão usam só o trecho final com a forma da mão estável (`features.quadros_estaveis`).
- **∅ aprendido com clipes reais.** O modelo de movimento aprende o que "não é letra com
  movimento" com os próprios clipes de letras paradas da equipe e com trocas entre duas
  poses reais da mesma pessoa. Assim a mão se ajeitando no começo do clipe não vira REPITA.
- **Webcam.** Uma máquina de estados acompanha a mão: parada por 8 quadros → decide uma
  letra estática; em movimento → grava até parar e passa o trecho ao modelo de movimento;
  uma troca de letra (∅) é ignorada.

**Por que pontos da mão e não a imagem crua (CNN)?** É a resposta direta às armadilhas do
tema, e foi medido: `python -m libras.comparar_cnn` treina uma MobileNetV3 com transfer
learning no recorte da mão, com o mesmo protocolo, e ela fica 27 pontos abaixo do modelo
de pontos na validação por pessoa (aprende quem gravou e onde, não a mão).

| Armadilha | Como o projeto trata |
|---|---|
| Fundo homogêneo vira atalho | O classificador nunca vê pixels de fundo, só os 21 pontos. A comparação com CNN mede o efeito. |
| Mãos de tamanhos diferentes | Tudo é dividido pelo tamanho da palma. |
| Treinar e testar com a mesma pessoa | Avaliação **por gravação**, deixando **uma pessoa de fora**, e teste final com pessoas reservadas (`--teste`). O relatório mostra quanto a divisão aleatória superestima. |
| Confusões A/S/E, M/N | Atributos da posição do polegar em relação aos dedos; margem maior exigida dentro desses grupos (gera REPITA em vez de chute); tabela de confusão por grupo no relatório. |
| Letras com movimento | Modelo de sequência separado, com classe ∅ aprendida de clipes reais de letras paradas. |

Relação com as aulas: classificação supervisionada com **métricas por classe** (aulas 5–6),
**CNN com transfer learning** como comparação (aulas 5–6, `libras/comparar_cnn.py`) e
**sequências com atenção** — Transformer com pooling por atenção sobre os 16 quadros
(aula 10, `libras/sequencia.py`), comparado com modelos clássicos na mesma validação.

## Tech stack

Python 3.12, CPU comum

| Biblioteca | Versão testada | Papel |
|---|---|---|
| MediaPipe (Hand Landmarker) | 1.2.0 | Acha a mão e devolve os 21 pontos por quadro |
| OpenCV (`opencv-python`) | 5.0.0 | Imagens, vídeo e webcam; rosto (YuNet + SFace) para separar as pessoas do zip |
| NumPy | 2.5.4 | Normalização, atributos e aumentos de dados |
| scikit-learn | 1.9.1 | MLP, Regressão Logística, Extra Trees; agrupamento de rostos; métricas |
| PyTorch | 2.15.0 | Transformer com atenção (letras com movimento) |
| torchvision | 0.29.2 | MobileNetV3 pré-treinada no ImageNet (comparação com CNN) |
| pandas | 3.1.0 | Tabelas por letra e por pessoa, CSVs |
| Matplotlib | 3.11.2 | Matrizes de confusão |
| joblib | 1.6.0 | Pontos extraídos e modelos treinados |
| Pillow | 13.0.0 | Folha de conferência das pessoas |
| pytest | 9.2.0 | Testes unitários |

Modelos de terceiros baixados na primeira execução: `hand_landmarker.task` (Google
MediaPipe), YuNet e SFace (OpenCV Zoo), MobileNetV3-Small (torchvision).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.10+ (confira se existe wheel do `mediapipe` para a sua versão). Na primeira
execução o modelo `hand_landmarker.task` (~8 MB) é baixado automaticamente do Google.

## Processo de dados e treinamento

### 1. Coletar dados locais (mínimo 150 imagens; várias pessoas)

```bash
python -m libras.coletar --sinalizador ana   --fotos 20 --repeticoes 8 --fundo "sala, parede branca"
python -m libras.coletar --sinalizador bruno --fotos 20 --repeticoes 8 --fundo "corredor, janela"
python -m libras.coletar --sinalizador carla --letras HJKXYZ --repeticoes 10
```
Cada pessoa fica em `dados/brutos/<sinalizador>/<LETRA>/`. Fotos ou clipes para letras
estáticas, clipes `.mp4` (ou pastas de quadros) para as com movimento.

Recomendações de protocolo:
- **pelo menos 4 pessoas** (3 para validação cruzada + 1 reservada para teste);
- fundos reais e variados, luz diferente, distâncias diferentes, mãos de tamanhos distintos;
- incluir pessoa canhota se possível;
- não espelhe fotos de celular (câmera frontal costuma espelhar — use a traseira ou desative);
- peça a alguém fluente (comunidade surda / Letras-LIBRAS) para conferir as configurações.

### 2. Dataset da equipe (`contributed_images.zip`)

O zip foi gravado pela equipe: 7.059 imagens em `images/<LETRA>/*.jpg`, com clipes de 16
quadros (`<id>__f0000…f0015.jpg`) e fotos avulsas. Ele **não diz quem sinalizou**, e sem
isso não dá para avaliar com "pessoa que não apareceu no treino". O script descobre as
pessoas pelo rosto:

```bash
python -m libras.contribuido agrupar   --zip ~/Downloads/contributed_images.zip
#   -> dados/contribuido/gravacoes.csv  (1 linha por gravação, coluna "pessoa")
#   -> dados/contribuido/pessoas.jpg    (1 linha por pessoa: CONFIRA a olho)
python -m libras.contribuido organizar --zip ~/Downloads/contributed_images.zip
#   -> dados/brutos/contrib_p01/<LETRA>/<id>/f0000.jpg ...  (clipe = pasta de quadros)
```
- Rosto: detector YuNet + identidade SFace (OpenCV Zoo). Gravações sem rosto herdam a
  pessoa pelo **cenário** (mesmo quarto/luz) quando a semelhança é alta; o resto fica sem
  pessoa e **é descartado** por padrão (`--sem-pessoa extras` manda para o treino, com
  risco de vazar a mão de quem está no teste).
- Juntar duas pessoas num grupo é seguro (avaliação mais rígida); separar a mesma pessoa
  em dois grupos é o que vaza. Se a folha `pessoas.jpg` mostrar erro, corrija a coluna
  `pessoa` do CSV e rode só o `organizar` de novo.
- Fotos avulsas de H/J/K/X/Y/Z são ignoradas (letra com movimento precisa de sequência).

Resultado com o zip atual: 11 grupos de pessoas (p01–p11), 467 gravações identificadas
pelo rosto, 57 pelo cenário e 190 descartadas.

### 3. Dataset externo — entra **só no treino**

```bash
# Brazilian Sign Language Alphabet (Mendeley, Pascal VOC) — 15 letras, CC BY 4.0
# download público (sem login) da versão 5:
curl -L -o mendeley.zip https://data.mendeley.com/public-api/zip/k4gs3bmx5k/download/5
unzip -q mendeley.zip 'Preview Images/*'
python -m libras.importar voc --origem "Preview Images" --nome mendeley
```
Não se sabe quem sinalizou cada imagem, então ele nunca entra na validação nem no teste.
Ajuda pouco, mas ajuda: +2,4 pontos na validação por pessoa (`treinar --sem-extras` mede
isso). Cite a licença CC BY 4.0 no trabalho. Os outros dois datasets sugeridos não são
usados: o ASL Alphabet é ASL (o Mendeley já é o recorte dele com as 15 letras iguais em
LIBRAS) e o MINDS-Libras tem sinais, não o alfabeto. O importador também aceita datasets
com uma pasta por letra (`libras.importar pastas`).

### 4. Extrair os pontos da mão

```bash
python -m libras.extrair
```
Incremental (só processa o que é novo; arquivos apagados saem do banco). Guarda a
sequência inteira de cada clipe — o treino escolhe os quadros. Mostra uma tabela
pessoa × letra e lista arquivos onde nenhuma mão foi encontrada — revise-os.

### 5. Treinar e avaliar

```bash
python -m libras.treinar --teste contrib_p02,contrib_p06
python -m libras.treinar --teste contrib_p02,contrib_p06 --sem-extras   # mede o efeito do Mendeley
python -m libras.comparar_cnn --teste contrib_p02,contrib_p06            # comparação com CNN
```
Como avalia:
- as pessoas de `--teste` são separadas antes de tudo e nunca entram no treino;
- com as demais, **deixa uma pessoa de fora por vez** (validação) para escolher os modelos
  e calibrar o REPITA;
- a unidade é a **gravação**: os quadros de um clipe nunca ficam metade no treino e metade
  na avaliação; um clipe é decidido pela média dos seus quadros estáveis;
- letras estáticas: compara MLP, Regressão Logística e Extra Trees; escolhe pela acurácia
  balanceada (média da acurácia por letra), desempatando pela **pior letra**;
- clipes (26 letras): compara logística, MLP e Transformer com atenção no modelo de movimento;
- calibra o limiar de "REPITA O SINAL" para ~95% de acerto nas respostas dadas (`--alvo`);
- mede a armadilha da mesma pessoa (divisão aleatória × pessoa nova) e o efeito do Mendeley;
- salva `modelos/estatico.joblib` e `modelos/dinamico.joblib`.

Relatório em `relatorios/relatorio.md` com: seção **Dados** (origem, pessoas, descartes,
licença), acurácia **por letra** (validação e teste), matrizes de confusão, desempenho por
pessoa, grupos confundíveis, divisão aleatória × pessoa nova, com × sem Mendeley e o
tratamento declarado das letras com movimento. O treino completo leva ~1h30 em CPU,
quase tudo no Transformer.

## Como usar

```bash
python -m libras.traduzir --camera 0                       # tempo real
python -m libras.traduzir --entrada a.jpg n.jpg a.jpg      # fotos, 1 letra cada
python -m libras.traduzir --entrada l.jpg j.mp4 quadros_z/ # mistura fotos e clipes
python -m libras.traduzir --video-continuo palavra.mp4     # palavra inteira em vídeo
```
Webcam: **ESPAÇO** insere espaço, **BACKSPACE** apaga, **C** limpa, **Q** sai.
Segure cada letra ~0,3 s. Para letra dobrada (ex.: "LL"), dê um pequeno quique com a mão
ou tire-a do quadro e volte.

Exemplo com clipes de uma pessoa que nunca entrou no treino: JOGO, CASA e LIVRO saem
certos; ZECA sai GECA (o Z é uma das letras fracas).

## Como o projeto foi construído

1. **Dados da equipe.** O zip chegou sem identificação de pessoa; um mosaico por letra
   mostrou o conteúdo, e o agrupamento por rosto separou 11 grupos.
2. **Mendeley.** Baixado pela API pública e importado só para treino.
3. **Primeiro treino com dados reais.** Letras paradas com 74,2% na validação por pessoa.
4. **Teste de ponta a ponta.** Com clipes de uma pessoa nova, JOGO saía "JO" e CASA saía
   vazio. Diagnóstico: em 78% dos clipes de letras paradas a mão se mexia o bastante para
   o limiar fixo de movimento mandar o clipe ao modelo de movimento, e os primeiros quadros
   mostravam a mão ainda se formando.
5. **Correções medidas na validação por pessoa.** Usar só os quadros estáveis subiu as
   letras paradas; treinar o ∅ com clipes reais de letras paradas levou os clipes de 24,4%
   para ~79%.
6. **Aulas 5, 6 e 10.** A CNN com transfer learning (MobileNetV3, ajuste fino) ficou bem
   abaixo dos pontos; o Transformer com atenção entrou como candidato do modelo de
   movimento e venceu por pouco a logística.
7. **Treino final e relatório.** Avaliação por gravação, com seção de dados e as
   armadilhas medidas.

Toda mudança só ficou se melhorou a validação deixando uma pessoa de fora; as pessoas de
teste (p02 e p06) foram escolhidas antes dos resultados e nunca usadas para decidir nada.

## Critério de aceite → onde está

| Critério | Onde |
|---|---|
| Acurácia por letra (não só média) | `relatorio.md`, tabelas "Acurácia por letra"; CSVs `*_por_letra.csv` |
| Sinalizador que não apareceu no treino | `--teste <pessoa>` + seções "Pessoa(s) nunca vista(s)"; validação deixando uma pessoa de fora |
| Tratamento declarado das letras com movimento | Aviso acima; seção "Letras com movimento e clipes" do relatório; `libras/reconhecedor.py` |

## Estrutura

```
libras/
  config.py        letras estáticas/dinâmicas, grupos confundíveis, limiares, pastas
  detector.py      MediaPipe Hand Landmarker (API Tasks)
  features.py      normalização, atributos, quadros estáveis, aumento de dados
  modelo.py        modelos candidatos, classe ∅, junção dos modelos, regra LETRA/REPITA
  sequencia.py     Transformer com atenção sobre os quadros (letras com movimento)
  metricas.py      métricas por letra/pessoa/grupo, matriz de confusão, limiar
  comparar_cnn.py  MobileNetV3 + transfer learning no recorte da mão (comparação)
  coletar.py       coleta pela webcam
  contribuido.py   zip da equipe -> pessoas (rosto/cenário) -> dados/brutos/contrib_pNN
  importar.py      importa Pascal VOC (Mendeley) e pastas por classe
  extrair.py       imagens/clipes -> pontos da mão
  treinar.py       validação por pessoa, teste, escolha de modelo, relatório
  reconhecedor.py  foto, clipe e fluxo contínuo (máquina de estados)
  traduzir.py      interface de linha de comando / webcam
tests/             testes unitários
```

## Testes

```bash
python -m pytest -q
```

## Ajustes finos (`libras/config.py`)

| Sintoma | Ajuste |
|---|---|
| Muitos "REPITA O SINAL" | baixe `LIMIAR_CONFIANCA`/`MARGEM_MINIMA` ou rode `treinar --alvo 0.90` |
| Troca de letra vira H/J/K… | suba `LIMIAR_MOVIMENTO`; colete clipes com a mão chegando e saindo da letra |
| Letra demora a sair | baixe `QUADROS_PARA_FIXAR` |
| Mesma letra sai repetida | suba `LIMIAR_MUDANCA_POSE` |
| Câmera com muito tremor | suba `LIMIAR_MOVIMENTO` e `CONCORDANCIA_MINIMA` |

## Limitações conhecidas

- Letras com movimento são o ponto fraco (51,7% na validação): só 3 pessoas fora do teste
  têm clipes dessas letras, cerca de 12 por letra.
- F/T e R/U diferem por detalhes de dedo que os pontos do MediaPipe capturam mal.
- Depende do MediaPipe achar a mão: luz fraca, mão cortada ou desfocada viram REPITA.
- Não lê expressão facial nem segunda mão (não fazem parte da datilologia básica).
- As trocas de letra do ∅ são montadas a partir de poses reais; clipes reais de palavras
  soletradas, rotulados, melhorariam a webcam.

## Extensões sugeridas

- **Mais clipes de H/J/K/X/Y/Z com mais pessoas** é o que mais falta: o modelo de
  movimento é o ponto fraco por falta de dados, não de arquitetura.
- Trocar o fundo das imagens de teste da CNN para medir o atalho do fundo diretamente.
- Corretor ortográfico/dicionário de nomes para sugerir a palavra completa.

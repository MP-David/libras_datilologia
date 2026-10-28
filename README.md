#Tradutor do alfabeto manual (datilologia) em LIBRAS

Reconhece letras do alfabeto manual a partir de fotos, clipes ou webcam e monta a
palavra soletrada. Saída sempre em um destes formatos:

```
LETRA RECONHECIDA: A (97%)  |  palavra: ANA
REPITA O SINAL (dúvida entre M e N)  |  palavra: AN
```

> ## ⚠️ Aviso à equipe (leia antes de coletar dados)
> **H, J, K, X, Y e Z têm movimento** e não podem ser reconhecidas a partir de uma
> única foto. O projeto trata essas letras com um **modelo separado, que recebe uma
> sequência de quadros** (vídeo). Por isso:
> - para essas letras, colete **vídeos** (o script de coleta já faz isso);
> - o dataset do Mendeley cobre só 15 letras e **nenhuma** das com movimento —
>   elas dependem 100% da coleta da equipe;
> - confirme a lista com o curso de Letras-LIBRAS: há variação regional, e o **Ç**
>   (C com movimento) costuma entrar. Para incluí-lo basta acrescentá-lo em
>   `LETRAS_DINAMICAS` (`libras/config.py`) e coletar vídeos.

---

## Como funciona

```
imagem/quadro ──► MediaPipe Hand Landmarker ──► 21 pontos (x,y,z) da mão
                                                   │
                         normalização: punho na origem, escala da palma,
                         mão esquerda espelhada (canhotos)
                                                   │
          ┌────────── mão parada ──────────┐     ┌──────── mão em movimento ───────┐
          │ 114 atributos por quadro:      │     │ trecho reamostrado p/ 16 quadros │
          │ coordenadas, distâncias entre  │     │ forma da mão ao longo do tempo + │
          │ pontas, polegar×dedos, ângulos │     │ trajetória do punho/indicador/   │
          │ das articulações, orientação   │     │ mínimo + giro da palma           │
          └──────► modelo ESTÁTICO (20) ───┘     └──► modelo DINÂMICO (6 + "∅") ────┘
                                    │                      │
                         regra de decisão: confiança, margem entre 1ª e 2ª
                         hipóteses (mais exigente em A/S/E, M/N…), concordância
                         entre quadros  ──►  LETRA  ou  REPITA O SINAL
```

**Por que pontos da mão e não a imagem crua (CNN)?** É a resposta direta às armadilhas
do tema:

| Armadilha | Como o projeto trata |
|---|---|
| Fundo homogêneo vira atalho | O classificador nunca vê pixels de fundo, só os 21 pontos. |
| Mãos de tamanhos diferentes | Tudo é dividido pelo tamanho da palma. |
| Treinar e testar com a mesma pessoa | Validação **deixando um sinalizador de fora** e teste final com pessoa reservada (`--teste`). O relatório mostra quanto a divisão aleatória superestima. |
| Confusões A/S/E, M/N | Atributos específicos da posição do polegar em relação aos dedos; margem maior exigida dentro desses grupos (gera REPITA em vez de chute); tabela de confusão por grupo no relatório. |
| Letras com movimento | Detector de movimento + modelo de sequência com classe "∅" (troca de letra), treinado com transições sintéticas. |

Relação com as aulas: o classificador estático é classificação supervisionada com
métricas por classe (aulas 5–6); o modelo de movimento trabalha com sequências
reamostradas (aula 10). Uma CNN com transfer learning sobre o recorte da mão é uma boa
**comparação** para o relatório da equipe (ver "Extensões").

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.10+ (confira se existe wheel do `mediapipe` para a sua versão). Na primeira
execução o modelo `hand_landmarker.task` (~8 MB) é baixado automaticamente do Google.

## Passo a passo

### 1. Coletar dados locais (mínimo 150 imagens; várias pessoas)

```bash
python -m libras.coletar --sinalizador ana   --fotos 20 --repeticoes 8 --fundo "sala, parede branca"
python -m libras.coletar --sinalizador bruno --fotos 20 --repeticoes 8 --fundo "corredor, janela"
python -m libras.coletar --sinalizador carla --letras HJKXYZ --repeticoes 10
```
Cada pessoa fica em `dados/brutos/<sinalizador>/<LETRA>/`. Fotos para letras estáticas,
vídeos `.mp4` para as com movimento. Também dá para colocar fotos/vídeos de celular
direto nessas pastas.

Recomendações de protocolo:
- **pelo menos 4 sinalizadores** (3 para validação cruzada + 1 reservado para teste);
- fundos reais e variados, luz diferente, distâncias diferentes, mãos de tamanhos distintos;
- incluir pessoa canhota se possível;
- não espelhe fotos de celular (câmera frontal costuma espelhar — use a traseira ou desative);
- peça a alguém fluente (comunidade surda / Letras-LIBRAS) para conferir as configurações.

### 2. (Opcional) Importar datasets externos — entram **só no treino**

```bash
# Brazilian Sign Language Alphabet (Mendeley, Pascal VOC) — 15 letras
python -m libras.importar voc --origem ~/Downloads/libras_alfabeto --nome mendeley

# ASL Alphabet — só letras que a equipe conferiu como iguais na LIBRAS
python -m libras.importar pastas --origem ~/Downloads/asl_alphabet_train --nome asl \
       --somente ABCDILOUVW --max-por-letra 300
```
Não sabemos quem sinalizou cada imagem desses datasets, então eles nunca entram na
validação nem no teste. Cite as licenças CC BY 4.0 no trabalho. O MINDS-Libras (sinais
dinâmicos, não alfabeto) não é usado diretamente; pode servir para estudar trajetórias.

### 3. Extrair os pontos da mão

```bash
python -m libras.extrair
```
Incremental (só processa o que é novo). Mostra uma tabela sinalizador × letra e lista
arquivos onde nenhuma mão foi encontrada — revise-os.

### 4. Treinar e avaliar

```bash
python -m libras.treinar --teste carla
```
- compara SVM, Extra Trees, MLP e Regressão Logística com validação deixando uma pessoa de fora;
- escolhe pela acurácia balanceada, desempatando pela **pior letra**;
- calibra o limiar de "REPITA O SINAL" para ~95% de acerto nas respostas dadas (`--alvo`);
- avalia em `carla`, que nunca foi vista no treino;
- salva `modelos/estatico.joblib` e `modelos/dinamico.joblib`.

Relatório em `relatorios/relatorio.md` com: acurácia **por letra** (validação e teste),
matriz de confusão, desempenho por sinalizador, grupos confundíveis, comparação
aleatória × pessoa nova, curva de rejeição e o tratamento declarado das letras com movimento.
Há um exemplo (com dados sintéticos) em `docs/exemplo_relatorio_sintetico/`.

### 5. Usar

```bash
python -m libras.traduzir --camera 0                       # tempo real
python -m libras.traduzir --entrada a.jpg n.jpg a.jpg      # fotos, 1 letra cada
python -m libras.traduzir --entrada l.jpg j.mp4 quadros_z/ # mistura fotos e clipes
python -m libras.traduzir --video-continuo palavra.mp4     # palavra inteira em vídeo
```
Webcam: **ESPAÇO** insere espaço, **BACKSPACE** apaga, **C** limpa, **Q** sai.
Segure cada letra ~0,3 s. Para letra dobrada (ex.: "LL"), dê um pequeno quique com a mão
ou tire-a do quadro e volte.

## Critério de aceite → onde está

| Critério | Onde |
|---|---|
| Acurácia por letra (não só média) | `relatorio.md`, tabelas "Acurácia por letra"; CSVs `*_por_letra.csv` |
| Sinalizador que não apareceu no treino | `--teste <pessoa>` + seção "Sinalizador(es) nunca vistos"; validação leave-one-signer-out |
| Tratamento declarado das letras com movimento | Aviso acima; seção "Letras com movimento" do relatório; `libras/reconhecedor.py` |

## Estrutura

```
libras/
  config.py        letras estáticas/dinâmicas, grupos confundíveis, limiares
  detector.py      MediaPipe Hand Landmarker (API Tasks)
  features.py      normalização, atributos estáticos e de sequência, aumento de dados
  modelo.py        modelos candidatos, classe "∅" de transição, regra LETRA/REPITA
  metricas.py      métricas por letra/sinalizador/grupo, matriz de confusão, limiar
  coletar.py       coleta pela webcam
  importar.py      importa Pascal VOC (Mendeley) e pastas por classe (ASL)
  extrair.py       imagens/vídeos -> pontos da mão
  treinar.py       validação por sinalizador, teste, relatório
  reconhecedor.py  foto, clipe e fluxo contínuo (máquina de estados)
  traduzir.py      interface de linha de comando / webcam
  sintetico.py     dados sintéticos só para testar o pipeline
tests/test_pipeline.py
```

## Testar sem câmera

```bash
python -m pytest -q                       # ou: python tests/test_pipeline.py
python -m libras.sintetico                # gera mãos sintéticas
python -m libras.treinar --dados dados/processados/amostras_sinteticas.joblib --teste s5
```
Os números com dados sintéticos **não** dizem nada sobre LIBRAS real — as configurações
de mão são inventadas. Servem só para conferir que tudo roda de ponta a ponta.

## Ajustes finos (`libras/config.py`)

| Sintoma | Ajuste |
|---|---|
| Muitos "REPITA O SINAL" | baixe `LIMIAR_CONFIANCA`/`MARGEM_MINIMA` ou rode `treinar --alvo 0.90` |
| Troca de letra vira H/J/K… | suba `LIMIAR_MOVIMENTO`; colete vídeos com a mão chegando e saindo da letra |
| Letra demora a sair | baixe `QUADROS_PARA_FIXAR` |
| Mesma letra sai repetida | suba `LIMIAR_MUDANCA_POSE` |
| Câmera com muito tremor | suba `LIMIAR_MOVIMENTO` e `CONCORDANCIA_MINIMA` |

## Limitações conhecidas

- Depende do MediaPipe achar a mão: luz fraca, mão cortada ou desfocada viram REPITA.
- M/N e A/S/E diferem pela posição do polegar, que às vezes fica escondido; o sistema
  prefere pedir repetição a chutar.
- Não lê expressão facial nem segunda mão (não fazem parte da datilologia básica).
- As transições sintéticas aproximam a troca entre letras; vídeos reais de palavras
  soletradas, rotulados, melhorariam o modelo de movimento.

## Extensões sugeridas

- **Comparação com CNN (aulas 5–6):** recorte a mão pela caixa do MediaPipe, treine uma
  MobileNet/EfficientNet com transfer learning (ImageNet → ASL → LIBRAS) e compare por
  letra e por sinalizador com este modelo de pontos — inclusive trocando o fundo das
  imagens de teste para medir o atalho.
- **Sequências com atenção (aula 10):** troque o modelo dinâmico por uma GRU/Transformer
  pequena sobre os 16 quadros; o pré-processamento em `features.preencher_e_reamostrar`
  já entrega a sequência pronta.
- Corretor ortográfico/dicionário de nomes para sugerir a palavra completa.

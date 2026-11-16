"""Treina e avalia o tradutor. Toda avaliação é POR GRAVAÇÃO (uma foto ou um clipe
= uma resposta) e com PESSOA QUE NÃO ESTAVA NO TREINO.

    python -m libras.treinar --teste contrib_p02,contrib_p06
    python -m libras.treinar --teste contrib_p02,contrib_p06 --sem-extras   # sem Mendeley

1. Modelo ESTÁTICO (forma da mão, 20 letras). Treina com os quadros em que a
   letra já está formada (features.quadros_estaveis). Uma gravação é decidida
   pela média das probabilidades desses quadros.
2. Modelo de MOVIMENTO ("porteiro"): H, J, K, X, Y, Z ou ∅. O ∅ é aprendido com
   clipes REAIS de letras paradas (a mão se ajeitando não é letra com movimento)
   e com trocas de letra montadas a partir de poses reais (para a webcam).
   Candidatos: logística, MLP e um Transformer com atenção (libras/sequencia.py).
3. CLIPE (qualquer das 26 letras) = os dois modelos juntos (modelo.probs_clipe).

Protocolo:
  * --teste: pessoas separadas ANTES de tudo; só aparecem na avaliação final.
  * Validação deixando UMA PESSOA DE FORA (as demais pessoas locais). Ela escolhe
    os modelos e calibra o limiar de "REPITA O SINAL".
  * Datasets externos (dados/extras) entram só no treino; nunca são avaliados.
  * Armadilha "mesma pessoa": a mesma validação com divisão aleatória por gravação.
Relatório: relatorios/relatorio.md (+ CSVs e matrizes de confusão).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from . import config
from . import metricas as M
from .extrair import carregar_banco
from .features import aumentar_estatico, features_estaticas, quadros_estaveis
from .modelo import (TRANSICAO, ModeloLetras, candidatos, candidatos_movimento, matriz_dinamica,
                     probs_clipe, sintetizar_transicoes)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
LETRAS = config.LETRAS_ESTATICAS + config.LETRAS_DINAMICAS


# ---------------------------------------------------------------------------
# Gravações
# ---------------------------------------------------------------------------
def _espacados(n: int, k: int = 10) -> list[int]:
    return list(range(n)) if n <= k else [int(round(i)) for i in np.linspace(0, n - 1, k)]


def montar_gravacoes(banco):
    """Gravações de letras estáticas (foto ou clipe -> quadros estáveis) e clipes (todas as letras)."""
    estaticas, clipes = [], []
    for f in banco["fotos"]:
        if f["letra"] in config.LETRAS_ESTATICAS:
            estaticas.append({**{k: f[k] for k in ("letra", "sinalizador", "origem", "arquivo", "mao")},
                              "tipo": "foto", "quadros": np.asarray(f["pts"], np.float32)[None]})
    for c in banco["clipes"]:
        clipes.append(c)
        if c["letra"] in config.LETRAS_ESTATICAS:
            idx = quadros_estaveis(c["seq"], c["mao"])
            if idx:
                estaticas.append({**{k: c[k] for k in ("letra", "sinalizador", "origem", "arquivo", "mao")},
                                  "tipo": "clipe", "quadros": c["seq"][idx]})
    return estaticas, clipes


def _dobras(grupos_idx: dict, extras: list):
    """leave-one-signer-out: [(nome, treino, validação)]"""
    return [(s, [i for t, idx in grupos_idx.items() if t != s for i in idx] + extras, idx)
            for s, idx in grupos_idx.items()]


def _dobras_aleatorias(locais, y, extras, seed, k=5):
    yl = y[locais]
    k = max(2, min(k, int(np.unique(yl, return_counts=True)[1].min())))
    il = np.array(locais)
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    return [(f"aleatória {j}", list(il[a]) + extras, list(il[b])) for j, (a, b) in enumerate(skf.split(il, yl))]


def _alinhar(P, classes_est, classes):
    out = np.zeros((len(P), len(classes)))
    for j, c in enumerate(classes_est):
        out[:, classes.index(c)] = P[:, j]
    return out


# ---------------------------------------------------------------------------
# 1. Modelo estático
# ---------------------------------------------------------------------------
class Estatico:
    """Atributos calculados uma vez: por gravação, quadros estáveis (para avaliar)
    e até 10 quadros espaçados x (1 + aumentos) (para treinar)."""

    def __init__(self, gravs, aumentos, seed):
        rng = np.random.default_rng(seed)
        self.g = gravs
        self.y = np.array([g["letra"] for g in gravs])
        self.classes = sorted(set(self.y))
        self.aval, self.treino = [], []
        for g in gravs:
            qs = g["quadros"]
            self.aval.append(np.stack([features_estaticas(q, g["mao"]) for q in qs]))
            vs = [v for k in _espacados(len(qs)) for v in
                  [qs[k]] + [aumentar_estatico(qs[k], rng) for _ in range(aumentos)]]
            self.treino.append(np.stack([features_estaticas(v, g["mao"]) for v in vs]))

    def ajustar(self, nome, idx, seed):
        X = np.concatenate([self.treino[i] for i in idx])
        Y = np.concatenate([[self.y[i]] * len(self.treino[i]) for i in idx])
        return clone(candidatos(seed)[nome]).fit(X, Y)

    def probs(self, est, idx):
        return np.array([_alinhar(est.predict_proba(self.aval[i]), list(est.classes_), self.classes).mean(0)
                         for i in idx])


def treinar_estatico(gravs, teste, args, rel):
    print(f"\n=== LETRAS ESTÁTICAS: {len(gravs)} gravações ===", flush=True)
    E = Estatico(gravs, args.aumentos, args.seed)
    g = np.array([x["sinalizador"] for x in gravs])
    o = np.array([x["origem"] for x in gravs])
    extras = [] if args.sem_extras else [i for i in range(len(gravs)) if o[i] == "extra"]
    locais = [i for i in range(len(gravs)) if o[i] == "local" and g[i] not in teste]
    idx_teste = [i for i in range(len(gravs)) if g[i] in teste]
    por_pessoa = {s: [i for i in locais if g[i] == s] for s in sorted(set(g[locais]))}
    sec = rel["estatico"] = {"classes": E.classes, "locais": list(por_pessoa)}
    if len(por_pessoa) >= 2:
        dobras = _dobras(por_pessoa, extras)
        sec["validacao"] = f"deixando uma pessoa de fora ({len(por_pessoa)} pessoas locais)"
    else:
        print("  AVISO: menos de 2 pessoas locais fora do teste -> validação aleatória (otimista)")
        dobras = _dobras_aleatorias(locais, E.y, extras, args.seed)
        sec["validacao"] = "ALEATÓRIA (otimista — colete mais pessoas!)"

    resultados = {}
    for nome in args.modelos.split(","):
        t0 = time.time()
        oof, modelos = np.zeros((len(gravs), len(E.classes))), {}
        for s, tr, va in dobras:
            modelos[s] = E.ajustar(nome, tr, args.seed)
            oof[va] = E.probs(modelos[s], va)
        idx = sorted(i for _, _, va in dobras for i in va)
        pred = np.array(E.classes)[oof[idx].argmax(1)]
        tab = M.tabela_por_letra(E.y[idx], pred, E.classes)
        resultados[nome] = {"oof": oof, "idx": idx, "pred": pred, "bal": M.acuracia_balanceada(E.y[idx], pred),
                            "min": float(tab.acuracia_letra.min()), "modelos": modelos}
        print(f"  {nome:12s} acurácia balanceada {resultados[nome]['bal']:.1%} | pior letra "
              f"{resultados[nome]['min']:.1%}  ({time.time() - t0:.0f}s)", flush=True)
    melhor = _escolher(resultados)
    r = resultados[melhor]
    print(f"  -> escolhido: {melhor}")
    idx, pred = r["idx"], r["pred"]
    limiar, curva = M.calibrar_limiar(E.y[idx], r["oof"][idx], E.classes, config.MARGEM_MINIMA, args.alvo)
    tab = M.tabela_por_letra(E.y[idx], pred, E.classes)
    tab.to_csv(config.DIR_RELATORIOS / "estatico_validacao_por_letra.csv", index=False)
    M.matriz_png(E.y[idx], pred, E.classes, f"Letras estáticas — validação ({sec['validacao']})",
                 config.DIR_RELATORIOS / "estatico_matriz_validacao.png")
    sec.update({"comparacao": {k: (v["bal"], v["min"]) for k, v in resultados.items()}, "modelo": melhor,
                "tabela_validacao": tab, "limiar": limiar, "curva": curva,
                "por_sinalizador": M.por_sinalizador(E.y[idx], pred, g[idx]),
                "por_tipo": _por_tipo(E.y[idx], pred, [gravs[i]["tipo"] for i in idx]),
                "grupos": M.analise_grupos(E.y[idx], pred)})

    if len(por_pessoa) >= 2:   # armadilha: mesma pessoa no treino e na validação
        oof = np.zeros_like(r["oof"])
        for _, tr, va in _dobras_aleatorias(locais, E.y, extras, args.seed):
            oof[va] = E.probs(E.ajustar(melhor, tr, args.seed), va)
        bal_r = M.acuracia_balanceada(E.y[locais], np.array(E.classes)[oof[locais].argmax(1)])
        sec["aleatoria_vs_pessoa"] = (bal_r, r["bal"])
        print(f"  divisão aleatória {bal_r:.1%} vs pessoa nova {r['bal']:.1%}")
        if extras:   # quanto o dataset externo ajuda
            oof = np.zeros_like(r["oof"])
            for _, tr, va in _dobras(por_pessoa, []):
                oof[va] = E.probs(E.ajustar(melhor, tr, args.seed), va)
            bal_s = M.acuracia_balanceada(E.y[idx], np.array(E.classes)[oof[idx].argmax(1)])
            sec["com_sem_extras"] = (r["bal"], bal_s)
            print(f"  com datasets externos {r['bal']:.1%} vs sem {bal_s:.1%}")

    est = E.ajustar(melhor, locais + extras, args.seed)
    modelo = ModeloLetras("estatico", est, list(est.classes_), limiar=limiar["limiar"],
                          margem=config.MARGEM_MINIMA,
                          info={"modelo": melhor, "treinado_em": datetime.now().isoformat(),
                                "sinalizadores_treino": sorted(set(g[locais + extras])),
                                "sinalizadores_teste": sorted(teste)})
    sec["faltando"] = [l for l in config.LETRAS_ESTATICAS if l not in modelo.classes]
    if idx_teste:
        P = E.probs(est, idx_teste)
        yt, pt = E.y[idx_teste], np.array(E.classes)[P.argmax(1)]
        tab_t = M.tabela_por_letra(yt, pt, E.classes)
        tab_t.to_csv(config.DIR_RELATORIOS / "estatico_teste_por_letra.csv", index=False)
        M.matriz_png(yt, pt, E.classes, f"Letras estáticas — pessoa(s) nunca vista(s): {', '.join(sorted(teste))}",
                     config.DIR_RELATORIOS / "estatico_matriz_teste.png")
        sec.update({"tabela_teste": tab_t, "bal_teste": M.acuracia_balanceada(yt, pt),
                    "acc_teste": float((yt == pt).mean()), "grupos_teste": M.analise_grupos(yt, pt),
                    "por_sinalizador_teste": M.por_sinalizador(yt, pt, g[idx_teste]),
                    "rejeicao_teste": M.efeito_rejeicao(yt, P, E.classes, modelo.limiar, modelo.margem)})
        print(f"  TESTE (pessoa nunca vista): acurácia balanceada {sec['bal_teste']:.1%}", flush=True)
    # modelos por dobra (sem a pessoa validada): reaproveitados na avaliação dos clipes
    return modelo, r["modelos"], E


def _escolher(resultados):
    """Acurácia balanceada; entre os que estão a até 1 ponto do melhor, fica o de PIOR LETRA mais alta."""
    top = max(v["bal"] for v in resultados.values())
    return max((k for k, v in resultados.items() if v["bal"] >= top - 0.01),
               key=lambda k: (resultados[k]["min"], resultados[k]["bal"]))


def _por_tipo(y, pred, tipos):
    t = np.array(tipos)
    return {k: M.acuracia_balanceada(y[t == k], pred[t == k]) for k in sorted(set(tipos))}


# ---------------------------------------------------------------------------
# 2 e 3. Modelo de movimento + clipes (26 letras)
# ---------------------------------------------------------------------------
def _amostras_movimento(clipes, idx, estaticas_treino, n_trans, seed):
    """Letras com movimento + ∅ (clipes reais de letras paradas + trocas de letra)."""
    am = []
    for i in idx:
        c = dict(clipes[i])
        if c["letra"] in config.LETRAS_ESTATICAS:
            c["letra"] = TRANSICAO
        am.append(c)
    poses = [{"letra": g["letra"], "sinalizador": g["sinalizador"], "origem": g["origem"],
              "mao": g["mao"], "pts": g["quadros"][-1]} for g in estaticas_treino]
    am += sintetizar_transicoes(poses, n_trans, seed)
    contexto = {}
    for p in poses:
        contexto.setdefault(p["sinalizador"], []).append(p["pts"])
    return am, contexto


def _ajustar_movimento(nome, clipes, idx, estaticas_treino, args):
    est, atrib = candidatos_movimento(args.seed)[nome]
    am, ctx = _amostras_movimento(clipes, idx, estaticas_treino, args.transicoes, args.seed)
    X, y, _ = matriz_dinamica(am, args.aumentos, args.seed, ctx, atrib)
    return clone(est).fit(X, y), atrib


def _probs_clipes(clipes, idx, est_estatico, classes_est, mov, atrib):
    """(n, 26) probabilidades combinadas, na ordem de LETRAS."""
    from .modelo import vetor_sequencia
    out = np.zeros((len(idx), len(LETRAS)))
    for k, i in enumerate(idx):
        c = clipes[i]
        q = quadros_estaveis(c["seq"], c["mao"])
        Pe = est_estatico.predict_proba(np.stack([features_estaticas(c["seq"][j], c["mao"]) for j in q]))
        p_est = dict(zip(est_estatico.classes_, Pe.mean(0)))
        p_est.update({l: 0.0 for l in classes_est if l not in p_est})
        v = vetor_sequencia(c["seq"], c["mao"], atrib)
        p_mov = dict(zip(mov.classes_, mov.predict_proba(v[None])[0])) if v is not None else None
        comb = probs_clipe(p_est, p_mov)
        out[k] = [comb.get(l, 0.0) for l in LETRAS]
    return out


def treinar_clipes(clipes, teste, est_final, est_dobras, E, args, rel):
    g = np.array([c["sinalizador"] for c in clipes])
    y = np.array([c["letra"] for c in clipes])
    locais = [i for i, c in enumerate(clipes) if c["origem"] == "local" and g[i] not in teste]
    idx_teste = [i for i, c in enumerate(clipes) if g[i] in teste]
    if not any(y[i] in config.LETRAS_DINAMICAS for i in locais):
        print("\nAVISO: nenhum clipe de H/J/K/X/Y/Z — o tradutor só reconhecerá letras estáticas.")
        return None
    print(f"\n=== CLIPES (26 letras): {len(locais)} clipes locais, {len(idx_teste)} de teste ===", flush=True)
    por_pessoa = {s: [i for i in locais if g[i] == s] for s in sorted(set(g[locais]))}
    sec = rel["clipes"] = {"locais": list(por_pessoa)}
    eg = np.array([x["sinalizador"] for x in E.g])
    eo = np.array([x["origem"] for x in E.g])

    def estaticas_de(excluir):
        return [x for x, s, o in zip(E.g, eg, eo) if s not in teste and s != excluir and
                (o == "local" or not args.sem_extras)]

    validos = [s for s in por_pessoa if s in est_dobras]
    if len(validos) >= 2:
        sec["validacao"] = f"deixando uma pessoa de fora ({len(validos)} pessoas com clipes)"
    else:
        sec["validacao"] = "sem validação por pessoa (menos de 2 pessoas com clipes)"
    resultados = {}
    for nome in args.modelos_movimento.split(","):
        t0 = time.time()
        oof = np.zeros((len(clipes), len(LETRAS)))
        for s in validos:
            tr = [i for i in locais if g[i] != s]
            mov, atrib = _ajustar_movimento(nome, clipes, tr, estaticas_de(s), args)
            oof[por_pessoa[s]] = _probs_clipes(clipes, por_pessoa[s], est_dobras[s], E.classes, mov, atrib)
        idx = sorted(i for s in validos for i in por_pessoa[s])
        if not idx:
            break
        pred = np.array(LETRAS)[oof[idx].argmax(1)]
        tab = M.tabela_por_letra(y[idx], pred, LETRAS)
        din = [k for k, i in enumerate(idx) if y[i] in config.LETRAS_DINAMICAS]
        resultados[nome] = {"oof": oof, "idx": idx, "pred": pred, "bal": M.acuracia_balanceada(y[idx], pred),
                            "min": float(tab.acuracia_letra.min()),
                            "bal_mov": M.acuracia_balanceada(y[idx][din], pred[din])}
        print(f"  {nome:12s} 26 letras {resultados[nome]['bal']:.1%} | só movimento "
              f"{resultados[nome]['bal_mov']:.1%} ({time.time() - t0:.0f}s)", flush=True)
    if resultados:
        melhor = _escolher(resultados)
        r = resultados[melhor]
        idx, pred = r["idx"], r["pred"]
        limiar, curva = M.calibrar_limiar(y[idx], r["oof"][idx], LETRAS, config.MARGEM_MINIMA, args.alvo)
        tab = M.tabela_por_letra(y[idx], pred, LETRAS)
        tab.to_csv(config.DIR_RELATORIOS / "clipes_validacao_por_letra.csv", index=False)
        M.matriz_png(y[idx], pred, LETRAS, f"Clipes, 26 letras — validação ({sec['validacao']})",
                     config.DIR_RELATORIOS / "clipes_matriz_validacao.png")
        sec.update({"comparacao": {k: (v["bal"], v["min"], v["bal_mov"]) for k, v in resultados.items()},
                    "modelo": melhor, "tabela_validacao": tab, "limiar": limiar,
                    "por_sinalizador": M.por_sinalizador(y[idx], pred, g[idx])})
        # armadilha da mesma pessoa, para os clipes
        oof = np.zeros((len(clipes), len(LETRAS)))
        el = [i for i in range(len(E.g)) if eg[i] not in teste and eo[i] == "local"]
        ext = [] if args.sem_extras else [i for i in range(len(E.g)) if eo[i] == "extra"]
        for _, tr, va in _dobras_aleatorias(locais, y, [], args.seed):
            arqs_va = {clipes[i]["arquivo"] for i in va}
            est = E.ajustar(rel["estatico"]["modelo"], [i for i in el if E.g[i]["arquivo"] not in arqs_va] + ext,
                            args.seed)
            mov, atrib = _ajustar_movimento(melhor, clipes, tr, [E.g[i] for i in el + ext
                                                                if E.g[i]["arquivo"] not in arqs_va], args)
            oof[va] = _probs_clipes(clipes, va, est, E.classes, mov, atrib)
        bal_r = M.acuracia_balanceada(y[locais], np.array(LETRAS)[oof[locais].argmax(1)])
        sec["aleatoria_vs_pessoa"] = (bal_r, r["bal"])
        print(f"  -> escolhido: {melhor} | divisão aleatória {bal_r:.1%} vs pessoa nova {r['bal']:.1%}")
    else:
        melhor, limiar = args.modelos_movimento.split(",")[0], {"limiar": config.LIMIAR_CONFIANCA}
        sec.update({"modelo": melhor})

    mov, atrib = _ajustar_movimento(melhor, clipes, locais, estaticas_de(None), args)
    modelo = ModeloLetras("dinamico", mov, list(mov.classes_), limiar=limiar["limiar"],
                          margem=config.MARGEM_MINIMA, atributos=atrib,
                          info={"modelo": melhor, "limiar_clipe": limiar["limiar"],
                                "treinado_em": datetime.now().isoformat(),
                                "sinalizadores_treino": sorted(set(g[locais])),
                                "sinalizadores_teste": sorted(teste)})
    sec["faltando"] = [l for l in config.LETRAS_DINAMICAS if l not in modelo.classes]
    if idx_teste:
        P = _probs_clipes(clipes, idx_teste, est_final.estimador, E.classes, mov, atrib)
        yt, pt = y[idx_teste], np.array(LETRAS)[P.argmax(1)]
        tab_t = M.tabela_por_letra(yt, pt, LETRAS)
        tab_t.to_csv(config.DIR_RELATORIOS / "clipes_teste_por_letra.csv", index=False)
        M.matriz_png(yt, pt, LETRAS, f"Clipes, 26 letras — pessoa(s) nunca vista(s): {', '.join(sorted(teste))}",
                     config.DIR_RELATORIOS / "clipes_matriz_teste.png")
        din = np.isin(yt, config.LETRAS_DINAMICAS)
        sec.update({"tabela_teste": tab_t, "bal_teste": M.acuracia_balanceada(yt, pt),
                    "bal_teste_mov": M.acuracia_balanceada(yt[din], pt[din]),
                    "bal_teste_est": M.acuracia_balanceada(yt[~din], pt[~din]),
                    "rejeicao_teste": M.efeito_rejeicao(yt, P, LETRAS, limiar["limiar"], config.MARGEM_MINIMA)})
        print(f"  TESTE (pessoa nunca vista): 26 letras {sec['bal_teste']:.1%} "
              f"(paradas {sec['bal_teste_est']:.1%}, movimento {sec['bal_teste_mov']:.1%})", flush=True)
    return modelo


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------
def _secao_dados(banco, estaticas, clipes, teste, args):
    L = ["## Dados", ""]
    cont = Counter()
    for f in banco["fotos"]:
        cont[(f["sinalizador"], "fotos")] += 1
    for c in banco["clipes"]:
        cont[(c["sinalizador"], "clipes")] += 1
        if c["letra"] in config.LETRAS_DINAMICAS:
            cont[(c["sinalizador"], "mov")] += 1
    pessoas = sorted({s for s, _ in cont})
    por_letras = {s: sorted({x["letra"] for x in banco["fotos"] + banco["clipes"] if x["sinalizador"] == s})
                  for s in pessoas}
    L += ["| fonte / pessoa | papel | fotos | clipes | clipes com movimento | letras |", "|---|---|---|---|---|---|"]
    for s in pessoas:
        papel = ("teste (nunca no treino)" if s in teste else
                 "só treino" if s.startswith("extra:") else "treino + validação")
        if s.startswith("extra:") and args.sem_extras:
            papel = "não usado (--sem-extras)"
        L.append(f"| {s} | {papel} | {cont[(s, 'fotos')]} | {cont[(s, 'clipes')]} | {cont[(s, 'mov')]} | "
                 f"{len(por_letras[s])} |")
    n_desc = Counter(m for _, m in banco["falhas"])
    L += ["", f"Arquivos sem mão detectada (descartados pelo extrator): {sum(n_desc.values())} "
              f"({', '.join(f'{v} {k}' for k, v in n_desc.items())}).", ""]
    csv_c = config.DIR_CONTRIBUIDO / "gravacoes.csv"
    if csv_c.exists():
        with open(csv_c, encoding="utf-8") as f:
            linhas = list(csv.DictReader(f))
        como = Counter(x["como"] for x in linhas)
        L += ["**Dataset da equipe** (`contributed_images.zip`, gravado pela equipe): "
              f"{len(linhas)} gravações — clipes de até 16 quadros (`f0000`…`f0015`) e fotos avulsas. "
              "Os arquivos não identificam a pessoa, então ela foi descoberta pelo **rosto** "
              "(detector YuNet + identidade SFace, OpenCV Zoo; `libras/contribuido.py`): "
              f"{como.get('rosto', 0)} gravações pelo rosto, {como.get('cenário', 0)} pelo cenário "
              f"(mesmo quarto e luz) e {como.get('sem_grupo', 0)} sem pessoa identificada, **descartadas** "
              "para não vazar a mão de quem está no teste. A conferência visual está em "
              "`dados/contribuido/pessoas.jpg`. Alguns grupos juntam mais de uma pessoa do mesmo "
              "quarto: isso só deixa a avaliação mais rígida (nunca a mesma pessoa em dois grupos).", ""]
    L += ["**Brazilian Sign Language Alphabet** (Passos, Fernandes e Comunello, Mendeley Data, "
          "DOI 10.17632/k4gs3bmx5k.5, licença CC BY 4.0): 15 letras (A B C D E I L M N O R S U V W), "
          "imagens 200×200 com fundo liso, derivadas do ASL Alphabet. Usado **só no treino**: não se "
          "sabe quem sinalizou, então nunca entra na avaliação.", "",
          "Não usados: **ASL Alphabet** (é ASL; o Mendeley já é o recorte dele com as 15 letras iguais "
          "em LIBRAS) e **MINDS-Libras** (sinais, não o alfabeto).", "",
          "**Unidade de avaliação:** a gravação. Uma foto ou um clipe inteiro recebe UMA resposta; "
          "os quadros de um mesmo clipe nunca são separados entre treino e validação.", ""]
    return L


def escrever_relatorio(rel, args, banco, estaticas, clipes, teste):
    L = ["# Relatório — tradutor de datilologia em LIBRAS", "",
         f"Gerado em {datetime.now():%d/%m/%Y %H:%M}. Pessoa(s) de teste: "
         f"**{', '.join(sorted(teste)) or 'nenhuma (use --teste)'}**. Aumentos por quadro: {args.aumentos}.", ""]
    L += _secao_dados(banco, estaticas, clipes, teste, args)
    s = rel.get("estatico")
    if s and "modelo" in s:
        L += ["## Letras estáticas (forma da mão) — por gravação", "",
              f"Validação: {s['validacao']}. Pessoas: {', '.join(s['locais'])}.", "",
              "Cada foto ou clipe é decidido pela média das probabilidades dos quadros em que a letra "
              "já está formada; os quadros iniciais, com a mão ainda chegando na letra, são ignorados "
              "no treino e na decisão.", "",
              "**Comparação de modelos (acurácia balanceada | pior letra):**", ""]
        L += [f"- {k}: {b:.1%} | {m:.1%}" + ("  ← escolhido" if k == s["modelo"] else "")
              for k, (b, m) in s["comparacao"].items()]
        if "aleatoria_vs_pessoa" in s:
            a, p = s["aleatoria_vs_pessoa"]
            L += ["", f"**Armadilha da mesma pessoa:** divisão aleatória {a:.1%} vs pessoa nova {p:.1%} "
                      f"(diferença {a - p:+.1%})."]
        if "com_sem_extras" in s:
            c, sm = s["com_sem_extras"]
            L += ["", f"**Dataset externo (Mendeley) ajuda?** com {c:.1%} vs sem {sm:.1%} "
                      f"(diferença {c - sm:+.1%}) na validação por pessoa."]
        li = s["limiar"]
        L += ["", f"**Limiar de REPITA O SINAL:** {li['limiar']:.2f} → responde em {li['cobertura']:.1%} "
                  f"das gravações, acertando {li['acuracia_aceitas']:.1%} delas (alvo {args.alvo:.0%}).", "",
              "Acurácia balanceada por tipo de entrada: " +
              ", ".join(f"{k} {v:.1%}" for k, v in s["por_tipo"].items()) + ".", "",
              "### Acurácia por letra — validação (pessoa fora do treino)", "",
              M.tabela_md(s["tabela_validacao"]), "", "![matriz](estatico_matriz_validacao.png)", "",
              "### Por pessoa (cada uma avaliada por um modelo que não a viu)", "",
              M.tabela_md(s["por_sinalizador"]), "", "### Grupos confundíveis", ""] + s["grupos"]
        if "tabela_teste" in s:
            rj = s["rejeicao_teste"]
            L += [f"### Pessoa(s) nunca vista(s): {', '.join(sorted(teste))}", "",
                  f"Acurácia {s['acc_teste']:.1%} | balanceada {s['bal_teste']:.1%} | com rejeição: "
                  f"responde {rj['cobertura']:.1%}, acerta {rj['acuracia_aceitas']:.1%}.", "",
                  M.tabela_md(s["tabela_teste"]), "", "![matriz](estatico_matriz_teste.png)", ""]
            L += s["grupos_teste"]
        if s.get("faltando"):
            L += [f"> **Sem dados para:** {', '.join(s['faltando'])}.", ""]
    c = rel.get("clipes")
    L += ["## Letras com movimento e clipes (26 letras)", "",
          "**Tratamento das letras com movimento.** " + ", ".join(config.LETRAS_DINAMICAS) +
          " não são reconhecidas por foto: precisam de uma sequência de quadros (clipe ou webcam). "
          f"O clipe é reamostrado para {config.QUADROS_SEQUENCIA} quadros e passa por um modelo de "
          "movimento que responde H/J/K/X/Y/Z ou ∅ (\"não é letra com movimento\"). O ∅ é aprendido com "
          "clipes REAIS de letras paradas da equipe — a mão se ajeitando no começo do clipe não é "
          "movimento de letra — e com trocas entre duas poses reais da mesma pessoa (para a webcam não "
          "ler a troca de letra como H/J/K/X/Y/Z). A resposta final junta os dois modelos: "
          "P(letra parada) = P(∅) × P_estático(letra) e P(letra com movimento) = P_movimento(letra).", ""]
    if c and "comparacao" in c:
        L += [f"Validação: {c['validacao']}.", "",
              "**Modelo de movimento (acurácia balanceada nas 26 letras | pior letra | só H/J/K/X/Y/Z):**", ""]
        L += [f"- {k}: {b:.1%} | {m:.1%} | {d:.1%}" + ("  ← escolhido" if k == c["modelo"] else "")
              for k, (b, m, d) in c["comparacao"].items()]
        L += ["", "`atencao` = Transformer de 2 camadas com pooling por atenção sobre os quadros "
                  "(`libras/sequencia.py`); `logistica`/`mlp` usam um vetor-resumo do movimento."]
        if "aleatoria_vs_pessoa" in c:
            a, p = c["aleatoria_vs_pessoa"]
            L += ["", f"**Armadilha da mesma pessoa:** divisão aleatória {a:.1%} vs pessoa nova {p:.1%} "
                      f"(diferença {a - p:+.1%})."]
        li = c["limiar"]
        L += ["", f"**Limiar de REPITA O SINAL (clipes):** {li['limiar']:.2f} → responde em "
                  f"{li['cobertura']:.1%} dos clipes, acertando {li['acuracia_aceitas']:.1%} deles.", "",
              "### Acurácia por letra — validação (pessoa fora do treino)", "",
              M.tabela_md(c["tabela_validacao"]), "", "![matriz](clipes_matriz_validacao.png)", "",
              "### Por pessoa", "", M.tabela_md(c["por_sinalizador"]), ""]
    if c and "tabela_teste" in c:
        rj = c["rejeicao_teste"]
        L += [f"### Pessoa(s) nunca vista(s): {', '.join(sorted(teste))}", "",
              f"Balanceada nas 26 letras {c['bal_teste']:.1%} (paradas {c['bal_teste_est']:.1%}, com movimento "
              f"{c['bal_teste_mov']:.1%}) | com rejeição: responde {rj['cobertura']:.1%}, acerta "
              f"{rj['acuracia_aceitas']:.1%}.", "", M.tabela_md(c["tabela_teste"]), "",
              "![matriz](clipes_matriz_teste.png)", ""]
    if c and c.get("faltando"):
        L += [f"> **Sem dados para:** {', '.join(c['faltando'])}.", ""]
    cnn = config.DIR_RELATORIOS / "comparacao_cnn.md"
    if cnn.exists():
        L += ["## Comparação com CNN (transfer learning)", "",
              f"Ver [comparacao_cnn.md](comparacao_cnn.md) (gerado por `python -m libras.comparar_cnn`).", ""]
    (config.DIR_RELATORIOS / "relatorio.md").write_text("\n".join(L), "utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teste", default="", help="pessoa(s) reservada(s) para o teste final, separadas por vírgula")
    ap.add_argument("--aumentos", type=int, default=3, help="cópias aumentadas por quadro/clipe de treino")
    ap.add_argument("--alvo", type=float, default=0.95, help="acerto desejado nas respostas aceitas")
    ap.add_argument("--modelos", default="mlp,logistica,extra_trees", help="candidatos do modelo estático")
    ap.add_argument("--modelos-movimento", default="logistica,mlp,atencao")
    ap.add_argument("--transicoes", type=int, default=40, help="trocas de letra montadas por pessoa")
    ap.add_argument("--sem-extras", action="store_true", help="não usa dados/extras (ex.: Mendeley)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dados", default=str(config.DIR_PROCESSADOS / "amostras.joblib"))
    args = ap.parse_args()

    banco = carregar_banco(args.dados)
    teste = {t.strip() for t in args.teste.split(",") if t.strip()}
    conhecidos = {s["sinalizador"] for s in banco["fotos"] + banco["clipes"]}
    if teste - conhecidos:
        raise SystemExit(f"Pessoa(s) de teste desconhecida(s): {teste - conhecidos}")
    if not teste:
        print("AVISO: sem --teste, não haverá avaliação com pessoa nunca vista (critério de aceite).")
    config.DIR_RELATORIOS.mkdir(parents=True, exist_ok=True)
    config.DIR_MODELOS.mkdir(parents=True, exist_ok=True)

    estaticas, clipes = montar_gravacoes(banco)
    rel = {}
    est, est_dobras, E = treinar_estatico(estaticas, teste, args, rel)
    est.salvar(config.DIR_MODELOS / "estatico.joblib")
    din = treinar_clipes(clipes, teste, est, est_dobras, E, args, rel)
    if din is not None:
        din.salvar(config.DIR_MODELOS / "dinamico.joblib")
    escrever_relatorio(rel, args, banco, estaticas, clipes, teste)
    resumo = {t: {k: v for k, v in s.items() if k in ("modelo", "validacao", "bal_teste", "acc_teste",
                                                        "bal_teste_mov", "bal_teste_est", "faltando")}
              for t, s in rel.items()}
    (config.DIR_RELATORIOS / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nRelatório: {config.DIR_RELATORIOS / 'relatorio.md'}")


if __name__ == "__main__":
    main()

"""Treina e avalia os dois modelos (letras estáticas e letras com movimento).

    python -m libras.treinar --teste carla
    python -m libras.treinar --teste carla,diego --aumentos 6 --alvo 0.95

O que acontece:
 1. Sinalizadores em --teste são separados ANTES de tudo e só aparecem na
    avaliação final ("sinalizador que não apareceu no treino").
 2. Com os demais sinalizadores locais é feita validação cruzada deixando UMA
    PESSOA DE FORA por vez (leave-one-signer-out). É ela que escolhe o modelo
    (SVM, Extra Trees, MLP, Regressão Logística) e calibra o limiar de
    "REPITA O SINAL". Datasets externos (dados/extras) entram sempre no treino
    e nunca na validação.
 3. Para mostrar a armadilha "treinar e testar com a mesma pessoa", o mesmo
    modelo é avaliado também com divisão aleatória; a diferença aparece no
    relatório.
 4. O modelo final é treinado com tudo menos --teste e salvo em modelos/.
Relatório: relatorios/relatorio.md (+ CSVs e matrizes de confusão em PNG).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from . import config
from . import metricas as M
from .modelo import (TRANSICAO, ModeloLetras, candidatos, matriz_dinamica, matriz_estatica,
                     sintetizar_transicoes)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")


def _features_por_amostra(amostras, tipo, aumentos, seed, contexto=None):
    """Para cada amostra: (features original, features aumentadas). Calculado uma
    vez e reaproveitado por todas as dobras e modelos."""
    if tipo == "estatico":
        fn = matriz_estatica
    else:
        def fn(a, n, sd):
            return matriz_dinamica(a, n, sd, contexto)
    base, aum = [], []
    for i, s in enumerate(amostras):
        Xb, _, _ = fn([s], 0, seed + i)
        Xa, _, _ = fn([s], aumentos, seed + i)
        base.append(Xb[0] if len(Xb) else None)
        aum.append(Xa if len(Xa) else None)
    return base, aum


def _juntar(idx, base, aum, y, usar_aum):
    X, Y = [], []
    for i in idx:
        bloco = aum[i] if usar_aum else (None if base[i] is None else base[i][None])
        if bloco is None:
            continue
        X.append(bloco)
        Y += [y[i]] * len(bloco)
    return np.concatenate(X), np.array(Y)


def _probs_alinhadas(est, X, classes):
    p = est.predict_proba(X)
    out = np.zeros((len(X), len(classes)))
    for j, c in enumerate(est.classes_):
        out[:, classes.index(c)] = p[:, j]
    return out


def treinar_tipo(tipo, amostras, teste, args, rel, contexto=None):
    nome = "LETRAS ESTÁTICAS" if tipo == "estatico" else "LETRAS COM MOVIMENTO"
    print(f"\n=== {nome}: {len(amostras)} amostras ===")
    ok_feat_base, ok_feat_aum = _features_por_amostra(amostras, tipo, args.aumentos, args.seed,
                                                             contexto)
    validas = [i for i, b in enumerate(ok_feat_base) if b is not None]
    y = np.array([s["letra"] for s in amostras])
    g = np.array([s["sinalizador"] for s in amostras])
    origem = np.array([s["origem"] for s in amostras])
    classes = sorted(set(y[validas]))

    idx_teste = [i for i in validas if g[i] in teste]
    idx_extra = [i for i in validas if origem[i] == "extra"]
    idx_local = [i for i in validas if origem[i] == "local" and g[i] not in teste]
    locais = sorted(set(g[idx_local]))
    sec = rel.setdefault(tipo, {"classes": classes, "locais": locais})
    if not idx_local and not idx_extra:
        print("  sem dados — pulando")
        return None

    # ---------------- validação: uma pessoa de fora por vez ----------------
    if len(locais) >= 2:
        dobras = [([i for i in idx_local if g[i] != s] + idx_extra,
                   [i for i in idx_local if g[i] == s]) for s in locais]
        sec["validacao"] = f"leave-one-signer-out ({len(locais)} sinalizadores locais)"
    else:
        print("  AVISO: menos de 2 sinalizadores locais fora do teste -> validação "
              "aleatória (otimista: mesma pessoa no treino e na validação).")
        sec["validacao"] = "ALEATÓRIA (otimista — colete mais sinalizadores!)"
        dobras = _dobras_aleatorias(idx_local, y, idx_extra, args.seed)

    nomes = args.modelos.split(",")
    resultados = {}
    for nm in nomes:
        t0 = time.time()
        oof = np.zeros((len(amostras), len(classes)))
        for tr, te in dobras:
            if not te:
                continue
            Xtr, ytr = _juntar(tr, ok_feat_base, ok_feat_aum, y, True)
            Xte, _ = _juntar(te, ok_feat_base, ok_feat_aum, y, False)
            est = clone(candidatos(args.seed)[nm]).fit(Xtr, ytr)
            oof[te] = _probs_alinhadas(est, Xte, classes)
        idx_av = sorted({i for _, te in dobras for i in te})
        pred = np.array(classes)[oof[idx_av].argmax(1)]
        bal = M.acuracia_balanceada(y[idx_av], pred)
        tab = M.tabela_por_letra(y[idx_av], pred, classes)
        resultados[nm] = {"oof": oof, "idx": idx_av, "bal": bal,
                          "min": tab.acuracia_letra.min(), "pred": pred}
        print(f"  {nm:12s} acurácia balanceada {bal:.1%} | pior letra {tab.acuracia_letra.min():.1%}"
              f"  ({time.time() - t0:.0f}s)")

    # critério: acurácia balanceada; entre modelos a até 1 ponto do melhor,
    # fica o que tem a PIOR LETRA mais alta (o tema cobra desempenho por letra)
    top_bal = max(v["bal"] for v in resultados.values())
    melhor = max((k for k, v in resultados.items() if v["bal"] >= top_bal - 0.01),
                 key=lambda k: (resultados[k]["min"], resultados[k]["bal"]))
    r = resultados[melhor]
    idx_av, pred = r["idx"], r["pred"]
    print(f"  -> escolhido: {melhor}")

    limiar_info, curva = M.calibrar_limiar(y[idx_av], r["oof"][idx_av], classes,
                                           config.MARGEM_MINIMA, args.alvo)
    print(f"  limiar de rejeição: {limiar_info['limiar']:.2f} -> cobertura "
          f"{limiar_info['cobertura']:.1%}, acerto nas respostas {limiar_info['acuracia_aceitas']:.1%}")

    tab = M.tabela_por_letra(y[idx_av], pred, classes)
    tab.to_csv(config.DIR_RELATORIOS / f"{tipo}_validacao_por_letra.csv", index=False)
    M.matriz_png(y[idx_av], pred, classes, f"{nome} — validação ({sec['validacao']})",
                 config.DIR_RELATORIOS / f"{tipo}_matriz_validacao.png")
    sec.update({
        "comparacao": {k: (v["bal"], v["min"]) for k, v in resultados.items()},
        "modelo": melhor, "tabela_validacao": tab,
        "por_sinalizador": M.por_sinalizador(y[idx_av], pred, g[idx_av]),
        "grupos": M.analise_grupos(y[idx_av], pred),
        "limiar": limiar_info, "curva": curva,
    })

    # ---------------- armadilha: mesma pessoa no treino e no teste ----------------
    if len(locais) >= 2:
        dob = _dobras_aleatorias(idx_local, y, idx_extra, args.seed)
        oof = np.zeros((len(amostras), len(classes)))
        for tr, te in dob:
            Xtr, ytr = _juntar(tr, ok_feat_base, ok_feat_aum, y, True)
            Xte, _ = _juntar(te, ok_feat_base, ok_feat_aum, y, False)
            oof[te] = _probs_alinhadas(clone(candidatos(args.seed)[melhor]).fit(Xtr, ytr),
                                       Xte, classes)
        idx_r = sorted({i for _, te in dob for i in te})
        bal_r = M.acuracia_balanceada(y[idx_r], np.array(classes)[oof[idx_r].argmax(1)])
        sec["aleatoria_vs_pessoa"] = (bal_r, r["bal"])
        print(f"  divisão aleatória {bal_r:.1%} vs pessoa nova {r['bal']:.1%} "
              f"(diferença = quanto o modelo 'decora a mão')")

    # ---------------- modelo final ----------------
    idx_final = idx_local + idx_extra
    Xf, yf = _juntar(idx_final, ok_feat_base, ok_feat_aum, y, True)
    est = clone(candidatos(args.seed)[melhor]).fit(Xf, yf)
    modelo = ModeloLetras(tipo, est, list(est.classes_), limiar=limiar_info["limiar"],
                          margem=config.MARGEM_MINIMA,
                          info={"modelo": melhor, "treinado_em": datetime.now().isoformat(),
                                "sinalizadores_treino": sorted(set(g[idx_final])),
                                "sinalizadores_teste": sorted(teste)})
    config.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    modelo.salvar(config.DIR_MODELOS / f"{tipo}.joblib")
    faltando = [l for l in (config.LETRAS_ESTATICAS if tipo == "estatico" else config.LETRAS_DINAMICAS)
                if l not in modelo.classes]
    sec["faltando"] = faltando
    if faltando:
        print(f"  ATENÇÃO: sem dados para {', '.join(faltando)} — o modelo não reconhece essas letras.")

    # ---------------- sinalizador(es) nunca vistos ----------------
    if idx_teste:
        Xt, yt = _juntar(idx_teste, ok_feat_base, ok_feat_aum, y, False)
        P = _probs_alinhadas(est, Xt, classes)
        pt = np.array(classes)[P.argmax(1)]
        tab_t = M.tabela_por_letra(yt, pt, classes)
        tab_t.to_csv(config.DIR_RELATORIOS / f"{tipo}_teste_por_letra.csv", index=False)
        M.matriz_png(yt, pt, classes, f"{nome} — sinalizador(es) novo(s): {', '.join(sorted(teste))}",
                     config.DIR_RELATORIOS / f"{tipo}_matriz_teste.png")
        gt = g[idx_teste]
        sec.update({"tabela_teste": tab_t, "bal_teste": M.acuracia_balanceada(yt, pt),
                    "acc_teste": float((yt == pt).mean()),
                    "por_sinalizador_teste": M.por_sinalizador(yt, pt, gt),
                    "grupos_teste": M.analise_grupos(yt, pt),
                    "rejeicao_teste": M.efeito_rejeicao(yt, P, classes, modelo.limiar, modelo.margem)})
        print(f"  TESTE (pessoa nunca vista): acurácia balanceada {sec['bal_teste']:.1%} | "
              f"pior letra {tab_t.acuracia_letra.min():.1%}")
    return modelo


def _dobras_aleatorias(idx_local, y, idx_extra, seed, k=5):
    if len(idx_local) < 2:
        return [(idx_extra, [])]
    yl = y[idx_local]
    k = max(2, min(k, int(min(np.unique(yl, return_counts=True)[1]))))
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    il = np.array(idx_local)
    return [(list(il[a]) + idx_extra, list(il[b])) for a, b in skf.split(il, yl)]


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------
def escrever_relatorio(rel, args):
    L = [f"# Relatório — tradutor de datilologia em LIBRAS",
         f"Gerado em {datetime.now():%d/%m/%Y %H:%M}. Sinalizador(es) de teste: "
         f"**{args.teste or 'nenhum (use --teste)'}**. Aumentos por amostra: {args.aumentos}.", ""]
    for tipo, titulo in (("estatico", "Letras estáticas"), ("dinamico", "Letras com movimento")):
        s = rel.get(tipo)
        if not s or "modelo" not in s:
            continue
        L += [f"## {titulo}", "",
              f"Validação: {s['validacao']}. Sinalizadores: {', '.join(s['locais']) or '—'}.", ""]
        if tipo == "dinamico":
            L += ["**Tratamento das letras com movimento.** " + ", ".join(config.LETRAS_DINAMICAS) +
                  " não são reconhecidas por foto. O sistema acompanha a mão quadro a quadro; quando o "
                  "punho ou as pontas dos dedos se deslocam mais que "
                  f"{config.LIMIAR_MOVIMENTO} tamanho de palma, o trecho é gravado até a mão parar e "
                  f"reamostrado para {config.QUADROS_SEQUENCIA} quadros. O modelo usa forma da mão ao "
                  "longo do tempo, trajetória do punho/indicador/mínimo e giro da palma. Ele tem uma "
                  f"classe extra (∅ = `{TRANSICAO}`) treinada com trocas sintéticas entre letras "
                  "estáticas, para que mudar de letra não seja lido como H/J/K/X/Y/Z.", ""]
        L += ["**Comparação de modelos (acurácia balanceada | pior letra):**", ""]
        L += [f"- {k}: {b:.1%} | {m:.1%}" + ("  ← escolhido" if k == s["modelo"] else "")
              for k, (b, m) in s["comparacao"].items()]
        if "aleatoria_vs_pessoa" in s:
            a, p = s["aleatoria_vs_pessoa"]
            L += ["", f"**Armadilha da mesma pessoa:** divisão aleatória {a:.1%} vs pessoa nova "
                      f"{p:.1%} (diferença {a - p:+.1%}). A divisão aleatória superestima o desempenho."]
        li = s["limiar"]
        L += ["", f"**Limiar de REPITA O SINAL:** {li['limiar']:.2f} → responde em "
                  f"{li['cobertura']:.1%} das tentativas, acertando {li['acuracia_aceitas']:.1%} delas "
                  f"(alvo {args.alvo:.0%}).", "",
              "### Acurácia por letra — validação (pessoa fora do treino)", "",
              M.tabela_md(s["tabela_validacao"]), "",
              f"![matriz]({tipo}_matriz_validacao.png)", "",
              "### Por sinalizador (cada um avaliado com modelo que não o viu)", "",
              M.tabela_md(s["por_sinalizador"]), "", "### Grupos confundíveis", ""] + s["grupos"]
        if "tabela_teste" in s:
            rj = s["rejeicao_teste"]
            L += [f"### Sinalizador(es) nunca vistos: {args.teste}", "",
                  f"Acurácia {s['acc_teste']:.1%} | balanceada {s['bal_teste']:.1%} | com rejeição: "
                  f"responde {rj['cobertura']:.1%}, acerta {rj['acuracia_aceitas']:.1%}.", "",
                  M.tabela_md(s["tabela_teste"]), "", f"![matriz]({tipo}_matriz_teste.png)", ""]
            L += s["grupos_teste"]
        if s.get("faltando"):
            L += [f"> **Sem dados para:** {', '.join(s['faltando'])} — colete antes de entregar.", ""]
    (config.DIR_RELATORIOS / "relatorio.md").write_text("\n".join(L), "utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teste", default="", help="sinalizador(es) reservados para teste final, separados por vírgula")
    ap.add_argument("--aumentos", type=int, default=5, help="cópias aumentadas por amostra de treino")
    ap.add_argument("--alvo", type=float, default=0.95, help="acerto desejado nas respostas aceitas")
    ap.add_argument("--modelos", default="svm_rbf,extra_trees,mlp,logistica")
    ap.add_argument("--transicoes", type=int, default=40, help="sequências de transição sintéticas por sinalizador")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dados", default=str(config.DIR_PROCESSADOS / "amostras.joblib"))
    args = ap.parse_args()

    banco = joblib.load(args.dados)
    teste = {t.strip() for t in args.teste.split(",") if t.strip()}
    conhecidos = {s["sinalizador"] for s in banco["estaticas"] + banco["dinamicas"]}
    if teste - conhecidos:
        raise SystemExit(f"Sinalizador(es) de teste desconhecido(s): {teste - conhecidos}")
    if not teste:
        print("AVISO: sem --teste, não haverá avaliação com sinalizador nunca visto "
              "(critério de aceite). A validação por pessoa ainda é feita.")
    config.DIR_RELATORIOS.mkdir(parents=True, exist_ok=True)

    rel = {}
    est = [s for s in banco["estaticas"] if s["letra"] in config.LETRAS_ESTATICAS]
    treinar_tipo("estatico", est, teste, args, rel)
    din = [s for s in banco["dinamicas"] if s["letra"] in config.LETRAS_DINAMICAS]
    if din:
        din = din + sintetizar_transicoes(est, args.transicoes, args.seed)
        contexto = {}
        for s in est:
            contexto.setdefault(s["sinalizador"], []).append(s["pts"])
        treinar_tipo("dinamico", din, teste, args, rel, contexto)
    else:
        print("\nAVISO: nenhuma sequência de H/J/K/X/Y/Z — o tradutor só reconhecerá letras estáticas.")
    escrever_relatorio(rel, args)
    resumo = {t: {k: v for k, v in s.items() if k in ("modelo", "validacao", "bal_teste", "acc_teste", "faltando")}
              for t, s in rel.items()}
    (config.DIR_RELATORIOS / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nRelatório: {config.DIR_RELATORIOS / 'relatorio.md'}")


if __name__ == "__main__":
    main()

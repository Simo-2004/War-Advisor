"""
War Advisor - Analisi della campagna sperimentale

Si usa solo tramite `python scripts/csv_batch_test.py --analizza`. Legge i CSV
gia' prodotti e non rigioca nulla. Tutto quello che produce finisce in
`<cartella del dataset>/analisi/`, che viene svuotata e riscritta a ogni
esecuzione: nessun file derivato puo' restare indietro rispetto ai dati.

  RIEPILOGO_DATI_TESI.md   tutti i numeri per la tesi, generati dai dati
  tabelle/*.csv            le tabelle da cui il riepilogo e' costruito
  figure/*.pdf, *.png      le quattro figure (PDF vettoriale, PNG a 300 dpi)

Se accanto alla cartella principale esistono cartelle `<nome>_*` con un proprio
manifest (i dataset di controllo), vengono confrontate partita per partita con
quella principale.

Le scelte statistiche seguono le osservazioni del relatore del 15/10/2026:
ipotesi primaria dichiarata e correzione di Holm sugli altri confronti; analisi
appaiata (McNemar) perche' in E2 ed E4 le condizioni condividono il seed; analisi
con lo scenario come unita' contro la pseudo-replicazione; E4 ristretto agli
scenari in cui la dottrina si attiva davvero; le partite non decise trattate come
esito a se'.
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
# COSTANTI
# ══════════════════════════════════════════════════════════════════

LIVELLI = ("tutte", "easy", "normal", "hard")
NOME_LIVELLO = {"tutte": "Tutti i livelli", "easy": "Facile",
                "normal": "Normale", "hard": "Difficile"}
CONDIZIONI_E2 = ("top_ranked", "intermedia", "worst_ranked")
COPPIE_E2 = (("top_ranked", "worst_ranked"), ("top_ranked", "intermedia"),
             ("intermedia", "worst_ranked"))
COPPIA_E4 = ("dottrine_on", "dottrine_off")

#: L'unica ipotesi confermativa. Tutti gli altri confronti sono esplorativi.
IPOTESI_PRIMARIA = {"esperimento": "E2", "livello": "tutte",
                    "condizione_a": "top_ranked", "condizione_b": "worst_ranked",
                    "metrica": "win_rate"}

#: Metriche continue. `rapporto_scambi` e' la metrica che non dipende dalla
#: durata: la forza residua divide per le unita' mai schierate, che crescono con
#: i reclutamenti e quindi con la lunghezza della partita.
METRICHE = (
    ("forza_residua_pct", "Forza residua", "%"),
    ("rapporto_scambi", "Rapporto di scambio", "%"),
    ("perdite_proprie", "Perdite proprie", "unita"),
    ("perdite_nemiche", "Perdite inflitte", "unita"),
    ("morale_residuo", "Morale residuo", "0-100"),
    ("turni_giocati", "Turni giocati", "turni"),
    ("scontri", "Scontri", "n"),
    ("celle_player", "Celle controllate", "n"),
)
METRICHE_APPAIATE = ("forza_residua_pct", "rapporto_scambi", "perdite_proprie",
                     "perdite_nemiche", "turni_giocati")

#: In E4 uno scenario entra nell'analisi solo se, con la dottrina accesa,
#: l'effetto e' attivo in media in almeno uno scontro per partita. S11 ne ha
#: 0,19: il requisito dell'Accerchiamento (quattro tipi di truppa vivi insieme)
#: si rompe alle prime perdite, e l'ablation li' confronterebbe il nulla.
SOGLIA_ATTIVAZIONE_E4 = 1.0

Z95 = 1.959963984540054
Z80 = 0.8416212335729143


# ══════════════════════════════════════════════════════════════════
# STATISTICA DI BASE (senza scipy: tutto verificabile riga per riga)
# ══════════════════════════════════════════════════════════════════

def _beta_frazione(a: float, b: float, x: float) -> float:
    """Frazione continua della beta incompleta (metodo di Lentz)."""
    minimo = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > minimo else minimo)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > minimo else minimo)
        c = 1.0 + aa / c
        c = c if abs(c) > minimo else minimo
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > minimo else minimo)
        c = 1.0 + aa / c
        c = c if abs(c) > minimo else minimo
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def beta_incompleta(a: float, b: float, x: float) -> float:
    """Beta incompleta regolarizzata I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    logaritmo = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log(1.0 - x))
    fattore = math.exp(logaritmo)
    if x < (a + 1.0) / (a + b + 2.0):
        return fattore * _beta_frazione(a, b, x) / a
    return 1.0 - fattore * _beta_frazione(b, a, 1.0 - x) / b


def p_t(t: float, gl: float) -> float:
    """p-value a due code della t di Student con `gl` gradi di liberta'."""
    if not math.isfinite(t):
        return 0.0
    return beta_incompleta(gl / 2.0, 0.5, gl / (gl + t * t))


def quantile_t(probabilita: float, gl: float) -> float:
    """Quantile della t (es. 0,975 con 10 gl = 2,228), per bisezione."""
    bersaglio = 2.0 * (1.0 - probabilita)
    basso, alto = 0.0, 1000.0
    for _ in range(200):
        medio = (basso + alto) / 2.0
        if p_t(medio, gl) > bersaglio:
            basso = medio
        else:
            alto = medio
    return (basso + alto) / 2.0


def p_normale(z: float) -> float:
    """p-value a due code della normale standard."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def p_binomiale_esatto(k: int, n: int) -> float:
    """p-value esatto a due code di Bin(n, 1/2): e' il test di McNemar esatto."""
    if n == 0:
        return 1.0
    minimo = min(k, n - k)
    coda = sum(math.comb(n, i) for i in range(minimo + 1))
    return min(1.0, 2 * coda / 2 ** n)


def holm(p_values: Sequence[float]) -> List[float]:
    """Correzione di Holm-Bonferroni (step-down), nell'ordine di ingresso."""
    m = len(p_values)
    ordine = sorted(range(m), key=lambda i: p_values[i])
    corretti = [0.0] * m
    massimo = 0.0
    for rango, i in enumerate(ordine):
        valore = min(1.0, (m - rango) * p_values[i])
        massimo = max(massimo, valore)
        corretti[i] = massimo
    return corretti


def wilson(successi: int, n: int) -> Tuple[float, float, float]:
    """(proporzione, ic95_min, ic95_max) in percentuale, intervallo di Wilson."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successi / n
    den = 1 + Z95 ** 2 / n
    centro = (p + Z95 ** 2 / (2 * n)) / den
    meta = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / den
    return p * 100, max(0.0, centro - meta) * 100, min(1.0, centro + meta) * 100


def media_ic(valori: Iterable[float]) -> Dict[str, float]:
    """Media, deviazione standard, errore standard e IC 95% con quantile t."""
    x = np.asarray([v for v in valori if v == v], dtype=float)
    n = len(x)
    if n == 0:
        return {"n": 0, "media": np.nan, "dev_std": np.nan, "err_std": np.nan,
                "ic95_min": np.nan, "ic95_max": np.nan}
    media = float(x.mean())
    if n < 2:
        return {"n": n, "media": media, "dev_std": np.nan, "err_std": np.nan,
                "ic95_min": np.nan, "ic95_max": np.nan}
    dev = float(x.std(ddof=1))
    err = dev / math.sqrt(n)
    q = quantile_t(0.975, n - 1)
    return {"n": n, "media": media, "dev_std": dev, "err_std": err,
            "ic95_min": media - q * err, "ic95_max": media + q * err}


def t_un_campione(valori: Iterable[float]) -> Dict[str, float]:
    """Test t a un campione contro zero, con IC 95%."""
    r = media_ic(valori)
    n = r["n"]
    if n < 2 or not r["err_std"]:
        return {**r, "t": np.nan, "gl": n - 1, "p_value": np.nan}
    t = r["media"] / r["err_std"]
    return {**r, "t": t, "gl": n - 1, "p_value": p_t(t, n - 1)}


def _r(valore: Any, cifre: int = 4) -> Any:
    """Arrotonda i numeri, lascia stare il resto."""
    if isinstance(valore, (float, np.floating)):
        if not math.isfinite(valore):
            return ""
        # un p-value di 1e-27 arrotondato a 4 decimali diventerebbe 0: per i
        # numeri piccolissimi si tengono le cifre significative
        if valore != 0 and abs(valore) < 10 ** -cifre:
            return float(f"{float(valore):.{cifre}g}")
        return round(float(valore), cifre)
    if isinstance(valore, np.integer):
        return int(valore)
    return valore


def _formatta_p(p: float) -> str:
    if p != p:
        return "n.d."
    if p < 0.001:
        return "p < 0,001"
    # vicino alla soglia tre decimali ingannano: 0,0504 diventerebbe "0,050"
    cifre = 4 if 0.045 <= p < 0.055 else 3
    return f"p = {p:.{cifre}f}".replace(".", ",")


def _num(x: float, cifre: int = 1, segno: bool = False) -> str:
    """Numero in formato italiano: virgola decimale."""
    if x != x:
        return "n.d."
    testo = f"{x:+.{cifre}f}" if segno else f"{x:.{cifre}f}"
    return testo.replace(".", ",").replace("-", "−")


# ══════════════════════════════════════════════════════════════════
# CARICAMENTO
# ══════════════════════════════════════════════════════════════════

def carica(cartella: Path, separatore: str) -> SimpleNamespace:
    """Legge un dataset e aggiunge le colonne derivate."""
    def leggi(nome: str) -> pd.DataFrame:
        percorso = cartella / nome
        if not percorso.exists():
            return pd.DataFrame()
        return pd.read_csv(percorso, sep=separatore, encoding="utf-8-sig", low_memory=False)

    runs = leggi("runs.csv")
    advisor = leggi("advisor.csv")
    manifest = {}
    if (cartella / "manifest.json").exists():
        manifest = json.loads((cartella / "manifest.json").read_text(encoding="utf-8"))

    turni_derivati = False
    if not runs.empty:
        runs["vinta"] = (runs["esito"] == "vittoria").astype(int)
        runs["persa"] = (runs["esito"] == "sconfitta").astype(int)
        runs["indecisa"] = (runs["esito"] == "non_decisa").astype(int)
        # Il turno 401: nei dataset prodotti prima della correzione la durata
        # vera si ricava togliendo uno, per TUTTE le partite (vedi riepilogo).
        if "turni_giocati" not in runs.columns:
            runs["turni_giocati"] = runs["turni_totali"] - 1
            turni_derivati = True
        totale = runs["perdite_proprie"] + runs["perdite_nemiche"]
        runs["rapporto_scambi"] = np.where(totale > 0,
                                           runs["perdite_nemiche"] / totale.where(totale > 0) * 100,
                                           np.nan)
    return SimpleNamespace(cartella=cartella, runs=runs, advisor=advisor,
                           manifest=manifest, turni_derivati=turni_derivati)


def _parametro(manifest: Dict[str, Any], chiave: str) -> Any:
    """Parametri del manifest, compatibili anche con i manifest precedenti."""
    parametri = manifest.get("parametri", {})
    if chiave == "combat_affinity_weight":
        if parametri.get(chiave) is not None:
            return parametri[chiave]
        return ((manifest.get("taratura") or {}).get("compat_layer") or {}).get(chiave)
    return parametri.get(chiave)


def _livello(df: pd.DataFrame, livello: str) -> pd.DataFrame:
    return df if livello == "tutte" else df[df["difficolta_ia"] == livello]


def _livelli_presenti(df: pd.DataFrame) -> List[str]:
    presenti = set(df["difficolta_ia"].unique()) if not df.empty else set()
    return ["tutte"] + [l for l in LIVELLI[1:] if l in presenti]


def _nome_condizione(condizione: str, runs: pd.DataFrame) -> str:
    """Etichetta leggibile, con il rank vero letto dai dati."""
    if condizione in CONDIZIONI_E2 and not runs.empty:
        rank = runs.loc[runs["condizione"] == condizione, "rank_advisor"]
        rank_txt = f" (rank {int(round(rank.median()))})" if len(rank) else ""
        base = {"top_ranked": "Più affine", "intermedia": "Intermedia",
                "worst_ranked": "Meno affine"}[condizione]
        return base + rank_txt
    return {"dottrine_on": "Dottrina attiva",
            "dottrine_off": "Dottrina disattivata"}.get(condizione, condizione)


def scenari_validi_e4(runs: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """Scenari di E4 in cui la dottrina si attiva davvero, e la tabella del perche'."""
    e4 = runs[runs["esperimento"] == "E4"]
    if e4.empty:
        return [], pd.DataFrame()
    attivazione = (e4[e4["condizione"] == "dottrine_on"]
                   .groupby("scenario_id")["dottrina_scontri_attiva"].mean())
    validi = sorted(attivazione[attivazione >= SOGLIA_ATTIVAZIONE_E4].index)
    return validi, attivazione.rename("attivazione_media").reset_index()


# ══════════════════════════════════════════════════════════════════
# TABELLE
# ══════════════════════════════════════════════════════════════════

def tabella_condizioni(runs: pd.DataFrame, validi_e4: List[str]) -> pd.DataFrame:
    """Per ogni condizione: n, media, deviazione standard, errore standard, IC."""
    righe = []
    for esperimento in ("E2", "E4"):
        df = runs[runs["esperimento"] == esperimento]
        if esperimento == "E4":
            df = df[df["scenario_id"].isin(validi_e4)]
        if df.empty:
            continue
        condizioni = CONDIZIONI_E2 if esperimento == "E2" else COPPIA_E4
        for livello in _livelli_presenti(df):
            for condizione in condizioni:
                g = _livello(df, livello)
                g = g[g["condizione"] == condizione]
                if g.empty:
                    continue
                base = {"esperimento": esperimento, "livello": livello,
                        "condizione": condizione,
                        "rank_advisor_medio": g["rank_advisor"].mean(),
                        "scenari": len(g["scenario_id"].unique())}
                n = len(g)
                for metrica, colonna in (("win_rate", "vinta"), ("tasso_sconfitte", "persa"),
                                         ("tasso_non_decise", "indecisa")):
                    p, lo, hi = wilson(int(g[colonna].sum()), n)
                    righe.append({**base, "metrica": metrica, "unita": "%", "n": n,
                                  "media": p, "dev_std": math.sqrt(p / 100 * (1 - p / 100)) * 100,
                                  "err_std": math.sqrt(p / 100 * (1 - p / 100) / n) * 100,
                                  "ic95_min": lo, "ic95_max": hi, "metodo_ic": "Wilson"})
                decise = g[g["indecisa"] == 0]
                p, lo, hi = wilson(int(decise["vinta"].sum()), len(decise))
                righe.append({**base, "metrica": "win_rate_decise", "unita": "%",
                              "n": len(decise), "media": p,
                              "dev_std": math.sqrt(p / 100 * (1 - p / 100)) * 100 if len(decise) else np.nan,
                              "err_std": math.sqrt(p / 100 * (1 - p / 100) / len(decise)) * 100 if len(decise) else np.nan,
                              "ic95_min": lo, "ic95_max": hi, "metodo_ic": "Wilson"})
                for colonna, _, unita in METRICHE:
                    r = media_ic(g[colonna])
                    righe.append({**base, "metrica": colonna, "unita": unita, **r,
                                  "metodo_ic": "t di Student"})
    return pd.DataFrame(righe)


def tabella_confronti(runs: pd.DataFrame, validi_e4: List[str]) -> pd.DataFrame:
    """Confronti a campioni indipendenti, con p-value, Holm e ipotesi primaria.

    Restano per continuita' con la versione precedente, ma il confronto da
    citare per l'ipotesi primaria e' quello appaiato (McNemar): qui gli
    intervalli ignorano che le condizioni condividono il seed.
    """
    righe = []
    for esperimento, coppie in (("E2", COPPIE_E2), ("E4", (COPPIA_E4,))):
        df = runs[runs["esperimento"] == esperimento]
        if esperimento == "E4":
            df = df[df["scenario_id"].isin(validi_e4)]
        if df.empty:
            continue
        for livello in _livelli_presenti(df):
            g = _livello(df, livello)
            for a, b in coppie:
                ga, gb = g[g["condizione"] == a], g[g["condizione"] == b]
                if ga.empty or gb.empty:
                    continue
                base = {"esperimento": esperimento, "livello": livello,
                        "condizione_a": a, "condizione_b": b,
                        "n_a": len(ga), "n_b": len(gb)}
                # vittorie: z a due proporzioni (varianza combinata per il test)
                pa, pb = ga["vinta"].mean(), gb["vinta"].mean()
                pooled = (ga["vinta"].sum() + gb["vinta"].sum()) / (len(ga) + len(gb))
                se_test = math.sqrt(pooled * (1 - pooled) * (1 / len(ga) + 1 / len(gb)))
                se_ic = math.sqrt(pa * (1 - pa) / len(ga) + pb * (1 - pb) / len(gb))
                diff = pa - pb
                righe.append({**base, "metrica": "win_rate", "unita": "punti %",
                              "media_a": pa * 100, "media_b": pb * 100,
                              "differenza": diff * 100, "err_std": se_ic * 100,
                              "ic95_min": (diff - Z95 * se_ic) * 100,
                              "ic95_max": (diff + Z95 * se_ic) * 100,
                              "p_value": p_normale(diff / se_test) if se_test else 1.0,
                              "test": "z a due proporzioni"})
                for colonna, _, unita in METRICHE:
                    xa, xb = ga[colonna].dropna(), gb[colonna].dropna()
                    if len(xa) < 2 or len(xb) < 2:
                        continue
                    va, vb = xa.var(ddof=1) / len(xa), xb.var(ddof=1) / len(xb)
                    se = math.sqrt(va + vb)
                    diff = xa.mean() - xb.mean()
                    # gradi di liberta' di Welch-Satterthwaite
                    gl = (va + vb) ** 2 / (va ** 2 / (len(xa) - 1) + vb ** 2 / (len(xb) - 1)) \
                        if va + vb > 0 else len(xa) + len(xb) - 2
                    q = quantile_t(0.975, gl)
                    righe.append({**base, "metrica": colonna, "unita": unita,
                                  "media_a": xa.mean(), "media_b": xb.mean(),
                                  "differenza": diff, "err_std": se,
                                  "ic95_min": diff - q * se, "ic95_max": diff + q * se,
                                  "p_value": p_t(diff / se, gl) if se else 1.0,
                                  "test": "t di Welch"})
    df = pd.DataFrame(righe)
    if df.empty:
        return df
    chiave = ["esperimento", "livello", "condizione_a", "condizione_b", "metrica"]
    doppie = df[df.duplicated(chiave, keep=False)]
    if not doppie.empty:                       # non deve mai succedere
        raise RuntimeError(f"Confronti duplicati:\n{doppie[chiave]}")
    df["ic95_esclude_zero"] = (df["ic95_min"] > 0) | (df["ic95_max"] < 0)
    df["p_holm"] = holm(df["p_value"].tolist())
    df["ipotesi_primaria"] = ((df["esperimento"] == IPOTESI_PRIMARIA["esperimento"])
                              & (df["livello"] == IPOTESI_PRIMARIA["livello"])
                              & (df["condizione_a"] == IPOTESI_PRIMARIA["condizione_a"])
                              & (df["condizione_b"] == IPOTESI_PRIMARIA["condizione_b"])
                              & (df["metrica"] == IPOTESI_PRIMARIA["metrica"]))
    return df


def _appaia(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Accoppia le partite delle due condizioni che condividono il seed."""
    chiave = ["scenario_id", "difficolta_ia", "replica"]
    colonne = chiave + ["seed", "vinta"] + list(METRICHE_APPAIATE)
    ga = df[df["condizione"] == a][colonne].set_index(chiave)
    gb = df[df["condizione"] == b][colonne].set_index(chiave)
    coppie = ga.join(gb, lsuffix="_a", rsuffix="_b", how="inner")
    diversi = (coppie["seed_a"] != coppie["seed_b"]).sum()
    if diversi:
        raise RuntimeError(f"{diversi} coppie {a}/{b} non condividono il seed: "
                           "il disegno appaiato non regge.")
    return coppie


def mcnemar(vinte_a: pd.Series, vinte_b: pd.Series) -> Dict[str, float]:
    """Tabella 2x2 delle coppie e test di McNemar (esatto e chi quadro)."""
    a = int(((vinte_a == 1) & (vinte_b == 1)).sum())
    b = int(((vinte_a == 1) & (vinte_b == 0)).sum())
    c = int(((vinte_a == 0) & (vinte_b == 1)).sum())
    d = int(((vinte_a == 0) & (vinte_b == 0)).sum())
    n = a + b + c + d
    denominatore = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = (a * d - b * c) / denominatore if denominatore else np.nan
    discordanti = b + c
    diff = (b - c) / n if n else np.nan
    se = math.sqrt(max(0.0, discordanti - (b - c) ** 2 / n)) / n if n else np.nan
    return {
        "n_coppie": n, "entrambe_vincono": a, "solo_a_vince": b,
        "solo_b_vince": c, "nessuna_vince": d, "correlazione_phi": phi,
        "mcnemar_chi2_corretto": (abs(b - c) - 1) ** 2 / discordanti if discordanti else 0.0,
        "mcnemar_chi2": (b - c) ** 2 / discordanti if discordanti else 0.0,
        "p_value_esatto": p_binomiale_esatto(b, discordanti),
        "diff_appaiata_pct": diff * 100, "err_std_pct": se * 100,
        "ic95_min": (diff - Z95 * se) * 100, "ic95_max": (diff + Z95 * se) * 100,
    }


def tabelle_appaiate(runs: pd.DataFrame, validi_e4: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """McNemar sulle vittorie e t appaiato sulle metriche continue."""
    vittorie, metriche = [], []
    for esperimento, coppie in (("E2", COPPIE_E2), ("E4", (COPPIA_E4,))):
        df = runs[runs["esperimento"] == esperimento]
        if esperimento == "E4":
            df = df[df["scenario_id"].isin(validi_e4)]
        if df.empty:
            continue
        for livello in _livelli_presenti(df):
            g = _livello(df, livello)
            for a, b in coppie:
                coppie_df = _appaia(g, a, b)
                if coppie_df.empty:
                    continue
                base = {"esperimento": esperimento, "livello": livello,
                        "condizione_a": a, "condizione_b": b}
                primaria = (esperimento == IPOTESI_PRIMARIA["esperimento"]
                            and livello == IPOTESI_PRIMARIA["livello"]
                            and a == IPOTESI_PRIMARIA["condizione_a"]
                            and b == IPOTESI_PRIMARIA["condizione_b"])
                vittorie.append({**base, **mcnemar(coppie_df["vinta_a"], coppie_df["vinta_b"]),
                                 "ipotesi_primaria": primaria})
                for colonna in METRICHE_APPAIATE:
                    d = (coppie_df[f"{colonna}_a"] - coppie_df[f"{colonna}_b"]).dropna()
                    r = t_un_campione(d)
                    metriche.append({**base, "metrica": colonna, "n_coppie": r["n"],
                                     "media_differenze": r["media"], "dev_std_differenze": r["dev_std"],
                                     "err_std": r["err_std"], "ic95_min": r["ic95_min"],
                                     "ic95_max": r["ic95_max"], "t": r["t"], "gl": r["gl"],
                                     "p_value": r["p_value"]})
    vittorie_df = pd.DataFrame(vittorie)
    if not vittorie_df.empty:
        vittorie_df["p_holm"] = holm(vittorie_df["p_value_esatto"].tolist())
    metriche_df = pd.DataFrame(metriche)
    if not metriche_df.empty:
        metriche_df["p_holm"] = holm(metriche_df["p_value"].tolist())
    return vittorie_df, metriche_df


def tabelle_per_scenario(runs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lo scenario come unita' statistica: contro la pseudo-replicazione.

    Le 990 partite per condizione vengono da 11 scenari. Per generalizzare agli
    scenari si calcola prima la differenza dentro ogni scenario, poi si fa
    l'inferenza sugli 11 valori: l'errore standard e' quello fra scenari.
    """
    e2 = runs[runs["esperimento"] == "E2"]
    if e2.empty:
        return pd.DataFrame(), pd.DataFrame()
    dettaglio, sintesi = [], []
    for livello in _livelli_presenti(e2):
        g = _livello(e2, livello)
        for colonna, etichetta in (("vinta", "win_rate"), ("forza_residua_pct", "forza_residua_pct"),
                                   ("rapporto_scambi", "rapporto_scambi")):
            scala = 100 if colonna == "vinta" else 1
            medie = g.groupby(["scenario_id", "condizione"])[colonna].mean().unstack() * scala
            medie = medie.reindex(columns=list(CONDIZIONI_E2))
            delta = medie["top_ranked"] - medie["worst_ranked"]
            for scenario, riga in medie.iterrows():
                dettaglio.append({"livello": livello, "metrica": etichetta, "scenario_id": scenario,
                                  "partite_per_condizione": int((g["scenario_id"] == scenario).sum() / 3),
                                  "piu_affine": riga["top_ranked"], "intermedia": riga["intermedia"],
                                  "meno_affine": riga["worst_ranked"],
                                  "delta_piu_meno_affine": delta[scenario]})
            r = t_un_campione(delta)
            # errore standard "ingenuo" a campioni indipendenti, per confronto
            x = g[g["condizione"] == "top_ranked"][colonna] * scala
            y = g[g["condizione"] == "worst_ranked"][colonna] * scala
            se_ingenuo = math.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
            sintesi.append({"livello": livello, "metrica": etichetta, "n_scenari": r["n"],
                            "scenari_positivi": int((delta > 1e-9).sum()),
                            "scenari_nulli": int((delta.abs() <= 1e-9).sum()),
                            "scenari_negativi": int((delta < -1e-9).sum()),
                            "scenari_nulli_elenco": " ".join(delta[delta.abs() <= 1e-9].index),
                            "media_delta": r["media"], "dev_std_fra_scenari": r["dev_std"],
                            "err_std_per_scenario": r["err_std"],
                            "err_std_partite_indipendenti": se_ingenuo,
                            "t": r["t"], "gl": r["gl"], "p_value": r["p_value"],
                            "quantile_t_0975": quantile_t(0.975, r["gl"]) if r["gl"] > 0 else np.nan,
                            "ic95_min": r["ic95_min"], "ic95_max": r["ic95_max"]})
    return pd.DataFrame(dettaglio), pd.DataFrame(sintesi)


def tabella_e4(runs: pd.DataFrame, validi: List[str],
               attivazione: pd.DataFrame) -> pd.DataFrame:
    """E4 scenario per scenario, poi l'aggregato sui soli scenari validi."""
    e4 = runs[runs["esperimento"] == "E4"]
    if e4.empty:
        return pd.DataFrame()
    righe = []

    def blocco(df: pd.DataFrame) -> Dict[str, Any]:
        on, off = df[df["condizione"] == "dottrine_on"], df[df["condizione"] == "dottrine_off"]
        p_on, lo_on, hi_on = wilson(int(on["vinta"].sum()), len(on))
        p_off, lo_off, hi_off = wilson(int(off["vinta"].sum()), len(off))
        coppie = _appaia(df, "dottrine_on", "dottrine_off")
        m = mcnemar(coppie["vinta_a"], coppie["vinta_b"])
        return {"n_per_condizione": len(on), "win_rate_on": p_on, "ic95_on_min": lo_on,
                "ic95_on_max": hi_on, "win_rate_off": p_off, "ic95_off_min": lo_off,
                "ic95_off_max": hi_off, "diff_appaiata_pct": m["diff_appaiata_pct"],
                "ic95_diff_min": m["ic95_min"], "ic95_diff_max": m["ic95_max"],
                "solo_on_vince": m["solo_a_vince"], "solo_off_vince": m["solo_b_vince"],
                "p_mcnemar_esatto": m["p_value_esatto"],
                # effetto minimo rilevabile all'80% di potenza (alfa 0,05, due code)
                "effetto_minimo_rilevabile_80": (Z95 + Z80) * m["err_std_pct"],
                "forza_residua_on": on["forza_residua_pct"].mean(),
                "forza_residua_off": off["forza_residua_pct"].mean()}

    att = dict(zip(attivazione["scenario_id"], attivazione["attivazione_media"]))
    for scenario in sorted(e4["scenario_id"].unique()):
        g = e4[e4["scenario_id"] == scenario]
        incluso = scenario in validi
        righe.append({"tipo": "scenario", "scenario_id": scenario, "livello": "tutte",
                      "strategia": g["strategia_nome"].iloc[0],
                      "composizione": g["composizione"].iloc[0],
                      "attivazione_media": att.get(scenario, np.nan), "incluso": incluso,
                      "motivo": ("" if incluso else
                                 f"dottrina attiva in media in {_num(att.get(scenario, 0), 2)} "
                                 f"scontri per partita, sotto la soglia di "
                                 f"{_num(SOGLIA_ATTIVAZIONE_E4, 0)}"),
                      **blocco(g)})
    validi_df = e4[e4["scenario_id"].isin(validi)]
    for livello in _livelli_presenti(validi_df):
        righe.append({"tipo": "aggregato", "scenario_id": " ".join(validi), "livello": livello,
                      "strategia": "", "composizione": "", "attivazione_media": np.nan,
                      "incluso": True, "motivo": f"{len(validi)} scenari validi",
                      **blocco(_livello(validi_df, livello))})
    df = pd.DataFrame(righe)
    # i confronti per scenario sono piu' d'uno: il p si corregge fra di loro
    scelti = (df["tipo"] == "scenario") & df["incluso"]
    df["p_holm_fra_scenari"] = np.nan
    if scelti.any():
        df.loc[scelti, "p_holm_fra_scenari"] = holm(df.loc[scelti, "p_mcnemar_esatto"].tolist())
    return df


def tabella_esiti_per_livello(runs: pd.DataFrame) -> pd.DataFrame:
    """L'anomalia easy/normal: perche' contro 'easy' si vince meno che contro 'normal'."""
    e2 = runs[runs["esperimento"] == "E2"]
    righe = []
    for livello in _livelli_presenti(e2):
        g = _livello(e2, livello)
        nd = g[g["indecisa"] == 1]
        decise = g[g["indecisa"] == 0]
        righe.append({
            "livello": livello, "n": len(g),
            "vittorie_pct": g["vinta"].mean() * 100, "sconfitte_pct": g["persa"].mean() * 100,
            "non_decise_pct": g["indecisa"].mean() * 100,
            "win_rate_decise_pct": decise["vinta"].mean() * 100 if len(decise) else np.nan,
            "turni_giocati_medi": g["turni_giocati"].mean(), "scontri_medi": g["scontri"].mean(),
            "celle_ia_medie": g["celle_ai"].mean(), "unita_ia_finali_medie": g["unita_ai_finali"].mean(),
            "hp_castello_giocatore_pct": g["hp_castello_player_pct"].mean(),
            "hp_castello_ia_pct": g["hp_castello_ai_pct"].mean(),
            "non_decise_n": len(nd),
            "non_decise_scontri_medi": nd["scontri"].mean() if len(nd) else np.nan,
            "non_decise_celle_ia_medie": nd["celle_ai"].mean() if len(nd) else np.nan,
            "non_decise_celle_giocatore_medie": nd["celle_player"].mean() if len(nd) else np.nan,
            "non_decise_hp_castello_giocatore_pct": nd["hp_castello_player_pct"].mean() if len(nd) else np.nan,
            "non_decise_hp_castello_ia_pct": nd["hp_castello_ai_pct"].mean() if len(nd) else np.nan,
            "non_decise_unita_ia_finali": nd["unita_ai_finali"].mean() if len(nd) else np.nan,
            "non_decise_unita_giocatore_vive": nd["unita_vive"].mean() if len(nd) else np.nan,
            "primo_scontro_medio": g["turno_primo_scontro"].mean(),
        })
    return pd.DataFrame(righe)


def tabella_durata(runs: pd.DataFrame) -> pd.DataFrame:
    """Quanto ogni metrica dipende dalla durata della partita (correlazione)."""
    e2 = runs[runs["esperimento"] == "E2"]
    righe = []
    for livello in _livelli_presenti(e2):
        g = _livello(e2, livello)
        for colonna, etichetta, _ in METRICHE[:2]:
            coppia = g[["turni_giocati", colonna]].dropna()
            r = float(np.corrcoef(coppia["turni_giocati"], coppia[colonna])[0, 1]) \
                if len(coppia) > 2 else np.nan
            righe.append({"livello": livello, "metrica": colonna,
                          "correlazione_con_turni_giocati": r, "n": len(coppia)})
    return pd.DataFrame(righe)


def tabella_e1(ctx: SimpleNamespace, advisor: pd.DataFrame) -> pd.DataFrame:
    """E1 scenario per scenario, con le motivazioni e le vere contro-prove."""
    righe = []

    def valuta(unita: Sequence[str], terreno: str, meteo: str, stato: str) -> Dict[str, Any]:
        classifica = ctx.valuta_scenario(unita, terreno, meteo, stato)
        vettore = ctx.aggregate_army(list(unita), ctx.DATA["units"])
        modificato, _ = ctx.apply_modifiers(army_vector=vettore, terrain_name=terreno,
                                            weather_name=meteo, troop_status_name=stato,
                                            modifiers_data=ctx.DATA)
        primi = sorted(modificato.items(), key=lambda kv: -kv[1])[:2]
        return {"classifica": classifica,
                "attributi_principali": "; ".join(
                    f"{ctx.NOMI_ATTRIBUTI.get(k, k)} {_num(v, 2)}" for k, v in primi),
                "vettore_modificato": json.dumps({k: round(v, 3) for k, v in modificato.items()})}

    for scenario in ctx.SCENARI:
        v = valuta(scenario["unita"], scenario["terreno"], scenario["meteo"], scenario["stato"])
        cl = v["classifica"]
        attese = scenario["attese"]
        fuori = [x for x in cl if attese and x["strategia_id"] not in attese]
        # verifica d'integrita': i numeri devono coincidere con advisor.csv
        if not advisor.empty:
            orig = advisor[(advisor["esperimento"] == "E1")
                           & (advisor["scenario_id"] == scenario["id"])
                           & (advisor["rank_advisor"] == 1)]
            if not orig.empty and orig["strategia_id"].iloc[0] != cl[0]["strategia_id"]:
                raise RuntimeError(f"E1 {scenario['id']}: il codice non riproduce advisor.csv")
        righe.append({
            "tipo": "scenario", "id": scenario["id"], "descrizione": scenario["descrizione"],
            "composizione": ctx.descrivi_composizione(scenario["unita"]),
            "terreno": scenario["terreno"], "meteo": scenario["meteo"], "stato": scenario["stato"],
            "attributi_principali_dopo_modificatori": v["attributi_principali"],
            "top1": cl[0]["strategia_nome"], "top1_compat_pct": cl[0]["compat_advisor_pct"],
            "top2": cl[1]["strategia_nome"], "top2_compat_pct": cl[1]["compat_advisor_pct"],
            "scarto_top1_top2_pt": cl[0]["compat_advisor_pct"] - cl[1]["compat_advisor_pct"],
            "migliore_non_attesa": fuori[0]["strategia_nome"] if fuori else "",
            "migliore_non_attesa_compat_pct": fuori[0]["compat_advisor_pct"] if fuori else np.nan,
            "attese": "; ".join(_nomi_strategie(ctx, attese)),
            "top1_coerente": (cl[0]["strategia_id"] in attese) if attese else "",
            "motivazione_attese": scenario.get("motivazione", ""),
            "vettore_modificato": v["vettore_modificato"],
        })

    for prova in ctx.CONTROPROVE:
        famiglia = ctx.famiglia_attributo(prova["attributo_identitario"])
        esiti = []
        for ruolo in ("base", "variante"):
            contesto = prova[ruolo]
            v = valuta(prova["unita"], contesto["terreno"], contesto["meteo"], contesto["stato"])
            cl = v["classifica"]
            esiti.append(cl[0]["strategia_id"])
            nella_famiglia = cl[0]["strategia_id"] in famiglia
            nome_attr = ctx.NOMI_ATTRIBUTI.get(prova["attributo_identitario"], prova["attributo_identitario"])
            previsione = (f"in testa una strategia fondata sulla {nome_attr}"
                          if ruolo == "base" else
                          f"la strategia in testa non è più fondata sulla {nome_attr}")
            confermata = nella_famiglia if ruolo == "base" else not nella_famiglia
            righe.append({
                "tipo": f"contro-prova {ruolo}", "id": prova["id"],
                "descrizione": prova["descrizione"],
                "composizione": ctx.descrivi_composizione(prova["unita"]),
                "terreno": contesto["terreno"], "meteo": contesto["meteo"], "stato": contesto["stato"],
                "attributi_principali_dopo_modificatori": v["attributi_principali"],
                "top1": cl[0]["strategia_nome"], "top1_compat_pct": cl[0]["compat_advisor_pct"],
                "top2": cl[1]["strategia_nome"], "top2_compat_pct": cl[1]["compat_advisor_pct"],
                "scarto_top1_top2_pt": cl[0]["compat_advisor_pct"] - cl[1]["compat_advisor_pct"],
                "migliore_non_attesa": "", "migliore_non_attesa_compat_pct": np.nan,
                "attese": f"previsione: {previsione} ({', '.join(_nomi_strategie(ctx, famiglia))})",
                "top1_coerente": confermata,
                "motivazione_attese": prova["motivazione"],
                "vettore_modificato": v["vettore_modificato"],
            })
    return pd.DataFrame(righe)


def _nomi_strategie(ctx: SimpleNamespace, ids: Iterable[str]) -> List[str]:
    nomi = {x["id"]: x["name"] for x in ctx.DATA["strategies"]}
    return [nomi.get(i, i) for i in ids]


def tabella_e3(advisor: pd.DataFrame) -> pd.DataFrame:
    """E3 scenario per scenario: quanto e perche' il consiglio cambia col contesto."""
    e3 = advisor[advisor["esperimento"] == "E3"] if not advisor.empty else advisor
    if e3.empty:
        return pd.DataFrame()
    righe = []
    for scenario, g in e3.groupby("scenario_id"):
        primi = g[g["rank_advisor"] == 1]
        riferimento = primi[(primi["terreno"] == "Pianura") & (primi["meteo"] == "Sereno")
                            & (primi["stato_truppe"] == "Fresche")]
        strategia_rif = riferimento["strategia_id"].iloc[0] if len(riferimento) else primi["strategia_id"].mode()[0]
        compat_rif = g[g["strategia_id"] == strategia_rif]["compat_advisor_pct"]
        secondi = g[(g["rank_advisor"] == 2) & (g["terreno"] == "Pianura")
                    & (g["meteo"] == "Sereno") & (g["stato_truppe"] == "Fresche")]
        scarto = (riferimento["compat_advisor_pct"].iloc[0] - secondi["compat_advisor_pct"].iloc[0]
                  if len(riferimento) and len(secondi) else np.nan)
        cambi = int((primi["strategia_id"] != strategia_rif).sum())
        frequente = primi["strategia_nome"].value_counts()
        righe.append({
            "scenario_id": scenario, "descrizione": g["descrizione"].iloc[0],
            "contesti": len(primi), "strategie_distinte_in_testa": primi["strategia_id"].nunique(),
            "cambi_rispetto_al_riferimento": cambi, "cambi_pct": cambi / len(primi) * 100,
            "strategia_di_riferimento": riferimento["strategia_nome"].iloc[0] if len(riferimento) else "",
            "strategia_piu_frequente": frequente.index[0],
            "quota_piu_frequente_pct": frequente.iloc[0] / len(primi) * 100,
            "scarto_top1_top2_nel_riferimento_pt": scarto,
            "escursione_compat_riferimento_pt": compat_rif.max() - compat_rif.min(),
            "divergenza_media_pt": g["divergenza_pt"].abs().mean(),
            "divergenza_max_pt": g["divergenza_pt"].abs().max(),
            "classe": ("strategia in testa invariata" if primi["strategia_id"].nunique() == 1
                       else "sensibile al contesto"),
        })
    return pd.DataFrame(righe)


def tabella_controllo(principale: pd.DataFrame, controllo: pd.DataFrame,
                      ctx: SimpleNamespace) -> pd.DataFrame:
    """Confronto partita per partita fra il dataset principale e uno di controllo.

    Le partite si accoppiano per scenario, livello, condizione e replica, e si
    verifica che il seed sia lo stesso: le uniche differenze ammesse sono quelle
    dichiarate nei due manifest.
    """
    chiave = ["scenario_id", "difficolta_ia", "condizione", "replica"]
    p = principale[principale["esperimento"] == "E2"]
    c = controllo[controllo["esperimento"] == "E2"]
    colonne = chiave + ["seed", "esito", "vinta", "indecisa", "compat_combat_pct",
                        "forza_residua_pct", "rapporto_scambi"]
    unite = p[colonne].merge(c[colonne], on=chiave, suffixes=("_p", "_c"))
    diversi = (unite["seed_p"] != unite["seed_c"]).sum()
    if diversi:
        raise RuntimeError(f"{diversi} partite di controllo hanno un seed diverso dalla principale.")
    righe = []
    for livello in _livelli_presenti(unite):
        g = _livello(unite, livello)
        for condizione in CONDIZIONI_E2:
            x = g[g["condizione"] == condizione]
            if x.empty:
                continue
            cp, cc = x["compat_combat_pct_p"].mean(), x["compat_combat_pct_c"].mean()
            righe.append({
                "livello": livello, "condizione": condizione, "n": len(x),
                "win_rate_principale": x["vinta_p"].mean() * 100,
                "win_rate_controllo": x["vinta_c"].mean() * 100,
                "non_decise_principale_pct": x["indecisa_p"].mean() * 100,
                "non_decise_controllo_pct": x["indecisa_c"].mean() * 100,
                "forza_residua_principale": x["forza_residua_pct_p"].mean(),
                "forza_residua_controllo": x["forza_residua_pct_c"].mean(),
                "compat_combat_media_principale": cp, "compat_combat_media_controllo": cc,
                "fattore_medio_principale": float(np.mean([ctx.fattore(v / 100) for v in x["compat_combat_pct_p"]])),
                "fattore_medio_controllo": float(np.mean([ctx.fattore(v / 100) for v in x["compat_combat_pct_c"]])),
                "esiti_identici": int((x["esito_p"] == x["esito_c"]).sum()),
                "da_non_decisa_a_vittoria": int(((x["esito_p"] == "non_decisa") & (x["esito_c"] == "vittoria")).sum()),
                "da_non_decisa_a_sconfitta": int(((x["esito_p"] == "non_decisa") & (x["esito_c"] == "sconfitta")).sum()),
                "altri_cambi": int(((x["esito_p"] != x["esito_c"]) & (x["esito_p"] != "non_decisa")).sum()),
            })
        # la differenza prima meno ultima, appaiata, in ciascuno dei due dataset
        top = g[g["condizione"] == "top_ranked"].set_index(["scenario_id", "difficolta_ia", "replica"])
        worst = g[g["condizione"] == "worst_ranked"].set_index(["scenario_id", "difficolta_ia", "replica"])
        accoppiate = top.join(worst, lsuffix="_t", rsuffix="_w", how="inner")
        if accoppiate.empty:
            continue
        mp = mcnemar(accoppiate["vinta_p_t"], accoppiate["vinta_p_w"])
        mc = mcnemar(accoppiate["vinta_c_t"], accoppiate["vinta_c_w"])
        righe.append({
            "livello": livello, "condizione": "differenza più affine meno affine",
            "n": mp["n_coppie"],
            "win_rate_principale": mp["diff_appaiata_pct"], "win_rate_controllo": mc["diff_appaiata_pct"],
            "ic95_principale": f"[{mp['ic95_min']:.1f}; {mp['ic95_max']:.1f}]",
            "ic95_controllo": f"[{mc['ic95_min']:.1f}; {mc['ic95_max']:.1f}]",
            "coppie_discordanti_principale": f"{mp['solo_a_vince']} contro {mp['solo_b_vince']}",
            "coppie_discordanti_controllo": f"{mc['solo_a_vince']} contro {mc['solo_b_vince']}",
            "p_mcnemar_principale": mp["p_value_esatto"], "p_mcnemar_controllo": mc["p_value_esatto"],
        })
    return pd.DataFrame(righe)


#: Colonne che, per una partita gia' decisa, devono coincidere esattamente se il
#: codice di oggi riproduce quello che ha generato la campagna.
COLONNE_RIPRODUZIONE = ("esito", "perdite_proprie", "perdite_nemiche", "scontri", "unita_vive",
                        "unita_schierate_totali", "turni_giocati", "celle_player", "celle_ai",
                        "unita_ai_finali")


def verifica_riproducibilita(principale: pd.DataFrame,
                             controllo: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Le partite decise entro il limite della campagna devono essere identiche.

    Vale solo per un controllo con lo stesso peso e un limite di turni piu' alto:
    il gioco e' deterministico, quindi fino al limite della campagna le partite
    con lo stesso seed sono le stesse. Se coincidono colonna per colonna, il
    codice attuale riproduce quello che ha generato la campagna.
    """
    chiave = ["scenario_id", "difficolta_ia", "condizione", "replica"]
    p = principale[(principale["esperimento"] == "E2") & (principale["indecisa"] == 0)]
    c = controllo[controllo["esperimento"] == "E2"]
    unite = p[chiave + list(COLONNE_RIPRODUZIONE)].merge(
        c[chiave + list(COLONNE_RIPRODUZIONE)], on=chiave, suffixes=("_p", "_c"))
    if unite.empty:
        return None
    uguali = np.ones(len(unite), dtype=bool)
    for colonna in COLONNE_RIPRODUZIONE:
        uguali &= (unite[f"{colonna}_p"].astype(str) == unite[f"{colonna}_c"].astype(str)).to_numpy()
    return {"partite_confrontate": len(unite), "partite_identiche": int(uguali.sum()),
            "colonne": list(COLONNE_RIPRODUZIONE)}


# ══════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════

#: Palette validate con lo script del sistema di visualizzazione:
#: le tre condizioni sono ordinate (rank 1, 6, 10), quindi una sola tinta dal
#: scuro al chiaro, leggibile anche stampata in scala di grigi.
COLORE_CONDIZIONE = {"top_ranked": "#104281", "intermedia": "#2a78d6", "worst_ranked": "#86b6ef"}
#: Dottrine ed esiti hanno tinte proprie, diverse dai blu delle condizioni: nella
#: stessa figura un colore non deve mai significare due cose. Viola e arancio
#: passano tutti i controlli; verde e rosso stanno nella fascia 6-8 per i
#: daltonici, ammessa perche' ogni segmento ha anche il numero e una posizione
#: fissa (vittorie sotto, non decise in mezzo, sconfitte sopra).
COLORE_E4 = {"dottrine_on": "#6250d6", "dottrine_off": "#eb6834"}
COLORE_ESITO = {"vittoria": "#008300", "non_decisa": "#c3c2b7", "sconfitta": "#e34948"}
INCHIOSTRO, INCHIOSTRO_2, MUTO = "#0b0b0b", "#52514e", "#898781"
GRIGLIA, ASSE, SUPERFICIE = "#e1e0d9", "#c3c2b7", "#ffffff"


def _stile(plt: Any) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": ASSE, "axes.labelcolor": INCHIOSTRO_2, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        # griglia a linea continua sottile: il tratteggio si legge come soglia
        "grid.color": GRIGLIA, "grid.linestyle": "-", "grid.linewidth": 0.6,
        "xtick.color": MUTO, "ytick.color": MUTO, "xtick.labelcolor": INCHIOSTRO_2,
        "ytick.labelcolor": INCHIOSTRO_2, "text.color": INCHIOSTRO,
        "legend.frameon": False, "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _asse_percentuale(ax: Any, massimo: float = 100, passo: int = 20,
                      etichetta: str = "") -> None:
    """Asse in percentuale: tacche regolari, mai oltre 100."""
    tetto = min(100, max(passo, math.ceil(massimo / passo) * passo))
    ax.set_ylim(0, tetto)
    ax.set_yticks(range(0, tetto + 1, passo))
    ax.set_yticklabels([str(v) for v in range(0, tetto + 1, passo)])
    if etichetta:
        ax.set_ylabel(etichetta)


def _salva(fig: Any, cartella: Path, nome: str) -> List[str]:
    cartella.mkdir(parents=True, exist_ok=True)
    fig.savefig(cartella / f"{nome}.pdf", bbox_inches="tight")
    fig.savefig(cartella / f"{nome}.png", dpi=300, bbox_inches="tight")
    return [f"{nome}.pdf", f"{nome}.png"]


def figure(runs: pd.DataFrame, validi_e4: List[str], cartella: Path) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("   matplotlib non installato: figure saltate (pip install -r requirements.txt)")
        return []
    _stile(plt)
    prodotti: List[str] = []
    e2 = runs[runs["esperimento"] == "E2"]
    if not e2.empty:
        prodotti += _figura_esiti(plt, e2, cartella)
        prodotti += _figura_forza(plt, e2, cartella)
        prodotti += _figura_scenari(plt, e2, cartella)
    e4 = runs[(runs["esperimento"] == "E4") & (runs["scenario_id"].isin(validi_e4))]
    if not e4.empty:
        prodotti += _figura_e4(plt, e4, cartella)
    plt.close("all")
    return prodotti


def _figura_esiti(plt: Any, e2: pd.DataFrame, cartella: Path) -> List[str]:
    """A: tasso di vittoria con IC di Wilson. B: composizione dei tre esiti."""
    livelli = _livelli_presenti(e2)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.6, 3.0),
                                     gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.28})
    larghezza = 0.25
    massimo = 0
    for i, condizione in enumerate(CONDIZIONI_E2):
        valori, errori_lo, errori_hi = [], [], []
        for livello in livelli:
            g = _livello(e2, livello)
            g = g[g["condizione"] == condizione]
            p, lo, hi = wilson(int(g["vinta"].sum()), len(g))
            valori.append(p)
            errori_lo.append(p - lo)
            errori_hi.append(hi - p)
            massimo = max(massimo, hi)
        x = np.arange(len(livelli)) + (i - 1) * larghezza
        ax_a.bar(x, valori, larghezza * 0.9, color=COLORE_CONDIZIONE[condizione],
                 label=_nome_condizione(condizione, e2), zorder=2)
        ax_a.errorbar(x, valori, yerr=[errori_lo, errori_hi], fmt="none",
                      ecolor=INCHIOSTRO, elinewidth=0.8, capsize=2, zorder=3)
    ax_a.set_xticks(range(len(livelli)))
    ax_a.set_xticklabels([NOME_LIVELLO[l].replace("Tutti i livelli", "Tutti i\nlivelli")
                          for l in livelli], fontsize=8)
    ax_a.axvline(0.5, color=GRIGLIA, linewidth=0.8)
    ax_a.set_xlabel("Livello dell'IA")
    _asse_percentuale(ax_a, massimo + 6, 20, "Vittorie (%)")
    ax_a.set_title("A   Tasso di vittoria (IC 95%, Wilson)", loc="left", fontsize=9,
                   color=INCHIOSTRO)
    ax_a.legend(loc="upper left", fontsize=7.5, handlelength=1.2)

    # B: barre impilate al 100%, una per livello x condizione
    posizioni, etichette, gruppi = [], [], []
    passo = 0
    for livello in livelli:
        for condizione in CONDIZIONI_E2:
            posizioni.append(passo)
            rank = int(round(e2.loc[e2["condizione"] == condizione, "rank_advisor"].median()))
            etichette.append(f"r{rank}")
            passo += 1
        gruppi.append((posizioni[-3] + posizioni[-1]) / 2)
        passo += 0.6
    fondo = np.zeros(len(posizioni))
    for esito, colonna, nome in (("vittoria", "vinta", "Vittorie"),
                                 ("non_decisa", "indecisa", "Non decise"),
                                 ("sconfitta", "persa", "Sconfitte")):
        valori = []
        for livello in livelli:
            for condizione in CONDIZIONI_E2:
                g = _livello(e2, livello)
                valori.append(g[g["condizione"] == condizione][colonna].mean() * 100)
        valori = np.array(valori)
        ax_b.bar(posizioni, valori, 0.86, bottom=fondo, color=COLORE_ESITO[esito],
                 label=nome, edgecolor=SUPERFICIE, linewidth=0.8, zorder=2)
        for x, v, b in zip(posizioni, valori, fondo):
            if v >= 12:
                ax_b.text(x, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=6,
                          color="#ffffff" if esito != "non_decisa" else INCHIOSTRO)
        fondo += valori
    ax_b.set_xticks(posizioni)
    ax_b.set_xticklabels(etichette, fontsize=6.5)
    ax_b.axvline((posizioni[2] + posizioni[3]) / 2, color=GRIGLIA, linewidth=0.8)
    for centro, livello in zip(gruppi, livelli):
        ax_b.text(centro, -15, NOME_LIVELLO[livello], ha="center", va="top", fontsize=8,
                  color=INCHIOSTRO_2, transform=ax_b.transData, clip_on=False)
    _asse_percentuale(ax_b, 100, 20, "Partite (%)")
    ax_b.set_title("B   Composizione degli esiti", loc="left", fontsize=9, color=INCHIOSTRO)
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=7.5,
                handlelength=1.2)
    return _salva(fig, cartella, "fig1_e2_esiti")


def _figura_forza(plt: Any, e2: pd.DataFrame, cartella: Path) -> List[str]:
    """Boxplot della forza residua, un pannello per livello, con la media."""
    livelli = _livelli_presenti(e2)
    fig, assi = plt.subplots(1, len(livelli), figsize=(6.6, 2.7), sharey=True)
    for ax, livello in zip(np.atleast_1d(assi), livelli):
        g = _livello(e2, livello)
        dati = [g[g["condizione"] == c]["forza_residua_pct"].dropna() for c in CONDIZIONI_E2]
        scatole = ax.boxplot(dati, widths=0.6, patch_artist=True, showfliers=True,
                             medianprops={"color": INCHIOSTRO, "linewidth": 1.2},
                             whiskerprops={"color": MUTO, "linewidth": 0.8},
                             capprops={"color": MUTO, "linewidth": 0.8},
                             flierprops={"marker": "o", "markersize": 2, "markerfacecolor": MUTO,
                                         "markeredgecolor": "none", "alpha": 0.5})
        for scatola, condizione in zip(scatole["boxes"], CONDIZIONI_E2):
            scatola.set_facecolor(COLORE_CONDIZIONE[condizione])
            scatola.set_alpha(0.85)
            scatola.set_edgecolor(SUPERFICIE)
        medie = [d.mean() for d in dati]
        ax.scatter(range(1, 4), medie, marker="D", s=16, color="#ffffff",
                   edgecolor=INCHIOSTRO, linewidth=0.8, zorder=4)
        ax.set_xticks(range(1, 4))
        ranks = [int(round(e2.loc[e2["condizione"] == c, "rank_advisor"].median())) for c in CONDIZIONI_E2]
        ax.set_xticklabels([f"r{r}" for r in ranks])
        ax.set_title(NOME_LIVELLO[livello], fontsize=8.5, color=INCHIOSTRO)
        _asse_percentuale(ax, 100, 20, "Forza residua (%)" if livello == livelli[0] else "")
    fig.text(0.5, -0.04, "Strategia per rank dell'advisor (r1 più affine, r10 meno affine)",
             ha="center", fontsize=8, color=INCHIOSTRO_2)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=INCHIOSTRO, linewidth=1.2, label="Mediana"),
                        Line2D([], [], marker="D", linestyle="none", markersize=4.5,
                               markerfacecolor="#ffffff", markeredgecolor=INCHIOSTRO,
                               label="Media")],
               loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.5)
    fig.subplots_adjust(wspace=0.12)
    return _salva(fig, cartella, "fig2_e2_forza_residua")


def _figura_scenari(plt: Any, e2: pd.DataFrame, cartella: Path) -> List[str]:
    """Slopegraph: una linea per scenario, il confronto avviene dentro lo scenario."""
    ordine = ["worst_ranked", "intermedia", "top_ranked"]
    fig, assi = plt.subplots(1, 2, figsize=(6.6, 3.1), gridspec_kw={"wspace": 0.42})
    for ax, (colonna, scala, etichetta) in zip(
            assi, (("vinta", 100, "Vittorie (%)"), ("forza_residua_pct", 1, "Forza residua (%)"))):
        medie = (e2.groupby(["scenario_id", "condizione"])[colonna].mean().unstack() * scala)[ordine]
        for scenario, riga in medie.iterrows():
            ax.plot(range(3), riga.values, color="#b8b6ad", linewidth=0.9, zorder=2)
        # etichette degli scenari a destra, distanziate per non sovrapporsi
        finali = medie[ordine[-1]].sort_values()
        posizioni, ultimo = {}, -1e9
        for scenario, valore in finali.items():
            y = max(valore, ultimo + 3.4)
            posizioni[scenario] = y
            ultimo = y
        for scenario, y in posizioni.items():
            # l'etichetta puo' essere spostata per non sovrapporsi: un filo la
            # collega al punto vero, cosi' non indica mai un valore sbagliato
            ax.plot([2.0, 2.07], [finali[scenario], y], color=ASSE, linewidth=0.6, zorder=1)
            ax.text(2.09, y, scenario, fontsize=6, color=MUTO, va="center")
        r = [media_ic(medie[c]) for c in ordine]
        centro = [x["media"] for x in r]
        ax.fill_between(range(3), [x["ic95_min"] for x in r], [x["ic95_max"] for x in r],
                        color=COLORE_CONDIZIONE["intermedia"], alpha=0.15, linewidth=0, zorder=1)
        ax.plot(range(3), centro, color=COLORE_CONDIZIONE["top_ranked"], linewidth=2,
                marker="o", markersize=5, zorder=3, label="Media degli 11 scenari (IC 95%)")
        ax.set_xticks(range(3))
        ranks = {c: int(round(e2.loc[e2["condizione"] == c, "rank_advisor"].median())) for c in ordine}
        ax.set_xticklabels([f"Meno affine\n(r{ranks[ordine[0]]})", f"Intermedia\n(r{ranks[ordine[1]]})",
                            f"Più affine\n(r{ranks[ordine[2]]})"], fontsize=7.5)
        ax.set_xlim(-0.15, 2.3)
        massimo = max(medie.max().max(), max(x["ic95_max"] for x in r))
        _asse_percentuale(ax, massimo + 5, 20, etichetta)
    assi[0].set_title("A   Vittorie, tutti i livelli", loc="left", fontsize=9)
    assi[1].set_title("B   Forza residua, tutti i livelli", loc="left", fontsize=9)
    assi[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.2), fontsize=7.5, handlelength=1.6)
    return _salva(fig, cartella, "fig3_e2_per_scenario")


def _figura_e4(plt: Any, e4: pd.DataFrame, cartella: Path) -> List[str]:
    """Ablation delle dottrine sui soli scenari validi, piu' l'aggregato."""
    scenari = sorted(e4["scenario_id"].unique())
    gruppi = scenari + ["Aggregato"]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    larghezza = 0.36
    massimo = 0
    for i, condizione in enumerate(COPPIA_E4):
        valori, lo_e, hi_e = [], [], []
        for gruppo in gruppi:
            g = e4 if gruppo == "Aggregato" else e4[e4["scenario_id"] == gruppo]
            g = g[g["condizione"] == condizione]
            p, lo, hi = wilson(int(g["vinta"].sum()), len(g))
            valori.append(p)
            lo_e.append(p - lo)
            hi_e.append(hi - p)
            massimo = max(massimo, hi)
        x = np.arange(len(gruppi)) + (i - 0.5) * larghezza
        ax.bar(x, valori, larghezza * 0.92, color=COLORE_E4[condizione],
               label=_nome_condizione(condizione, e4), zorder=2)
        ax.errorbar(x, valori, yerr=[lo_e, hi_e], fmt="none", ecolor=INCHIOSTRO,
                    elinewidth=0.8, capsize=2, zorder=3)
    # differenza appaiata sopra ogni gruppo: e' il numero che conta
    for j, gruppo in enumerate(gruppi):
        g = e4 if gruppo == "Aggregato" else e4[e4["scenario_id"] == gruppo]
        m = mcnemar(*[_appaia(g, *COPPIA_E4)[c] for c in ("vinta_a", "vinta_b")])
        ax.text(j, massimo + 3, f"{_num(m['diff_appaiata_pct'], 1, True)}", ha="center",
                fontsize=7.5, color=INCHIOSTRO_2)
    nomi = []
    for gruppo in gruppi:
        if gruppo == "Aggregato":
            nomi.append(f"Aggregato\n({len(scenari)} scenari)")
        else:
            nome = e4[e4["scenario_id"] == gruppo]["strategia_nome"].iloc[0]
            nomi.append(f"{gruppo}\n" + "\n".join(textwrap.wrap(nome, 11)))
    ax.set_xticks(range(len(gruppi)))
    ax.set_xticklabels(nomi, fontsize=7.5)
    ax.axvline(len(gruppi) - 1.5, color=GRIGLIA, linewidth=0.8)
    _asse_percentuale(ax, massimo + 10, 20, "Vittorie (%)")
    ax.legend(loc="upper left", fontsize=7.5, ncol=2, handlelength=1.2)
    ax.text(len(gruppi) - 0.5, massimo + 9, "differenza appaiata\n(punti %)", ha="right",
            va="bottom", fontsize=6.5, color=MUTO)
    return _salva(fig, cartella, "fig4_e4_dottrine")


# ══════════════════════════════════════════════════════════════════
# RIEPILOGO
# ══════════════════════════════════════════════════════════════════

def _riga(df: pd.DataFrame, **filtri: Any) -> pd.Series:
    sel = df
    for k, v in filtri.items():
        sel = sel[sel[k] == v]
    return sel.iloc[0] if len(sel) else pd.Series(dtype=object)


def _costanti_ia() -> Dict[str, Dict[str, Any]]:
    """Le costanti di aggressione dei profili IA, lette dal codice del gioco."""
    try:
        from gamecore.session.ai_core import (ai_easy_difficulty, ai_hard_difficulty,
                                              ai_normal_difficulty)
    except ImportError:
        return {}
    return {livello: {nome: getattr(modulo, nome, None)
                      for nome in ("CASTLE_PUSH_FROM_TURN", "CASTLE_PUSH_MIN_UNITS",
                                   "CASTLE_PUSH_CHANCE")}
            for livello, modulo in (("easy", ai_easy_difficulty), ("normal", ai_normal_difficulty),
                                    ("hard", ai_hard_difficulty))}


def _peso(valore: Any) -> str:
    """0,25 / 0 / 0,5: il peso dell'affinita' in formato italiano."""
    try:
        x = float(valore)
    except (TypeError, ValueError):
        return "n.d."
    return str(int(x)) if x == int(x) else _num(x, 2).rstrip("0")


def _lettura(testo: str) -> str:
    return f"*Lettura.* {testo[:1].upper()}{testo[1:]}"


def scrivi_riepilogo(percorso: Path, ds: SimpleNamespace, t: Dict[str, pd.DataFrame],
                     controlli: List[Tuple[str, SimpleNamespace, pd.DataFrame]],
                     figure_prodotte: List[str], validi_e4: List[str]) -> None:
    """Il documento con i numeri della tesi. Ogni numero viene da una tabella.

    I dati e le interpretazioni stanno separati: ogni paragrafo interpretativo
    comincia con "Lettura" e si appoggia solo a numeri gia' riportati sopra.
    """
    m = ds.manifest
    prov = m.get("provenienza_codice") or {}
    runs = ds.runs
    e2 = runs[runs["esperimento"] == "E2"]
    peso_p = _parametro(m, "combat_affinity_weight")
    turni_p = _parametro(m, "turni_massimi")
    nomi_metriche = {c: e for c, e, _ in METRICHE}

    # che cosa varia in ciascun controllo rispetto al dataset principale
    controlli_peso, controlli_turni = [], []
    for nome, controllo, tab in controlli:
        peso_c = _parametro(controllo.manifest, "combat_affinity_weight")
        turni_c = _parametro(controllo.manifest, "turni_massimi")
        if peso_c is not None and peso_p is not None and float(peso_c) != float(peso_p):
            controlli_peso.append((nome, controllo, tab, peso_c))
        elif turni_c is not None and turni_p is not None and int(turni_c) != int(turni_p):
            controlli_turni.append((nome, controllo, tab, int(turni_c)))

    out: List[str] = []
    w = out.append

    w("# Riepilogo dei dati per la tesi")
    w("")
    w("Documento generato da `scripts/csv_batch_test.py --analizza`. Ogni numero è calcolato dai "
      "file del dataset indicato sotto; le tabelle complete sono in `tabelle/`, le figure in "
      "`figure/`. Non va modificato a mano: se i dati cambiano, si rigenera. I paragrafi che "
      "cominciano con *Lettura* sono interpretazioni e si appoggiano solo ai dati riportati prima.")
    w("")

    # ── dataset ────────────────────────────────────────────────────
    w("## Dataset")
    w("")
    w(f"- Cartella: `{ds.cartella.name}`; generato il {m.get('generato', 'n.d.')}")
    w(f"- Comando: `{m.get('comando', 'n.d.')}`")
    w(f"- Partite: {len(runs)} (E2: {len(e2)}, E4: {int((runs['esperimento'] == 'E4').sum())}); "
      f"righe advisor: {len(ds.advisor)}")
    w(f"- combat_affinity_weight {_peso(peso_p)}; limite di {turni_p} turni; "
      f"{_parametro(m, 'repliche')} repliche")
    w(f"- Commit registrato nel manifest: `{str(m.get('commit', 'n.d.'))[:7]}`")
    if prov:
        stato = ("codice identico al commit" if prov.get("codice_identico_al_commit")
                 else "con modifiche non committate a " + ", ".join(prov.get("file_modificati_non_committati", [])))
        w(f"- Provenienza del codice: {stato}; impronta dello script `{prov.get('impronta_script_sha256')}`")
    else:
        w("- Provenienza del codice: il commit registrato non contiene ancora "
          "`scripts/csv_batch_test.py`; la campagna è stata generata da codice non ancora "
          "committato, e il manifest di allora non lo segnalava.")
    for nome, controllo, _, turni_c in controlli_turni:
        verifica = verifica_riproducibilita(runs, controllo.runs)
        if not verifica:
            continue
        commit_c = str(controllo.manifest.get("commit", "n.d."))[:7]
        w(f"- Riproducibilità verificata sui dati: il dataset `{nome}`, generato con il codice "
          f"del commit `{commit_c}` e con gli stessi seed, contiene "
          f"{verifica['partite_confrontate']} partite che nella campagna si erano decise entro "
          f"{turni_p} turni; **{verifica['partite_identiche']} su "
          f"{verifica['partite_confrontate']} sono identiche** in tutte le colonne confrontate "
          f"({', '.join(verifica['colonne'])}).")
    w("")

    # ── 1. ipotesi primaria ────────────────────────────────────────
    w("## 1. Ipotesi primaria (E2)")
    w("")
    w("Ipotesi dichiarata prima dell'analisi: **la strategia più affine (rank 1) vince più spesso "
      "della meno affine (rank 10), su tutti i livelli dell'IA**. È l'unico confronto confermativo; "
      "gli altri sono esplorativi e riportano il p-value corretto con il metodo di Holm.")
    w("")
    app = t["appaiata_vittorie"]
    prim = _riga(app, esperimento="E2", livello="tutte", condizione_a="top_ranked",
                 condizione_b="worst_ranked")
    if len(prim):
        w(f"- Coppie appaiate (stesso seed): {int(prim['n_coppie'])}")
        w(f"- Coppie discordanti: **{int(prim['solo_a_vince'])}** vinte solo dalla più affine, "
          f"**{int(prim['solo_b_vince'])}** vinte solo dalla meno affine")
        w(f"- Test di McNemar esatto: {_formatta_p(prim['p_value_esatto'])}; "
          f"χ² con correzione di continuità {_num(prim['mcnemar_chi2_corretto'], 1)}")
        w(f"- Differenza appaiata: {_num(prim['diff_appaiata_pct'], 1, True)} punti percentuali, "
          f"IC 95% [{_num(prim['ic95_min'], 1)}; {_num(prim['ic95_max'], 1)}]")
        w(f"- Correlazione fra gli esiti delle due condizioni sullo stesso seed: "
          f"φ = {_num(prim['correlazione_phi'], 2)}")
    sint = t["per_scenario_sintesi"]
    ps = _riga(sint, livello="tutte", metrica="win_rate")
    if len(ps):
        w("")
        w("Con **lo scenario come unità statistica**, per generalizzare agli scenari e non alle "
          "singole partite:")
        w("")
        w(f"- Differenza positiva in **{int(ps['scenari_positivi'])} scenari su {int(ps['n_scenari'])}**, "
          f"nulla in {int(ps['scenari_nulli'])} ({ps['scenari_nulli_elenco'] or 'nessuno'}), "
          f"negativa in {int(ps['scenari_negativi'])}")
        w(f"- Media delle differenze {_num(ps['media_delta'], 1, True)} punti; deviazione standard "
          f"fra scenari {_num(ps['dev_std_fra_scenari'], 1)}")
        w(f"- Errore standard fra scenari **{_num(ps['err_std_per_scenario'], 2)}** punti; trattando "
          f"le partite come indipendenti risultava {_num(ps['err_std_partite_indipendenti'], 2)}")
        w(f"- t({int(ps['gl'])}) = {_num(ps['t'], 2)}, {_formatta_p(ps['p_value'])}; IC 95% "
          f"[{_num(ps['ic95_min'], 1)}; {_num(ps['ic95_max'], 1)}] con quantile t = "
          f"{_num(ps['quantile_t_0975'], 3)}")
        w("")
        w("I due intervalli rispondono a domande diverse: quello sulle partite appaiate riguarda "
          "nuove partite degli stessi 11 scenari; quello sugli scenari riguarda scenari simili a "
          "questi 11. Nel capitolo va dichiarato a quale popolazione si riferisce ciascuno.")
    w("")

    # ── 2. che cosa misura E2 ──────────────────────────────────────
    w("## 2. Che cosa misura E2")
    w("")
    w("La compatibilità che ordina la classifica dell'advisor e quella che fissa il "
      "moltiplicatore di forza in combattimento (da 0,30 a 2,35) derivano dalla stessa distanza "
      "euclidea fra il vettore dell'esercito e quello della strategia: una strategia in cima alla "
      "classifica riceve, per costruzione, un moltiplicatore più alto.")
    w("")
    for nome, controllo, tab, peso_c in controlli_peso:
        d = _riga(tab, livello="tutte", condizione="differenza più affine meno affine")
        w(f"**Esperimento di controllo** `{nome}`: combat_affinity_weight {_peso(peso_c)} invece di "
          f"{_peso(peso_p)}, stessi seed.")
        w("")
        if len(d):
            w(f"- Differenza appaiata più affine meno affine: {_num(d['win_rate_principale'], 1, True)} "
              f"punti con peso {_peso(peso_p)}; {_num(d['win_rate_controllo'], 1, True)} punti con "
              f"peso {_peso(peso_c)}")
            w(f"- Coppie discordanti: {d['coppie_discordanti_principale']} con peso {_peso(peso_p)}; "
              f"{d['coppie_discordanti_controllo']} con peso {_peso(peso_c)} (McNemar "
              f"{_formatta_p(d['p_mcnemar_controllo'])})")
        fattori = tab[tab["livello"].eq("tutte") & tab["condizione"].isin(CONDIZIONI_E2)]
        if len(fattori):
            testo = "; ".join(
                f"{_nome_condizione(r['condizione'], e2)} da {_num(r['fattore_medio_principale'], 2)} "
                f"a {_num(r['fattore_medio_controllo'], 2)}" for _, r in fattori.iterrows())
            w(f"- Moltiplicatore di forza medio: {testo}")
            w(f"- Partite con esito identico nei due dataset: {int(fattori['esiti_identici'].sum())} "
              f"su {int(fattori['n'].sum())}")
        w("")
        w(_lettura(
            "azzerare il peso rimuove solo lo strato di affinità unità-ambiente; la distanza "
            "euclidea, che lega la classifica al moltiplicatore, resta la stessa. Il moltiplicatore "
            "delle tre strategie cambia di pochi centesimi e il loro ordine non cambia, quindi il "
            "gradiente non poteva sparire. Che resti **non** dimostra che l'advisor colga qualcosa "
            "oltre al proprio moltiplicatore: dimostra che il gradiente non dipende dallo strato di "
            "affinità. E2 va quindi presentato come **verifica di coerenza fra advisor e motore di "
            "gioco**. Separare davvero le due cose richiederebbe di neutralizzare il moltiplicatore "
            "della strategia del giocatore; l'unico canale rimasto sarebbero gli effetti "
            "dottrinali, il cui effetto in E4 non è stato rilevato."))
        w("")

    # ── 3. esplorativi ─────────────────────────────────────────────
    w("## 3. Confronti esplorativi di E2")
    w("")
    w("Tutti appaiati sullo stesso seed: McNemar esatto per le vittorie, t appaiato per le metriche "
      "continue; p corretto con Holm sull'insieme dei confronti della tabella.")
    w("")
    w("| Livello | Confronto | Coppie discordanti | Differenza (punti %) | IC 95% | p (Holm) |")
    w("|---|---|---|---|---|---|")
    app_e2 = app[app["esperimento"] == "E2"]
    for _, r in app_e2.iterrows():
        w(f"| {NOME_LIVELLO[r['livello']]} | {_nome_condizione(r['condizione_a'], e2)} contro "
          f"{_nome_condizione(r['condizione_b'], e2)} | {int(r['solo_a_vince'])} contro "
          f"{int(r['solo_b_vince'])} | {_num(r['diff_appaiata_pct'], 1, True)} | "
          f"[{_num(r['ic95_min'], 1)}; {_num(r['ic95_max'], 1)}] | {_formatta_p(r['p_holm'])} |")
    w("")
    non_sig = app_e2[app_e2["p_holm"] >= 0.05]
    if len(non_sig):
        w("Confronti non significativi dopo la correzione: " + "; ".join(
            f"{NOME_LIVELLO[r['livello']].lower()}, rank {int(round(e2.loc[e2.condizione == r['condizione_a'], 'rank_advisor'].median()))} "
            f"contro rank {int(round(e2.loc[e2.condizione == r['condizione_b'], 'rank_advisor'].median()))} "
            f"({_formatta_p(r['p_holm'])})" for _, r in non_sig.iterrows()) + ".")
        w("")
    met = t["appaiata_metriche"]
    hard = met[(met["esperimento"] == "E2") & (met["livello"] == "hard")
               & (met["metrica"].isin(["forza_residua_pct", "rapporto_scambi"]))]
    if len(hard):
        vinte = e2[e2["difficolta_ia"] == "hard"].groupby("condizione")["vinta"].agg(["sum", "count"])
        w("Contro l'IA difficile il tasso di vittoria ha un effetto pavimento: "
          + ", ".join(f"{_nome_condizione(c, e2).split(' (')[0].lower()} {int(vinte.loc[c, 'sum'])} su "
                      f"{int(vinte.loc[c, 'count'])}" for c in CONDIZIONI_E2 if c in vinte.index)
          + ". Lì la metrica principale è la forza residua, affiancata dal rapporto di scambio:")
        w("")
        w("| Confronto | Metrica | Differenza media appaiata | IC 95% | p (Holm) |")
        w("|---|---|---|---|---|")
        for _, r in hard.iterrows():
            w(f"| {_nome_condizione(r['condizione_a'], e2)} contro {_nome_condizione(r['condizione_b'], e2)} "
              f"| {nomi_metriche[r['metrica']]} | {_num(r['media_differenze'], 1, True)} "
              f"| [{_num(r['ic95_min'], 1)}; {_num(r['ic95_max'], 1)}] | {_formatta_p(r['p_holm'])} |")
        w("")

    # ── 4. E4 ──────────────────────────────────────────────────────
    e4t = t["e4"]
    if not e4t.empty:
        w("## 4. E4: dottrine (ablation)")
        w("")
        esclusi = e4t[(e4t["tipo"] == "scenario") & (~e4t["incluso"].astype(bool))]
        w(f"Criterio di inclusione: con la dottrina accesa, l'effetto deve essere attivo in media in "
          f"almeno {_num(SOGLIA_ATTIVAZIONE_E4, 0)} scontro per partita. Scenari validi: "
          f"**{', '.join(validi_e4)}**. Escluso: "
          + ("; ".join(f"{r['scenario_id']} ({r['motivo']})" for _, r in esclusi.iterrows()) or "nessuno")
          + ". Gli scenari senza alcuna dottrina attivabile dal loro esercito erano già esclusi in "
            "fase di generazione.")
        w("")
        w("| Scenario | Dottrina | Attivazione media | Vittorie, attiva | Vittorie, disattivata | "
          "Differenza appaiata | IC 95% | McNemar | p (Holm fra scenari) |")
        w("|---|---|---|---|---|---|---|---|---|")
        for _, r in e4t.iterrows():
            if r["tipo"] == "aggregato" and r["livello"] != "tutte":
                continue
            nome = r["scenario_id"] if r["tipo"] == "scenario" else f"Aggregato ({len(validi_e4)} scenari)"
            holm_txt = (_formatta_p(r["p_holm_fra_scenari"])
                        if r.get("p_holm_fra_scenari") == r.get("p_holm_fra_scenari") else "-")
            w(f"| {nome}{'' if r['incluso'] else ' (escluso)'} | {r['strategia'] or '-'} | "
              f"{_num(r['attivazione_media'], 2) if r['tipo'] == 'scenario' else '-'} | "
              f"{_num(r['win_rate_on'], 1)}% | {_num(r['win_rate_off'], 1)}% | "
              f"{_num(r['diff_appaiata_pct'], 1, True)} | [{_num(r['ic95_diff_min'], 1)}; "
              f"{_num(r['ic95_diff_max'], 1)}] | {_formatta_p(r['p_mcnemar_esatto'])} | {holm_txt} |")
        w("")
        agg = _riga(e4t, tipo="aggregato", livello="tutte")
        if len(agg):
            w(f"Sull'aggregato dei {len(validi_e4)} scenari validi la differenza è "
              f"{_num(agg['diff_appaiata_pct'], 1, True)} punti, IC 95% [{_num(agg['ic95_diff_min'], 1)}; "
              f"{_num(agg['ic95_diff_max'], 1)}], {_formatta_p(agg['p_mcnemar_esatto'])}. L'effetto "
              f"minimo rilevabile all'80% di potenza (alfa 0,05) è di circa "
              f"{_num(agg['effetto_minimo_rilevabile_80'], 1)} punti.")
            w("")
            w(_lettura(
                "l'effetto delle dottrine **non è stato rilevato con questa potenza**: l'intervallo è "
                "compatibile sia con nessun effetto sia con un effetto di qualche punto, quindi non si "
                "può scrivere \"nessun effetto\". Il dato aggregato nasconde un'eterogeneità forte fra "
                "scenari, con differenze di segno opposto; dopo la correzione per i confronti fra "
                "scenari, nessuna singola differenza resta sotto la soglia di 0,05."
                if (e4t.loc[(e4t["tipo"] == "scenario") & e4t["incluso"].astype(bool), "p_holm_fra_scenari"] >= 0.05).all()
                else
                "l'effetto delle dottrine **non è stato rilevato con questa potenza** sull'aggregato; "
                "il dato aggregato nasconde un'eterogeneità forte fra scenari, visibile nella tabella."))
            w("")

    # ── 5. esiti e anomalie ────────────────────────────────────────
    w("## 5. Partite non decise e differenze fra i livelli dell'IA")
    w("")
    liv = t["esiti_per_livello"]
    w("Le partite non decise sono un esito a sé, riportato accanto a vittorie e sconfitte:")
    w("")
    w("| Livello | Vittorie | Sconfitte | Non decise | Vittorie sulle sole decise | Primo scontro (turno medio) |")
    w("|---|---|---|---|---|---|")
    for _, r in liv.iterrows():
        w(f"| {NOME_LIVELLO[r['livello']]} | {_num(r['vittorie_pct'], 1)}% | {_num(r['sconfitte_pct'], 1)}% | "
          f"{_num(r['non_decise_pct'], 1)}% | {_num(r['win_rate_decise_pct'], 1)}% | "
          f"{_num(r['primo_scontro_medio'], 1)} |")
    w("")
    w("Stato a fine partita nelle sole partite **non decise**:")
    w("")
    w("| Livello | Partite | Unità del giocatore | Celle del giocatore | Celle dell'IA | "
      "Castello del giocatore | Castello dell'IA |")
    w("|---|---|---|---|---|---|---|")
    for _, r in liv[liv["livello"] != "tutte"].iterrows():
        w(f"| {NOME_LIVELLO[r['livello']]} | {int(r['non_decise_n'])} | "
          f"{_num(r['non_decise_unita_giocatore_vive'], 1)} | {_num(r['non_decise_celle_giocatore_medie'], 1)} | "
          f"{_num(r['non_decise_celle_ia_medie'], 1)} | {_num(r['non_decise_hp_castello_giocatore_pct'], 1)}% | "
          f"{_num(r['non_decise_hp_castello_ia_pct'], 1)}% |")
    w("")
    easy, normal = _riga(liv, livello="easy"), _riga(liv, livello="normal")
    conversioni = None
    for nome, controllo, tab, turni_c in controlli_turni:
        righe_easy = tab[(tab["livello"] == "easy") & tab["condizione"].isin(CONDIZIONI_E2)]
        if len(righe_easy):
            conversioni = (turni_c, int(righe_easy["da_non_decisa_a_vittoria"].sum()),
                           int(righe_easy["da_non_decisa_a_sconfitta"].sum()))
    if len(easy) and len(normal):
        costanti = _costanti_ia()
        w(f"Contro l'IA facile il giocatore vince meno che contro la normale: "
          f"{_num(easy['vittorie_pct'], 1)}% contro {_num(normal['vittorie_pct'], 1)}% in assoluto, "
          f"{_num(easy['win_rate_decise_pct'], 1)}% contro {_num(normal['win_rate_decise_pct'], 1)}% "
          f"sulle sole partite decise. Contro la facile il primo scontro arriva prima (turno "
          f"{_num(easy['primo_scontro_medio'], 0)} contro {_num(normal['primo_scontro_medio'], 0)}) e "
          f"{_num(easy['non_decise_pct'], 1)}% delle partite non si decide entro {turni_p} turni.")
        if conversioni:
            w(f"Portando il limite a {conversioni[0]} turni, delle partite non decise contro la "
              f"facile {conversioni[1]} diventano vittorie e {conversioni[2]} sconfitte.")
        w("")
        if costanti and all(costanti.get(l) for l in ("easy", "normal", "hard")):
            def descr(l: str) -> str:
                c = costanti[l]
                return (f"{NOME_LIVELLO[l].lower()} dal turno {c['CASTLE_PUSH_FROM_TURN']} con almeno "
                        f"{c['CASTLE_PUSH_MIN_UNITS']} unità e probabilità "
                        f"{_num(c['CASTLE_PUSH_CHANCE'], 2)}")
            w(_lettura(
                "i dati non indicano un'IA passiva: nelle partite non decise contro la facile il "
                "giocatore ha perso quasi tutto e il castello dell'IA è quasi intatto, e con più "
                "turni quelle partite finiscono per lo più in sconfitta. Sono sconfitte lente. Il "
                "codice dei profili IA è coerente con questo: la facile è quella che attacca il "
                f"castello avversario prima e con meno truppe ({descr('easy')}; {descr('normal')}; "
                f"{descr('hard')}). Nel codice è descritta come un profilo avventato i cui eserciti "
                "piccoli sono comodi da distruggere; lo sono per un giocatore umano che difende, non "
                "per il giocatore automatico, che non presidia mai il proprio castello. La differenza "
                "fra i livelli dipende quindi anche dalla politica del giocatore automatico, e va "
                "dichiarata come limite; per lo stesso motivo l'asse dei livelli non va letto come "
                "una scala di durezza crescente."))
            w("")
    w("**Il turno 401.** La colonna `turni_totali` registrava il numero della mappa *dopo* l'ultimo "
      "turno: la mappa parte dal turno 1 e `execute_turn` lo incrementa alla fine di ogni turno, anche "
      "di quello in cui cade un castello. Il valore era quindi turni giocati + 1 in **tutte** le "
      "partite, non solo nelle non decise, che mostravano 401 con limite 400. Nessuna partita ha "
      "superato il limite: le non decise hanno giocato esattamente 400 turni. Le differenze fra "
      "condizioni non cambiano, perché lo scarto è costante. Lo script ora registra anche "
      "`turni_giocati`" + ("; per questo dataset è ricavata come `turni_totali` meno 1."
                           if ds.turni_derivati else "."))
    w("")
    for nome, controllo, tab, turni_c in controlli_turni:
        repliche_c = _parametro(controllo.manifest, "repliche")
        w(f"**Prova di sensibilità sul limite di turni** `{nome}`: {turni_c} turni invece di "
          f"{turni_p}, {repliche_c} repliche (da 0 a {int(repliche_c) - 1}), stessi seed. Le "
          f"colonne a {turni_p} turni sono calcolate sulle stesse partite, non sull'intero dataset.")
        w("")
        w(f"| Livello | Condizione | Partite | Non decise a {turni_p} | Non decise a {turni_c} | "
          f"Diventate vittorie | Diventate sconfitte | Vittorie a {turni_p} | Vittorie a {turni_c} |")
        w("|---|---|---|---|---|---|---|---|---|")
        for _, r in tab[tab["condizione"].isin(CONDIZIONI_E2)].iterrows():
            w(f"| {NOME_LIVELLO[r['livello']]} | {_nome_condizione(r['condizione'], e2)} | {int(r['n'])} | "
              f"{_num(r['non_decise_principale_pct'], 1)}% | {_num(r['non_decise_controllo_pct'], 1)}% | "
              f"{int(r['da_non_decisa_a_vittoria'])} | {int(r['da_non_decisa_a_sconfitta'])} | "
              f"{_num(r['win_rate_principale'], 1)}% | {_num(r['win_rate_controllo'], 1)}% |")
        w("")
        cambi = int(tab[tab["condizione"].isin(CONDIZIONI_E2) & tab["livello"].eq("tutte")]["altri_cambi"].sum())
        d = _riga(tab, livello="tutte", condizione="differenza più affine meno affine")
        frase = (f"Le partite già decise entro {turni_p} turni hanno lo stesso esito anche con "
                 f"{turni_c}: i cambi di esito che non vengono da partite non decise sono {cambi}. "
                 f"Il gioco è deterministico, quindi i primi {turni_p} turni sono identici.")
        if len(d):
            frase += (f" La differenza appaiata più affine meno affine passa da "
                      f"{_num(d['win_rate_principale'], 1, True)} a {_num(d['win_rate_controllo'], 1, True)} "
                      f"punti (coppie discordanti {d['coppie_discordanti_principale']} e "
                      f"{d['coppie_discordanti_controllo']}): il gradiente non dipende dal limite di turni.")
        w(frase)
        w("")

    # ── 6. forza residua e durata ──────────────────────────────────
    dur = t["durata"]
    w("## 6. Forza residua e durata")
    w("")
    w("La forza residua è superstiti diviso unità mai schierate: le partite lunghe, con più "
      "reclutamenti, hanno un denominatore più grande e quindi una forza residua più bassa a parità "
      "di esito. Per questo è affiancata dal **rapporto di scambio**, perdite inflitte diviso perdite "
      "totali, che non dipende dal numero di unità reclutate.")
    w("")
    for _, r in dur[dur["livello"] == "tutte"].iterrows():
        w(f"- Correlazione con i turni giocati, {nomi_metriche[r['metrica']].lower()}: "
          f"r = {_num(r['correlazione_con_turni_giocati'], 2)}")
    w("")
    w("Differenza più affine meno affine con lo scenario come unità (t con 10 gradi di libertà):")
    w("")
    w("| Metrica | Livello | Differenza media | Scenari positivi | t | p |")
    w("|---|---|---|---|---|---|")
    for metrica in ("forza_residua_pct", "rapporto_scambi"):
        for _, r in sint[sint["metrica"] == metrica].iterrows():
            w(f"| {nomi_metriche[metrica]} | {NOME_LIVELLO[r['livello']]} | {_num(r['media_delta'], 1, True)} | "
              f"{int(r['scenari_positivi'])} su {int(r['n_scenari'])} | {_num(r['t'], 2)} | "
              f"{_formatta_p(r['p_value'])} |")
    w("")

    # ── 7. E1 ──────────────────────────────────────────────────────
    e1 = t["e1"]
    if not e1.empty:
        sc = e1[e1["tipo"] == "scenario"]
        coerenti = int(sc["top1_coerente"].apply(lambda v: v is True).sum())
        con_attese = int((sc["attese"] != "").sum())
        w("## 7. E1: sanity check e contro-prove")
        w("")
        w(f"Strategia in testa coerente con le attese in **{coerenti} scenari su {con_attese}** con "
          "attesa dichiarata. Le attese sono state fissate dallo sperimentatore; ognuna è motivata "
          "nella tabella `e1.csv` (colonna `motivazione_attese`) a partire dalla descrizione dichiarata "
          "delle unità e dai modificatori. Restano attese dello sperimentatore: una validazione da "
          "parte di un esperto esterno le renderebbe indipendenti.")
        w("")
        s02, s03 = _riga(e1, tipo="scenario", id="S02"), _riga(e1, tipo="scenario", id="S03")
        if len(s02) and len(s03):
            w(f"**S03 non è una contro-prova.** Prevede la stessa risposta di S02 e la ottiene "
              f"({s03['top1']}); cambia solo la compatibilità, da {_num(s02['top1_compat_pct'], 1)}% "
              f"a {_num(s03['top1_compat_pct'], 1)}% ({_num(s03['top1_compat_pct'] - s02['top1_compat_pct'], 1, True)} "
              "punti). La furtività di base degli assassini è 0,95 e i modificatori di quei contesti "
              "possono solo aumentarla: S03 misura la sensibilità della compatibilità, non quella "
              "della raccomandazione.")
            w("")
        cp = e1[e1["tipo"].str.startswith("contro-prova")]
        if len(cp):
            esiti_prove = {}
            for pid, g in cp.groupby("id", sort=True):
                base = g[g["tipo"].str.endswith("base")].iloc[0]
                var = g[g["tipo"].str.endswith("variante")].iloc[0]
                if not base["top1_coerente"]:
                    verdetto = "non informativa"
                elif var["top1_coerente"]:
                    verdetto = "confermata"
                else:
                    verdetto = "fallita"
                esiti_prove[pid] = (verdetto, base, var)
            w("**Contro-prove generate da una regola.** Non sono scelte a mano: per ogni tipo di unità "
              "si prende l'esercito puro, il suo attributo più alto, e ogni modificatore di terreno o "
              "meteo che lo dimezza. La previsione, fissata prima di guardare l'output, è che nel "
              "contesto neutro (pianura, sereno, truppe fresche) sia in testa una strategia con "
              "quell'attributo al primo posto del profilo ideale, e nella variante no.")
            w("")
            w("| Prova | Esercito | Contesto neutro | Variante | Verdetto |")
            w("|---|---|---|---|---|")
            for pid, (verdetto, base, var) in esiti_prove.items():
                w(f"| {pid} | {base['composizione']} | {base['top1']} ({_num(base['top1_compat_pct'], 1)}%) | "
                  f"{var['terreno'] if var['terreno'] != base['terreno'] else var['meteo']}: {var['top1']} "
                  f"({_num(var['top1_compat_pct'], 1)}%) | {verdetto} |")
            w("")
            conteggio = {v: [p for p, (vv, _, _) in esiti_prove.items() if vv == v]
                         for v in ("confermata", "fallita", "non informativa")}
            frasi = []
            if conteggio["confermata"]:
                frasi.append(f"confermate {len(conteggio['confermata'])} ({', '.join(conteggio['confermata'])})")
            if conteggio["fallita"]:
                frasi.append(f"fallite {len(conteggio['fallita'])} ({', '.join(conteggio['fallita'])}): "
                             "la strategia in testa resta la stessa anche con l'attributo identitario "
                             "dimezzato")
            if conteggio["non informativa"]:
                frasi.append(f"non informative {len(conteggio['non informativa'])} "
                             f"({', '.join(conteggio['non informativa'])}): già nel contesto neutro la "
                             "strategia in testa non è fondata sull'attributo identitario, quindi la "
                             "prova non può dire nulla")
            w(f"Su {len(esiti_prove)} contro-prove: " + "; ".join(frasi) + ".")
            w("")

    # ── 8. E3 ──────────────────────────────────────────────────────
    e3 = t["e3"]
    if not e3.empty:
        tot = int(e3["contesti"].sum())
        cambi = int(e3["cambi_rispetto_al_riferimento"].sum())
        w("## 8. E3: effetto del contesto")
        w("")
        w(f"Contesti valutati: {tot} ({len(e3)} scenari per {int(e3['contesti'].iloc[0])} combinazioni "
          f"di terreno, meteo e stato). La strategia in testa differisce da quella del contesto di "
          f"riferimento (pianura, sereno, truppe fresche) in **{cambi} contesti su {tot} "
          f"({_num(cambi / tot * 100, 1)}%)**.")
        w("")
        w("Il dato è **descrittivo**: non esiste un tasso atteso per un advisor \"sensibile al "
          "contesto\". Come riferimento ci sono solo i due estremi: 0% per un advisor che ignora il "
          "contesto, 90% per una classifica casuale fra 10 strategie, dove la prima coincide con quella "
          "di riferimento una volta su dieci.")
        w("")
        w("| Scenario | Strategie distinte in testa | Cambi | Scarto fra 1ª e 2ª nel riferimento | "
          "Escursione della compatibilità | Classe |")
        w("|---|---|---|---|---|---|")
        for _, r in e3.iterrows():
            w(f"| {r['scenario_id']} | {int(r['strategie_distinte_in_testa'])} | "
              f"{int(r['cambi_rispetto_al_riferimento'])} su {int(r['contesti'])} | "
              f"{_num(r['scarto_top1_top2_nel_riferimento_pt'], 1)} punti | "
              f"{_num(r['escursione_compat_riferimento_pt'], 1)} punti | {r['classe']} |")
        w("")
        invariati = e3[e3["classe"] == "strategia in testa invariata"]
        variabili = e3[e3["classe"] != "strategia in testa invariata"]
        if len(invariati) and len(variabili):
            w(_lettura(
                f"la strategia in testa non cambia mai negli scenari {', '.join(invariati['scenario_id'])}, "
                f"che sono anche quelli con lo scarto più ampio fra prima e seconda strategia nel "
                f"contesto di riferimento (fra {_num(invariati['scarto_top1_top2_nel_riferimento_pt'].min(), 1)} "
                f"e {_num(invariati['scarto_top1_top2_nel_riferimento_pt'].max(), 1)} punti, contro "
                f"valori fra {_num(variabili['scarto_top1_top2_nel_riferimento_pt'].min(), 1)} e "
                f"{_num(variabili['scarto_top1_top2_nel_riferimento_pt'].max(), 1)} negli altri). Il "
                f"contesto sposta la compatibilità in tutti gli scenari (escursione fra "
                f"{_num(e3['escursione_compat_riferimento_pt'].min(), 1)} e "
                f"{_num(e3['escursione_compat_riferimento_pt'].max(), 1)} punti), ma in questi non "
                "abbastanza da cambiare l'ordine. Per eserciti mono-specialistici l'advisor è quindi "
                "poco sensibile al contesto nella raccomandazione, pur restandolo nella compatibilità."))
            w("")

    # ── 9. figure, 10. definizioni ─────────────────────────────────
    w("## 9. Figure")
    w("")
    didascalie = {
        "fig1_e2_esiti": "E2. A: tasso di vittoria per strategia e livello dell'IA, con intervallo "
                         "di confidenza di Wilson al 95%. B: composizione degli esiti per strategia "
                         "e livello dell'IA.",
        "fig2_e2_forza_residua": "E2. Distribuzione della forza residua per strategia e livello "
                                 "dell'IA; la linea è la mediana, il rombo la media.",
        "fig3_e2_per_scenario": "E2. Esito per scenario al variare della strategia: ogni linea "
                                "grigia è uno degli 11 scenari, la linea blu è la media con il "
                                "suo intervallo al 95% calcolato fra scenari.",
        "fig4_e4_dottrine": "E4. Tasso di vittoria con dottrina attiva e disattivata nei soli "
                            "scenari in cui la dottrina si attiva, e aggregato; sopra ogni gruppo "
                            "la differenza appaiata in punti percentuali.",
    }
    for nome in sorted({f.rsplit('.', 1)[0] for f in figure_prodotte}):
        w(f"- `figure/{nome}.pdf` e `.png`: {didascalie.get(nome, '')}")
    w("")
    w("## 10. Definizioni usate")
    w("")
    w("- **Intervallo di confidenza al 95%.** Il 95% è una proprietà della procedura, non del "
      "singolo intervallo: ripetendo l'esperimento molte volte, il 95% degli intervalli costruiti in "
      "questo modo conterrebbe il valore vero. Non significa che il valore vero cada in questo "
      "intervallo con probabilità del 95%.")
    w("- **Wilson.** Intervallo per le proporzioni che resta fra 0% e 100% anche con tassi vicini a "
      "zero, come contro l'IA difficile.")
    w("- **McNemar.** Test per esiti binari appaiati: usa solo le coppie discordanti, cioè i seed in "
      "cui vince una sola delle due condizioni. Il p riportato è quello esatto binomiale.")
    w("- **Holm.** Correzione per confronti multipli: controlla la probabilità di almeno un falso "
      "positivo nell'insieme dei confronti esplorativi.")
    w("- **Scenario come unità.** Le partite dello stesso scenario sono correlate; per affermare "
      "qualcosa sugli scenari si calcola una differenza per scenario e si fa l'inferenza sulle 11 "
      "differenze, con una t a 10 gradi di libertà.")
    w("")
    percorso.write_text("\n".join(out) + "\n", encoding="utf-8")


# ══════════════════════════════════════════════════════════════════
# INGRESSO
# ══════════════════════════════════════════════════════════════════

def _svuota(cartella: Path) -> List[Path]:
    """Toglie i file dell'esecuzione precedente; ritorna quelli che non si possono togliere.

    Si cancellano i file e non le cartelle: su Windows una cartella aperta in
    Esplora risorse non si puo' eliminare, e un CSV aperto in Excel e'
    bloccato. Se qualcosa e' bloccato ci si ferma prima di scrivere, cosi' in
    `analisi/` non restano mai file vecchi mescolati a file nuovi.
    """
    if not cartella.exists():
        return []
    # prima fase: si rinomina tutto; un file aperto altrove non si lascia
    # rinominare, e in quel caso si rimette a posto quanto gia' spostato
    file = [p for p in sorted(cartella.rglob("*")) if p.is_file()]
    spostati, bloccati = [], []
    for percorso in file:
        provvisorio = percorso.with_name(percorso.name + ".da_cancellare")
        try:
            percorso.rename(provvisorio)
            spostati.append((percorso, provvisorio))
        except PermissionError:
            bloccati.append(percorso)
    if bloccati:
        for originale, provvisorio in spostati:
            provvisorio.rename(originale)
        return bloccati
    # seconda fase: nessun file e' bloccato, si cancella
    for _, provvisorio in spostati:
        provvisorio.unlink()
    return []


def _cerca_controlli(cartella: Path) -> List[Path]:
    """Cartelle sorelle `<nome>_*` con un proprio manifest e un runs.csv."""
    genitore, nome = cartella.parent, cartella.name
    return sorted(p for p in genitore.glob(f"{nome}_*")
                  if p.is_dir() and (p / "manifest.json").exists() and (p / "runs.csv").exists())


def esegui(cartella: Path, separatore: str, ctx: SimpleNamespace) -> int:
    ds = carica(cartella, separatore)
    if ds.runs.empty and ds.advisor.empty:
        print(f"Nessun dato leggibile in {ctx.percorso_leggibile(cartella)}.")
        return 1
    print(f"Analisi di {ctx.percorso_leggibile(cartella)}: {len(ds.runs)} partite, "
          f"{len(ds.advisor)} righe advisor")

    uscita = cartella / "analisi"
    bloccati = _svuota(uscita)
    if bloccati:
        print("Non posso sovrascrivere questi file, probabilmente aperti in un altro programma "
              "(Excel, un lettore PDF):")
        for f in bloccati:
            print(f"   {ctx.percorso_leggibile(f)}")
        print("Chiudili e rilancia: nessun file nuovo e' stato scritto.")
        return 1
    cartella_tab, cartella_fig = uscita / "tabelle", uscita / "figure"
    cartella_tab.mkdir(parents=True, exist_ok=True)

    validi_e4, attivazione = (scenari_validi_e4(ds.runs) if not ds.runs.empty
                              else ([], pd.DataFrame()))
    t: Dict[str, pd.DataFrame] = {}
    if not ds.runs.empty:
        t["condizioni"] = tabella_condizioni(ds.runs, validi_e4)
        t["confronti"] = tabella_confronti(ds.runs, validi_e4)
        t["appaiata_vittorie"], t["appaiata_metriche"] = tabelle_appaiate(ds.runs, validi_e4)
        t["per_scenario"], t["per_scenario_sintesi"] = tabelle_per_scenario(ds.runs)
        t["e4"] = tabella_e4(ds.runs, validi_e4, attivazione)
        t["esiti_per_livello"] = tabella_esiti_per_livello(ds.runs)
        t["durata"] = tabella_durata(ds.runs)
    if not ds.advisor.empty:
        t["e1"] = tabella_e1(ctx, ds.advisor)
        t["e3"] = tabella_e3(ds.advisor)
    for chiave in ("e1", "e3", "e4", "per_scenario_sintesi", "appaiata_vittorie",
                   "appaiata_metriche", "esiti_per_livello", "durata"):
        t.setdefault(chiave, pd.DataFrame())

    controlli = []
    for altra in _cerca_controlli(cartella):
        controllo = carica(altra, separatore)
        tab = tabella_controllo(ds.runs, controllo.runs, ctx)
        t[altra.name.removeprefix(cartella.name + "_")] = tab
        controlli.append((altra.name, controllo, tab))
        print(f"   confronto con il dataset di controllo {altra.name}: "
              f"{int(tab[tab['condizione'].isin(CONDIZIONI_E2) & tab['livello'].eq('tutte')]['n'].sum())} "
              "partite accoppiate")

    for nome, df in t.items():
        if df.empty:
            continue
        df.map(_r).to_csv(cartella_tab / f"{nome}.csv", sep=separatore, index=False,
                          encoding="utf-8-sig")
    prodotte = figure(ds.runs, validi_e4, cartella_fig) if not ds.runs.empty else []
    scrivi_riepilogo(uscita / "RIEPILOGO_DATI_TESI.md", ds, t, controlli, prodotte, validi_e4)

    print(f"   {sum(1 for d in t.values() if not d.empty)} tabelle, {len(prodotte)} file di figura, "
          f"riepilogo in {ctx.percorso_leggibile(uscita / 'RIEPILOGO_DATI_TESI.md')}")
    return 0

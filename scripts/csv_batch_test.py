"""
War Advisor - Modalita esperimento / batch (CSV)

Ripete scenari controllati con seed esplicito e scrive i risultati in CSV,
senza passare dal browser. E' lo strumento della campagna sperimentale: i
numeri che finiscono nella tesi escono da qui, non da partite giocate a mano.

I numeri NON sono ricalcolati in questo file: vengono chiesti al motore vero
(`GameSession`, `engine.compute_ranking`), le stesse funzioni che girano in
partita. Se cambia il gioco, cambiano i dati - e' il punto.

──────────────────────────────────────────────────────────────────────
I QUATTRO ESPERIMENTI

  E1  sanity check del recommender
      Scenari intuitivi e controllati (cavalleria in pianura, assassini in
      foresta di notte, ...). Nessuna partita: si guarda solo se la strategia
      in testa e' coerente con la semantica dichiarata dei vettori.

  E2  validita predittiva del ranking            <- l'esperimento principale
      Stesso esercito, stesso scenario, stesso seed: varia SOLO la strategia.
      Si confrontano la prima, una intermedia e l'ultima della classifica.
      Ipotesi attesa: salendo di rank migliora, in media, l'esito.

  E3  effetto del contesto e coerenza Advisor/Combat
      Si fa variare terreno, meteo e stato truppe e si misura quanto spesso
      cambia la strategia in testa, e quanto divergono i due punteggi.

  E4  dottrine e difficolta IA
      Ablation: stesso scenario con e senza il layer dottrine, e lo stesso
      confronto fra i livelli di difficolta.

──────────────────────────────────────────────────────────────────────
I FILE PRODOTTI                       (cartella `reports/esperimenti/`)

  runs.csv       una riga per partita: lo schema del punto 3.5 della specifica
  battles.csv    una riga per scontro: strategia, compatibilita, dottrina
                 attiva, bonus/malus applicati ed esito di quel singolo
                 scontro. E' la parte che rende osservabili le dottrine.
  advisor.csv    una riga per (scenario x strategia): rank, compatibilita
                 advisor e compatibilita combat, senza giocare la partita.
  manifest.json  come e' stato prodotto il dataset: commit, data, parametri,
                 taratura del layer compatibilita. Serve a poter dire quale
                 versione del software ha generato i numeri della tesi.

Con `--analizza` si aggiungono le sintesi del punto 4.5, ricavate dai CSV
sopra senza rigiocare niente:

  analisi_condizioni.csv   media, deviazione standard, numerosita' e
                           intervallo al 95% per ogni condizione e metrica
  analisi_confronti.csv    differenze fra condizioni con il loro intervallo
  analisi_scenari.csv      lo stesso quadro scenario per scenario
  grafico_*.svg            barre con barre d'errore, boxplot, compatibilita'
                           contro esito, ablation delle dottrine

──────────────────────────────────────────────────────────────────────
USO

  python scripts/csv_batch_test.py --esperimento tutti
  python scripts/csv_batch_test.py --esperimento E2 --repliche 40 --seed 12345
  python scripts/csv_batch_test.py --esperimento E2 --difficolta hard normal
  python scripts/csv_batch_test.py --lista-scenari
  python scripts/csv_batch_test.py --verifica      # requisiti e cooldown dottrine
  python scripts/csv_batch_test.py --analizza      # sintesi e grafici dei dati gia' prodotti

Rifare un solo esperimento senza rieseguire tutto, e poi ricomporre il dataset:

  python scripts/csv_batch_test.py --esperimento E4 --out reports/nuovo_e4
  python scripts/csv_batch_test.py --unisci reports/vecchio reports/nuovo_e4 \
                                   --out reports/dataset

Nell'unione, per ogni esperimento vince l'ULTIMA cartella che lo contiene: il
vecchio E4 sparisce e gli altri esperimenti restano intatti. Concatenare i CSV
a mano non funziona — i due E4 resterebbero dentro insieme.

E1 ed E3 non giocano partite e durano meno di un secondo. E2 ed E4 giocano:
contare circa 0.4 s a partita, cioe' ~5 minuti per una campagna completa con
30 repliche su tre difficolta'.

Ogni partita e' identificata da `scenario_id` e `seed`: ripetendo il comando
con lo stesso `--seed` si riottengono esattamente le stesse partite.
"""

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import load_data, aggregate_army, apply_modifiers, compute_ranking
from gamecore.gameflow import start_game_session
from gamecore.session.session import PLAYER, AI, SessionState
from gamecore.session import troop_condition as tc

# [COMPAT-LAYER] La taratura finisce nel manifest: senza, un dataset non si
# puo' interpretare a distanza di mesi.
try:
    from gamecore import combat_compat as cmp_layer
except ImportError:
    cmp_layer = None

# [DOCTRINE-LAYER] Due riferimenti distinti, e servono entrambi:
#   `sessione_modulo.doc`  e' la leva dell'ablation (E4 la azzera e la rimette)
#   `_DOTTRINE`            resta sempre valido, per sapere quali strategie
#                          hanno un effetto a prescindere dalla condizione
try:
    from gamecore.session import session as sessione_modulo
except ImportError:                                           # pragma: no cover
    sessione_modulo = None
try:
    from gamecore import doctrines as _DOTTRINE
except ImportError:                                           # layer rimosso
    _DOTTRINE = None


DATA = load_data()
UNITA = {u["id"]: u for u in DATA["units"]}
STRATEGIE = {s["id"]: s for s in DATA["strategies"]}
TERRENI = list(DATA["terrain"].keys())
METEO_BASE = list(DATA["weather"].keys())
STATI = list(DATA["troop_status"].keys())

OUTPUT_DIR = PROJECT_ROOT / "reports" / "esperimenti"

# Orizzonte di default. Misurato: a 300 turni meta' delle partite resta
# indecisa, a 600 si decidono quasi tutte. 400 e' il compromesso; chi vuole
# un win rate pulito alzi `--turni`.
TURNI_DEFAULT = 400
REPLICHE_DEFAULT = 30
SEED_DEFAULT = 20260101
DIFFICOLTA = ("easy", "normal", "hard", "nightmare")

ESITO_VITTORIA = "vittoria"
ESITO_SCONFITTA = "sconfitta"
ESITO_NON_DECISA = "non_decisa"


# ══════════════════════════════════════════════════════════════════
# SCENARI DI RIFERIMENTO
# ══════════════════════════════════════════════════════════════════
#
# Set controllato per E1: composizioni dall'identita' netta, messe nel
# contesto che la tesi dichiara essere il loro. `attese` elenca le strategie
# considerate semanticamente coerenti; serve solo a marcare la riga, non
# entra in nessun calcolo.

SCENARI: List[Dict[str, Any]] = [
    {
        "id": "S01",
        "descrizione": "Cavalleria leggera in pianura, giorno sereno",
        "unita": ["light_cavalry"] * 3,
        "terreno": "Pianura", "meteo": "Sereno", "stato": "Fresche",
        "attese": ["blitz", "flank_maneuver", "encirclement"],
    },
    {
        "id": "S02",
        "descrizione": "Assassini in foresta di notte",
        "unita": ["assassins"] * 3,
        "terreno": "Foresta", "meteo": "Notte", "stato": "Fresche",
        "attese": ["ambush", "guerrilla"],
    },
    {
        "id": "S03",
        "descrizione": "Assassini in pianura in pieno sole (contro-prova di S02)",
        "unita": ["assassins"] * 3,
        "terreno": "Pianura", "meteo": "Sereno", "stato": "Fresche",
        "attese": ["ambush", "guerrilla"],
    },
    {
        "id": "S04",
        "descrizione": "Fanteria pesante in pianura",
        "unita": ["heavy_infantry"] * 3,
        "terreno": "Pianura", "meteo": "Sereno", "stato": "Fresche",
        "attese": ["frontal_assault", "breakthrough", "defense_depth"],
    },
    {
        "id": "S05",
        "descrizione": "Artiglieria e picchieri in montagna",
        "unita": ["artillery", "artillery", "pikemen"],
        "terreno": "Montagna", "meteo": "Sereno", "stato": "Fresche",
        "attese": ["screen_and_fire", "defense_depth"],
    },
    {
        "id": "S06",
        "descrizione": "Arcieri e picchieri in difesa, truppe stanche",
        "unita": ["archers", "archers", "pikemen"],
        "terreno": "Foresta", "meteo": "Sereno", "stato": "Stanche",
        "attese": ["screen_and_fire", "defense_depth", "ambush"],
    },
    {
        "id": "S07",
        "descrizione": "Esploratori in palude, nebbia",
        "unita": ["scouts"] * 3,
        "terreno": "Palude", "meteo": "Nebbia", "stato": "Fresche",
        "attese": ["guerrilla", "ambush", "tactical_retreat"],
    },
    {
        "id": "S08",
        "descrizione": "Armata mista, nessuna identita' dominante",
        "unita": ["heavy_infantry", "archers", "light_cavalry"],
        "terreno": "Pianura", "meteo": "Sereno", "stato": "Fresche",
        "attese": [],
    },
    # ── Scenari costruiti sui requisiti delle dottrine ──────────────
    # Gli otto qui sopra sono fatti per E1: identita' semantica netta. Ma una
    # dottrina si accende solo se l'esercito ne soddisfa il requisito, e su
    # quelle composizioni succede in due casi su otto — l'ablation di E4
    # finirebbe a confrontare due condizioni identiche. Questi tre completano
    # la copertura dei cinque effetti implementati.
    {
        "id": "S09",
        "descrizione": "Muro di picchieri: requisito di Difesa in Profondita'",
        "unita": ["pikemen"] * 4,
        "terreno": "Montagna", "meteo": "Sereno", "stato": "Fresche",
        "attese": ["defense_depth", "screen_and_fire", "frontal_assault"],
    },
    {
        "id": "S10",
        "descrizione": "Cavalleria con scorta: requisito di Manovra sui Fianchi",
        "unita": ["light_cavalry"] * 3 + ["archers"],
        "terreno": "Pianura", "meteo": "Sereno", "stato": "Fresche",
        "attese": ["flank_maneuver", "encirclement", "blitz"],
    },
    {
        "id": "S11",
        "descrizione": "Armata combinata completa: requisito di Accerchiamento",
        "unita": ["archers", "assassins", "light_cavalry", "heavy_infantry"],
        "terreno": "Pianura", "meteo": "Sereno", "stato": "Fresche",
        "attese": ["encirclement", "flank_maneuver"],
    },
]
SCENARI_PER_ID = {s["id"]: s for s in SCENARI}


# ══════════════════════════════════════════════════════════════════
# IL GIOCATORE AUTOMATICO
# ══════════════════════════════════════════════════════════════════

class GiocatoreScriptato:
    """Politica di gioco fissa, uguale in tutte le condizioni.

    Serve a E2: se il giocatore cambiasse comportamento fra una condizione e
    l'altra, la differenza di esito non sarebbe attribuibile alla strategia.
    Qui non c'e' nessuna decisione presa sulla strategia in corso: la politica
    e' cieca ad essa.

      1. recluta mantenendo la composizione iniziale, a rotazione. Reclutare
         sempre la stessa truppa sposterebbe il profilo dell'esercito e
         favorirebbe le strategie affini a quella truppa.
      2. assorbe la riserva: se l'armata e' a casa e ci sono truppe nuove, la
         richiama e la riforma con tutto. E' l'unico modo che il gioco offre
         per rinforzare una legione gia' in campo.
      3. marcia sul castello nemico, e ci torna ogni volta che resta senza
         ordini (dopo un assalto respinto o un ripiegamento).

    Non fortifica, non presidia e non ricerca abilita': sono leve che
    aggiungerebbero varianza senza servire alla domanda di ricerca.
    """

    def __init__(self, composizione: Sequence[str]) -> None:
        self.composizione = list(composizione)
        self.prossima_recluta = 0

    def agisci(self, sessione: Any, casa: Tuple[int, int], nemico: Tuple[int, int]) -> None:
        self._recluta(sessione)
        self._assorbi_riserva(sessione, casa, nemico)
        self._rimanda_al_fronte(sessione, nemico)

    def _recluta(self, sessione: Any) -> None:
        # Il numero di tentativi per turno segue la dimensione della
        # composizione, ma non introduce un vantaggio: a fermare gli acquisti
        # sono il budget e il cooldown molto prima del conteggio del ciclo.
        # Misurato: ~0.17 reclute per turno, e gli scenari da 4 unita' ne
        # reclutano MENO di quelli da 3, non di piu'.
        for _ in range(len(self.composizione)):
            unita = self.composizione[self.prossima_recluta % len(self.composizione)]
            try:
                sessione.recruit_player_unit(unita)
            except Exception:
                return                      # budget finito o cooldown attivo
            self.prossima_recluta += 1

    def _assorbi_riserva(self, sessione: Any, casa: Tuple[int, int],
                         nemico: Tuple[int, int]) -> None:
        try:
            if sessione.player_legions:
                legione_id, legione = next(iter(sessione.player_legions.items()))
                a_casa = tuple(legione.get("pos", ())) == tuple(casa)
                if a_casa and len(sessione.player_units) >= 3:
                    sessione.recall_player_legion(legione_id)
            if sessione.player_units and not sessione.player_legions:
                sessione.create_player_legion(
                    "Armata", _conteggio(sessione.player_units), nemico
                )
        except Exception:
            return

    def _rimanda_al_fronte(self, sessione: Any, nemico: Tuple[int, int]) -> None:
        for legione_id, legione in list(sessione.player_legions.items()):
            if legione.get("target") is None:
                try:
                    sessione.retarget_player_legion(legione_id, nemico)
                except Exception:
                    continue


def _conteggio(unita: Sequence[str]) -> Dict[str, int]:
    conteggio: Dict[str, int] = {}
    for unit_id in unita:
        conteggio[unit_id] = conteggio.get(unit_id, 0) + 1
    return conteggio


def _descrivi_composizione(unita: Sequence[str]) -> str:
    """"2 Cavalleria Leggera + 1 Esploratori", come nell'esempio della specifica."""
    conteggio = _conteggio(unita)
    pezzi = [
        f"{quante} {UNITA.get(unit_id, {}).get('name', unit_id)}"
        for unit_id, quante in sorted(conteggio.items(), key=lambda x: (-x[1], x[0]))
    ]
    return " + ".join(pezzi)


# ══════════════════════════════════════════════════════════════════
# VALUTAZIONE SENZA PARTITA (E1 / E3)
# ══════════════════════════════════════════════════════════════════

def valuta_scenario(
    unita: Sequence[str],
    terreno: str,
    meteo: Optional[str],
    stato: Optional[str],
    dati: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Classifica completa dell'advisor piu' la compatibilita' di combattimento.

    Ritorna una riga per strategia, gia' ordinata per rank. I due punteggi
    vengono dalle stesse funzioni che usano la schermata consigli e il motore
    di battaglia, cosi' il confronto e' legittimo.
    """
    dati = dati or DATA
    vettore = aggregate_army(list(unita), dati["units"])
    modificato, avvisi = apply_modifiers(
        army_vector=vettore, terrain_name=terreno, weather_name=meteo,
        troop_status_name=stato, modifiers_data=dati,
    )
    classifica = compute_ranking(
        army_vector=modificato, strategies_list=dati["strategies"],
        unit_ids=list(unita), terrain_name=terreno, weather_name=meteo,
        affinities_data=dati.get("unit_affinities", {}),
    )

    righe: List[Dict[str, Any]] = []
    for posizione, voce in enumerate(classifica, start=1):
        combat = _compatibilita_combat(modificato, voce, unita, terreno, meteo, dati)
        righe.append({
            "strategia_id": voce["id"],
            "strategia_nome": voce["name"],
            "rank_advisor": posizione,
            "distanza": round(float(voce["distance"]), 4),
            "compat_advisor_pct": round(float(voce["compatibility"]), 2),
            "compat_combat_pct": round(combat * 100, 2),
            "divergenza_pt": round(float(voce["compatibility"]) - combat * 100, 2),
            "avvisi_critical": len(avvisi),
        })
    return righe


def _compatibilita_combat(
    modificato: Dict[str, float],
    strategia: Dict[str, Any],
    unita: Sequence[str],
    terreno: str,
    meteo: Optional[str],
    dati: Dict[str, Any],
) -> float:
    """La compatibilita' che entra nel moltiplicatore di forza, in [0..1]."""
    from engine import euclidean_distance
    distanza = euclidean_distance(modificato, strategia["ideal_attributes"])
    if cmp_layer is None:
        return max(0.0, min(1.0, 1.0 - distanza / (8 ** 0.5)))
    combat, _ = cmp_layer.scores(
        distance=distanza, unit_ids=list(unita), strategy_id=strategia["id"],
        terrain=terreno, weather=meteo,
        affinities=dati.get("unit_affinities", {}),
    )
    return combat


# ══════════════════════════════════════════════════════════════════
# UNA PARTITA
# ══════════════════════════════════════════════════════════════════

def gioca_partita(
    *,
    scenario_id: str,
    unita: Sequence[str],
    strategia_id: str,
    difficolta: str,
    seed: int,
    turni_massimi: int = TURNI_DEFAULT,
    terreno_scenario: str = "Pianura",
    meteo_scenario: Optional[str] = None,
    stato_scenario: Optional[str] = None,
    etichetta_condizione: str = "",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Gioca una partita completa e ritorna (riga_run, righe_battaglie).

    `terreno_scenario` / `meteo_scenario` / `stato_scenario` descrivono lo
    scenario valutato dall'advisor. [SETUP-RULE] NON sono le condizioni della
    partita: il meteo lo sorteggia il gioco dal seed e le truppe partono
    sempre fresche. Le due cose vanno registrate separatamente, altrimenti il
    dataset direbbe una cosa falsa.
    """
    vettore = aggregate_army(list(unita), DATA["units"])
    modificato, _ = apply_modifiers(
        army_vector=vettore, terrain_name=terreno_scenario,
        weather_name=meteo_scenario, troop_status_name=stato_scenario,
        modifiers_data=DATA,
    )

    avvio = start_game_session(
        data=DATA, player_units=list(unita), terrain=terreno_scenario,
        weather=meteo_scenario, troop_status=stato_scenario,
        strategy_id=strategia_id, army_profile=vettore,
        modified_profile=modificato, map_seed=seed, ai_difficulty=difficolta,
    )
    sessione = avvio["session"]
    seed_effettivo = avvio.get("map_seed", seed)

    meteo_iniziale = sessione.weather
    casa = sessione.game_map.castle_positions.get(PLAYER)
    nemico = sessione.game_map.castle_positions.get(AI)
    # Senza castelli il giocatore automatico non potrebbe fare nulla, e la
    # partita finirebbe nel dataset come una riga valida di zeri. Meglio
    # fermarsi rumorosamente che produrre dati silenziosamente sbagliati.
    if casa is None or nemico is None:
        raise RuntimeError(
            f"Mappa senza castelli (seed {seed}): giocatore={casa}, IA={nemico}."
        )
    giocatore = GiocatoreScriptato(unita)

    battaglie: List[Dict[str, Any]] = []
    ultimo_evento = 0
    turno_primo_scontro: Optional[int] = None
    errore: str = ""

    for _ in range(turni_massimi):
        if sessione.state != SessionState.ACTIVE:
            break
        giocatore.agisci(sessione, casa, nemico)
        try:
            sessione.execute_turn()
        except Exception as eccezione:                # pragma: no cover
            errore = f"{type(eccezione).__name__}: {eccezione}"
            break

        # Gli eventi si drenano a ogni turno: la coda del motore tiene solo la
        # parte recente, e una partita lunga perderebbe gli scontri iniziali.
        nuovi, ultimo_evento = _drena_eventi(sessione, ultimo_evento)
        for evento in nuovi:
            riga = _riga_battaglia(evento, scenario_id, seed_effettivo,
                                   strategia_id, difficolta, etichetta_condizione)
            if riga is not None:
                battaglie.append(riga)
                if turno_primo_scontro is None:
                    turno_primo_scontro = riga["turno"]

    riga = _riga_run(
        sessione=sessione, scenario_id=scenario_id, seed=seed_effettivo,
        unita=unita, strategia_id=strategia_id, difficolta=difficolta,
        terreno_scenario=terreno_scenario, meteo_scenario=meteo_scenario,
        stato_scenario=stato_scenario, meteo_iniziale=meteo_iniziale,
        turni_massimi=turni_massimi, battaglie=battaglie,
        turno_primo_scontro=turno_primo_scontro,
        etichetta_condizione=etichetta_condizione, errore=errore,
    )
    return riga, battaglie


def _drena_eventi(sessione: Any, ultimo_id: int) -> Tuple[List[Dict[str, Any]], int]:
    registro = getattr(sessione, "event_log", None)
    if registro is None:
        return [], ultimo_id
    eventi = [e for e in registro.to_list() if e.get("id", 0) > ultimo_id]
    if eventi:
        ultimo_id = max(e.get("id", 0) for e in eventi)
    return eventi, ultimo_id


def _riga_battaglia(
    evento: Dict[str, Any], scenario_id: str, seed: int,
    strategia_id: str, difficolta: str, condizione: str,
) -> Optional[Dict[str, Any]]:
    """Una riga per scontro: e' qui che le dottrine diventano osservabili."""
    if evento.get("tipo") != "battaglia":
        return None
    dettaglio = evento.get("dettaglio") or {}
    lati = dettaglio.get("scontro") or {}
    pl = lati.get("player") or {}
    ia = lati.get("ai") or {}
    attaccante = dettaglio.get("attaccante")
    posizione = evento.get("pos") or [None, None]

    forza_pl = dettaglio.get("forza_attaccante" if attaccante == "player" else "forza_difensore")
    forza_ia = dettaglio.get("forza_attaccante" if attaccante == "ai" else "forza_difensore")

    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "condizione": condizione,
        "strategia_run": strategia_id,
        "difficolta_ia": difficolta,
        "turno": evento.get("turno"),
        "riga": posizione[0], "colonna": posizione[1],
        "terreno": dettaglio.get("terreno"),
        "meteo": dettaglio.get("meteo"),
        "attaccante": attaccante,
        "forza_player": forza_pl,
        "forza_ai": forza_ia,
        "rapporto_forze": (round(forza_pl / forza_ia, 4)
                           if forza_pl and forza_ia else ""),
        "esito_scontro": _esito_scontro(forza_pl, forza_ia),
        # ── lato giocatore ────────────────────────────────────────
        "pl_strategia": pl.get("strategia"),
        "pl_strategia_nome": pl.get("strategia_nome"),
        "pl_compat_combat": _arrotonda(pl.get("compat_combat")),
        "pl_compat_advisor": _arrotonda(pl.get("compat_advisor")),
        "pl_fattore_strategia": _arrotonda(pl.get("fattore_strategia")),
        "pl_unita": pl.get("unita"),
        "pl_stato_truppe": pl.get("stato_truppe"),
        "pl_morale": _arrotonda(pl.get("morale"), 1),
        "pl_fatica": _arrotonda(pl.get("fatica"), 1),
        "pl_dottrina_attiva": pl.get("dottrina_attiva"),
        "pl_dottrina_effetto": pl.get("dottrina_effetto"),
        "pl_dottrina_requisito_ok": pl.get("dottrina_requisito_ok"),
        # "quando scatta": turno di adozione e turni trascorsi da allora.
        "pl_dottrina_dal_turno": pl.get("dottrina_dal_turno"),
        "pl_dottrina_turni_attiva": _turni_da(evento.get("turno"),
                                              pl.get("dottrina_dal_turno")),
        "pl_bonus_difesa": _arrotonda(pl.get("bonus_difesa"), 1),
        "pl_fortificazione": pl.get("fortificazione"),
        "pl_fattore_marcia": _arrotonda(pl.get("fattore_marcia")),
        # ── lato IA ───────────────────────────────────────────────
        "ia_strategia": ia.get("strategia"),
        "ia_strategia_nome": ia.get("strategia_nome"),
        "ia_compat_combat": _arrotonda(ia.get("compat_combat")),
        "ia_compat_advisor": _arrotonda(ia.get("compat_advisor")),
        "ia_fattore_strategia": _arrotonda(ia.get("fattore_strategia")),
        "ia_unita": ia.get("unita"),
        "ia_stato_truppe": ia.get("stato_truppe"),
        "ia_morale": _arrotonda(ia.get("morale"), 1),
        "ia_dottrina_attiva": ia.get("dottrina_attiva"),
        "ia_dottrina_effetto": ia.get("dottrina_effetto"),
        "ia_dottrina_dal_turno": ia.get("dottrina_dal_turno"),
        "ia_dottrina_turni_attiva": _turni_da(evento.get("turno"),
                                              ia.get("dottrina_dal_turno")),
        "ia_bonus_difesa": _arrotonda(ia.get("bonus_difesa"), 1),
        "ia_fortificazione": ia.get("fortificazione"),
        "artiglieria_in_campo": dettaglio.get("artiglieria"),
    }


def _esito_scontro(forza_pl: Optional[float], forza_ia: Optional[float]) -> str:
    if not forza_pl or not forza_ia:
        return ""
    if forza_pl > forza_ia:
        return "player_piu_forte"
    if forza_ia > forza_pl:
        return "ai_piu_forte"
    return "pari"


def _turni_da(turno: Any, dal_turno: Any) -> Any:
    """Da quanti turni la legione ha quella dottrina, al momento dello scontro."""
    try:
        return max(0, int(turno) - int(dal_turno))
    except (TypeError, ValueError):
        return ""


def _arrotonda(valore: Any, cifre: int = 4) -> Any:
    try:
        return round(float(valore), cifre)
    except (TypeError, ValueError):
        return ""


def _riga_run(
    *, sessione: Any, scenario_id: str, seed: int, unita: Sequence[str],
    strategia_id: str, difficolta: str, terreno_scenario: str,
    meteo_scenario: Optional[str], stato_scenario: Optional[str],
    meteo_iniziale: str, turni_massimi: int, battaglie: List[Dict[str, Any]],
    turno_primo_scontro: Optional[int], etichetta_condizione: str, errore: str,
) -> Dict[str, Any]:
    """Lo schema del punto 3.5, campo per campo."""
    classifica = valuta_scenario(unita, terreno_scenario, meteo_scenario, stato_scenario)
    voce = next((r for r in classifica if r["strategia_id"] == strategia_id), None)

    vive_in_campo = sum(len(l.get("units") or []) for l in sessione.player_legions.values())
    vive_riserva = len(sessione.player_units)
    vive = vive_in_campo + vive_riserva
    perse = int(sessione.troops_lost.get(PLAYER, 0))
    schierate = vive + perse
    perse_ia = int(sessione.troops_lost.get(AI, 0))

    esito = ESITO_NON_DECISA
    if sessione.winner == "player":
        esito = ESITO_VITTORIA
    elif sessione.winner == "ai":
        esito = ESITO_SCONFITTA

    pl_attive = sum(1 for b in battaglie if b.get("pl_dottrina_attiva"))
    effetti = sorted({b.get("pl_dottrina_effetto") for b in battaglie
                      if b.get("pl_dottrina_attiva") and b.get("pl_dottrina_effetto")})

    castello_pl = sessione.castle_hp.get(PLAYER)
    castello_ia = sessione.castle_hp.get(AI)
    max_pl = sessione.castle_hp_max.get(PLAYER) or 1
    max_ia = sessione.castle_hp_max.get(AI) or 1

    return {
        # scenario_id / seed
        "scenario_id": scenario_id,
        "seed": seed,
        "condizione": etichetta_condizione,
        # composizione esercito
        "composizione": _descrivi_composizione(unita),
        "composizione_id": "+".join(sorted(unita)),
        "n_unita_iniziali": len(unita),
        # terreno / meteo / stato
        "terreno_scenario": terreno_scenario,
        "meteo_scenario": meteo_scenario or "",
        "stato_scenario": stato_scenario or "",
        # [SETUP-RULE] le condizioni VERE della partita, che il giocatore non sceglie
        "meteo_iniziale_partita": meteo_iniziale,
        "meteo_finale_partita": sessione.weather,
        "stato_truppe_iniziale": tc.STATUS_FRESH,
        "terreno_casa": sessione.player_home_terrain,
        # strategia scelta
        "strategia_id": strategia_id,
        "strategia_nome": STRATEGIE.get(strategia_id, {}).get("name", strategia_id),
        # rank advisor + le due compatibilita
        "rank_advisor": voce["rank_advisor"] if voce else "",
        "strategie_totali": len(classifica),
        "compat_advisor_pct": voce["compat_advisor_pct"] if voce else "",
        "compat_combat_pct": voce["compat_combat_pct"] if voce else "",
        "divergenza_pt": voce["divergenza_pt"] if voce else "",
        # dottrina / effetti
        "dottrina_ha_effetto": _dottrina_ha_effetto(strategia_id),
        "dottrina_scontri_attiva": pl_attive,
        "dottrina_scontri_totali": len(battaglie),
        "dottrina_effetti": " ; ".join(effetti),
        # difficolta IA
        "difficolta_ia": difficolta,
        # esito
        "esito": esito,
        "vincitore": sessione.winner or "",
        # perdite proprie / nemiche
        "perdite_proprie": perse,
        "perdite_nemiche": perse_ia,
        # forza residua / morale
        "forza_residua_pct": round(vive / schierate * 100, 1) if schierate else "",
        "unita_vive": vive,
        "unita_schierate_totali": schierate,
        "morale_residuo": _morale_residuo(sessione),
        # turni allo scontro o fine
        "turno_primo_scontro": turno_primo_scontro if turno_primo_scontro else "",
        "turni_totali": sessione.game_map.turn,
        "limite_turni": turni_massimi,
        # contorno utile a interpretare l'esito
        "scontri": len(battaglie),
        "celle_player": sessione.game_map.count_occupied(PLAYER),
        "celle_ai": sessione.game_map.count_occupied(AI),
        "unita_ai_finali": len(sessione.ai_units),
        "hp_castello_player_pct": round((castello_pl or 0) / max_pl * 100, 1),
        "hp_castello_ai_pct": round((castello_ia or 0) / max_ia * 100, 1),
        "errore": errore,
    }


def _dottrina_attivabile(strategia_id: str, unita: Sequence[str]) -> bool:
    """La dottrina ha un effetto E questo esercito ne soddisfa il requisito.

    Le due cose sono distinte: Manovra sui Fianchi ha un effetto, ma pretende
    tre cavallerie leggere — su un'armata di assassini non si accendera' mai.
    """
    if not _dottrina_ha_effetto(strategia_id) or _DOTTRINE is None:
        return False
    try:
        return bool(_DOTTRINE.gate_passed(strategia_id, list(unita)))
    except Exception:
        return False


def _dottrina_ha_effetto(strategia_id: str) -> bool:
    """[DOCTRINE-LAYER] Solo 5 strategie su 10 hanno un effetto implementato.

    Serve a filtrare: su una strategia senza effetto l'ablation di E4 non puo'
    misurare nulla, e `dottrina_scontri_attiva = 0` e' corretto, non un difetto.

    Legge il modulo dottrine direttamente e non `session.doc`, che l'ablation
    azzera: questa e' una proprieta' della strategia, non della condizione
    sperimentale, e deve valere uguale nelle due fasi.
    """
    if _DOTTRINE is None:
        return False
    try:
        return bool(_DOTTRINE.has_effect(strategia_id))
    except Exception:
        return False


def _morale_residuo(sessione: Any) -> float:
    """Morale medio delle legioni in campo; se non ce ne sono, quello della riserva."""
    valori = [
        float(sessione._legion_condition(PLAYER, legione).get("morale", 0.0))
        for legione in sessione.player_legions.values()
    ]
    if not valori:
        valori = [float(sessione.reserve_condition[PLAYER].get("morale", 0.0))]
    return round(sum(valori) / len(valori), 1)


# ══════════════════════════════════════════════════════════════════
# GLI ESPERIMENTI
# ══════════════════════════════════════════════════════════════════

def esperimento_e1(_args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    """Sanity check: la strategia in testa e' coerente con la semantica?"""
    righe: List[Dict[str, Any]] = []
    for scenario in SCENARI:
        classifica = valuta_scenario(
            scenario["unita"], scenario["terreno"], scenario["meteo"], scenario["stato"]
        )
        attese = scenario["attese"]
        prima = classifica[0]["strategia_id"]
        for riga in classifica:
            righe.append({
                "esperimento": "E1",
                "scenario_id": scenario["id"],
                "descrizione": scenario["descrizione"],
                "composizione": _descrivi_composizione(scenario["unita"]),
                "terreno": scenario["terreno"],
                "meteo": scenario["meteo"] or "",
                "stato_truppe": scenario["stato"] or "",
                **riga,
                "attesa_semantica": " | ".join(attese),
                "e_attesa": riga["strategia_id"] in attese if attese else "",
                "top1_coerente": (prima in attese) if attese else "",
            })
    return {"advisor": righe}


def esperimento_e2(args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    """Validita' predittiva: stesso tutto, varia solo la strategia."""
    runs: List[Dict[str, Any]] = []
    battaglie: List[Dict[str, Any]] = []

    for scenario in _scenari_scelti(args):
        classifica = valuta_scenario(
            scenario["unita"], scenario["terreno"], scenario["meteo"], scenario["stato"]
        )
        # prima, intermedia e ultima della classifica: le tre condizioni che
        # la specifica chiede di confrontare.
        mediana = classifica[len(classifica) // 2]
        condizioni = [
            ("top_ranked", classifica[0]),
            ("intermedia", mediana),
            ("worst_ranked", classifica[-1]),
        ]
        for difficolta in args.difficolta:
            for etichetta, voce in condizioni:
                for replica in range(args.repliche):
                    seed = _seed_di(args.seed, scenario["id"], difficolta, replica)
                    riga, scontri = gioca_partita(
                        scenario_id=scenario["id"], unita=scenario["unita"],
                        strategia_id=voce["strategia_id"], difficolta=difficolta,
                        seed=seed, turni_massimi=args.turni,
                        terreno_scenario=scenario["terreno"],
                        meteo_scenario=scenario["meteo"],
                        stato_scenario=scenario["stato"],
                        etichetta_condizione=etichetta,
                    )
                    riga["esperimento"] = "E2"
                    riga["replica"] = replica
                    runs.append(riga)
                    for scontro in scontri:
                        scontro["esperimento"] = "E2"
                        scontro["replica"] = replica
                    battaglie.extend(scontri)
                    _avanzamento(len(runs), _totale_e2(args))
    return {"runs": runs, "battles": battaglie}


def esperimento_e3(args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    """Effetto del contesto: quanto cambia il consiglio al variare dell'ambiente."""
    righe: List[Dict[str, Any]] = []
    for scenario in _scenari_scelti(args):
        base = None
        for terreno in TERRENI:
            for meteo in METEO_BASE:
                for stato in STATI:
                    classifica = valuta_scenario(scenario["unita"], terreno, meteo, stato)
                    prima = classifica[0]
                    if base is None:
                        base = prima["strategia_id"]
                    for riga in classifica:
                        righe.append({
                            "esperimento": "E3",
                            "scenario_id": scenario["id"],
                            "descrizione": scenario["descrizione"],
                            "composizione": _descrivi_composizione(scenario["unita"]),
                            "terreno": terreno,
                            "meteo": meteo,
                            "stato_truppe": stato,
                            **riga,
                            "top1_contesto": prima["strategia_id"],
                            "top1_cambiata": prima["strategia_id"] != base,
                        })
    return {"advisor": righe}


def esperimento_e4(args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    """Dottrine e difficolta': ablation del layer, poi confronto fra livelli."""
    runs: List[Dict[str, Any]] = []
    battaglie: List[Dict[str, Any]] = []
    originale = getattr(sessione_modulo, "doc", None) if sessione_modulo else None

    # Le strategie si scelgono PRIMA di toccare il layer, e valgono per
    # entrambe le fasi: se le due fasi usassero strategie diverse il confronto
    # non sarebbe piu' un'ablation.
    #
    # Servono DUE condizioni, non una. La strategia deve avere un effetto
    # implementato (solo 5 su 10 ce l'hanno) *e* l'esercito dello scenario deve
    # poterne soddisfare il requisito di attivazione. Senza la seconda, si
    # finisce a confrontare due condizioni identiche: misurato su una campagna
    # intera, la dottrina non si accendeva in 6 scenari su 8 — gli assassini non
    # diventano tre cavallerie leggere, e l'ablation non misurava nulla.
    strategie_per_scenario: Dict[str, str] = {}
    saltati: List[str] = []
    for scenario in _scenari_scelti(args):
        classifica = valuta_scenario(
            scenario["unita"], scenario["terreno"], scenario["meteo"], scenario["stato"]
        )
        scelta = next(
            (v["strategia_id"] for v in classifica
             if _dottrina_attivabile(v["strategia_id"], scenario["unita"])),
            None,
        )
        if scelta is None:
            saltati.append(scenario["id"])
            continue
        strategie_per_scenario[scenario["id"]] = scelta
    if saltati:
        print(f"   scenari esclusi da E4 (nessuna dottrina attivabile da questo "
              f"esercito): {', '.join(saltati)}")
    if not strategie_per_scenario:
        print("   nessuno scenario con una dottrina attivabile: E4 non produce righe.")
        return {"runs": [], "battles": []}

    # Il totale va contato sugli scenari rimasti, non su quelli chiesti.
    totale = len(strategie_per_scenario) * len(args.difficolta) * args.repliche * 2

    for con_dottrine in (True, False):
        if sessione_modulo is not None:
            sessione_modulo.doc = originale if con_dottrine else None
        etichetta = "dottrine_on" if con_dottrine else "dottrine_off"
        for scenario in _scenari_scelti(args):
            if scenario["id"] not in strategie_per_scenario:
                continue
            strategia = strategie_per_scenario[scenario["id"]]
            for difficolta in args.difficolta:
                for replica in range(args.repliche):
                    seed = _seed_di(args.seed, scenario["id"], difficolta, replica)
                    riga, scontri = gioca_partita(
                        scenario_id=scenario["id"], unita=scenario["unita"],
                        strategia_id=strategia, difficolta=difficolta, seed=seed,
                        turni_massimi=args.turni,
                        terreno_scenario=scenario["terreno"],
                        meteo_scenario=scenario["meteo"],
                        stato_scenario=scenario["stato"],
                        etichetta_condizione=etichetta,
                    )
                    riga["esperimento"] = "E4"
                    riga["replica"] = replica
                    runs.append(riga)
                    for scontro in scontri:
                        scontro["esperimento"] = "E4"
                        scontro["replica"] = replica
                    battaglie.extend(scontri)
                    _avanzamento(len(runs), totale)

    if sessione_modulo is not None:
        sessione_modulo.doc = originale          # il layer torna com'era
    return {"runs": runs, "battles": battaglie}


# ══════════════════════════════════════════════════════════════════
# VERIFICA DELLE DOTTRINE
# ══════════════════════════════════════════════════════════════════
#
# Terzo punto del 3.3: controllare in automatico che i requisiti di
# attivazione e i cooldown producano davvero gli effetti descritti. Sta qui e
# non in una suite a parte perche' e' la stessa domanda del dataset — se una
# dottrina non si accende quando dovrebbe, le colonne `*_dottrina_*` dei CSV
# descrivono un gioco diverso da quello raccontato nella tesi.

def verifica_dottrine() -> int:
    """Controlla requisiti di attivazione, ritardo e cooldown. Ritorna i fallimenti."""
    if _DOTTRINE is None:
        print("Layer dottrine assente: niente da verificare.")
        return 0

    doc = _DOTTRINE
    esiti: List[Tuple[bool, str, str]] = []

    def controlla(nome: str, condizione: bool, dettaglio: str = "") -> None:
        esiti.append((bool(condizione), nome, dettaglio))

    # ── requisiti di attivazione, uno per dottrina con effetto ──────
    casi_gate = [
        (doc.BLITZ, [doc.LIGHT_CAVALRY] * 3, True, "solo cavalleria leggera"),
        (doc.BLITZ, [doc.LIGHT_CAVALRY] * 2 + [doc.ARCHERS], False, "una truppa estranea basta a bloccarla"),
        (doc.FLANK_MANEUVER, [doc.LIGHT_CAVALRY] * doc.FLANK_MIN_CAVALRY, True, "al minimo di cavalleria"),
        (doc.FLANK_MANEUVER, [doc.LIGHT_CAVALRY] * (doc.FLANK_MIN_CAVALRY - 1), False, "una sotto il minimo"),
        (doc.DEFENSE_DEPTH, [doc.PIKEMEN] * doc.DEFENSE_DEPTH_MIN_PIKEMEN, True, "al minimo di picchieri"),
        (doc.DEFENSE_DEPTH, [doc.PIKEMEN] * (doc.DEFENSE_DEPTH_MIN_PIKEMEN - 1), False, "una sotto il minimo"),
        (doc.SCREEN_AND_FIRE, [doc.ARTILLERY, doc.PIKEMEN], True, "artiglieria piu' scudo"),
        (doc.SCREEN_AND_FIRE, [doc.ARTILLERY] * 3, False, "artiglieria senza scudo"),
        (doc.ENCIRCLEMENT, list(doc.ENCIRCLEMENT_REQUIRED_TYPES), True, "tutti i tipi richiesti"),
        (doc.ENCIRCLEMENT, list(doc.ENCIRCLEMENT_REQUIRED_TYPES)[:-1], False, "manca un tipo"),
    ]
    for strategia, unita, atteso, nota in casi_gate:
        ottenuto = doc.gate_passed(strategia, unita)
        controlla(f"requisito {strategia}: {nota}", ottenuto == atteso,
                  f"atteso {atteso}, ottenuto {ottenuto}")

    # Una dottrina senza effetto non deve mai risultare attiva.
    for strategia in STRATEGIE:
        if not doc.has_effect(strategia):
            controlla(
                f"{strategia} non ha effetto e non si attiva",
                not doc.is_active(strategia, [doc.LIGHT_CAVALRY] * 5, turn=99, since_turn=1),
            )

    # ── ritardo di attivazione ──────────────────────────────────────
    unita_blitz = [doc.LIGHT_CAVALRY] * 3
    ritardo = doc.ACTIVATION_DELAY_TURNS
    for trascorsi in range(0, ritardo + 3):
        attiva = doc.is_active(doc.BLITZ, unita_blitz, turn=10 + trascorsi, since_turn=10)
        controlla(
            f"ritardo attivazione: {trascorsi} turni dopo la scelta",
            attiva == (trascorsi >= ritardo),
            f"atteso {trascorsi >= ritardo}, ottenuto {attiva}",
        )
    # `since_turn=None` vale come "adottata da sempre": e' il ripiego
    # documentato per le legioni nate prima che il layer esistesse.
    controlla("senza since_turn vale come adottata da sempre",
              doc.is_active(doc.BLITZ, unita_blitz, turn=10, since_turn=None))
    controlla("ma il requisito resta obbligatorio anche senza since_turn",
              not doc.is_active(doc.BLITZ, [doc.ARCHERS] * 3, turn=10, since_turn=None))

    # ── cooldown di cambio ──────────────────────────────────────────
    attesa = doc.CHANGE_COOLDOWN_TURNS
    for trascorsi in range(0, attesa + 3):
        puo = doc.can_change(turn=20 + trascorsi, last_change_turn=20)
        controlla(
            f"cooldown: cambio {trascorsi} turni dopo l'ultimo",
            puo == (trascorsi >= attesa),
            f"atteso {trascorsi >= attesa}, ottenuto {puo}",
        )
        mancano = doc.turns_before_change(20 + trascorsi, 20)
        controlla(
            f"cooldown: turni mancanti a {trascorsi}",
            mancano == max(0, attesa - trascorsi),
            f"atteso {max(0, attesa - trascorsi)}, ottenuto {mancano}",
        )
    controlla("al primo cambio non c'e' attesa", doc.can_change(turn=1, last_change_turn=None))

    # ── gli effetti valgono solo a dottrina accesa ───────────────────
    acc = dict(turn=10 + ritardo, since_turn=10)
    spenta = dict(turn=10, since_turn=10)
    controlla(
        "accerchiamento: moltiplicatore solo da attiva e in attacco",
        doc.enemy_loss_multiplier(doc.ENCIRCLEMENT, list(doc.ENCIRCLEMENT_REQUIRED_TYPES),
                                  attacking=True, **acc) == doc.ENCIRCLEMENT_LOSS_MULTIPLIER
        and doc.enemy_loss_multiplier(doc.ENCIRCLEMENT, list(doc.ENCIRCLEMENT_REQUIRED_TYPES),
                                      attacking=False, **acc) == 1.0
        and doc.enemy_loss_multiplier(doc.ENCIRCLEMENT, list(doc.ENCIRCLEMENT_REQUIRED_TYPES),
                                      attacking=True, **spenta) == 1.0,
    )
    controlla(
        "accerchiamento: le perdite aumentano davvero",
        doc.scaled_losses(4, doc.ENCIRCLEMENT_LOSS_MULTIPLIER) > 4
        and doc.scaled_losses(4, 1.0) == 4,
    )

    fallimenti = [e for e in esiti if not e[0]]
    print(f"Verifica dottrine ({len(esiti)} controlli su requisiti, ritardo e cooldown)")
    for _, nome, dettaglio in fallimenti:
        print(f"   KO  {nome}" + (f"  -> {dettaglio}" if dettaglio else ""))
    senza_effetto = [s for s in STRATEGIE if not doc.has_effect(s)]
    print(f"   {len(esiti) - len(fallimenti)}/{len(esiti)} superati")
    print(f"   dottrine con effetto implementato: "
          f"{len(STRATEGIE) - len(senza_effetto)}/{len(STRATEGIE)}"
          f"   (senza: {', '.join(senza_effetto)})")
    return len(fallimenti)


def _scenari_scelti(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if not args.scenari:
        return SCENARI
    scelti = [SCENARI_PER_ID[s] for s in args.scenari if s in SCENARI_PER_ID]
    if not scelti:
        raise SystemExit(f"Nessuno scenario valido fra {args.scenari}. "
                         f"Disponibili: {', '.join(SCENARI_PER_ID)}")
    return scelti


def _seed_di(base: int, scenario_id: str, difficolta: str, replica: int) -> int:
    """Seed deterministico e stabile: la stessa replica si rigioca identica.

    Dipende dallo scenario e dalla difficolta' ma NON dalla strategia: le tre
    condizioni di E2 devono vedere la stessa mappa e la stessa IA.
    """
    sorgente = f"{base}|{scenario_id}|{difficolta}|{replica}"
    return random.Random(sorgente).randrange(2 ** 31)


def _totale_e2(args: argparse.Namespace) -> int:
    return len(_scenari_scelti(args)) * len(args.difficolta) * 3 * args.repliche


_ultimo_avanzamento = [-1]


def _avanzamento(fatte: int, totale: int) -> None:
    percentuale = int(fatte / max(1, totale) * 100)
    if percentuale != _ultimo_avanzamento[0] and percentuale % 5 == 0:
        _ultimo_avanzamento[0] = percentuale
        print(f"   ... {fatte}/{totale} partite ({percentuale}%)", flush=True)


# ══════════════════════════════════════════════════════════════════
# SCRITTURA
# ══════════════════════════════════════════════════════════════════

def scrivi_csv(righe: List[Dict[str, Any]], percorso: Path, separatore: str) -> None:
    if not righe:
        return
    # L'unione delle chiavi: esperimenti diversi aggiungono colonne diverse.
    colonne: List[str] = []
    for riga in righe:
        for chiave in riga:
            if chiave not in colonne:
                colonne.append(chiave)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: senza BOM Excel sbaglia gli accenti nei nomi delle unita.
    with percorso.open("w", encoding="utf-8-sig", newline="") as flusso:
        scrittore = csv.DictWriter(flusso, fieldnames=colonne, delimiter=separatore,
                                   extrasaction="ignore")
        scrittore.writeheader()
        for riga in righe:
            scrittore.writerow(riga)
    print(f"   scritto {_percorso_leggibile(percorso)}  "
          f"({len(righe)} righe, {len(colonne)} colonne)")


def _percorso_leggibile(percorso: Path) -> str:
    """Relativo al progetto quando ci sta dentro, assoluto quando `--out` e' altrove."""
    try:
        return str(percorso.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(percorso)


def _commit_corrente() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "sconosciuto"
    except Exception:
        return "sconosciuto"


def scrivi_manifest(args: argparse.Namespace, conteggi: Dict[str, int],
                    percorso: Path, *, completati: Optional[List[str]] = None,
                    parziale: bool = False, rumoroso: bool = True) -> None:
    """Come e' stato prodotto il dataset. Senza, i numeri non sono rintracciabili."""
    manifest = {
        "generato": datetime.now().isoformat(timespec="seconds"),
        "commit": _commit_corrente(),
        "comando": " ".join(sys.argv),
        # Un dataset interrotto a meta' non va confuso con uno completo.
        "completo": not parziale,
        "esperimenti_completati": list(completati or []),
        "parametri": {
            "esperimenti": args.esperimento,
            "seed_base": args.seed,
            "repliche": args.repliche,
            "turni_massimi": args.turni,
            "difficolta": list(args.difficolta),
            "scenari": args.scenari or [s["id"] for s in SCENARI],
        },
        "taratura": {
            "compat_layer": cmp_layer.describe() if cmp_layer else None,
            "dottrine_presenti": bool(getattr(sessione_modulo, "doc", None)),
        },
        "politica_giocatore": GiocatoreScriptato.__doc__,
        "righe_prodotte": conteggi,
        "note": (
            "[SETUP-RULE] terreno/meteo/stato nelle colonne *_scenario sono le "
            "condizioni valutate dall'advisor, non quelle della partita: in "
            "partita il meteo e' sorteggiato dal seed e le truppe partono "
            "sempre fresche."
        ),
    }
    percorso.parent.mkdir(parents=True, exist_ok=True)
    percorso.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    if rumoroso:
        print(f"   scritto {_percorso_leggibile(percorso)}")


# ══════════════════════════════════════════════════════════════════
# ANALISI DEI RISULTATI
# ══════════════════════════════════════════════════════════════════
#
# Punto 4.5 della specifica: per ogni condizione media, deviazione standard e
# numero di repliche; per gli esiti binari il win rate con intervallo di
# confidenza; grafici semplici. Legge i CSV gia' prodotti, non rigioca nulla.
#
# I grafici sono SVG scritti a mano invece che matplotlib: evita una
# dipendenza in piu' da installare, resta vettoriale per la stampa della tesi
# e si apre in qualsiasi browser.

#: Metriche continue da riassumere, con etichetta leggibile e unita'.
METRICHE = [
    ("forza_residua_pct", "Forza residua", "%"),
    ("perdite_proprie", "Perdite proprie", "unita"),
    ("perdite_nemiche", "Perdite inflitte", "unita"),
    ("morale_residuo", "Morale residuo", "0-100"),
    ("turni_totali", "Durata", "turni"),
    ("scontri", "Scontri", "n"),
    ("celle_player", "Celle controllate", "n"),
]

#: Ordine di presentazione delle condizioni.
ORDINE_CONDIZIONI = ["top_ranked", "intermedia", "worst_ranked",
                     "dottrine_on", "dottrine_off"]

Z95 = 1.959964


def _media_e_ic(valori: Sequence[float]) -> Dict[str, Any]:
    """Media, deviazione standard campionaria e intervallo di confidenza al 95%."""
    puliti = [float(v) for v in valori if v == v and v != ""]
    n = len(puliti)
    if n == 0:
        return {"n": 0, "media": "", "dev_std": "", "err_std": "",
                "ic95_min": "", "ic95_max": ""}
    media = sum(puliti) / n
    if n < 2:
        return {"n": n, "media": round(media, 4), "dev_std": "", "err_std": "",
                "ic95_min": "", "ic95_max": ""}
    varianza = sum((v - media) ** 2 for v in puliti) / (n - 1)
    dev = math.sqrt(varianza)
    err = dev / math.sqrt(n)
    return {"n": n, "media": round(media, 4), "dev_std": round(dev, 4),
            "err_std": round(err, 4),
            "ic95_min": round(media - Z95 * err, 4),
            "ic95_max": round(media + Z95 * err, 4)}


def _proporzione_e_ic(successi: int, totale: int) -> Dict[str, Any]:
    """Proporzione con intervallo di Wilson.

    Il metodo di Wilson invece dell'approssimazione normale: con tassi vicini
    a 0 o a 1 — il caso della difficolta' 'hard', dove il giocatore vince quasi
    mai — l'intervallo normale sborda sotto zero e non significa piu' nulla.
    """
    if totale == 0:
        return {"n": 0, "media": "", "dev_std": "", "err_std": "",
                "ic95_min": "", "ic95_max": ""}
    p = successi / totale
    denominatore = 1 + Z95 ** 2 / totale
    centro = (p + Z95 ** 2 / (2 * totale)) / denominatore
    meta = (Z95 * math.sqrt(p * (1 - p) / totale
                            + Z95 ** 2 / (4 * totale ** 2))) / denominatore
    dev = math.sqrt(p * (1 - p)) if 0 <= p <= 1 else 0.0
    return {"n": totale, "media": round(p * 100, 4),
            "dev_std": round(dev * 100, 4),
            "err_std": round(math.sqrt(p * (1 - p) / totale) * 100, 4),
            "ic95_min": round(max(0.0, centro - meta) * 100, 4),
            "ic95_max": round(min(1.0, centro + meta) * 100, 4)}


def _differenza_medie(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """Differenza fra due medie indipendenti, con intervallo al 95%."""
    pa = [float(v) for v in a if v == v and v != ""]
    pb = [float(v) for v in b if v == v and v != ""]
    if len(pa) < 2 or len(pb) < 2:
        return {}
    ma, mb = sum(pa) / len(pa), sum(pb) / len(pb)
    va = sum((v - ma) ** 2 for v in pa) / (len(pa) - 1)
    vb = sum((v - mb) ** 2 for v in pb) / (len(pb) - 1)
    err = math.sqrt(va / len(pa) + vb / len(pb))
    diff = ma - mb
    return {"differenza": round(diff, 4), "err_std": round(err, 4),
            "ic95_min": round(diff - Z95 * err, 4),
            "ic95_max": round(diff + Z95 * err, 4),
            "ic95_esclude_zero": bool(abs(diff) > Z95 * err)}


def _differenza_proporzioni(sa: int, na: int, sb: int, nb: int) -> Dict[str, Any]:
    """Differenza fra due tassi, con intervallo al 95%."""
    if na == 0 or nb == 0:
        return {}
    pa, pb = sa / na, sb / nb
    err = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    diff = pa - pb
    return {"differenza": round(diff * 100, 4), "err_std": round(err * 100, 4),
            "ic95_min": round((diff - Z95 * err) * 100, 4),
            "ic95_max": round((diff + Z95 * err) * 100, 4),
            "ic95_esclude_zero": bool(abs(diff) > Z95 * err)}


def _quartili(valori: Sequence[float]) -> Optional[Dict[str, float]]:
    """Cinque numeri per il boxplot, con baffi a 1.5 volte lo scarto interquartile."""
    puliti = sorted(float(v) for v in valori if v == v and v != "")
    if not puliti:
        return None

    def percentile(q: float) -> float:
        if len(puliti) == 1:
            return puliti[0]
        posizione = q * (len(puliti) - 1)
        basso = int(math.floor(posizione))
        alto = min(basso + 1, len(puliti) - 1)
        return puliti[basso] + (puliti[alto] - puliti[basso]) * (posizione - basso)

    q1, mediana, q3 = percentile(0.25), percentile(0.5), percentile(0.75)
    iqr = q3 - q1
    dentro = [v for v in puliti if q1 - 1.5 * iqr <= v <= q3 + 1.5 * iqr] or puliti
    return {"min": dentro[0], "q1": q1, "mediana": mediana, "q3": q3,
            "max": dentro[-1], "n": len(puliti)}


# ── Grafici SVG ───────────────────────────────────────────────────

LARGHEZZA, ALTEZZA = 760, 420
MARGINE = {"sinistra": 72, "destra": 24, "alto": 52, "basso": 86}
COLORI = ["#2f6f9f", "#8f4fa0", "#b5602f", "#3f8f6f", "#9f4f5f"]


def _telaio(titolo: str, etichetta_y: str, valore_max: float,
            tacche: int = 5) -> Tuple[List[str], float, float, float]:
    """Assi, griglia e titolo. Ritorna (pezzi, x0, y0, altezza utile)."""
    x0 = MARGINE["sinistra"]
    y0 = ALTEZZA - MARGINE["basso"]
    utile = ALTEZZA - MARGINE["alto"] - MARGINE["basso"]
    larghezza_utile = LARGHEZZA - MARGINE["sinistra"] - MARGINE["destra"]
    pezzi = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{LARGHEZZA}" '
        f'height="{ALTEZZA}" viewBox="0 0 {LARGHEZZA} {ALTEZZA}" '
        f'font-family="Segoe UI, Arial, sans-serif">',
        f'<rect width="{LARGHEZZA}" height="{ALTEZZA}" fill="#ffffff"/>',
        f'<text x="{LARGHEZZA/2}" y="28" text-anchor="middle" font-size="16" '
        f'font-weight="600" fill="#1a1a1a">{_xml(titolo)}</text>',
    ]
    for i in range(tacche + 1):
        valore = valore_max * i / tacche
        y = y0 - utile * i / tacche
        pezzi.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+larghezza_utile}" '
                     f'y2="{y:.1f}" stroke="#e4e4e4" stroke-width="1"/>')
        pezzi.append(f'<text x="{x0-9}" y="{y+4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="#555">{valore:.0f}</text>')
    pezzi.append(f'<line x1="{x0}" y1="{y0}" x2="{x0+larghezza_utile}" y2="{y0}" '
                 f'stroke="#333" stroke-width="1.4"/>')
    pezzi.append(f'<line x1="{x0}" y1="{MARGINE["alto"]}" x2="{x0}" y2="{y0}" '
                 f'stroke="#333" stroke-width="1.4"/>')
    pezzi.append(f'<text x="18" y="{MARGINE["alto"]+utile/2}" font-size="12" '
                 f'fill="#333" transform="rotate(-90 18 {MARGINE["alto"]+utile/2})" '
                 f'text-anchor="middle">{_xml(etichetta_y)}</text>')
    return pezzi, x0, y0, utile


def _xml(testo: str) -> str:
    return (str(testo).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _etichetta_x(pezzi: List[str], x: float, y0: float, testo: str) -> None:
    pezzi.append(f'<text x="{x:.1f}" y="{y0+18}" text-anchor="middle" '
                 f'font-size="11" fill="#333">{_xml(testo)}</text>')


def grafico_barre(titolo: str, etichetta_y: str, gruppi: List[str],
                  serie: List[Tuple[str, List[Optional[Dict[str, Any]]]]]) -> str:
    """Barre raggruppate con barre d'errore (intervallo di confidenza al 95%)."""
    valori = [v["ic95_max"] for _, dati in serie for v in dati
              if v and isinstance(v.get("ic95_max"), (int, float))]
    valori += [v["media"] for _, dati in serie for v in dati
               if v and isinstance(v.get("media"), (int, float))]
    massimo = max(valori) * 1.12 if valori else 1.0
    pezzi, x0, y0, utile = _telaio(titolo, etichetta_y, massimo)
    larghezza_utile = LARGHEZZA - MARGINE["sinistra"] - MARGINE["destra"]

    passo = larghezza_utile / max(1, len(gruppi))
    larghezza_barra = passo * 0.72 / max(1, len(serie))
    for indice_gruppo, gruppo in enumerate(gruppi):
        centro = x0 + passo * (indice_gruppo + 0.5)
        _etichetta_x(pezzi, centro, y0, gruppo)
        for indice_serie, (nome, dati) in enumerate(serie):
            voce = dati[indice_gruppo] if indice_gruppo < len(dati) else None
            if not voce or not isinstance(voce.get("media"), (int, float)):
                continue
            offset = (indice_serie - (len(serie) - 1) / 2) * larghezza_barra
            x = centro + offset - larghezza_barra / 2
            altezza = utile * voce["media"] / massimo
            colore = COLORI[indice_serie % len(COLORI)]
            pezzi.append(f'<rect x="{x:.1f}" y="{y0-altezza:.1f}" '
                         f'width="{larghezza_barra*0.92:.1f}" height="{altezza:.1f}" '
                         f'fill="{colore}" opacity="0.88"/>')
            # Centro della barra: serve sia alla barra d'errore sia all'etichetta,
            # quindi va calcolato comunque, anche quando l'intervallo manca.
            xm = x + larghezza_barra * 0.46
            cima = y0 - altezza
            if isinstance(voce.get("ic95_min"), (int, float)):
                y_basso = y0 - utile * voce["ic95_min"] / massimo
                y_alto = y0 - utile * voce["ic95_max"] / massimo
                cima = min(cima, y_alto)
                pezzi.append(f'<line x1="{xm:.1f}" y1="{y_basso:.1f}" x2="{xm:.1f}" '
                             f'y2="{y_alto:.1f}" stroke="#222" stroke-width="1.3"/>')
                for yy in (y_basso, y_alto):
                    pezzi.append(f'<line x1="{xm-4:.1f}" y1="{yy:.1f}" '
                                 f'x2="{xm+4:.1f}" y2="{yy:.1f}" stroke="#222" '
                                 f'stroke-width="1.3"/>')
            # L'etichetta va sopra il baffo superiore, non sopra la barra:
            # altrimenti i due si sovrappongono e il numero diventa illeggibile.
            pezzi.append(f'<text x="{xm:.1f}" y="{cima-6:.1f}" text-anchor="middle" '
                         f'font-size="10" fill="#333">{voce["media"]:.1f}</text>')
    pezzi.append(_legenda([nome for nome, _ in serie]))
    pezzi.append("</svg>")
    return "\n".join(pezzi)


def grafico_box(titolo: str, etichetta_y: str,
                categorie: List[Tuple[str, Optional[Dict[str, float]]]]) -> str:
    """Boxplot: mediana, quartili e baffi."""
    valori = [v for _, q in categorie if q for v in (q["max"], q["q3"])]
    massimo = max(valori) * 1.12 if valori else 1.0
    pezzi, x0, y0, utile = _telaio(titolo, etichetta_y, massimo)
    larghezza_utile = LARGHEZZA - MARGINE["sinistra"] - MARGINE["destra"]
    passo = larghezza_utile / max(1, len(categorie))
    larghezza_box = min(74.0, passo * 0.46)

    def y_di(v: float) -> float:
        return y0 - utile * v / massimo

    for indice, (nome, q) in enumerate(categorie):
        centro = x0 + passo * (indice + 0.5)
        _etichetta_x(pezzi, centro, y0, nome)
        if not q:
            continue
        colore = COLORI[indice % len(COLORI)]
        # I baffi solo FUORI dalla scatola, come vuole la convenzione: un'unica
        # linea da min a max attraverserebbe il box e si confonderebbe con la
        # mediana.
        pezzi.append(f'<line x1="{centro}" y1="{y_di(q["min"]):.1f}" x2="{centro}" '
                     f'y2="{y_di(q["q1"]):.1f}" stroke="#444" stroke-width="1.2"/>')
        pezzi.append(f'<line x1="{centro}" y1="{y_di(q["q3"]):.1f}" x2="{centro}" '
                     f'y2="{y_di(q["max"]):.1f}" stroke="#444" stroke-width="1.2"/>')
        for v in (q["min"], q["max"]):
            pezzi.append(f'<line x1="{centro-larghezza_box/4:.1f}" y1="{y_di(v):.1f}" '
                         f'x2="{centro+larghezza_box/4:.1f}" y2="{y_di(v):.1f}" '
                         f'stroke="#444" stroke-width="1.2"/>')
        alto, basso = y_di(q["q3"]), y_di(q["q1"])
        pezzi.append(f'<rect x="{centro-larghezza_box/2:.1f}" y="{alto:.1f}" '
                     f'width="{larghezza_box:.1f}" height="{max(1.0, basso-alto):.1f}" '
                     f'fill="{colore}" opacity="0.30" stroke="{colore}" stroke-width="1.4"/>')
        pezzi.append(f'<line x1="{centro-larghezza_box/2:.1f}" y1="{y_di(q["mediana"]):.1f}" '
                     f'x2="{centro+larghezza_box/2:.1f}" y2="{y_di(q["mediana"]):.1f}" '
                     f'stroke="{colore}" stroke-width="2.4"/>')
        pezzi.append(f'<text x="{centro}" y="{y0+34}" text-anchor="middle" '
                     f'font-size="10" fill="#777">n={q["n"]}</text>')
    pezzi.append("</svg>")
    return "\n".join(pezzi)


def grafico_linea(titolo: str, etichetta_x: str, etichetta_y: str,
                  punti: List[Tuple[float, Dict[str, Any]]]) -> str:
    """Spezzata con banda di confidenza: compatibilita' -> esito."""
    if not punti:
        return ""
    massimo = max(p[1]["ic95_max"] for p in punti
                  if isinstance(p[1].get("ic95_max"), (int, float))) * 1.15 or 1.0
    pezzi, x0, y0, utile = _telaio(titolo, etichetta_y, massimo)
    larghezza_utile = LARGHEZZA - MARGINE["sinistra"] - MARGINE["destra"]
    xs = [p[0] for p in punti]
    x_min, x_max = min(xs), max(xs)
    campo = (x_max - x_min) or 1.0

    def x_di(v: float) -> float:
        return x0 + larghezza_utile * 0.06 + (larghezza_utile * 0.88) * (v - x_min) / campo

    def y_di(v: float) -> float:
        return y0 - utile * v / massimo

    banda_alto = " ".join(f"{x_di(x):.1f},{y_di(d['ic95_max']):.1f}" for x, d in punti)
    banda_basso = " ".join(f"{x_di(x):.1f},{y_di(d['ic95_min']):.1f}"
                           for x, d in reversed(punti))
    pezzi.append(f'<polygon points="{banda_alto} {banda_basso}" fill="{COLORI[0]}" '
                 f'opacity="0.16"/>')
    linea = " ".join(f"{x_di(x):.1f},{y_di(d['media']):.1f}" for x, d in punti)
    pezzi.append(f'<polyline points="{linea}" fill="none" stroke="{COLORI[0]}" '
                 f'stroke-width="2.2"/>')
    for x, d in punti:
        pezzi.append(f'<circle cx="{x_di(x):.1f}" cy="{y_di(d["media"]):.1f}" r="4" '
                     f'fill="{COLORI[0]}"/>')
        _etichetta_x(pezzi, x_di(x), y0, f"{x:.0f}")
        pezzi.append(f'<text x="{x_di(x):.1f}" y="{y0+34}" text-anchor="middle" '
                     f'font-size="10" fill="#777">n={d["n"]}</text>')
    pezzi.append(f'<text x="{x0+larghezza_utile/2}" y="{ALTEZZA-26}" '
                 f'text-anchor="middle" font-size="12" fill="#333">{_xml(etichetta_x)}</text>')
    pezzi.append("</svg>")
    return "\n".join(pezzi)


def _legenda(nomi: List[str]) -> str:
    pezzi = []
    x = MARGINE["sinistra"]
    y = ALTEZZA - 30
    for indice, nome in enumerate(nomi):
        colore = COLORI[indice % len(COLORI)]
        pezzi.append(f'<rect x="{x}" y="{y-9}" width="12" height="12" fill="{colore}" '
                     f'opacity="0.88"/>')
        pezzi.append(f'<text x="{x+17}" y="{y+1}" font-size="11" fill="#333">'
                     f'{_xml(nome)}</text>')
        x += 24 + 7.4 * len(nome)
    return "\n".join(pezzi)


# ── Lettura e aggregazione ────────────────────────────────────────

def _leggi_csv(percorso: Path, separatore: str) -> List[Dict[str, Any]]:
    if not percorso.exists():
        return []
    with percorso.open(encoding="utf-8-sig", newline="") as flusso:
        return list(csv.DictReader(flusso, delimiter=separatore))


def _numero(riga: Dict[str, Any], chiave: str) -> Optional[float]:
    valore = riga.get(chiave, "")
    if valore in ("", None):
        return None
    try:
        return float(valore)
    except (TypeError, ValueError):
        return None


def _chiave_ordine(condizione: str) -> int:
    return (ORDINE_CONDIZIONI.index(condizione)
            if condizione in ORDINE_CONDIZIONI else len(ORDINE_CONDIZIONI))


def _ordina_difficolta(valori: Iterable[str]) -> List[str]:
    """Per durezza crescente, non in ordine alfabetico.

    Alfabeticamente verrebbe easy, hard, normal: un grafico in cui la
    difficolta' non cresce da sinistra a destra si legge male.
    """
    return sorted(set(valori),
                  key=lambda d: DIFFICOLTA.index(d) if d in DIFFICOLTA else len(DIFFICOLTA))


def analizza(cartella: Path, separatore: str) -> int:
    """Aggrega i CSV gia' prodotti in tabelle di sintesi e grafici.

    Non rigioca nulla: legge `runs.csv` e produce medie, deviazioni standard,
    numerosita', win rate con intervallo di confidenza, i confronti fra
    condizioni e i grafici del punto 4.5.
    """
    runs = _leggi_csv(cartella / "runs.csv", separatore)
    if not runs:
        print(f"Nessun runs.csv leggibile in {_percorso_leggibile(cartella)}.")
        return 1
    print(f"Analisi di {len(runs)} partite da {_percorso_leggibile(cartella)}")

    righe_condizioni = _tabella_condizioni(runs)
    righe_confronti = _tabella_confronti(runs)
    righe_scenari = _tabella_scenari(runs)

    scrivi_csv(righe_condizioni, cartella / "analisi_condizioni.csv", separatore)
    scrivi_csv(righe_confronti, cartella / "analisi_confronti.csv", separatore)
    scrivi_csv(righe_scenari, cartella / "analisi_scenari.csv", separatore)
    _scrivi_grafici(runs, cartella)
    _stampa_sintesi(righe_condizioni, righe_confronti)
    return 0


def _gruppi(runs: List[Dict[str, Any]], esperimento: str) -> List[Tuple[str, str]]:
    """Coppie (difficolta, condizione) presenti, piu' la riga aggregata."""
    ordine_d = _ordina_difficolta(r["difficolta_ia"] for r in runs
                                  if r.get("esperimento") == esperimento)
    presenti = sorted({(r["difficolta_ia"], r["condizione"]) for r in runs
                       if r.get("esperimento") == esperimento},
                      key=lambda x: (ordine_d.index(x[0]), _chiave_ordine(x[1])))
    condizioni = sorted({c for _, c in presenti}, key=_chiave_ordine)
    return [("tutte", c) for c in condizioni] + presenti


def _filtra(runs: List[Dict[str, Any]], esperimento: str, difficolta: str,
            condizione: str) -> List[Dict[str, Any]]:
    return [r for r in runs
            if r.get("esperimento") == esperimento
            and r.get("condizione") == condizione
            and (difficolta == "tutte" or r.get("difficolta_ia") == difficolta)]


def _tabella_condizioni(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Una riga per (esperimento, difficolta, condizione, metrica)."""
    righe: List[Dict[str, Any]] = []
    for esperimento in sorted({r.get("esperimento", "") for r in runs}):
        for difficolta, condizione in _gruppi(runs, esperimento):
            gruppo = _filtra(runs, esperimento, difficolta, condizione)
            if not gruppo:
                continue
            comune = {"esperimento": esperimento, "difficolta_ia": difficolta,
                      "condizione": condizione,
                      "rank_advisor_medio": _media_e_ic(
                          [v for v in (_numero(r, "rank_advisor") for r in gruppo)
                           if v is not None])["media"]}
            # esito binario: win rate con intervallo di Wilson
            vittorie = sum(1 for r in gruppo if r.get("esito") == ESITO_VITTORIA)
            righe.append({**comune, "metrica": "win_rate", "unita": "%",
                          **_proporzione_e_ic(vittorie, len(gruppo))})
            sconfitte = sum(1 for r in gruppo if r.get("esito") == ESITO_SCONFITTA)
            righe.append({**comune, "metrica": "tasso_sconfitte", "unita": "%",
                          **_proporzione_e_ic(sconfitte, len(gruppo))})
            indecise = sum(1 for r in gruppo if r.get("esito") == ESITO_NON_DECISA)
            righe.append({**comune, "metrica": "tasso_non_decise", "unita": "%",
                          **_proporzione_e_ic(indecise, len(gruppo))})
            for colonna, etichetta, unita in METRICHE:
                valori = [v for v in (_numero(r, colonna) for r in gruppo)
                          if v is not None]
                righe.append({**comune, "metrica": colonna, "unita": unita,
                              **_media_e_ic(valori)})
    return righe


#: I confronti che la specifica chiede di mettere in evidenza.
CONFRONTI = {
    "E2": [("top_ranked", "worst_ranked"), ("top_ranked", "intermedia"),
           ("intermedia", "worst_ranked")],
    "E4": [("dottrine_on", "dottrine_off")],
}


def _tabella_confronti(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Differenze fra condizioni, con intervallo al 95%."""
    righe: List[Dict[str, Any]] = []
    for esperimento, coppie in CONFRONTI.items():
        difficolta_presenti = ["tutte"] + _ordina_difficolta(
            r["difficolta_ia"] for r in runs if r.get("esperimento") == esperimento)
        for difficolta in difficolta_presenti:
            for a, b in coppie:
                ga = _filtra(runs, esperimento, difficolta, a)
                gb = _filtra(runs, esperimento, difficolta, b)
                if not ga or not gb:
                    continue
                comune = {"esperimento": esperimento, "difficolta_ia": difficolta,
                          "condizione_a": a, "condizione_b": b,
                          "n_a": len(ga), "n_b": len(gb)}
                va = sum(1 for r in ga if r.get("esito") == ESITO_VITTORIA)
                vb = sum(1 for r in gb if r.get("esito") == ESITO_VITTORIA)
                diff = _differenza_proporzioni(va, len(ga), vb, len(gb))
                if diff:
                    righe.append({**comune, "metrica": "win_rate", "unita": "punti %",
                                  "media_a": round(va / len(ga) * 100, 2),
                                  "media_b": round(vb / len(gb) * 100, 2), **diff})
                for colonna, etichetta, unita in METRICHE:
                    xa = [v for v in (_numero(r, colonna) for r in ga) if v is not None]
                    xb = [v for v in (_numero(r, colonna) for r in gb) if v is not None]
                    diff = _differenza_medie(xa, xb)
                    if diff:
                        righe.append({**comune, "metrica": colonna, "unita": unita,
                                      "media_a": round(sum(xa) / len(xa), 2),
                                      "media_b": round(sum(xb) / len(xb), 2), **diff})
    return righe


def _tabella_scenari(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lo stesso quadro, scenario per scenario: serve a vedere se un effetto
    medio nasconde comportamenti opposti fra composizioni diverse."""
    righe: List[Dict[str, Any]] = []
    chiavi = sorted({(r.get("esperimento", ""), r.get("scenario_id", ""),
                      r.get("condizione", "")) for r in runs},
                    key=lambda x: (x[0], x[1], _chiave_ordine(x[2])))
    for esperimento, scenario, condizione in chiavi:
        gruppo = [r for r in runs if r.get("esperimento") == esperimento
                  and r.get("scenario_id") == scenario
                  and r.get("condizione") == condizione]
        if not gruppo:
            continue
        vittorie = sum(1 for r in gruppo if r.get("esito") == ESITO_VITTORIA)
        forza = [v for v in (_numero(r, "forza_residua_pct") for r in gruppo)
                 if v is not None]
        riepilogo_forza = _media_e_ic(forza)
        righe.append({
            "esperimento": esperimento, "scenario_id": scenario,
            "composizione": gruppo[0].get("composizione", ""),
            "condizione": condizione,
            "strategia_nome": gruppo[0].get("strategia_nome", ""),
            "n": len(gruppo),
            "win_rate_pct": round(vittorie / len(gruppo) * 100, 2),
            **{f"win_{k}": v for k, v in
               _proporzione_e_ic(vittorie, len(gruppo)).items()
               if k in ("ic95_min", "ic95_max")},
            "forza_residua_media": riepilogo_forza["media"],
            "forza_residua_dev_std": riepilogo_forza["dev_std"],
            "rank_advisor": gruppo[0].get("rank_advisor", ""),
            "compat_combat_pct": gruppo[0].get("compat_combat_pct", ""),
            "dottrina_scontri_attiva_media": round(
                sum((_numero(r, "dottrina_scontri_attiva") or 0) for r in gruppo)
                / len(gruppo), 2),
        })
    return righe


def _scrivi_grafici(runs: List[Dict[str, Any]], cartella: Path) -> None:
    """I tre grafici chiesti dalla specifica: barre con errore, boxplot, linea."""
    prodotti: List[str] = []

    # 1. barre con barre d'errore: win rate per condizione e difficolta (E2)
    e2 = [r for r in runs if r.get("esperimento") == "E2"]
    if e2:
        difficolta = _ordina_difficolta(r["difficolta_ia"] for r in e2)
        condizioni = sorted({r["condizione"] for r in e2}, key=_chiave_ordine)
        serie = []
        for condizione in condizioni:
            dati = []
            for d in difficolta:
                gruppo = _filtra(runs, "E2", d, condizione)
                vittorie = sum(1 for r in gruppo if r.get("esito") == ESITO_VITTORIA)
                dati.append(_proporzione_e_ic(vittorie, len(gruppo)) if gruppo else None)
            serie.append((condizione, dati))
        svg = grafico_barre(
            "E2 - Tasso di vittoria per strategia scelta (IC 95%)",
            "vittorie (%)", difficolta, serie)
        (cartella / "grafico_e2_win_rate.svg").write_text(svg, encoding="utf-8")
        prodotti.append("grafico_e2_win_rate.svg")

        # 2. boxplot: distribuzione della forza residua per condizione
        categorie = []
        for condizione in condizioni:
            gruppo = _filtra(runs, "E2", "tutte", condizione)
            valori = [v for v in (_numero(r, "forza_residua_pct") for r in gruppo)
                      if v is not None]
            categorie.append((condizione, _quartili(valori)))
        svg = grafico_box("E2 - Distribuzione della forza residua",
                          "forza residua (%)", categorie)
        (cartella / "grafico_e2_forza_residua.svg").write_text(svg, encoding="utf-8")
        prodotti.append("grafico_e2_forza_residua.svg")

        # 3. linea: compatibilita di combattimento -> esito
        punti = _curva_compatibilita(e2)
        if punti:
            svg = grafico_linea(
                "E2 - Compatibilita di combattimento e tasso di vittoria",
                "compatibilita combat (%), centro della fascia",
                "vittorie (%)", punti)
            (cartella / "grafico_e2_compatibilita_esito.svg").write_text(svg, encoding="utf-8")
            prodotti.append("grafico_e2_compatibilita_esito.svg")

    # 4. ablation delle dottrine
    e4 = [r for r in runs if r.get("esperimento") == "E4"]
    if e4:
        difficolta = _ordina_difficolta(r["difficolta_ia"] for r in e4)
        serie = []
        for condizione in ("dottrine_on", "dottrine_off"):
            dati = []
            for d in difficolta:
                gruppo = _filtra(runs, "E4", d, condizione)
                vittorie = sum(1 for r in gruppo if r.get("esito") == ESITO_VITTORIA)
                dati.append(_proporzione_e_ic(vittorie, len(gruppo)) if gruppo else None)
            serie.append((condizione, dati))
        svg = grafico_barre("E4 - Effetto del layer dottrine (IC 95%)",
                            "vittorie (%)", difficolta, serie)
        (cartella / "grafico_e4_dottrine.svg").write_text(svg, encoding="utf-8")
        prodotti.append("grafico_e4_dottrine.svg")

    for nome in prodotti:
        print(f"   scritto {_percorso_leggibile(cartella / nome)}")


#: Ampiezza delle fasce di compatibilita, in punti percentuali.
PASSO_FASCIA = 5.0


def _curva_compatibilita(runs: List[Dict[str, Any]]) -> List[Tuple[float, Dict[str, Any]]]:
    """Win rate per fascia di compatibilita di combattimento.

    E' la relazione che l'esperimento vuole mettere alla prova, letta senza
    passare dalle tre condizioni: ogni partita entra nella fascia della sua
    compatibilita, e si guarda se il tasso di vittoria sale con essa.
    """
    fasce: Dict[float, List[Dict[str, Any]]] = {}
    for riga in runs:
        compat = _numero(riga, "compat_combat_pct")
        if compat is None:
            continue
        centro = math.floor(compat / PASSO_FASCIA) * PASSO_FASCIA + PASSO_FASCIA / 2
        fasce.setdefault(centro, []).append(riga)
    punti = []
    for centro in sorted(fasce):
        gruppo = fasce[centro]
        if len(gruppo) < 20:          # fasce troppo rade non dicono nulla
            continue
        vittorie = sum(1 for r in gruppo if r.get("esito") == ESITO_VITTORIA)
        punti.append((centro, _proporzione_e_ic(vittorie, len(gruppo))))
    return punti


def _stampa_sintesi(condizioni: List[Dict[str, Any]],
                    confronti: List[Dict[str, Any]]) -> None:
    """Le due tabelle che servono davvero a colpo d'occhio."""
    print("\nRisultati osservati (tutte le difficolta):")
    print(f"   {'esp':<4} {'condizione':<14} {'n':>5} {'vittorie':>9} "
          f"{'IC 95%':>16} {'forza res.':>11} {'dev.std':>8}")
    for riga in condizioni:
        if riga["difficolta_ia"] != "tutte" or riga["metrica"] != "win_rate":
            continue
        forza = next((x for x in condizioni
                      if x["esperimento"] == riga["esperimento"]
                      and x["difficolta_ia"] == "tutte"
                      and x["condizione"] == riga["condizione"]
                      and x["metrica"] == "forza_residua_pct"), {})
        ic = f"[{riga['ic95_min']:.1f}, {riga['ic95_max']:.1f}]"
        print(f"   {riga['esperimento']:<4} {riga['condizione']:<14} {riga['n']:>5} "
              f"{riga['media']:>8.1f}% {ic:>16} "
              f"{forza.get('media', 0):>10.1f}% {forza.get('dev_std', 0):>8.1f}")

    print("\nConfronti (differenza A - B, intervallo al 95%):")
    print(f"   {'esp':<4} {'A':<13} {'B':<14} {'metrica':<20} {'diff':>8} "
          f"{'IC 95%':>18}  esclude 0")
    for riga in confronti:
        if riga["difficolta_ia"] != "tutte":
            continue
        if riga["metrica"] not in ("win_rate", "forza_residua_pct"):
            continue
        ic = f"[{riga['ic95_min']:.1f}, {riga['ic95_max']:.1f}]"
        print(f"   {riga['esperimento']:<4} {riga['condizione_a']:<13} "
              f"{riga['condizione_b']:<14} {riga['metrica']:<20} "
              f"{riga['differenza']:>+8.1f} {ic:>18}  "
              f"{'SI' if riga['ic95_esclude_zero'] else 'no'}")
    print("\n   (I numeri qui sopra sono risultati osservati. "
          "L'interpretazione va scritta a parte.)")


# ══════════════════════════════════════════════════════════════════
# AVVIO
# ══════════════════════════════════════════════════════════════════

ESPERIMENTI = {
    "E1": esperimento_e1,
    "E2": esperimento_e2,
    "E3": esperimento_e3,
    "E4": esperimento_e4,
}


def _argomenti(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    analizzatore = argparse.ArgumentParser(
        description="Esegue gli esperimenti di valutazione e scrive i CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    analizzatore.add_argument(
        "--esperimento", nargs="+", default=["E1"],
        choices=list(ESPERIMENTI) + ["tutti"],
        help="quali esperimenti eseguire (default: E1)",
    )
    analizzatore.add_argument("--seed", type=int, default=SEED_DEFAULT,
                              help=f"seed base, deterministico (default: {SEED_DEFAULT})")
    analizzatore.add_argument("--repliche", type=int, default=REPLICHE_DEFAULT,
                              help=f"partite per condizione (default: {REPLICHE_DEFAULT})")
    analizzatore.add_argument("--turni", type=int, default=TURNI_DEFAULT,
                              help=f"tetto di turni per partita (default: {TURNI_DEFAULT})")
    analizzatore.add_argument("--difficolta", nargs="+", default=["normal"],
                              choices=list(DIFFICOLTA),
                              help="livelli IA da confrontare (default: normal)")
    analizzatore.add_argument("--scenari", nargs="*", default=None,
                              help="id degli scenari (default: tutti)")
    analizzatore.add_argument("--out", type=Path, default=OUTPUT_DIR,
                              help="cartella di output")
    analizzatore.add_argument("--separatore", default=",",
                              help="separatore CSV (per Excel italiano: ';')")
    analizzatore.add_argument("--lista-scenari", action="store_true",
                              help="elenca gli scenari e termina")
    analizzatore.add_argument("--verifica", action="store_true",
                              help="controlla requisiti e cooldown delle dottrine, poi termina")
    analizzatore.add_argument("--unisci", nargs="+", type=Path, default=None,
                              metavar="CARTELLA",
                              help="fonde piu' esecuzioni in --out: per ogni esperimento "
                                   "vince l'ultima cartella indicata che lo contiene")
    analizzatore.add_argument("--analizza", action="store_true",
                              help="aggrega i CSV gia' presenti in --out (medie, deviazioni "
                                   "standard, win rate con IC, grafici) senza rigiocare nulla")
    return analizzatore.parse_args(argv)


def unisci_dataset(cartelle: Sequence[Path], destinazione: Path,
                   separatore: str) -> int:
    """Fonde piu' esecuzioni in un dataset solo, esperimento per esperimento.

    Serve quando si rifa' un esperimento senza rieseguire tutto: la regola e'
    che **per ogni esperimento vince l'ultima cartella che lo contiene**. Cosi'
    rigenerare E4 e passarlo per ultimo sostituisce il vecchio E4 e lascia
    intatti E1, E2 ed E3.

    Non si possono semplicemente concatenare i file: il vecchio esperimento
    resterebbe dentro accanto al nuovo, e le medie risulterebbero sporche.
    """
    if len(cartelle) < 2:
        print("Servono almeno due cartelle da unire.")
        return 1

    # provenienza[nome_file][esperimento] = (cartella, righe)
    provenienza: Dict[str, Dict[str, Tuple[Path, List[Dict[str, Any]]]]] = {}
    for cartella in cartelle:
        if not cartella.is_dir():
            print(f"Cartella inesistente: {cartella}")
            return 1
        for nome_file in ("runs", "battles", "advisor"):
            percorso = cartella / f"{nome_file}.csv"
            if not percorso.exists():
                continue
            with percorso.open(encoding="utf-8-sig", newline="") as flusso:
                righe = list(csv.DictReader(flusso, delimiter=separatore))
            per_esperimento: Dict[str, List[Dict[str, Any]]] = {}
            for riga in righe:
                per_esperimento.setdefault(riga.get("esperimento", "?"), []).append(riga)
            for codice, blocco in per_esperimento.items():
                provenienza.setdefault(nome_file, {})[codice] = (cartella, blocco)

    print(f"Unione di {len(cartelle)} cartelle in {_percorso_leggibile(destinazione)}")
    conteggi: Dict[str, int] = {}
    dettaglio_provenienza: Dict[str, Dict[str, str]] = {}
    for nome_file, per_esperimento in sorted(provenienza.items()):
        unite: List[Dict[str, Any]] = []
        for codice in sorted(per_esperimento):
            cartella, righe = per_esperimento[codice]
            unite.extend(righe)
            dettaglio_provenienza.setdefault(nome_file, {})[codice] = str(cartella)
            print(f"   {nome_file}.csv  {codice}: {len(righe):>6} righe da {cartella.name}")
        if unite:
            scrivi_csv(unite, destinazione / f"{nome_file}.csv", separatore)
            conteggi[nome_file] = len(unite)

    manifest = {
        "generato": datetime.now().isoformat(timespec="seconds"),
        "commit": _commit_corrente(),
        "comando": " ".join(sys.argv),
        "tipo": "unione",
        "completo": True,
        "cartelle_unite": [str(c) for c in cartelle],
        "provenienza": dettaglio_provenienza,
        "manifest_originali": _manifest_originali(cartelle),
        "righe_prodotte": conteggi,
        "nota": ("Per ogni esperimento sono state tenute le righe dell'ultima "
                 "cartella che lo conteneva."),
    }
    destinazione.mkdir(parents=True, exist_ok=True)
    (destinazione / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"   scritto {_percorso_leggibile(destinazione / 'manifest.json')}")
    return 0


def _manifest_originali(cartelle: Sequence[Path]) -> List[Dict[str, Any]]:
    """I manifest di partenza, conservati: il dataset unito resta tracciabile."""
    raccolti = []
    for cartella in cartelle:
        percorso = cartella / "manifest.json"
        if percorso.exists():
            try:
                raccolti.append({"cartella": str(cartella),
                                 **json.loads(percorso.read_text(encoding="utf-8"))})
            except Exception:
                continue
    return raccolti


def _elenca_scenari() -> None:
    print("Scenari disponibili:\n")
    for scenario in SCENARI:
        print(f"  {scenario['id']}  {scenario['descrizione']}")
        print(f"       {_descrivi_composizione(scenario['unita'])}  |  "
              f"{scenario['terreno']} / {scenario['meteo']} / {scenario['stato']}")
        if scenario["attese"]:
            print(f"       attese: {', '.join(scenario['attese'])}")
        print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _argomenti(argv)
    if args.lista_scenari:
        _elenca_scenari()
        return 0
    if args.verifica:
        return 1 if verifica_dottrine() else 0
    if args.unisci:
        return unisci_dataset(args.unisci, args.out, args.separatore)
    if args.analizza:
        return analizza(args.out, args.separatore)

    scelti = list(ESPERIMENTI) if "tutti" in args.esperimento else list(args.esperimento)
    raccolta: Dict[str, List[Dict[str, Any]]] = {"runs": [], "battles": [], "advisor": []}

    inizio = datetime.now()
    print(f"War Advisor - campagna sperimentale   seed base {args.seed}")
    print(f"esperimenti: {', '.join(scelti)}   repliche: {args.repliche}   "
          f"turni: {args.turni}   difficolta: {', '.join(args.difficolta)}\n")

    # Si scrive dopo OGNI esperimento, non solo alla fine: una campagna
    # completa dura decine di minuti e un'interruzione a meta' buttava via
    # tutto il lavoro gia' fatto. Cosi' quello che e' finito resta su disco.
    completati: List[str] = []
    interrotto = False
    for nome in scelti:
        print(f"[{nome}] {ESPERIMENTI[nome].__doc__.splitlines()[0]}")
        _ultimo_avanzamento[0] = -1
        try:
            for chiave, righe in ESPERIMENTI[nome](args).items():
                raccolta[chiave].extend(righe)
            completati.append(nome)
        except KeyboardInterrupt:
            print(f"   interrotto durante {nome}: salvo quello che c'e'.")
            interrotto = True
            break
        _salva(args, raccolta, completati, parziale=True)

    print("\nScrittura:")
    conteggi = _salva(args, raccolta, completati, parziale=interrotto, rumoroso=True)

    durata = (datetime.now() - inizio).total_seconds()
    print(f"\nFatto in {durata:.1f}s." if not interrotto
          else f"\nInterrotto dopo {durata:.1f}s. Esperimenti completi: "
               f"{', '.join(completati) or 'nessuno'}.")
    if raccolta["runs"]:
        _riepilogo(raccolta["runs"])
    return 0 if conteggi and not interrotto else (1 if interrotto else 0)


def _salva(args: argparse.Namespace, raccolta: Dict[str, List[Dict[str, Any]]],
           completati: List[str], *, parziale: bool,
           rumoroso: bool = False) -> Dict[str, int]:
    """Riscrive i CSV e il manifest con quanto raccolto finora."""
    conteggi: Dict[str, int] = {}
    for nome_file, righe in raccolta.items():
        if not righe:
            continue
        conteggi[nome_file] = len(righe)
        percorso = args.out / f"{nome_file}.csv"
        if rumoroso:
            scrivi_csv(righe, percorso, args.separatore)
        else:
            _scrivi_zitto(righe, percorso, args.separatore)
    scrivi_manifest(args, conteggi, args.out / "manifest.json",
                    completati=completati, parziale=parziale, rumoroso=rumoroso)
    return conteggi


def _scrivi_zitto(righe: List[Dict[str, Any]], percorso: Path, separatore: str) -> None:
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        scrivi_csv(righe, percorso, separatore)


def _riepilogo(runs: List[Dict[str, Any]]) -> None:
    """Due righe a schermo per capire subito se il dataset ha senso."""
    print("\nRiepilogo partite:")
    chiavi = sorted({(r.get("difficolta_ia"), r.get("condizione")) for r in runs})
    print(f"   {'difficolta':<11} {'condizione':<14} {'n':>4} {'vitt.':>6} "
          f"{'perdite':>8} {'forza res.':>11} {'turni':>7}")
    for difficolta, condizione in chiavi:
        gruppo = [r for r in runs
                  if r.get("difficolta_ia") == difficolta and r.get("condizione") == condizione]
        if not gruppo:
            continue
        vittorie = sum(1 for r in gruppo if r["esito"] == ESITO_VITTORIA)
        perdite = sum(r["perdite_proprie"] for r in gruppo) / len(gruppo)
        forza = [r["forza_residua_pct"] for r in gruppo if r["forza_residua_pct"] != ""]
        turni = sum(r["turni_totali"] for r in gruppo) / len(gruppo)
        print(f"   {str(difficolta):<11} {str(condizione):<14} {len(gruppo):>4} "
              f"{vittorie / len(gruppo) * 100:>5.0f}% {perdite:>8.1f} "
              f"{(sum(forza) / len(forza) if forza else 0):>10.1f}% {turni:>7.0f}")


if __name__ == "__main__":
    raise SystemExit(main())

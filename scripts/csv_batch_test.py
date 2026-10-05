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

──────────────────────────────────────────────────────────────────────
USO

  python scripts/csv_batch_test.py --esperimento tutti
  python scripts/csv_batch_test.py --esperimento E2 --repliche 40 --seed 12345
  python scripts/csv_batch_test.py --esperimento E2 --difficolta hard normal
  python scripts/csv_batch_test.py --lista-scenari
  python scripts/csv_batch_test.py --verifica      # requisiti e cooldown dottrine

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
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

"""
War Advisor - Meteo e ciclo giorno/notte

Il gioco ha già una configurazione meteo bilanciata in `data/modifiers.json`:
Sereno, Pioggia, Nebbia e Notte, con i loro moltiplicatori sugli attributi.
Quei valori NON si toccano — sono la taratura fatta a suo tempo e restano il
metro di valutazione.

Qui si fa una cosa sola: separare i due assi che nel file stanno mescolati.

    ciclo   Giorno | Notte      (Giorno è neutro: non cambia nulla)
    meteo   Sereno | Pioggia | Nebbia

I due assi si sommano, quindi "Notte + Pioggia" è uno stato valido mentre
"Giorno + Notte" non esiste per costruzione: sono valori dello stesso asse.

La somma è realizzata **componendo le voci di configurazione** in una singola
entry sintetica che viene registrata nella mappa meteo della sessione. Così è
`engine.apply_modifiers` ad applicarla, con la sua logica di sempre (CRITICAL
compreso), e `engine.py` non va toccato.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

# ── Assi ───────────────────────────────────────────────────────────
CYCLE_DAY = "Giorno"
CYCLE_NIGHT = "Notte"
CYCLES: Tuple[str, ...] = (CYCLE_DAY, CYCLE_NIGHT)

WEATHER_CLEAR = "Sereno"
WEATHER_RAIN = "Pioggia"
WEATHER_FOG = "Nebbia"
WEATHERS: Tuple[str, ...] = (WEATHER_CLEAR, WEATHER_RAIN, WEATHER_FOG)

#: Chiavi di `modifiers.json` da cui prendere i moltiplicatori di ogni asse.
#: Giorno non ha una voce: è il default che non cambia nulla.
CYCLE_SOURCE_KEY = {CYCLE_DAY: None, CYCLE_NIGHT: "Notte"}

SEPARATOR = " · "

# ── Ritmo dei cambiamenti ──────────────────────────────────────────
# Un solo orologio per entrambi gli assi. Con due orologi indipendenti gli
# intervalli si intrecciavano e le condizioni cambiavano ogni ~10 turni pur
# rispettando il minimo su ciascun asse: qui il minimo vale su ciò che il
# giocatore vede davvero cambiare.
CHANGE_MIN_TURNS = 20
CHANGE_MAX_TURNS = 26
#: A ogni cambio il ciclo si alterna sempre; il meteo si ritira solo a volte,
#: altrimenti non esisterebbero notti serene di seguito a giorni sereni.
WEATHER_REROLL_CHANCE = 0.55

#: Con che peso esce ogni meteo al sorteggio. Il sereno resta il caso comune:
#: le condizioni avverse devono essere un evento, non la norma.
WEATHER_WEIGHTS = {WEATHER_CLEAR: 0.5, WEATHER_RAIN: 0.3, WEATHER_FOG: 0.2}

# ── Presentazione (emoji e colori per l'indicatore in alto) ────────
CYCLE_UI = {
    CYCLE_DAY:   {"emoji": "☀️", "color": "#b45309", "background": "#fffbeb", "border": "#fcd34d"},
    CYCLE_NIGHT: {"emoji": "🌙", "color": "#c7d2fe", "background": "#312e81", "border": "#4338ca"},
}
WEATHER_UI = {
    WEATHER_CLEAR: {"emoji": "🌤", "label": "Sereno"},
    WEATHER_RAIN:  {"emoji": "🌧", "label": "Pioggia"},
    WEATHER_FOG:   {"emoji": "🌫", "label": "Nebbia"},
}

#: Descrizione dell'effetto, per il tooltip. Deriva dai valori reali del file
#: di configurazione, non è testo decorativo.
EFFECT_HINTS = {
    CYCLE_NIGHT: "furtività molto alta, disciplina a rischio",
    WEATHER_RAIN: "tiro e mobilità ridotti",
    WEATHER_FOG: "furtività alta, tiro dimezzato",
}


# ── Effetto sulla singola unità ────────────────────────────────────
# `engine.apply_modifiers` applica i moltiplicatori al vettore MEDIO della
# legione: è la valutazione strategica e resta esattamente com'era. Ma sul
# valore in battaglia della singola unità il meteo non arrivava, quindi
# artiglieria e picchieri prendevano la stessa pioggia. Qui si legge la stessa
# tabella, con la stessa regola CRITICAL, applicata agli attributi di UNA unità.
CRITICAL_MARK = "CRITICAL"
CRITICAL_THRESHOLD = 0.5
CRITICAL_PENALTY = 0.5

#: Quanto pesa il meteo sul valore della singola unità.
#: 1.0 = esattamente lo scarto che esce dai moltiplicatori. A 1.0 l'effetto si
#: notava appena (l'artiglieria perdeva il 6% sotto la pioggia) perché il tiro
#: è una voce su otto nel valore di combattimento: 1.8 lo rende una scelta
#: tattica vera senza ribaltare gli scontri. È l'unica manopola da toccare per
#: alzare o abbassare il peso del meteo.
UNIT_IMPACT = 1.8
#: Rete di sicurezza: nessuna condizione può dimezzare o raddoppiare un'unità.
UNIT_FACTOR_MIN = 0.60
UNIT_FACTOR_MAX = 1.40


def unit_weather_factor(
    attributes: Dict[str, float],
    weather_key: Optional[str],
    modifiers_data: Dict[str, Any],
    weights: Dict[str, float],
) -> float:
    """Quanto le condizioni correnti alzano o abbassano il valore di UNA unità.

    `weights` sono i pesi con cui gli attributi formano il valore in battaglia:
    passandoli si ottiene lo scarto reale su quel valore, non su una media
    astratta. Un'unità che non ha nulla a che fare con gli attributi toccati
    dal meteo torna 1.0 e non viene disturbata.
    """
    modifiers = (modifiers_data.get("weather") or {}).get(weather_key or "") or {}
    if not modifiers:
        return 1.0

    base = 0.0
    modified = 0.0
    for key, weight in weights.items():
        value = float(attributes.get(key, 0.0))
        base += value * weight

        modifier = modifiers.get(key)
        if modifier == CRITICAL_MARK:
            # Stessa soglia dell'engine, ma letta sulla singola unità: di notte
            # a rischiare è chi ha poca disciplina, non tutta la legione.
            if value < CRITICAL_THRESHOLD:
                value *= CRITICAL_PENALTY
        elif isinstance(modifier, (int, float)):
            value *= float(modifier)
        modified += value * weight

    if base <= 0.0:
        return 1.0

    raw = modified / base
    return max(UNIT_FACTOR_MIN, min(UNIT_FACTOR_MAX, 1.0 + ((raw - 1.0) * UNIT_IMPACT)))


def unit_effects(
    units_list: List[Dict[str, Any]],
    weather_key: Optional[str],
    modifiers_data: Dict[str, Any],
    weights: Dict[str, float],
    *,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    """Chi guadagna e chi perde con le condizioni correnti, per UI e log."""
    rows: List[Dict[str, Any]] = []
    for unit in units_list:
        factor = unit_weather_factor(
            unit.get("attributes", {}), weather_key, modifiers_data, weights
        )
        if abs(factor - 1.0) < 0.005:
            continue
        rows.append(
            {
                "unit_id": unit.get("id"),
                "unit_name": unit.get("name", unit.get("id")),
                "factor": round(factor, 3),
                "percent": int(round((factor - 1.0) * 100)),
            }
        )

    rows.sort(key=lambda row: -abs(row["factor"] - 1.0))
    return rows[:limit] if limit else rows


def combined_key(cycle: str, weather: str) -> str:
    """Nome della voce composta, usato come chiave meteo per l'engine."""
    return f"{cycle}{SEPARATOR}{weather}"


def split_key(key: Optional[str]) -> Tuple[str, str]:
    """Scompone una chiave composta; tollera i nomi semplici di una partita vecchia."""
    if not key:
        return CYCLE_DAY, WEATHER_CLEAR
    if SEPARATOR in key:
        cycle, weather = key.split(SEPARATOR, 1)
        return (
            cycle if cycle in CYCLES else CYCLE_DAY,
            weather if weather in WEATHERS else WEATHER_CLEAR,
        )
    # Chiave a un solo valore: può essere un meteo o "Notte".
    if key == CYCLE_NIGHT:
        return CYCLE_NIGHT, WEATHER_CLEAR
    if key in WEATHERS:
        return CYCLE_DAY, key
    return CYCLE_DAY, WEATHER_CLEAR


def _merge_modifiers(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Somma due set di modificatori.

    I moltiplicatori si moltiplicano fra loro; il marcatore CRITICAL vince su
    tutto, perché è una condizione, non un numero.
    """
    merged: Dict[str, Any] = dict(base)
    for attribute, modifier in extra.items():
        current = merged.get(attribute)
        if modifier == "CRITICAL" or current == "CRITICAL":
            merged[attribute] = "CRITICAL"
        elif current is None:
            merged[attribute] = modifier
        else:
            merged[attribute] = current * modifier
    return merged


def build_combined_weather_map(weather_config: Dict[str, Any]) -> Dict[str, Any]:
    """Genera tutte le combinazioni ciclo × meteo dai valori già bilanciati."""
    combined: Dict[str, Any] = {}
    for cycle in CYCLES:
        source = CYCLE_SOURCE_KEY.get(cycle)
        cycle_mods = dict(weather_config.get(source, {})) if source else {}
        for weather in WEATHERS:
            weather_mods = dict(weather_config.get(weather, {}))
            combined[combined_key(cycle, weather)] = _merge_modifiers(cycle_mods, weather_mods)
    return combined


def _merge_affinity(base: float, extra: float) -> float:
    """Somma due affinità in [0..1] con 0.5 come neutro.

    Stessa logica di `_merge_modifiers`: le condizioni si compongono invece di
    diluirsi, quindi si sommano gli scostamenti dal neutro. Notte + Nebbia per
    un assassino resta vicino al massimo; Notte + Sereno resta favorevole
    perché la notte pesa più del cielo sereno.
    """
    deviazione = (base - 0.5) + (extra - 0.5)
    return max(0.0, min(1.0, 0.5 + deviazione))


def build_combined_affinity_map(
    weather_affinity: Dict[str, Any], default: float = 0.5
) -> Dict[str, float]:
    """Affinità delle voci composte, dedotte da quelle dei due assi singoli.

    Serve perché `unit_affinities.json` elenca solo le quattro condizioni
    semplici: in partita la chiave meteo è composta ("Notte · Sereno") e la
    ricerca cadeva sul neutro, azzerando di fatto l'asse meteo dell'affinità.
    """
    combinate: Dict[str, float] = {}
    for cycle in CYCLES:
        # Giorno non ha una voce propria: è l'assenza di notte, quindi neutro.
        quota_ciclo = default if cycle == CYCLE_DAY else float(
            weather_affinity.get(cycle, default)
        )
        for weather in WEATHERS:
            quota_meteo = float(weather_affinity.get(weather, default))
            combinate[combined_key(cycle, weather)] = _merge_affinity(
                quota_ciclo, quota_meteo
            )
    return combinate


def affinities_with_combined_weather(affinities: Dict[str, Any]) -> Dict[str, Any]:
    """Copia delle affinità con le chiavi meteo composte già risolte."""
    if not affinities:
        return affinities
    default = float(affinities.get("default_affinity", 0.5))
    ambiente = affinities.get("environment_affinity") or {}
    nuovo_ambiente: Dict[str, Any] = {}
    for unit_id, voce in ambiente.items():
        if not isinstance(voce, dict) or "weather" not in voce:
            nuovo_ambiente[unit_id] = voce
            continue
        meteo = dict(voce["weather"])
        meteo.update(build_combined_affinity_map(meteo, default))
        nuovo_ambiente[unit_id] = {**voce, "weather": meteo}
    return {**affinities, "environment_affinity": nuovo_ambiente}


def data_with_combined_weather(data: Dict[str, Any]) -> Dict[str, Any]:
    """Copia dei dati con le voci composte aggiunte alla mappa meteo.

    Si lavora su una copia: registrarle nel dizionario globale le farebbe
    comparire anche nel selettore meteo della schermata iniziale.
    """
    weather_config = dict(data.get("weather", {}))
    weather_config.update(build_combined_weather_map(weather_config))
    return {
        **data,
        "weather": weather_config,
        # Le affinità vanno composte sullo stesso asse, altrimenti in partita
        # leggono una chiave che non esiste e tornano neutre.
        "unit_affinities": affinities_with_combined_weather(
            data.get("unit_affinities") or {}
        ),
    }


def roll_weather(rng: random.Random, exclude: Optional[str] = None) -> str:
    """Estrae un meteo secondo i pesi, evitando di ripetere quello corrente."""
    candidates = [w for w in WEATHERS if w != exclude] or list(WEATHERS)
    weights = [WEATHER_WEIGHTS.get(w, 1.0) for w in candidates]
    return rng.choices(candidates, weights=weights, k=1)[0]


def next_change_delay(rng: random.Random) -> int:
    return rng.randint(CHANGE_MIN_TURNS, CHANGE_MAX_TURNS)


def initial_conditions(rng: random.Random) -> Tuple[str, str]:
    """[SETUP-RULE] Condizioni meteo di inizio partita: sorteggiate, mai scelte.

    La schermata iniziale è un simulatore — ci si prova qualunque scenario —
    ma il meteo con cui si scende in campo non è una scelta del giocatore,
    altrimenti basterebbe selezionare la condizione che più favorisce il
    proprio esercito. L'estrazione passa dall'RNG della partita, quindi a
    parità di seed la partita resta riproducibile.
    """
    cycle = rng.choice(CYCLES)
    return cycle, roll_weather(rng)


def advance(cycle: str, weather: str, rng: random.Random) -> Tuple[str, str]:
    """Prossime condizioni: il ciclo si alterna, il meteo a volte cambia."""
    next_cycle = CYCLE_NIGHT if cycle == CYCLE_DAY else CYCLE_DAY
    next_weather = weather
    if rng.random() < WEATHER_REROLL_CHANCE:
        next_weather = roll_weather(rng, exclude=weather)
    return next_cycle, next_weather


def describe(cycle: str, weather: str, *, changes_in: int = 0) -> Dict[str, Any]:
    """Payload per l'indicatore: emoji, colori, etichette ed effetti."""
    cycle_ui = CYCLE_UI.get(cycle, CYCLE_UI[CYCLE_DAY])
    weather_ui = WEATHER_UI.get(weather, WEATHER_UI[WEATHER_CLEAR])

    effects: List[str] = []
    for key in (cycle, weather):
        hint = EFFECT_HINTS.get(key)
        if hint:
            effects.append(f"{key}: {hint}")
    if not effects:
        effects.append("Nessun effetto: condizioni ideali")

    return {
        "cycle": cycle,
        "weather": weather,
        "key": combined_key(cycle, weather),
        "label": f"{cycle}{SEPARATOR}{weather}",
        "emoji": f"{cycle_ui['emoji']}{weather_ui['emoji']}",
        "cycle_emoji": cycle_ui["emoji"],
        "weather_emoji": weather_ui["emoji"],
        "color": cycle_ui["color"],
        "background": cycle_ui["background"],
        "border": cycle_ui["border"],
        "is_night": cycle == CYCLE_NIGHT,
        "effects": effects,
        "changes_in": max(0, int(changes_in)),
    }

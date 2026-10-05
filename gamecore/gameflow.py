"""
War Advisor - Game Flow Glue

Collante minimale tra il ranking strategico e l'avvio della partita.
Riceve i dati del giocatore gia' calcolati dal backend, genera l'esercito
dell'IA e crea una GameSession pronta per il frontend.
"""

import random
from typing import Any, Dict, Optional

from gamecore.economy import STARTING_GRUX, calculate_army_cost, get_unit_costs
from gamecore.maps import TERRAIN_TYPES
from gamecore.session import GameSession, build_ai_army
from gamecore.session.ai_core.ai_easy_difficulty import AI_EASY_ID
from gamecore.session import weather_cycle as wc


def start_game_session(
    *,
    data: Dict[str, Any],
    player_units: list[str],
    terrain: str,
    weather: Optional[str],
    troop_status: Optional[str],
    strategy_id: str,
    army_profile: Dict[str, float],
    modified_profile: Dict[str, float],
    map_seed: Optional[int] = None,
    ai_difficulty: Optional[str] = None,
) -> Dict[str, Any]:
    """Crea una sessione di gioco completa partendo dal risultato di /calculate."""
    ai_difficulty = ai_difficulty or AI_EASY_ID
    if terrain not in TERRAIN_TYPES:
        raise ValueError(f"Terreno non valido: '{terrain}'. Valori ammessi: {TERRAIN_TYPES}")

    unit_costs = get_unit_costs(data["units"])
    player_army_cost = calculate_army_cost(player_units, unit_costs)
    if player_army_cost > STARTING_GRUX:
        raise ValueError("L'esercito selezionato supera il budget iniziale disponibile.")

    # Una partita senza seed se ne dà comunque uno: così resta replicabile
    # (serve agli esperimenti) e, soprattutto, il sorteggio del meteo non
    # collassa sempre sullo stesso valore come farebbe con `map_seed=None`.
    if map_seed is None:
        map_seed = random.randrange(2 ** 31)

    # [SETUP-RULE] Il meteo con cui si scende in campo si sorteggia: `weather`
    # è la condizione che il giocatore ha usato per SIMULARE, non una scelta
    # valida per la partita. Il sorteggio passa da `map_seed`, quindi a parità
    # di seed la partita è identica.
    meteo_rng = random.Random(map_seed * 7919 + 101)
    ciclo_iniziale, meteo_iniziale = wc.initial_conditions(meteo_rng)
    meteo_partita = wc.combined_key(ciclo_iniziale, meteo_iniziale)

    # L'IA costruisce il proprio esercito sulle condizioni vere dello scontro.
    # I dati vanno arricchiti con le voci composte, altrimenti la chiave
    # "Notte · Nebbia" non esiste nella tabella e i modificatori saltano.
    dati_partita = wc.data_with_combined_weather(data)

    ai_data = build_ai_army(
        data=dati_partita,
        ai_terrain="Montagna",
        weather=meteo_partita,
        n_units=3,
        budget=STARTING_GRUX,
        seed=map_seed,
        difficulty=ai_difficulty,
    )

    # `troop_status` e `terrain` restano la configurazione della simulazione:
    # la sessione li registra e poi impone le proprie condizioni di partenza.
    session = GameSession(
        player_units=player_units,
        player_strategy_id=strategy_id,
        player_army=army_profile,
        player_modified=modified_profile,
        player_troop_status=troop_status,
        player_budget=STARTING_GRUX - player_army_cost,
        player_army_cost=player_army_cost,
        ai_data=ai_data,
        weather=meteo_partita,
        data=data,
        player_home_terrain=terrain,
        map_seed=map_seed,
        ai_difficulty=ai_difficulty,
        simulated_weather=weather,
    )

    units_map = {unit["id"]: unit.get("name", unit["id"]) for unit in data["units"]}
    ai_units_names = [units_map.get(unit_id, unit_id) for unit_id in ai_data["units"]]

    return {
        "session": session,
        "ai_data": ai_data,
        # Il seed effettivamente usato: con questo la partita si rigioca uguale.
        "map_seed": map_seed,
        "message": (
            f"Partita avviata! L'IA ha scelto: {ai_data['strategy']['name']} "
            f"con {', '.join(ai_units_names)}."
        ),
    }
"""
War Advisor - Compatibilita di combattimento (SGANCIABILE)

Un solo punto in cui si decide quanto l'affinita unita-ambiente pesa nella
compatibilita che entra nel moltiplicatore di forza.

Il problema
-----------
Advisor e combattimento partivano dalla stessa distanza semantica, ma solo
l'Advisor sommava `compute_environment_adjustment()`. Misurato su 13.120
scenari: strategia migliore diversa nel 7,74% dei casi, scarto fino a 10,6
punti di compatibilita, e un'inversione vera sugli assassini - in battaglia
un'imboscata in pianura col sereno (92,1%) batteva la stessa imboscata in
foresta di notte (91,9%), mentre l'Advisor diceva 84,4% contro 100%.

La scelta
---------
Una funzione sola per tutti, e un peso dichiarato:

    advisory_score        distanza + affinita a peso pieno    (il consiglio)
    combat_compatibility  distanza + affinita x WEIGHT        (la battaglia)

Il peso non e pieno perche `apply_modifiers()` ha gia applicato terreno e meteo
al vettore d'esercito prima dello scontro: l'affinita a peso 1.0 li conterebbe
due volte. L'Advisor se lo puo permettere perche il suo compito e distinguere
fra strategie, non misurare una forza.

Calibrazione di COMBAT_AFFINITY_WEIGHT, misurata sul codice vero:

    peso   assassini foresta/notte vs pianura/sereno   fattore medio IA facile
    0.00   -0,17 pt  (ordine INVERSO)                  +0,00%
    0.25   +4,42 pt  (ordine corretto)                 +0,59%
    0.50   +9,01 pt                                    +1,54%
    1.00  +15,65 pt  (allineamento completo)           +3,01%

0.25 e il valore piu basso che rende la preferenza ambientale leggibile in
battaglia (9,4% di forza fra il contesto migliore e il peggiore) restando un
ordine di grandezza sotto il rumore con cui si misurano i tassi di vittoria.

WEIGHT = 0.0 riproduce esattamente il comportamento precedente: e il controllo
dell'ablation. 1.0 e l'allineamento completo con l'Advisor.

Rimozione
---------
Cancella questo file e i punti marcati [COMPAT-LAYER] in
`gamecore/session/session.py`. La compatibilita torna alla distanza euclidea
pura, cioe a WEIGHT = 0.0, e la partita e identica a prima del layer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from engine import compute_environment_adjustment

#: Massima distanza euclidea fra due vettori a 8 dimensioni con attributi in [0..1].
MAX_DISTANCE = 8 ** 0.5

#: Quanto l'affinita unita-ambiente pesa nella compatibilita di combattimento.
#: Vedi la tabella di calibrazione nel docstring del modulo.
COMBAT_AFFINITY_WEIGHT = 0.25

#: Peso dell'affinita nel punteggio dell'Advisor: pieno, com'e sempre stato in
#: `engine.compute_ranking()`. Serve a calcolare i due punteggi nello stesso
#: punto, per il log sperimentale.
ADVISORY_WEIGHT = 1.0


def normalize(distance: float) -> float:
    """Distanza semantica -> compatibilita [0..1]. Stessa scala di `compute_ranking`."""
    return max(0.0, min(1.0, 1.0 - (float(distance) / MAX_DISTANCE)))


def affinity_adjustment(
    *,
    unit_ids: Optional[Sequence[str]],
    strategy_id: str,
    terrain: Optional[str],
    weather: Optional[str],
    affinities: Optional[Dict[str, Any]],
) -> float:
    """Aggiustamento di affinita grezzo (negativo = bonus). 0.0 se manca un dato."""
    if not unit_ids or not strategy_id or not affinities:
        return 0.0
    try:
        return float(compute_environment_adjustment(
            unit_ids=list(unit_ids),
            strategy_id=strategy_id,
            terrain_name=terrain,
            weather_name=weather,
            affinities_data=affinities,
        ))
    except Exception:
        # Il moltiplicatore di forza non deve poter far cadere un turno.
        return 0.0


def scores(
    *,
    distance: float,
    unit_ids: Optional[Sequence[str]],
    strategy_id: str,
    terrain: Optional[str],
    weather: Optional[str],
    affinities: Optional[Dict[str, Any]],
    weight: Optional[float] = None,
) -> Tuple[float, float]:
    """`(combat_compatibility, advisory_score)`, entrambi in [0..1].

    I due punteggi differiscono solo per il peso dell'affinita: con
    `weight=1.0` coincidono, con `weight=0.0` il primo e la distanza euclidea
    pura. Il secondo e sempre a peso pieno, cioe uguale a quello che l'Advisor
    mostra al giocatore.
    """
    peso = COMBAT_AFFINITY_WEIGHT if weight is None else float(weight)
    adjustment = affinity_adjustment(
        unit_ids=unit_ids, strategy_id=strategy_id,
        terrain=terrain, weather=weather, affinities=affinities,
    )
    combat = normalize(max(0.0, distance + peso * adjustment))
    advisory = normalize(max(0.0, distance + ADVISORY_WEIGHT * adjustment))
    return combat, advisory


def describe() -> Dict[str, Any]:
    """Taratura corrente, per il log sperimentale e il pannello debug."""
    return {
        "combat_affinity_weight": COMBAT_AFFINITY_WEIGHT,
        "advisory_weight": ADVISORY_WEIGHT,
        "aligned": COMBAT_AFFINITY_WEIGHT == ADVISORY_WEIGHT,
    }

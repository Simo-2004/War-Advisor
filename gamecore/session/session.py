"""
War Advisor - GameSession

Gestisce una partita dove l'obiettivo è conquistare il castello avversario.
Le armate sul campo possono lasciare guarnigioni per rallentare l'assalto e,
quando perdono uno scontro, si ritirano al proprio castello invece di sparire.
"""

import random
import time
from enum import Enum
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

from engine import aggregate_army, apply_modifiers, compute_ranking, euclidean_distance
from gamecore.economy import (
    MINE_TILES_PER_SLOT,
    MINE_YIELD_PER_ROUND,
    STARTING_GRUX,
    available_mine_slots,
    get_unit_costs,
    mine_yield,
)
from gamecore.maps import GameMap, Occupation
from gamecore.session import abilities as ab
from gamecore.session.abilities import DOMAIN_ENGINEERING_ID, build_default_ability_states
from gamecore.session.black_market import BlackMarketState
# [CARAVAN] Le reclute dell'IA marciano dal castello al fronte invece di
# comparirci dentro. Il perché, con i numeri, sta nel docstring del modulo.
from gamecore.session import caravans as cv
# [EVENT-CHANNEL] Eventi strutturati per gli effetti del frontend. Se il
# file sparisce, `_evento()` diventa un no-op e la partita non cambia.
try:
    from gamecore.session import turn_events as ev
except ImportError:                                           # canale rimosso
    ev = None
from gamecore.session.ai_core.ai_builder import (
    build_ai_army,
    build_ai_policy,
    get_ai_difficulty_labels,
    normalize_ai_difficulty,
)
from gamecore.session.ai_core.ai_easy_difficulty import AI_EASY_ID
from gamecore.session.ai_core import ai_doctrine
from gamecore.session.in_game_advisor import build_in_game_advisor_payload
from gamecore.session.movement_points import MovementPointsSystem
from gamecore.session import troop_condition as tc
from gamecore.session import weather_cycle as wc

# [DOCTRINE-LAYER] Effetti unici delle strategie. Se il file sparisce le
# strategie tornano a valere solo come moltiplicatore di forza.
try:
    from gamecore import doctrines as doc
except ImportError:                                       # layer rimosso
    doc = None

# [BALANCE-LAYER] Correzioni di bilanciamento truppe (ruolo d'assedio
# dell'artiglieria, guaritori che assorbono perdite). Se il file non c'è, il
# gioco gira con i numeri grezzi del motore: ogni uso è protetto da un
# controllo su `balance`.
try:
    from gamecore import troop_balance as balance
except ImportError:                                           # layer rimosso
    balance = None

# [COMPAT-LAYER] Quanto l'affinità unità-ambiente pesa nella compatibilità che
# entra nel moltiplicatore di forza. Se il file sparisce si torna alla distanza
# euclidea pura, cioè al comportamento di prima del layer.
try:
    from gamecore import combat_compat as cmp_layer
except ImportError:                                           # layer rimosso
    cmp_layer = None

try:
    from debug.strength_debug import log_strength_debug
except Exception:
    def log_strength_debug(event: str, payload: Dict[str, Any]) -> None:
        return

try:
    from debug.battle_log_capture import create_battle_log_capture
except Exception:
    def create_battle_log_capture(session: Any = None) -> List[str]:
        return []

PLAYER = Occupation.PLAYER
AI = Occupation.AI

AI_LEGION_NAMES = [
    "Falange Nera",
    "Corvo di Ferro",
    "Lupi Scarlatti",
    "Ariete d'Acciaio",
    "Ombre del Nord",
    "Legione del Drago",
    "Spade Vermiglie",
    "Guardia Ferrea",
    "Orda Grigia",
    "Legione della Tempesta",
    "Artigli Neri",
    "Falce Cremisi",
    "Legione Spezzata",
    "Vento di Guerra",
    "Fauci d'Acciaio",
]

PLAYER_CONTROL_MODE_OPTIONS = ["orders", "manual"]
PLAYER_CONTROL_MODES = set(PLAYER_CONTROL_MODE_OPTIONS)
PLAYER_MOVEMENT_ORDER_OPTIONS = [
    "advance_castle",
    "engage_ai",
    "expand_front",
    "defend_castle",
    "hold",
]
PLAYER_MOVEMENT_ORDERS = set(PLAYER_MOVEMENT_ORDER_OPTIONS)
PLAYER_BUILD_ORDER_OPTIONS = [
    "balanced",
    "economy",
    "fortify",
    "garrison",
    "none",
]
PLAYER_BUILD_ORDERS = set(PLAYER_BUILD_ORDER_OPTIONS)

#: Rendimenti decrescenti delle miniere: le prime quattro pagano pieno, le
#: quattro dopo l'80%, e così via. Serve a impedire che la partita si decida
#: occupando mezza mappa di miniere e aspettando.
MINE_INCOME_BANDS = (1.0, 0.8, 0.6, 0.4)
MINE_INCOME_BAND_SIZE = 4

CASTLE_BASE_HP = 220
CASTLE_HP_PER_UNIT = 8

# ── Difesa del castello ────────────────────────────────────────────
# Manopola unica per bilanciare gli assedi. Il castello si difende in due modi,
# entrambi con un tetto: presidi sulla cella e livelli di fortificazione. Le
# legioni ferme sopra NON contano — erano cumulabili senza limite e rendevano
# l'assedio impossibile. Sulle celle normali le fortificazioni restano senza
# tetto (bonus calcolato in `_legion_battle_strength`).
CASTLE_MAX_FORTIFICATION_LEVEL = 4        # tetto di costruzioni sul castello
CASTLE_FORT_DAMAGE_REDUCTION_PER_LEVEL = 0.10   # -10% danni per livello (max -40%)
CASTLE_MAX_GARRISON_UNITS = 4             # tetto di presidi sulla cella del castello
CASTLE_DEFENDER_WEIGHT = 0.50             # quanto vale in difesa una truppa di presidio
CASTLE_DEFENSE_BASE = 44.0                # difesa di un castello completamente sguarnito
# La riserva non difende: finché contava, formare una legione toglieva difesa
# al castello e il gioco puniva chi giocava invece di accumulare truppe.

# Danno da assedio: dipende dal RAPPORTO forza/difesa, non dalla differenza.
# Con la differenza, oltre le ~10 unità l'assalto saturava sempre il tetto e
# da lì in poi né l'esercito attaccante né la difesa cambiavano più nulla.
CASTLE_DAMAGE_MAX = 65.0                  # danno massimo per assalto
CASTLE_DAMAGE_MIN = 8                     # pavimento: garantisce che il castello cada sempre
CASTLE_DAMAGE_HALF_RATIO = 15.0           # rapporto forza/difesa a metà del danno massimo

# ── Presidi e legioni sulle celle normali ──────────────────────────
# I presidi non si impilano più senza freno: quanti ne regge una cella
# dipende da quanto è fortificata. Serve a due cose insieme — toglie il
# muro di truppe su una casella qualsiasi e dà alle fortificazioni uno
# scopo oltre al bonus difensivo, creando un ordine di costruzione.
CELL_GARRISON_BASE_CAPACITY = 1           # presidi ospitabili senza fortificazioni
CELL_GARRISON_PER_FORT_LEVEL = 1          # presidi in più per ogni livello costruito
CELL_MAX_GARRISON_UNITS = 4               # tetto assoluto, fortificazioni comprese

# Tetto al numero di legioni contemporanee. Senza, si creavano decine di
# legioni da una unità: conquista a sciame e legioni accatastate sulla
# stessa cella. L'IA ha già il proprio limite nei file di difficoltà.
MAX_LEGIONS_PER_SIDE = 4

# Quanti turni di fila un piano di autoreclutamento può restare fermo ad
# aspettare i grux prima di arrendersi. I turni in attesa non consumano il
# piano, quindi senza questa valvola un piano avviato a casse vuote resterebbe
# acceso all'infinito.
AUTO_RECRUIT_MAX_WAIT = 8

# Terreni che le legioni non attraversano. Il fiume era già trattato come
# invalicabile dalle celle di schieramento, dalla pianificazione dell'IA e dal
# percorso disegnato sulla mappa — ma NON dal movimento vero, che guardava solo
# le legioni alleate e il castello nemico. Risultato: le legioni camminavano
# sull'acqua, e il segnalino mostrava una strada diversa da quella percorsa.
# Con un fiume centrale e pochi guadi la differenza è tutta la mappa.
TERRENI_INVALICABILI = ("Fiume",)

# [DOCTRINE-LAYER] Terreni che costano più di un turno di marcia: sono
# quelli su cui la Manovra sui Fianchi fa la differenza.
TERRENI_FATICOSI = ("Foresta", "Palude", "Montagna")

# ── Reclutamento IA ────────────────────────────────────────────────
# Il punteggio di una recluta guarda tre cose, e queste sono le manopole.
# Con la vecchia somma fissa di attributi il massimo era sempre la stessa
# truppa e l'esercito diventava mono-tipo (misurato: 90% arcieri a fine
# partita).
#   VARIETA   quanto penalizza una truppa di cui l'IA è già piena
#   PREZZO    quanto conta il costo (0 = lo ignora, 1 = ragiona a valore/grux)
#   STRATEGIA banda relativa: fra le candidate, chi avvicina di più l'esercito
#             alla strategia in uso prende +25%, chi meno -25%
RECRUIT_PESO_VARIETA = 0.55
RECRUIT_ESPONENTE_PREZZO = 0.6
RECRUIT_BANDA_STRATEGIA = 0.25

# Quanto l'IA è decisa a prendere la truppa col punteggio migliore: 0 = a caso,
# numeri alti = quasi sempre la prima. Ogni profilo dichiara la sua in
# `recruit_sharpness()`.
RECRUIT_SHARPNESS_DEFAULT = 4.0

# Unità minime perché una legione IA in più sia un reparto e non un drappello:
# spezzare l'esercito in gruppetti da tre la rende più debole, non più
# minacciosa — contro le mura ognuno fa il danno minimo.
AI_MIN_UNITS_PER_LEGION = 6

# Sotto questo esercito l'IA non mette da parte un grux per la ricerca: prima
# due reparti pieni, poi la tecnologia. Vedi `_ai_research_savings`.
AI_RESEARCH_MIN_ARMY = 2 * AI_MIN_UNITS_PER_LEGION

# Caselle massime in un ordine di cattura d'area. Non è un limite tecnico: una
# legione che deve prendere mezza mappa non torna più indietro, e l'ordine
# diventa impossibile da leggere sulla carta.
CAPTURE_AREA_MAX_CELLS = 40

# ── Peso degli attributi nel valore di combattimento ───────────────
# Erano scritti a mano dentro `_unit_battle_value`. Stanno qui perché li usa
# anche il calcolo dell'effetto meteo sulla singola unità: i due conti devono
# per forza pesare gli attributi allo stesso modo, altrimenti il fattore meteo
# non corrisponderebbe allo scarto reale sul valore.
UNIT_BATTLE_WEIGHTS: Dict[str, float] = {
    "U1_attack": 24.0,
    "U2_defense": 20.0,
    "U3_mobility": 12.0,
    "U4_stealth": 10.0,
    "U5_discipline": 14.0,
    "U6_terrain_adapt": 10.0,
    "U7_range_power": 8.0,
    "U8_support": 6.0,
}

LEGION_TYPE_ARMY = "army"
LEGION_TYPE_MINING = "mining"
LEGION_TYPE_CONSTRUCTION = "construction"
LEGION_TYPES = (LEGION_TYPE_ARMY, LEGION_TYPE_MINING, LEGION_TYPE_CONSTRUCTION)
LEGION_TYPE_LABELS = {
    LEGION_TYPE_ARMY: "Esercito",
    LEGION_TYPE_MINING: "Mineraria",
    LEGION_TYPE_CONSTRUCTION: "Costruzione",
}


class SessionState(str, Enum):
    ACTIVE    = "active"
    GAME_OVER = "game_over"


class GameSession:
    """
    Sessione di gioco completa.

    Attributi principali:
        player_units        — ID unità del giocatore
        player_strategy_id  — ID strategia confermata
        player_army         — vettore raw del giocatore
        player_modified     — vettore modificato per il terreno iniziale
        player_troop_status — stato truppe del giocatore

        ai_units            — ID unità dell'IA (scelte da ai_builder)
        ai_strategy_id/name — strategia ottimale dell'IA
        ai_army             — vettore raw dell'IA
        ai_modified         — vettore modificato per il terreno IA
        ai_troop_status     — stato truppe IA (sempre "Fresche" all'inizio)

        weather             — condizione meteo globale (opzionale)
        game_map            — istanza GameMap
        state               — SessionState corrente
        winner              — "player" | "ai" | None
        battle_log          — registro cronologico degli scontri
    """

    def __init__(
        self,
        *,
        # Player
        player_units: List[str],
        player_strategy_id: str,
        player_army: Dict[str, float],
        player_modified: Dict[str, float],
        player_troop_status: Optional[str],
        player_budget: int,
        player_army_cost: int,
        # AI (output di ai_builder.build_ai_army)
        ai_data: Dict[str, Any],
        # Ambiente
        weather: Optional[str],
        # Dati engine
        data: Dict[str, Any],
        player_home_terrain: str,
        # Mappa
        map_seed: Optional[int] = None,
        ai_difficulty: str = AI_EASY_ID,
        # [SETUP-RULE] Il meteo con cui il giocatore ha SIMULATO. Solo memoria:
        # `weather` resta la condizione vera dello scontro.
        simulated_weather: Optional[str] = None,
    ) -> None:
        # Mappa meteo arricchita con le combinazioni ciclo × meteo. È una copia:
        # registrarle nel dizionario globale le farebbe comparire nel selettore
        # meteo della schermata iniziale, che deve restare con le sue 4 voci.
        self.data = wc.data_with_combined_weather(data)
        self.units_map = {unit["id"]: unit for unit in self.data["units"]}
        self.strategies_map = {strategy["id"]: strategy for strategy in self.data["strategies"]}
        self.unit_costs = get_unit_costs(self.data["units"])
        self.player_home_terrain = player_home_terrain
        self.ai_home_terrain = "Montagna"

        # [SETUP-RULE] Terreno, meteo e stato truppe della schermata iniziale
        # sono parametri di SIMULAZIONE: servono a provare uno scenario, non a
        # sceglierne uno vantaggioso per la partita vera. Scendendo in campo
        # con le condizioni scelte un giocatore poteva regalarsi fino al 75% di
        # forza in più (misurato su 3 assassini in foresta). Qui restano come
        # documentazione di ciò che è stato simulato; le condizioni vere le
        # impone la partita.
        self.simulation_setup: Dict[str, Any] = {
            "terrain": player_home_terrain,
            "weather": simulated_weather if simulated_weather is not None else weather,
            "troop_status": player_troop_status,
        }

        # --- Giocatore ---
        self.player_units         = player_units
        self.player_strategy_id   = player_strategy_id
        self.player_army          = player_army
        self.player_modified      = player_modified
        # [SETUP-RULE] Si parte sempre da truppe fresche, come l'IA: un reparto
        # che non ha ancora marciato non può essere veterano né demoralizzato.
        self.player_troop_status  = tc.STATUS_FRESH
        self.player_army_cost     = player_army_cost

        # --- IA ---
        self.ai_units         = ai_data["units"]
        self.ai_strategy_id   = ai_data["strategy"]["id"]
        self.ai_strategy_name = ai_data["strategy"]["name"]
        self.ai_army          = ai_data["army_vector"]
        self.ai_modified      = ai_data["modified_vector"]
        self.ai_troop_status  = tc.STATUS_FRESH      # [SETUP-RULE] come il giocatore
        self.ai_army_cost     = ai_data.get("army_cost", 0)

        # Condizione della riserva nel castello. Le legioni la ereditano alla
        # nascita e ci rifondono la propria quando vengono richiamate: senza,
        # richiamare e riformare una legione sarebbe un azzeramento gratuito
        # della stanchezza. [SETUP-RULE] Entrambi partono da fresche: fatica e
        # morale si guadagnano in campo, non si scelgono a tavolino.
        self.reserve_condition: Dict[Occupation, Dict[str, Any]] = {
            PLAYER: tc.new_condition(self.player_troop_status),
            AI: tc.new_condition(self.ai_troop_status),
        }

        # --- Ambiente ---
        # Due assi che si sommano: ciclo (Giorno/Notte) e meteo (Sereno/Pioggia/
        # Nebbia). `self.weather` resta la chiave passata all'engine, ora composta,
        # così ogni punto che la usava continua a funzionare senza modifiche.
        self.day_cycle, self.weather_base = wc.split_key(weather)
        self.weather = wc.combined_key(self.day_cycle, self.weather_base)
        self.weather_rng = random.Random((map_seed or 0) * 977 + 13)
        self.turns_to_weather_change = wc.next_change_delay(self.weather_rng)

        # --- Legioni (Nuovo Sistema) ---
        self.player_legions: Dict[str, Dict[str, Any]] = {}
        self.ai_legions: Dict[str, Dict[str, Any]] = {}
        self.next_legion_id: int = 1
        # [CARAVAN] Rinforzi IA in marcia. Le truppe che stanno qui NON sono
        # in `ai_units`: sono fuori dall'esercito finché non arrivano, quindi
        # non contano in battaglia.
        self.ai_convoys = cv.Convogli()
        # [EVENT-CHANNEL] Id delle legioni che hanno già annunciato di
        # essere diventate armate: l'annuncio si fa una volta sola.
        self._armate_annunciate: set = set()
        self.ai_legion_respawn_delay_turns: int = 2
        self.ai_last_legion_loss_turn: Optional[int] = None

        # --- Mappa ---
        self.game_map: GameMap = GameMap(seed=map_seed)

        # [SETUP-RULE] Il terreno di casa è quello della cella su cui sorge il
        # castello, non quello scelto nel simulatore: è dove la riserva aspetta
        # davvero. Il vettore modificato va rifatto sulle condizioni vere,
        # altrimenti il pannello mostrerebbe ancora i numeri della simulazione.
        castello = self.game_map.castle_positions.get(PLAYER)
        cella_castello = self.game_map.get_cell(*castello) if castello else None
        if cella_castello is not None:
            self.player_home_terrain = cella_castello.terrain
        self.player_modified, _ = apply_modifiers(
            army_vector=self.player_army,
            terrain_name=self.player_home_terrain,
            weather_name=self.weather,
            troop_status_name=self.player_troop_status,
            modifiers_data=self.data,
        )

        # --- Stato ---
        self.state:      SessionState  = SessionState.ACTIVE
        self.winner:     Optional[str] = None
        self.battle_log: List[str]     = create_battle_log_capture(self)
        # [ENDGAME-STATS] Due numeri che il gioco non teneva da nessuna parte
        # e che non si possono ricostruire a posteriori: quando è cominciata
        # la partita e quante truppe ha perso ciascuno. Servono alla
        # schermata di fine partita; li legge solo `to_dict`, nessuna
        # formula di gioco li guarda. Rimozione: grep ENDGAME-STATS.
        self.started_at: float = time.time()
        # [EVENT-CHANNEL] Coda degli eventi del turno.
        self.event_log = ev.EventLog() if ev is not None else None
        self.troops_lost: Dict[Occupation, int] = {PLAYER: 0, AI: 0}
        self.grux_balance: Dict[Occupation, int] = {
            PLAYER: player_budget,
            AI: ai_data.get("remaining_grux", STARTING_GRUX - self.ai_army_cost),
        }
        self.base_fortification_cost: int = 45
        # Le ricerche dell'IA scorrono più in fretta man mano che la difficoltà
        # sale: le partite dure durano meno turni, e a tempi pieni l'IA forte
        # finirebbe con meno abilità di quella debole.
        _ai_difficulty = normalize_ai_difficulty(ai_difficulty)
        self.ability_states: Dict[Occupation, Dict[str, Any]] = {
            PLAYER: build_default_ability_states(),
            AI: build_default_ability_states(ab.ai_research_scale(_ai_difficulty)),
        }
        # Il banco del Mercato Nero esiste da subito ma resta chiuso finché
        # l'abilità non è sbloccata: nessuna offerta viene generata prima.
        self.black_market: Dict[Occupation, BlackMarketState] = {
            PLAYER: BlackMarketState(),
            AI: BlackMarketState(),
        }
        self.black_market_rng = random.Random((map_seed or 0) + 7717)
        # [DOCTRINE-LAYER] Sorteggio suo, come meteo e mercato nero: pescando
        # dallo stream della policy IA sposterebbe ogni decisione successiva.
        self.doctrine_rng = random.Random((map_seed or 0) * 613 + 29)
        self.recruit_cooldown_turns: int = 2
        self.last_recruit_turn: Dict[Occupation, Optional[int]] = {
            PLAYER: None,
            AI: None,
        }
        # [DEBUG-MODULE] Flag del kill switch IA, pilotato da gamecore/debug_module.
        # Senza il modulo resta False per sempre e il gioco si comporta come se
        # non esistesse. Rimozione: gamecore/debug_module/README.md
        self.debug_ai_kill_switch: bool = False
        self.ai_difficulty: str = normalize_ai_difficulty(ai_difficulty)
        self.ai_policy = build_ai_policy(self.ai_difficulty, seed=map_seed)
        self.ai_policy_seed = map_seed
        # La dottrina è il "come manovra", separata dal "cosa vuole" dei profili
        # di difficoltà: corsie, aggiramenti, giri larghi e attese.
        self.ai_doctrine = ai_doctrine.for_difficulty(self.ai_difficulty, seed=map_seed)
        self.movement_system = MovementPointsSystem()
        self.player_auto_recruit: Dict[str, Any] = {
            "enabled": False,
            "unit_id": None,
            "unit_name": None,
            "turns_total": 0,
            "turns_remaining": 0,
            "attempted_turns": 0,
            "successful_recruits": 0,
            "funds_wait": 0,
            "last_result": "inactive",
        }
        # Rimosso il vecchio sistema a ordini. Il turno avanza con execute_turn() e muove le legioni.
        self.castle_hp_max: Dict[Occupation, int] = self._build_castle_hp_pool()
        self.castle_hp: Dict[Occupation, int] = dict(self.castle_hp_max)

    def _build_castle_hp_pool(self) -> Dict[Occupation, int]:
        player_max = CASTLE_BASE_HP + (len(self.player_units) * CASTLE_HP_PER_UNIT)
        ai_max = CASTLE_BASE_HP + (len(self.ai_units) * CASTLE_HP_PER_UNIT)
        return {
            PLAYER: player_max,
            AI: ai_max,
        }

    def _ai_mine_income_multiplier(self) -> float:
        """Moltiplicatore sul rendimento delle miniere IA (vantaggio di difficoltà)."""
        getter = getattr(self.ai_policy, "mine_income_multiplier", None)
        if callable(getter):
            return max(1.0, float(getter()))
        return 1.0

    def _mine_income_for_terrains(
        self, terrains: Sequence[str], entity: Occupation = PLAYER
    ) -> int:
        """Rendimento delle miniere: prima il terreno, poi i rendimenti decrescenti.

        Ogni miniera rende in base a dove è scavata — montagna, pianura,
        foresta, palude in quest'ordine. Le rese si mettono in fila dalla più
        alta alla più bassa e sopra ci passano le bande di sempre: le prime
        quattro miniere pagano pieno, poi si scende. L'ordinamento serve a
        rendere il conto indipendente da quale miniera è stata scavata prima.
        """
        if not terrains:
            return 0

        rese = sorted((mine_yield(terreno) for terreno in terrains), reverse=True)
        ultima_banda = len(MINE_INCOME_BANDS) - 1
        income = sum(
            resa * MINE_INCOME_BANDS[min(posto // MINE_INCOME_BAND_SIZE, ultima_banda)]
            for posto, resa in enumerate(rese)
        )

        if entity == AI:
            income *= self._ai_mine_income_multiplier()

        # [ABILITY-EFFECTS] Linee di Rifornimento: meno perdite per strada.
        income *= self._ability_economy_factor(entity, ab.ECO_MINE_INCOME)
        return int(round(income))

    def _mine_income_for(self, entity: Occupation = PLAYER) -> int:
        """Quanto incassa l'entità a fine round dalle miniere che possiede."""
        return self._mine_income_for_terrains(self.game_map.mine_terrains(entity), entity)

    def _compute_castle_damage(self, attacker_strength: float, defender_score: float) -> int:
        """Danno inflitto al castello, in funzione del rapporto forza/difesa.

        Curva satura: più truppe porti all'assalto, più danno fai, ma con
        rendimenti decrescenti; più difesa c'è, meno ne fai. Il tetto e il
        pavimento restano gli estremi, non il caso normale.
        """
        # [BALANCE-LAYER] Curva d'assedio del layer: la versione qui sotto
        # premiava troppo poco chi porta un esercito d'assedio vero.
        if balance is not None:
            return balance.castle_damage(attacker_strength, defender_score, CASTLE_DAMAGE_MIN)

        ratio = attacker_strength / max(1.0, defender_score)
        intensity = ratio / (ratio + CASTLE_DAMAGE_HALF_RATIO)
        raw_damage = CASTLE_DAMAGE_MAX * intensity
        return max(CASTLE_DAMAGE_MIN, min(int(CASTLE_DAMAGE_MAX), int(round(raw_damage))))

    # ──────────────────────────────────────────────────────────
    # LEGIONI
    # ──────────────────────────────────────────────────────────

    def _get_free_spawn_cell(self, entity: Occupation, start_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Trova una cella adiacente libera per spawnare una legione (o il castello stesso se vuoto)."""
        r, c = start_pos
        cell = self.game_map.get_cell(r, c)

        own_legions = self.player_legions if entity == PLAYER else self.ai_legions
        castle_has_legion = any(tuple(leg.get("pos", ())) == (r, c) for leg in own_legions.values())
        if not castle_has_legion and cell and cell.terrain != "Fiume":
            return (r, c)

        # Cerca adiacenti liberi
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            adj = self.game_map.get_cell(nr, nc)
            if adj and adj.terrain != "Fiume":
                if not any(tuple(leg.get("pos", ())) == (nr, nc) for leg in own_legions.values()):
                    return (nr, nc)

        # Castello e adiacenti tutti presi: allarga la ricerca invece di
        # accettare una sovrapposizione, che è proprio ciò che vogliamo evitare.
        return self._free_cell_for(entity, (r, c))

    def _normalize_capture_area(
        self,
        cells: Optional[List[Any]],
    ) -> List[Tuple[int, int]]:
        """Valida l'area di cattura e la riduce a celle sensate.

        Si buttano via i doppioni, le celle fuori mappa e quelle già nostre:
        un ordine di cattura su roba che possediamo già è solo tempo perso.
        """
        if not cells:
            return []

        pulite: List[Tuple[int, int]] = []
        viste = set()
        for raw in cells:
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise ValueError("Area di cattura non valida: servono coppie [riga, colonna].")
            pos = (int(raw[0]), int(raw[1]))
            if pos in viste:
                continue
            cell = self.game_map.get_cell(*pos)
            if cell is None:
                raise ValueError(f"La cella {pos} è fuori dalla mappa.")
            viste.add(pos)
            if cell.occupation == PLAYER:
                continue
            # Sul fiume non ci si arriva, quindi non lo si conquista: tenerlo
            # in coda avrebbe fatto girare la legione a vuoto per sempre.
            if cell.terrain in TERRENI_INVALICABILI:
                continue
            pulite.append(pos)

        if len(pulite) > CAPTURE_AREA_MAX_CELLS:
            raise ValueError(
                f"Area troppo grande: {len(pulite)} caselle da conquistare, "
                f"il massimo è {CAPTURE_AREA_MAX_CELLS}."
            )
        return pulite

    def create_player_legion(
        self,
        name: str,
        units_dict: Dict[str, int],
        target: Optional[Tuple[int, int]],
        legion_type: str = LEGION_TYPE_ARMY,
        capture_area: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        if self.state != SessionState.ACTIVE:
            raise ValueError("Partita terminata.")

        if legion_type not in LEGION_TYPES:
            raise ValueError(f"Tipo legione non valido: {legion_type}")

        # O una destinazione o un'area, mai le due cose insieme: sono due ordini
        # diversi e la legione ne può eseguire uno solo.
        area = self._normalize_capture_area(capture_area)
        if area and target is not None:
            raise ValueError(
                "Scegli una destinazione oppure un'area da catturare, non entrambe."
            )
        if capture_area and not area:
            raise ValueError("Nell'area selezionata non c'è niente da conquistare: è già tua.")

        if target is not None:
            self._rifiuta_destinazione_impraticabile(
                (int(target[0]), int(target[1]))
            )

        if len(self.player_legions) >= MAX_LEGIONS_PER_SIDE:
            raise ValueError(
                f"Hai già {MAX_LEGIONS_PER_SIDE} legioni in campo, il massimo consentito: "
                f"richiamane una prima di formarne un'altra."
            )

        # Verifica disponibilità truppe nella riserva
        from collections import Counter
        reserve_counts = Counter(self.player_units)
        
        for uid, qty in units_dict.items():
            if reserve_counts.get(uid, 0) < qty:
                raise ValueError(f"Non hai abbastanza unità di tipo {uid} nel castello.")
                
        # Rimuovi le truppe dalla riserva
        legion_units = []
        for uid, qty in units_dict.items():
            for _ in range(qty):
                self.player_units.remove(uid)
                legion_units.append(uid)

        # Le truppe che partono non sono più in riserva, e il vettore di forza
        # descrive la riserva: senza questa riga `player_army` continuava a
        # contare anche chi era in campo, e l'advisor consigliava la strategia
        # per un esercito che nel castello non c'era più.
        self._recompute_entity_army_vector(PLAYER)

        # Spawn position
        castle_pos = self.game_map.castle_positions[PLAYER]
        spawn_pos = self._get_free_spawn_cell(PLAYER, castle_pos) or castle_pos
            
        legion_id = f"L_{self.next_legion_id}"
        self.next_legion_id += 1
        
        # Con un'area assegnata la prima meta è la casella più vicina da prendere:
        # da lì in poi ci pensa `_refresh_capture_area` a passare alla successiva.
        if area:
            target = min(area, key=lambda pos: self._order_distance(spawn_pos, pos))

        self.player_legions[legion_id] = {
            "id": legion_id,
            "name": name,
            "units": legion_units,
            "pos": spawn_pos,
            "target": target,
            "capture_area": [list(pos) for pos in area],
            "legion_type": legion_type,
            # Nasce con la strategia generale, poi la si cambia per legione.
            "strategy_id": self.player_strategy_id,
            # [DOCTRINE-LAYER] Anche appena nata l'effetto parte dal turno dopo:
            # creare una legione non è la scorciatoia per averlo subito.
            "doctrine_since_turn": self.game_map.turn,
            # Eredita la condizione della riserva da cui è stata formata.
            "condition": tc.new_condition(
                fatigue=self.reserve_condition[PLAYER]["fatigue"],
                morale=self.reserve_condition[PLAYER]["morale"],
                veteran=self.reserve_condition[PLAYER]["veteran"],
                victories=self.reserve_condition[PLAYER].get("victories", 0),
            ),
            "path": [],
            "path_step": 0,
            "movement": self.movement_system.export_legion_state(
                self._legion_movement_key(PLAYER, legion_id)
            ),
        }

        type_label = LEGION_TYPE_LABELS[legion_type]
        self._evento(                                         # [EVENT-CHANNEL]
            ev.LEGIONE_CREATA if ev else "legione_creata",
            entita=PLAYER, pos=spawn_pos,
            quantita=len(legion_units),
            dettaglio={"nome": name, "tipo": legion_type},
        )
        self.battle_log.append(
            f"⚔️ PLAYER addestra la legione '{name}' ({type_label}) e la invia verso "
            f"{target if target else 'attesa'}."
        )
        
        # Aggiorna controllo mappa
        cell = self.game_map.get_cell(*spawn_pos)
        if cell and cell.occupation != PLAYER:
            cell.occupation = PLAYER

        # [EVENT-CHANNEL] Il player forma le sue legioni fuori dal turno:
        # senza questa chiamata l'annuncio arriverebbe solo al turno dopo.
        self._annuncia_armate()

        return {
            "ok": True,
            "message": f"Legione {name} creata",
            "session": self.to_dict()
        }

    def recall_player_legion(self, legion_id: str) -> Dict[str, Any]:
        """Richiama una legione PLAYER: le sue unità tornano istantaneamente in riserva."""
        if self.state != SessionState.ACTIVE:
            raise ValueError("Partita terminata.")

        legion = self.player_legions.get(legion_id)
        if legion is None:
            raise ValueError(f"Legione non trovata: {legion_id}")

        name = legion.get("name", legion_id)
        legion_units = legion.get("units", [])
        # La condizione va fusa PRIMA di versare le truppe in riserva, così la
        # media pesa sulla riserva com'era: le truppe stanche che rientrano
        # stancano la riserva, e riformare la legione non azzera nulla.
        stato = self._legion_troop_status(PLAYER, legion)
        self._merge_legion_into_reserve(PLAYER, legion)
        self.player_units.extend(legion_units)
        del self.player_legions[legion_id]
        self._recompute_entity_army_vector(PLAYER)

        log_entry = (
            f"[Turno {self.game_map.turn}] 🏳 PLAYER: Legione '{name}' richiamata "
            f"— {len(legion_units)} unità ({stato}) tornano in riserva."
        )
        self.battle_log.append(log_entry)

        return {
            "ok": True,
            "message": log_entry,
            "session": self.to_dict()
        }

    def retarget_player_legion(self, legion_id: str, target: Tuple[int, int]) -> Dict[str, Any]:
        """Assegna una nuova destinazione a una legione PLAYER già in campo."""
        if self.state != SessionState.ACTIVE:
            raise ValueError("Partita terminata.")

        legion = self.player_legions.get(legion_id)
        if legion is None:
            raise ValueError(f"Legione non trovata: {legion_id}")

        target = (int(target[0]), int(target[1]))
        if self.game_map.get_cell(*target) is None:
            raise ValueError("Cella di destinazione fuori dai limiti della mappa.")
        self._rifiuta_destinazione_impraticabile(target)

        legion["target"] = target
        # Una destinazione annulla l'ordine di cattura: sono due ordini diversi
        # e la legione ne esegue uno solo, anche a legione già in campo.
        area_annullata = len(legion.get("capture_area") or [])
        legion["capture_area"] = []
        name = legion.get("name", legion_id)

        log_entry = (
            f"[Turno {self.game_map.turn}] 🧭 PLAYER: Legione '{name}' ridiretta verso {target}."
        )
        if area_annullata:
            log_entry += f" Ordine di cattura annullato ({area_annullata} caselle)."
        self.battle_log.append(log_entry)

        return {
            "ok": True,
            "message": log_entry,
            "session": self.to_dict()
        }

    def set_legion_capture_area(self, legion_id: str, cells: List[Any]) -> Dict[str, Any]:
        """Assegna un nuovo ordine di cattura d'area a una legione già in campo."""
        if self.state != SessionState.ACTIVE:
            raise ValueError("Partita terminata.")

        legion = self.player_legions.get(legion_id)
        if legion is None:
            raise ValueError(f"Legione non trovata: {legion_id}")

        area = self._normalize_capture_area(cells)
        if not area:
            raise ValueError("Nell'area selezionata non c'è niente da conquistare: è già tua.")

        current_pos = tuple(legion.get("pos", ()))
        legion["capture_area"] = [list(pos) for pos in area]
        # La prima meta è la più vicina: l'ordine parte da dove si trova adesso.
        if len(current_pos) == 2:
            legion["target"] = list(min(area, key=lambda pos: self._order_distance(current_pos, pos)))

        name = legion.get("name", legion_id)
        log_entry = (
            f"[Turno {self.game_map.turn}] 🗺 PLAYER: Legione '{name}' riceve l'ordine di "
            f"catturare {len(area)} caselle."
        )
        self.battle_log.append(log_entry)

        return {"ok": True, "message": log_entry, "session": self.to_dict()}

    def _ai_desired_legion_count(self) -> int:
        """Quante legioni l'IA vuole in campo adesso (profilo di difficoltà + situazione)."""
        max_getter = getattr(self.ai_policy, "max_legions", None)
        max_legions = max(1, int(max_getter())) if callable(max_getter) else 1
        # Stesso tetto del player: i profili di difficoltà stanno già sotto,
        # ma la regola non deve dipendere da come sono tarati.
        # La manovra può volerne più della difesa: con una legione sola non si
        # accerchia nessuno, e l'accerchiamento è ciò che distingue i profili
        # alti. Il tetto di gioco resta comunque l'ultima parola.
        doctrine_wanted = self.ai_doctrine.offensive_legions(self.game_map.turn)
        max_legions = min(MAX_LEGIONS_PER_SIDE, max(max_legions, doctrine_wanted))
        if max_legions < 2:
            return 1

        min_getter = getattr(self.ai_policy, "second_legion_min_units", None)
        min_units = max(2, int(min_getter())) if callable(min_getter) else 6

        # Risposta difensiva: seconda legione se c'è un'incursione in corso e
        # l'esercito regge la divisione.
        defensive = 2 if (self._ai_intruder_legions() and len(self.ai_units) >= min_units) else 1

        # Manovra: quante ne vuole la dottrina, purché ognuna resti un reparto
        # vero e non un pugno di uomini sparso per la mappa.
        maneuver = min(doctrine_wanted, len(self.ai_units) // AI_MIN_UNITS_PER_LEGION)

        return max(1, min(max_legions, max(defensive, maneuver)))

    def _ensure_ai_legions_initialized(self) -> None:
        """Mantiene in campo il numero di legioni IA voluto, con cooldown dopo annientamento."""
        if not self.ai_units:
            return

        if self.debug_ai_kill_switch:
            return

        if len(self.ai_legions) >= self._ai_desired_legion_count():
            return

        # Il cooldown vale solo per il ritorno in campo dopo l'annientamento totale,
        # non per l'aggiunta di una legione difensiva a fianco di una già viva.
        if not self.ai_legions and self.ai_last_legion_loss_turn is not None:
            turns_since_loss = self.game_map.turn - self.ai_last_legion_loss_turn
            if turns_since_loss < self.ai_legion_respawn_delay_turns:
                return

        castle_pos = self.game_map.castle_positions[AI]
        spawn_pos = self._get_free_spawn_cell(AI, castle_pos) or castle_pos

        legion_id = f"AI_{self.next_legion_id}"
        self.next_legion_id += 1

        used_names = {legion.get("name") for legion in self.ai_legions.values()}
        free_names = [name for name in AI_LEGION_NAMES if name not in used_names]
        legion_name = random.choice(free_names or AI_LEGION_NAMES)

        self.ai_legions[legion_id] = {
            "id": legion_id,
            "name": legion_name,
            # Le truppe le assegna la ripartizione qui sotto: con due legioni
            # copiare l'intero esercito in ciascuna lo duplicherebbe.
            "units": [],
            "pos": spawn_pos,
            "target": None,
            "legion_type": LEGION_TYPE_ARMY,
            "doctrine_since_turn": self.game_map.turn,   # [DOCTRINE-LAYER]
            "path": [],
            "path_step": 0,
            # Come per il player: eredita la condizione della riserva IA.
            "condition": tc.new_condition(
                fatigue=self.reserve_condition[AI]["fatigue"],
                morale=self.reserve_condition[AI]["morale"],
                veteran=self.reserve_condition[AI]["veteran"],
                victories=self.reserve_condition[AI].get("victories", 0),
            ),
            "movement": self.movement_system.export_legion_state(
                self._legion_movement_key(AI, legion_id)
            ),
        }

        self._sync_ai_legion_units()

        # [EVENT-CHANNEL] L'annuncio arriva solo adesso: la legione IA nasce
        # vuota e sono le righe qui sopra a darle le truppe. Emetterlo alla
        # creazione avrebbe portato sempre `quantita=0`, e chi ascolta non
        # avrebbe potuto distinguere un'armata da una pattuglia. Se la
        # ripartizione non le ha assegnato nessuno l'ha già sciolta, e allora
        # non c'è nessuna legione da annunciare.
        nata = self.ai_legions.get(legion_id)
        if nata is not None:
            self._evento(                                     # [EVENT-CHANNEL]
                ev.LEGIONE_CREATA if ev else "legione_creata",
                entita=AI, pos=spawn_pos,
                quantita=len(nata.get("units") or []),
                dettaglio={"nome": legion_name, "tipo": LEGION_TYPE_ARMY},
            )

        cell = self.game_map.get_cell(*spawn_pos)
        if cell is not None:
            cell.occupation = AI

    def _legion_movement_key(self, entity: Occupation, legion_id: str) -> str:
        """Chiave dello stato movimento di una legione."""
        return self.movement_system.legion_key(entity, legion_id)

    def _legion_is_movement_blocked(
        self,
        entity: Occupation,
        legion_id: str,
        legion: Dict[str, Any],
        logs: List[str],
    ) -> bool:
        """Consuma un turno di attraversamento: True se la legione non può muoversi ora.

        Riusa il sistema punti-movimento già esistente (100 punti/turno, terreni
        più difficili costano di più e il surplus diventa turni di attesa),
        applicandolo però alla singola legione invece che all'intera armata.
        """
        key = self._legion_movement_key(entity, legion_id)
        block = self.movement_system.consume_legion_block_if_any(key)
        legion["movement"] = self.movement_system.export_legion_state(key)
        if not block["blocked"]:
            return False

        # Anche restare impantanati a metà guado costa: senza questo un terreno
        # duro sarebbe MENO faticoso di uno facile, perché blocca la legione per
        # più turni e i turni fermi verrebbero contati come riposo.
        legion["traversing_terrain"] = block["last_terrain"]

        icon = "⚔️" if entity == PLAYER else "🤖"
        remaining = int(block["remaining_blocked_turns"]) + 1
        logs.append(
            f"{icon} La legione {entity.value.upper()} '{legion.get('name', legion_id)}' "
            f"attraversa {str(block['last_terrain']).lower()} "
            f"(costo {block['last_cost']} punti): ancora {remaining} turno/i di marcia."
        )
        return True

    def _register_legion_move_cost(
        self,
        entity: Occupation,
        legion_id: str,
        legion: Dict[str, Any],
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Registra il costo del terreno appena raggiunto dalla legione."""
        key = self._legion_movement_key(entity, legion_id)
        cell = self.game_map.get_cell(*to_pos)
        terrain = cell.terrain if cell is not None else "Pianura"
        # [DOCTRINE-LAYER] Manovra sui Fianchi: la colonna a cavallo marcia
        # con più punti, quindi bosco e palude non la bloccano per un turno.
        punti = None
        if doc is not None:
            stato = self._doctrine_state(entity, legion)
            if stato is not None:
                punti = doc.movement_points(
                    stato["strategy_id"], stato["units"],
                    turn=stato["turn"], since_turn=stato["since_turn"],
                    base_points=self.movement_system.points_per_turn,
                )
                if punti == self.movement_system.points_per_turn:
                    punti = None
        info = self.movement_system.register_legion_move(
            key, terrain, from_pos, to_pos, points_per_turn=punti
        )
        legion["movement"] = self.movement_system.export_legion_state(key)
        return info

    def _prune_legion_movement_states(self) -> None:
        """Scarta lo stato movimento delle legioni non più in campo."""
        active = {self._legion_movement_key(PLAYER, lid) for lid in self.player_legions}
        active |= {self._legion_movement_key(AI, lid) for lid in self.ai_legions}
        self.movement_system.prune_legions(active)

    def _sync_ai_legion_units(self) -> bool:
        """Ripartisce l'esercito IA (`self.ai_units`) fra le legioni in campo.

        Per il player `player_units` è una riserva da cui la legione sottrae; per
        l'IA `ai_units` resta l'esercito autoritativo e le legioni ne sono una
        vista, se no le reclute non arrivano al fronte e le perdite non riducono
        le legioni. Con più legioni le unità si alternano per valore, così
        nessuna eredita solo gli scarti.

        Returns:
            True se qualcosa è stato effettivamente modificato.
        """
        if not self.ai_legions:
            return False

        if not self.ai_units:
            # Esercito azzerato: sciogli le legioni invece di lasciarne di vuote in
            # campo (varrebbero forza 1 negli scontri e bloccherebbero il respawn).
            self.ai_legions.clear()
            self.ai_last_legion_loss_turn = self.game_map.turn
            return True

        legion_ids = sorted(self.ai_legions.keys())
        if len(legion_ids) == 1:
            legion = self.ai_legions[legion_ids[0]]
            if legion.get("units", []) == self.ai_units:
                return False
            legion["units"] = list(self.ai_units)
            return True

        # Distribuzione a serpentina sul valore in combattimento: le legioni
        # restano equilibrate anche con unità di qualità molto diversa.
        ordered = sorted(self.ai_units, key=lambda uid: self._unit_battle_value(uid), reverse=True)
        buckets: Dict[str, List[str]] = {lid: [] for lid in legion_ids}
        for index, unit_id in enumerate(ordered):
            slot = index % len(legion_ids)
            if (index // len(legion_ids)) % 2 == 1:
                slot = len(legion_ids) - 1 - slot
            buckets[legion_ids[slot]].append(unit_id)

        changed = False
        for legion_id in legion_ids:
            legion = self.ai_legions[legion_id]
            if legion.get("units", []) != buckets[legion_id]:
                legion["units"] = buckets[legion_id]
                changed = True

        # Una legione rimasta senza truppe va sciolta, non lasciata a vuoto.
        for legion_id in legion_ids:
            if not self.ai_legions[legion_id].get("units"):
                del self.ai_legions[legion_id]
                changed = True

        return changed

    def _ai_intruder_legions(self) -> List[Tuple[int, int]]:
        """Posizioni delle legioni player che hanno superato la metà campo.

        La metà è calcolata fra i due castelli, così la regola resta valida
        qualunque sia l'orientamento della mappa.
        """
        ai_castle = self.game_map.castle_positions.get(AI)
        player_castle = self.game_map.castle_positions.get(PLAYER)
        if ai_castle is None or player_castle is None or not self.player_legions:
            return []

        midline = (ai_castle[0] + player_castle[0]) / 2.0
        ai_is_north = ai_castle[0] < player_castle[0]

        intruders: List[Tuple[int, int]] = []
        for legion in self.player_legions.values():
            pos = tuple(legion.get("pos", ()))
            if len(pos) != 2:
                continue
            row = int(pos[0])
            crossed = row <= midline if ai_is_north else row >= midline
            if crossed:
                intruders.append((row, int(pos[1])))
        return intruders

    def _find_legion_at(self, entity: Occupation, pos: Tuple[int, int]) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Restituisce (id, legione) se l'entità ha una legione sulla posizione richiesta."""
        source = self.player_legions if entity == PLAYER else self.ai_legions
        for legion_id, legion in source.items():
            if tuple(legion.get("pos", ())) == pos:
                return legion_id, legion
        return None

    def _active_legion_positions(self, entity: Occupation) -> List[Tuple[int, int]]:
        """Posizioni correnti delle legioni attive di una entità, senza duplicati."""
        source = self.player_legions if entity == PLAYER else self.ai_legions
        seen: set[Tuple[int, int]] = set()
        positions: List[Tuple[int, int]] = []
        for legion in source.values():
            pos = tuple(legion.get("pos", ()))
            if len(pos) != 2:
                continue
            cella = (int(pos[0]), int(pos[1]))
            if cella in seen:
                continue
            seen.add(cella)
            positions.append(cella)
        return positions

    def _ai_expansion_allows(self, pos: Tuple[int, int]) -> bool:
        """True se l'IA può espandersi su questa cella nella fase corrente.

        In fase difensiva alcuni profili restano nella propria metà campo:
        senza il vincolo l'espansione punta il miglior obiettivo ovunque sia e
        degenera in una marcia dentro il territorio nemico.
        """
        confine = getattr(self.ai_policy, "confine_expansion_to_own_half", None)
        if not callable(confine) or not confine():
            return True

        ai_castle = self.game_map.castle_positions.get(AI)
        player_castle = self.game_map.castle_positions.get(PLAYER)
        if ai_castle is None or player_castle is None:
            return True

        midline = (ai_castle[0] + player_castle[0]) / 2.0
        if ai_castle[0] < player_castle[0]:
            return pos[0] <= midline
        return pos[0] >= midline

    def _ai_expansion_distance_penalty(self) -> float:
        """Peso della distanza nella scelta degli obiettivi economici dell'IA."""
        getter = getattr(self.ai_policy, "expansion_distance_penalty", None)
        if callable(getter):
            return max(0.0, float(getter()))
        return 0.18

    def _collect_ai_economic_targets(self, ai_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Target economici/territoriali per l'IA (miniere, territori utili, espansione)."""
        distance_penalty = self._ai_expansion_distance_penalty()
        scored: List[Tuple[float, Tuple[int, int]]] = []
        for row in self.game_map.grid:
            for cell in row:
                if cell.occupation == AI or cell.is_castle or cell.terrain == "Fiume":
                    continue

                pos = (cell.row, cell.col)
                if not self._ai_expansion_allows(pos):
                    continue

                distance = self._order_distance(ai_pos, pos)

                score = 0.0
                if cell.is_mine:
                    score += 4.2
                if cell.is_strategic:
                    score += 2.1
                if cell.terrain in {"Pianura", "Montagna"}:
                    score += 1.2
                if cell.occupation == PLAYER:
                    score += 0.5

                score -= distance * distance_penalty
                scored.append((score, pos))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [pos for _, pos in scored[:8]]

    def _own_legion_positions_map(
        self,
        entity: Occupation,
        exclude_id: Optional[str] = None,
    ) -> Dict[Tuple[int, int], str]:
        """Celle occupate dalle legioni di `entity`, mappate al nome della legione.

        È la base della regola del muro: una legione occupa la sua cella in
        esclusiva verso gli alleati. Senza, due legioni finivano sovrapposte e
        negli scontri ne combatteva una sola mentre l'altra restava illesa
        sulla stessa casella.
        """
        source = self.player_legions if entity == PLAYER else self.ai_legions
        occupied: Dict[Tuple[int, int], str] = {}
        for legion_id, legion in source.items():
            if exclude_id is not None and legion_id == exclude_id:
                continue
            pos = tuple(legion.get("pos", ()))
            if len(pos) == 2:
                occupied[pos] = legion.get("name", legion_id)
        return occupied

    # [DOCTRINE-LAYER] Serve solo alla Difesa in Profondità.
    def _one_step_retreat(
        self,
        entity: Occupation,
        from_pos: Tuple[int, int],
        own_castle: Tuple[int, int],
        exclude_id: Optional[str] = None,
    ) -> Optional[Tuple[int, int]]:
        """La cella adiacente che accorcia la strada verso il proprio castello.

        Arretrare è cedere terreno, non girare in tondo né allontanarsi da
        casa: valgono solo i passi che riducono la distanza. Senza candidate
        restituisce None e vale il ripiegamento normale al castello.
        """
        occupate = self._own_legion_positions_map(entity, exclude_id=exclude_id)
        # Anche le nemiche: arretrare addosso a una legione avversaria la
        # lascerebbe sulla stessa casella senza che nessuno dei due attacchi.
        occupate.update(
            self._own_legion_positions_map(AI if entity == PLAYER else PLAYER)
        )
        r, c = from_pos
        distanza_attuale = abs(r - own_castle[0]) + abs(c - own_castle[1])
        candidate = []
        for vicino in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if vicino in occupate or not self._cella_attraversabile(vicino):
                continue
            distanza = abs(vicino[0] - own_castle[0]) + abs(vicino[1] - own_castle[1])
            if distanza >= distanza_attuale:
                continue
            candidate.append((distanza, vicino))
        if not candidate:
            return None
        return min(candidate)[1]

    def _free_cell_for(
        self,
        entity: Occupation,
        preferred: Tuple[int, int],
        exclude_id: Optional[str] = None,
    ) -> Optional[Tuple[int, int]]:
        """Cella libera da legioni alleate, partendo da `preferred` e allargandosi.

        Serve ai ripiegamenti: mandare un perdente sul proprio castello quando
        lì c'è già una legione ricreerebbe la sovrapposizione che stiamo togliendo.
        """
        preferred = tuple(preferred)
        occupied = self._own_legion_positions_map(entity, exclude_id=exclude_id)

        visited = {preferred}
        queue = deque([preferred])
        while queue:
            pos = queue.popleft()
            cell = self.game_map.get_cell(*pos)
            if cell is not None and cell.terrain != "Fiume" and pos not in occupied:
                return pos
            r, c = pos
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in visited:
                    continue
                if self.game_map.get_cell(*neighbor) is None:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        return None

    def _nearest_reachable_cell_to(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        blocked: set,
    ) -> Optional[Tuple[int, int]]:
        """Cella raggiungibile da `start` più vicina a `goal`, aggirando `blocked`.

        È il ripiego quando la destinazione voluta è occupata o irraggiungibile:
        invece di piantarsi, la legione punta al punto utile più vicino. A parità
        di distanza dall'obiettivo vince quello che costa meno strada.
        """
        start = tuple(start)
        goal = tuple(goal)
        best: Optional[Tuple[int, int]] = None
        best_key: Optional[Tuple[int, int]] = None

        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            pos, dist = queue.popleft()
            key = (abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]), dist)
            if best_key is None or key < best_key:
                best_key, best = key, pos

            r, c = pos
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in visited or neighbor in blocked:
                    continue
                if not self._cella_attraversabile(neighbor):
                    continue
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
        return best

    def _step_preference(
        self,
        current: Tuple[int, int],
        start: Tuple[int, int],
        goal: Tuple[int, int],
        lane_col: Optional[int],
    ) -> List[Tuple[int, int]]:
        """I quattro vicini, dal passo più sensato al meno sensato.

        Fra i tanti cammini minimi, quale esce dipende solo da quale vicino il
        BFS guarda per primo: con l'ordine fisso vinceva sempre quello che va
        prima in verticale, e ogni legione scendeva lungo la colonna del
        castello. L'ordine qui è: chi avvicina all'obiettivo (garantisce il
        cammino minimo), poi chi tiene la corsia, poi chi resta vicino alla
        retta partenza-obiettivo — è il terzo criterio a raddrizzare la marcia
        in diagonale, a parità di lunghezza.

        La corsia può solo scegliere fra cammini di pari lunghezza: se partenza
        e arrivo stanno sulla stessa colonna non c'è niente da piegare, e per
        aggirare davvero serve l'obiettivo intermedio che assegna la dottrina.
        """
        r, c = current
        goal_r, goal_c = goal
        delta_r = goal_r - start[0]
        delta_c = goal_c - start[1]

        def chiave(neighbor: Tuple[int, int]) -> Tuple[int, int, int]:
            nr, nc = neighbor
            avvicina = abs(goal_r - nr) + abs(goal_c - nc)
            corsia = abs(nc - lane_col) if lane_col is not None else 0
            # Distanza dalla retta partenza→obiettivo (prodotto vettoriale).
            scostamento = abs(delta_r * (nc - start[1]) - delta_c * (nr - start[0]))
            return (avvicina, corsia, scostamento)

        return sorted(((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)), key=chiave)

    def _bfs_next_step(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        blocked: set,
        lane_col: Optional[int] = None,
        origin: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Primo passo del percorso più breve start->goal, evitando le celle in `blocked`.

        Il conto si fa AL CONTRARIO, partendo dal traguardo: si misura quanto
        dista ogni cella dall'obiettivo, poi si sceglie fra i passi che
        accorciano davvero la strada. Prima invece si seguivano i "padri"
        lasciati da un BFS in avanti, e lì la forma del cammino non era
        governabile: il padre lo assegna chi scopre per primo, non chi sarebbe
        la scelta migliore. Ecco perché tutte le legioni scendevano lungo la
        colonna del castello.

        Fra i passi buoni — tutti ugualmente brevi — vince chi si tiene sulla
        corsia `lane_col`, e a parità chi resta vicino alla retta
        partenza-obiettivo. Il percorso resta minimo: cambia solo la sua forma.
        """
        start = tuple(start)
        goal = tuple(goal)
        if start == goal:
            return start

        distance = self._distance_map_from(goal, blocked)
        here = distance.get(start)
        if here is None:
            return None

        candidates = [
            neighbor
            for neighbor in self._step_preference(start, origin or start, goal, lane_col)
            if distance.get(neighbor) == here - 1
        ]
        return candidates[0] if candidates else None

    def _distance_map_from(
        self,
        goal: Tuple[int, int],
        blocked: set,
    ) -> Dict[Tuple[int, int], int]:
        """Distanza in caselle di ogni cella raggiungibile dall'obiettivo.

        Le celle bloccate non si attraversano; il traguardo invece è sempre
        raggiungibile, anche se occupato — chi ci arriva ci combatte.

        L'eccezione è il terreno impraticabile: una meta in mezzo al fiume non
        vale nemmeno come traguardo, se no la legione ci camminava dentro
        proprio perché era la destinazione. Con la mappa vuota il chiamante
        ripiega sulla cella utile più vicina, che è il comportamento giusto.
        """
        rows, cols = self.game_map.rows, self.game_map.cols
        if not self._cella_attraversabile(goal):
            return {}
        distance: Dict[Tuple[int, int], int] = {goal: 0}
        queue = deque([goal])
        while queue:
            r, c = queue.popleft()
            step = distance[(r, c)] + 1
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                nr, nc = neighbor
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if neighbor in distance or neighbor in blocked:
                    continue
                if not self._cella_attraversabile(neighbor):
                    continue
                distance[neighbor] = step
                queue.append(neighbor)
        return distance

    def _next_legion_step(
        self,
        current_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        mover: Occupation,
        legion_id: Optional[str] = None,
        lane_col: Optional[int] = None,
        origin: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tuple[int, int], Dict[str, Any]]:
        """Passo verso il target, aggirando gli ostacoli e ripiegando se serve.

        Sono ostacoli il castello nemico (a meno che sia la destinazione scelta)
        e le legioni alleate, che fanno da muro: una cella ne ospita una sola.
        Contro le nemiche invece si va addosso, quello è uno scontro. Se
        l'obiettivo è occupato o irraggiungibile la legione non si pianta: punta
        alla cella libera più vicina.

        Returns:
            (prossima posizione, esito) con `blocked_by` e `fallback`, che
            servono al log per rendere visibile la decisione.
        """
        current_pos = tuple(current_pos)
        target_pos = tuple(target_pos)
        # Da dove è partita questa marcia. Serve a tenere la rotta dritta: senza
        # memoria del punto di partenza, "il passo più diretto" è sempre quello
        # sull'asse più lungo, e la legione scende tutta dritta prima di girare.
        origin = tuple(origin) if origin and len(origin) == 2 else current_pos
        resolution: Dict[str, Any] = {"blocked_by": None, "fallback": None}
        if current_pos == target_pos:
            return current_pos, resolution

        defender_castle = self.game_map.castle_positions.get(mover.opposite())
        allies = self._own_legion_positions_map(mover, exclude_id=legion_id)

        blocked: set = set(allies)
        if defender_castle is not None and tuple(defender_castle) != target_pos:
            blocked.add(tuple(defender_castle))

        # 1. Rotta diretta: il BFS aggira già le alleate lungo il percorso.
        if target_pos not in allies:
            next_step = self._bfs_next_step(current_pos, target_pos, blocked, lane_col, origin)
            # Il BFS concede sempre la cella d'arrivo, anche se bloccata: qui la
            # regola del muro deve valere pure per l'ultimo passo.
            if next_step is not None and next_step not in allies:
                return next_step, resolution

        # 2. Obiettivo occupato o senza varco: ripiego sul punto utile più vicino.
        resolution["blocked_by"] = allies.get(target_pos)
        fallback = self._nearest_reachable_cell_to(current_pos, target_pos, blocked)
        if fallback is None:
            return current_pos, resolution

        resolution["fallback"] = fallback
        if fallback == current_pos:
            return current_pos, resolution

        next_step = self._bfs_next_step(current_pos, fallback, blocked, lane_col, origin)
        if next_step is None or next_step in allies:
            return current_pos, resolution
        return next_step, resolution

    def _refresh_capture_area(self, legion: Dict[str, Any], logs: List[str]) -> None:
        """Tiene aggiornato l'ordine di cattura d'area di una legione.

        Toglie dalla coda le caselle ormai nostre — prese da questa legione, da
        un'altra o da un presidio, non importa — e punta la legione sulla più
        vicina di quelle che restano. Quando la coda si svuota l'ordine è
        eseguito e la legione si ferma lì: non torna a casa da sola, resta a
        presidiare quello che ha appena preso.
        """
        area = legion.get("capture_area")
        if not area:
            return

        current_pos = tuple(legion.get("pos", ()))
        rimaste = []
        for raw in area:
            pos = (int(raw[0]), int(raw[1]))
            cell = self.game_map.get_cell(*pos)
            if cell is not None and cell.occupation != PLAYER:
                rimaste.append(pos)

        if not rimaste:
            legion["capture_area"] = []
            legion["target"] = list(current_pos) if len(current_pos) == 2 else None
            logs.append(
                f"🗺 La legione PLAYER '{legion.get('name')}' ha completato la cattura dell'area."
            )
            return

        legion["capture_area"] = [list(pos) for pos in rimaste]
        if len(current_pos) == 2:
            prossima = min(rimaste, key=lambda pos: self._order_distance(current_pos, pos))
            legion["target"] = list(prossima)

    def _march_origin(
        self,
        legion: Dict[str, Any],
        current_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Punto da cui è cominciata la marcia verso l'obiettivo corrente.

        Si aggiorna solo quando l'obiettivo cambia. È la memoria che tiene
        dritta la rotta: valutando ogni passo a partire da dove la legione si
        trova adesso, la scelta più diretta è sempre quella sull'asse più
        lungo, e viene fuori una L — tutto dritto e poi la svolta.
        """
        memorizzato = legion.get("march_target")
        if tuple(memorizzato or ()) != tuple(target_pos):
            legion["march_target"] = list(target_pos)
            legion["march_origin"] = list(current_pos)

        origin = legion.get("march_origin")
        if isinstance(origin, (list, tuple)) and len(origin) == 2:
            return (int(origin[0]), int(origin[1]))
        return current_pos

    def _log_legion_fallback(
        self,
        legion: Dict[str, Any],
        etichetta: str,
        target_pos: Tuple[int, int],
        next_pos: Tuple[int, int],
        resolution: Dict[str, Any],
        logs: List[str],
    ) -> None:
        """Annota il ripiego quando la legione si sistema al posto dell'obiettivo.

        Scrive solo all'arrivo sulla cella di ripiego e una volta sola: senza
        memoria, una legione ferma riscriverebbe la stessa riga a ogni turno.
        """
        fallback = resolution.get("fallback")
        if fallback is None:
            legion.pop("fallback_note", None)
            return
        if tuple(next_pos) != tuple(fallback) or legion.get("fallback_note") == fallback:
            return

        legion["fallback_note"] = fallback
        blocked_by = resolution.get("blocked_by")
        motivo = (
            f"la legione '{blocked_by}' è arrivata prima"
            if blocked_by else "non c'è un varco libero"
        )
        logs.append(
            f"🧭 La legione {etichetta} '{legion['name']}' non può occupare {tuple(target_pos)}: "
            f"{motivo}. Si sistema in {tuple(fallback)}, la cella libera più vicina all'obiettivo."
        )

    def _pick_ai_legion_target(self, ai_legion: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Target IA: insegui legioni player, altrimenti espandi su obiettivi strategici e solo poi castello."""
        ai_pos = tuple(ai_legion.get("pos", ()))
        if len(ai_pos) != 2:
            return None

        player_pos: Optional[Tuple[int, int]] = None
        if self.player_legions:
            nearest = min(
                self.player_legions.values(),
                key=lambda legion: self._order_distance(ai_pos, tuple(legion.get("pos", ai_pos))),
            )
            nearest_pos = tuple(nearest.get("pos", ()))
            if len(nearest_pos) == 2:
                player_pos = nearest_pos

        own_castle = self.game_map.castle_positions.get(AI)
        enemy_castle = self.game_map.castle_positions.get(PLAYER)
        player_threatens_own_castle = False
        if player_pos is not None and own_castle is not None:
            player_threatens_own_castle = self._order_distance(player_pos, own_castle) <= 3
        player_distance = self._order_distance(ai_pos, player_pos) if player_pos is not None else None

        strategic_targets = self.game_map.get_strategic_targets(
            entity=AI,
            army_vector=self.ai_army,
            terrain_modifiers=self.data["terrain"],
        )
        # `get_strategic_targets` ordina solo per resa del terreno, senza guardare
        # la distanza: da sola porterebbe l'IA dall'altra parte della mappa.
        strategic_targets = [
            item for item in strategic_targets
            if self._ai_expansion_allows((item[1].row, item[1].col))
        ]
        economic_targets = self._collect_ai_economic_targets(ai_pos)

        # Incursione oltre metà campo: l'IA molla l'espansione e converge
        # sull'intruso più vicino. Sostituisce anche un eventuale assalto in corso.
        intruders = self._ai_intruder_legions()
        if intruders:
            focus = getattr(self.ai_policy, "should_focus_intruder", None)
            if callable(focus) and focus(turn=self.game_map.turn, intruder_count=len(intruders)):
                ai_legion["castle_commit"] = False
                return min(intruders, key=lambda p: self._order_distance(ai_pos, p))

        # Un assalto già deciso non va abbandonato a metà strada: il target viene
        # ri-scelto ogni 2-3 turni, ma il castello nemico dista 13 caselle, quindi
        # senza impegno persistente l'IA non arriva mai sotto le mura.
        if ai_legion.get("castle_commit") and enemy_castle is not None:
            if player_threatens_own_castle:
                ai_legion["castle_commit"] = False  # difendere casa ha priorità
            else:
                return enemy_castle

        policy_target = self.ai_policy.choose_target(
            ai_pos=ai_pos,
            player_pos=player_pos,
            own_castle=own_castle,
            enemy_castle=enemy_castle,
            strategic_targets=strategic_targets,
            economic_targets=economic_targets,
            turn=self.game_map.turn,
            legion_size=len(ai_legion.get("units", [])),
        )
        if policy_target is not None:
            chosen = tuple(policy_target)

            # Il castello nemico è un obiettivo strategico, non un inseguimento:
            # va valutato prima del filtro sotto, altrimenti quando il player
            # presidia il proprio castello (player_pos == enemy_castle) l'assalto
            # verrebbe scambiato per una caccia alla legione e deviato.
            if enemy_castle is not None and chosen == tuple(enemy_castle):
                ai_legion["castle_commit"] = True
                return chosen

            if (
                player_pos is not None
                and chosen == player_pos
                and not player_threatens_own_castle
                and player_distance is not None
                and player_distance > 3
                and economic_targets
            ):
                return economic_targets[0]
            return chosen

        if economic_targets:
            return economic_targets[0]

        if strategic_targets:
            # Evita corsa cieca al castello: preferisci tra i migliori target quello più vicino.
            top_candidates = strategic_targets[: min(5, len(strategic_targets))]
            _, best_cell = min(
                top_candidates,
                key=lambda item: self._order_distance(ai_pos, (item[1].row, item[1].col)),
            )
            return best_cell.row, best_cell.col

        # Ripiego: se il profilo è in postura difensiva non deve finire in
        # trasferta per mancanza di alternative — si stringe attorno a casa.
        if enemy_castle is not None and not self._ai_expansion_allows(tuple(enemy_castle)):
            return own_castle or player_pos

        return enemy_castle or player_pos

    def _ai_doctrine_plan(
        self,
        legion: Dict[str, Any],
        legion_index: int,
        current_pos: Tuple[int, int],
        base_target: Tuple[int, int],
        under_threat: bool,
    ) -> Any:
        """Passa l'obiettivo grezzo alla dottrina e ne ricava la manovra.

        Il profilo di difficoltà ha già deciso *dove* andare; qui si decide
        *come*: su quale corsia, se conviene aggirare, se è il caso di fermarsi.
        """
        def percorribile(pos: Tuple[int, int]) -> bool:
            cell = self.game_map.get_cell(*pos)
            return cell is not None and cell.terrain != "Fiume"

        return self.ai_doctrine.plan(
            legion=legion,
            legion_index=legion_index,
            ai_pos=current_pos,
            base_target=base_target,
            enemy_castle=self.game_map.castle_positions.get(PLAYER),
            rows=self.game_map.rows,
            cols=self.game_map.cols,
            turn=self.game_map.turn,
            under_threat=under_threat,
            is_walkable=percorribile,
        )

    def _ai_legion_target_lock_turns(self) -> int:
        """Numero turni minimi prima del retarget IA per evitare zig-zag e inseguimenti artificiali.

        Il valore appartiene al profilo di difficoltà: ogni file lo dichiara.
        """
        getter = getattr(self.ai_policy, "target_lock_turns", None)
        if callable(getter):
            return max(1, int(getter()))
        return 3

    def _castle_defense(self, defender: Occupation, castle_pos: Tuple[int, int]) -> Dict[str, Any]:
        """Difesa del castello: presidi (max 4) e fortificazioni.

        Contano solo i presidi distaccati, fino a `CASTLE_MAX_GARRISON_UNITS`:
        le legioni ferme sulla cella erano accumulabili senza limite e rendevano
        il castello imprendibile, e con la riserva nel conto formare una legione
        *sottraeva* difesa, cioè giocare puniva.

        Returns:
            score              — punteggio difensivo da confrontare con l'attaccante
            damage_multiplier  — riduzione danni delle fortificazioni (1.0 = nessuna)
            defenders          — presidi che difendono la cella
            fortification_level— livelli di fortificazione sul castello
        """
        cell = self.game_map.get_cell(*castle_pos)

        # 1. Solo i presidi difendono la cella, e non oltre il tetto.
        defender_units: List[str] = []
        if cell is not None:
            defender_units = list(cell.garrison_unit_ids)[:CASTLE_MAX_GARRISON_UNITS]

        # 2. Ogni presidio vale una frazione del suo valore in battaglia.
        # [ABILITY-EFFECTS] Chi difende porta con sé i bonus difensivi sbloccati.
        defense_value_of = self._ability_value_of(defender, ab.CTX_DEFENSE)
        troop_score = sum(
            defense_value_of(uid) * CASTLE_DEFENDER_WEIGHT for uid in defender_units
        )

        # 3. Fortificazioni: solo quelle sul castello, e con tetto dedicato.
        fortification_level = 0
        if cell is not None:
            fortification_level = min(int(cell.fortification_level), CASTLE_MAX_FORTIFICATION_LEVEL)
        reduction = fortification_level * CASTLE_FORT_DAMAGE_REDUCTION_PER_LEVEL

        # [ABILITY-EFFECTS] Dottrina Fortezza: mura rinforzate ai punti deboli.
        damage_multiplier = max(0.0, 1.0 - reduction)
        damage_multiplier *= self._ability_economy_factor(defender, ab.ECO_CASTLE_DAMAGE_TAKEN)

        return {
            "score": CASTLE_DEFENSE_BASE + troop_score,
            "damage_multiplier": max(0.0, damage_multiplier),
            "defenders": len(defender_units),
            "fortification_level": fortification_level,
        }

    def _apply_legion_castle_assault(
        self,
        attacker: Occupation,
        legion: Dict[str, Any],
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        logs: List[str],
    ) -> None:
        """Assedio castello nel sistema legioni: usa HP castello, niente conquista istantanea."""
        defender = attacker.opposite()
        defender_castle = self.game_map.castle_positions.get(defender)
        if defender_castle != to_pos:
            return

        hp_before = self.castle_hp.get(defender, self.castle_hp_max.get(defender, CASTLE_BASE_HP))
        legion_unit_ids = legion.get("units", [])
        # [ABILITY-EFFECTS] Il valore d'assedio passa dai bonus dello stile di
        # gioco dell'attaccante (la Scuola d'Assedio si sente qui).
        siege_value_of = self._ability_value_of(attacker, ab.CTX_SIEGE)
        # [BALANCE-LAYER] Contro le mura l'artiglieria pesa più che in campo
        # aperto. Senza layer resta la somma nuda dei valori, come prima.
        if balance is not None:
            raw_strength = balance.siege_strength(legion_unit_ids, siege_value_of)
        else:
            raw_strength = sum(siege_value_of(uid) for uid in legion_unit_ids)
        attacker_strength = max(20.0, raw_strength)

        defense = self._castle_defense(defender, to_pos)
        damage = self._compute_castle_damage(attacker_strength, defense["score"])
        # Le fortificazioni del castello tagliano i danni in percentuale, DOPO il
        # calcolo base: così l'effetto è leggibile ("con 4 livelli incasso il 60%").
        damage = max(8, int(round(damage * defense["damage_multiplier"])))
        hp_after = max(0, hp_before - damage)
        self.castle_hp[defender] = hp_after

        # L'assedio logora comunque: costa fatica e, se respinto, morale — ma
        # in proporzione a quanto poco ha ottenuto.
        status_before = tc.resolve_status(self._legion_condition(attacker, legion))
        castle_max = max(1, self.castle_hp_max.get(defender, CASTLE_BASE_HP))
        tc.apply_siege(
            self._legion_condition(attacker, legion),
            breached=hp_after <= 0,
            damage_ratio=damage / castle_max,
        )
        legion["acted_this_turn"] = True

        castle_cell = self.game_map.get_cell(*to_pos)
        if hp_after <= 0:
            if castle_cell is not None:
                castle_cell.occupation = attacker
            self.state = SessionState.GAME_OVER
            self.winner = attacker.value
            logs.append(
                f"🏰 {attacker.value.upper()} abbatte il castello ({to_pos}) "
                f"con {damage} danni (HP {hp_before}->{hp_after})."
            )
            self._evento(                                     # [EVENT-CHANNEL]
                ev.CASTELLO_CADUTO if ev else "castello_caduto",
                entita=attacker, pos=to_pos, quantita=damage,
                dettaglio={"artiglieria": self._ha_artiglieria(legion_unit_ids)},
            )
            return

        self._log_condition_change(attacker, legion, status_before, logs)

        # Castello non distrutto: assalto respinto, le truppe rientrano subito al castello d'origine
        # (niente spam di assalti: il target viene azzerato, serve un nuovo ordine del giocatore/IA).
        own_castle = self.game_map.castle_positions.get(attacker)
        if own_castle is not None:
            # Come nei ripiegamenti dopo uno scontro: se il castello è occupato
            # da un'altra legione si rientra di fianco, mai sopra.
            legion["pos"] = self._free_cell_for(
                attacker, own_castle, exclude_id=legion.get("id")
            ) or own_castle
        else:
            legion["pos"] = from_pos
        legion["target"] = None
        # Assalto concluso: l'impegno va sciolto, la prossima offensiva si ridecide.
        legion["castle_commit"] = False
        if castle_cell is not None:
            castle_cell.occupation = defender
        from_cell = self.game_map.get_cell(*from_pos)
        if from_cell is not None:
            from_cell.occupation = attacker

        logs.append(
            f"🏰 Assalto respinto: {attacker.value.upper()} infligge {damage} danni "
            f"al castello (HP {hp_before}->{hp_after}) e le truppe rientrano al castello d'origine."
        )
        self._evento(                                         # [EVENT-CHANNEL]
            ev.ASSALTO_CASTELLO if ev else "assalto_castello",
            entita=attacker, pos=to_pos, quantita=damage,
            dettaglio={"hp_prima": hp_before, "hp_dopo": hp_after,
                       "hp_max": castle_max, "respinto": True,
                       # Un castello preso a cannonate suona diverso da uno
                       # preso d'assalto con le scale.
                       "artiglieria": self._ha_artiglieria(legion_unit_ids)},
        )

    def _legion_battle_strength(
        self,
        entity: Occupation,
        legion: Dict[str, Any],
        terrain: str,
        *,
        defending: bool,
        cell: Optional[Any],
        enemy_strength: float = 0.0,
        movement_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Forza di combattimento di una singola legione sulla cella dello scontro.

        Riusa le stesse regole del resto del motore: valore unità modulato dal
        terreno, bonus stack, fattore strategia/contesto e — solo per chi
        difende la propria cella — fortificazioni, presidi e vantaggio del
        terreno difensivo.
        """
        unit_ids = list(legion.get("units", []))
        empty = {
            "strength": 0.0,
            "units": 0,
            "fortification_level": 0,
            "garrison_strength": 0,
            "strategy_factor": 0.0,
            "defense_bonus": 0.0,
            "movement_factor": 1.0,
        }
        if not unit_ids:
            return empty

        # 1. Valore delle unità modulato dal terreno (stessa regola dei presidi).
        # [ABILITY-EFFECTS] Lo stile di gioco pesa qui: in attacco o in difesa
        # a seconda del ruolo che la legione ha in questo scontro.
        context = ab.CTX_DEFENSE if defending else ab.CTX_ATTACK
        unlocked = self._unlocked_abilities(entity)
        if ab.has_unit_effects(unlocked, context):
            base_total = sum(
                self._garrison_unit_defense_value(uid, terrain)
                * ab.unit_factor(unlocked, uid, context)
                for uid in unit_ids
            )
        else:
            base_total = sum(self._garrison_unit_defense_value(uid, terrain) for uid in unit_ids)

        # [BALANCE-LAYER] Chi assalta una cella fortificata guadagna il surplus
        # d'assedio (oggi: solo l'artiglieria). È un'aggiunta al valore già
        # calcolato: togliendo il layer resta esattamente il numero di prima.
        if balance is not None and not defending and cell is not None:
            base_total += balance.fortification_surplus(
                unit_ids, self._ability_value_of(entity, ab.CTX_SIEGE), int(cell.fortification_level)
            )

        # 2. Bonus stack per unità dello stesso tipo (coerente con _strength_breakdown).
        stack_bonus = 0.0
        for unit_id, count in Counter(unit_ids).items():
            if count <= 1:
                continue
            value = self._unit_battle_value(unit_id)
            bonus = value * 0.22 * ((count - 1) ** 1.08)
            stack_bonus += min(bonus, value * count * 0.55)

        # 3. Strategia ed effetti ambientali sul terreno dello scontro.
        #    Si valuta la composizione DI QUESTA legione, non l'esercito globale:
        #    è ciò che rende sensata una strategia per legione (la cavalleria
        #    manovra, la fanteria tiene) invece di una sola per tutto il campo.
        #    Lo stato truppe è quello DI QUESTA legione: la stessa scala della
        #    schermata iniziale, applicata dal solito `apply_modifiers`.
        troop_status = self._legion_troop_status(entity, legion)
        modified, _ = apply_modifiers(
            army_vector=aggregate_army(unit_ids, self.data["units"]),
            terrain_name=terrain,
            weather_name=self.weather,
            troop_status_name=troop_status,
            modifiers_data=self.data,
        )
        # [COMPAT-LAYER] `unit_ids` è la composizione di questa legione: serve
        # all'affinità unità-ambiente, che senza di essa non può entrare.
        strategy_id = self._legion_strategy_id(entity, legion)
        punteggi = self._strategy_scores(
            entity, modified, strategy_id, terrain=terrain, unit_ids=unit_ids
        )
        strategy_factor = self._factor_from_compatibility(punteggi["combat_compatibility"])
        context_factor = max(0.75, min(1.25, (sum(modified.values()) / 4.6)))

        strength = (base_total + stack_bonus) * strategy_factor * context_factor

        # 4. Vantaggi difensivi: valgono solo per la legione già presente sulla cella.
        fortification_level = 0
        garrison_strength = 0
        defense_bonus = 0.0
        if defending and cell is not None:
            fortification_level = int(cell.fortification_level)
            garrison_strength = int(cell.garrison_strength)

            if fortification_level > 0:
                # Come nella difesa statica: parte fissa + parte che scala
                # sull'intensità dell'assalto, altrimenti contro legioni grandi
                # un bonus piatto diventa irrilevante.
                defense_bonus += (fortification_level * 18.0) + (max(0, fortification_level - 1) * 16.0)
                defense_bonus += enemy_strength * min(0.32, 0.12 + (fortification_level * 0.05))
            if garrison_strength > 0:
                defense_bonus += garrison_strength * 18.0
                defense_bonus += enemy_strength * min(0.22, garrison_strength * 0.025)
            if cell.garrison_unit_ids:
                # [ABILITY-EFFECTS] Anche i presidi della cella difendono con
                # i bonus difensivi di chi li ha piazzati.
                garrison_factor = ab.unit_factor if ab.has_unit_effects(unlocked, ab.CTX_DEFENSE) else None
                defense_bonus += sum(
                    self._garrison_unit_defense_value(uid, terrain)
                    * (garrison_factor(unlocked, uid, ab.CTX_DEFENSE) if garrison_factor else 1.0)
                    for uid in cell.garrison_unit_ids
                ) * 2.0
            if fortification_level > 0 and garrison_strength > 0:
                # Sinergia presidio-dentro-fortificazione, come nella difesa statica.
                defense_bonus += fortification_level * garrison_strength * 7.0
            defense_bonus += 8.0 if terrain in {"Foresta", "Montagna", "Palude"} else 5.0

        # 5. Malus da marcia incompleta: una legione sorpresa mentre attraversa
        #    un terreno difficile combatte peggio (regola già del sistema movimento).
        movement_factor = 1.0
        if movement_key is not None:
            movement_factor = float(
                self.movement_system.get_legion_defense_modifier(movement_key)["factor"]
            )

        return {
            "strength": (strength + defense_bonus) * movement_factor,
            "units": len(unit_ids),
            "fortification_level": fortification_level,
            "garrison_strength": garrison_strength,
            "strategy_factor": strategy_factor,
            "defense_bonus": defense_bonus,
            "movement_factor": movement_factor,
            # [COMPAT-LAYER] Entrambi i punteggi, perché il log sperimentale
            # deve poter confrontare ciò che l'Advisor consiglia con ciò che la
            # battaglia premia davvero.
            "strategy_id": strategy_id,
            "combat_compatibility": punteggi["combat_compatibility"],
            "advisory_score": punteggi["advisory_score"],
        }

    def _apply_legion_losses(
        self,
        entity: Occupation,
        legion_id: str,
        legion: Dict[str, Any],
        losses: int,
    ) -> Tuple[int, str, bool]:
        """Applica le perdite a una legione. Ritorna (perdite, testo, sopravvissuta)."""
        unit_ids = list(legion.get("units", []))
        source = self.player_legions if entity == PLAYER else self.ai_legions

        if losses <= 0 or not unit_ids:
            return 0, "", bool(unit_ids)

        losses = min(losses, len(unit_ids))

        # [BALANCE-LAYER] I guaritori presenti nella legione assorbono parte
        # delle perdite. Non sono al riparo: cadono per prime le unità di minor
        # valore, loro comprese, quindi il tampone si consuma da sé.
        saved = 0
        if balance is not None:
            losses, saved = balance.reduced_losses(unit_ids, losses)

        # Cadono per prime le unità di minor valore.
        removed = sorted(unit_ids, key=lambda uid: self._unit_battle_value(uid))[:losses]

        remaining = list(unit_ids)
        for unit_id in removed:
            remaining.remove(unit_id)
        self.troops_lost[entity] += len(removed)  # [ENDGAME-STATS]

        if entity == AI:
            # `ai_units` è l'esercito autoritativo dell'IA: senza rimuoverle anche
            # da lì, _sync_ai_legion_units rimetterebbe in campo le unità perse.
            for unit_id in removed:
                if unit_id in self.ai_units:
                    self.ai_units.remove(unit_id)
            self._recompute_entity_army_state(AI)
            # La ripartizione può riassegnare le truppe fra le legioni superstiti,
            # quindi la sopravvivenza va letta dopo il sync, non da `remaining`.
            self._sync_ai_legion_units()
            survived = bool(source.get(legion_id, {}).get("units"))
            if not self.ai_legions:
                self.ai_last_legion_loss_turn = self.game_map.turn
        else:
            legion["units"] = remaining
            survived = bool(remaining)
            if not survived and legion_id in source:
                del source[legion_id]

        counts = Counter(removed)
        text = ", ".join(
            f"{count} {self.units_map.get(unit_id, {}).get('name', unit_id)}"
            for unit_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        )

        # [BALANCE-LAYER] L'intervento dei guaritori va scritto nel log, se no
        # è un effetto che il giocatore paga 90 grux e non vede mai.
        if saved and balance is not None:
            nota = balance.losses_note(saved)
            if nota:
                text = f"{text} — {nota}" if text else nota

        return losses, text, survived

    def _resolve_legion_clash_if_any(
        self,
        pos: Tuple[int, int],
        logs: List[str],
        attacker: Optional[Occupation] = None,
    ) -> None:
        """Scontro legione-vs-legione: attrito reale con terreno, strategia e difese."""
        player_entry = self._find_legion_at(PLAYER, pos)
        ai_entry = self._find_legion_at(AI, pos)
        if player_entry is None or ai_entry is None:
            return

        cell = self.game_map.get_cell(*pos)
        terrain = cell.terrain if cell is not None else self.player_home_terrain

        # Chi ha appena mosso attacca; l'altra legione difende la cella su cui già si trovava.
        if attacker is None:
            attacker = AI
        defender = attacker.opposite()

        entries = {PLAYER: player_entry, AI: ai_entry}
        atk_id, atk_legion = entries[attacker]
        def_id, def_legion = entries[defender]

        atk = self._legion_battle_strength(
            attacker, atk_legion, terrain, defending=False, cell=cell,
            movement_key=self._legion_movement_key(attacker, atk_id),
        )
        # Le difese della cella scalano sull'intensità dell'assalto: serve prima la forza attaccante.
        dfn = self._legion_battle_strength(
            defender, def_legion, terrain, defending=True, cell=cell,
            enemy_strength=atk["strength"],
            movement_key=self._legion_movement_key(defender, def_id),
        )

        # [EVENT-CHANNEL] L'artiglieria va fotografata adesso: l'evento viene
        # emesso a perdite già applicate, e se i pezzi cadono nello scontro le
        # liste sono già state ripulite — lo scontro suonerebbe di spade dopo
        # essere stato un bombardamento.
        artiglieria_in_campo = self._ha_artiglieria(
            atk_legion.get("units"), def_legion.get("units")
        )

        # Le difese della cella aumentano anche l'attrito subito dall'attaccante.
        atk_losses = self._calculate_losses_for_battle(
            atk["units"],
            atk["strength"],
            dfn["strength"],
            fortification_level=dfn["fortification_level"],
            garrison_strength=dfn["garrison_strength"],
        )
        def_losses = self._calculate_losses_for_battle(
            dfn["units"],
            dfn["strength"],
            atk["strength"],
        )
        # [DOCTRINE-LAYER] Accerchiamento: la morsa fa più morti, ma solo
        # quando è lei ad attaccare. Il tetto anti-sterminio del motore resta,
        # e non può abbassare le perdite che il motore aveva già deciso.
        if doc is not None:
            stato = self._doctrine_state(attacker, atk_legion)
            if stato is not None:
                morsa = doc.enemy_loss_multiplier(
                    stato["strategy_id"], stato["units"],
                    turn=stato["turn"], since_turn=stato["since_turn"],
                    attacking=True,
                )
                if morsa != 1.0 and def_losses > 0:
                    tetto = max(def_losses, dfn["units"] - 1) if dfn["units"] > 1 else dfn["units"]
                    def_losses = min(tetto, doc.scaled_losses(def_losses, morsa))

        # [DOCTRINE-LAYER] Fotografia delle due legioni PRIMA delle perdite:
        # le porte vanno guardate sulla composizione con cui sono entrate in
        # battaglia. Vedi il docstring di `_doctrine_state`.
        units_before = {
            atk_id: list(atk_legion.get("units") or []),
            def_id: list(def_legion.get("units") or []),
        }

        atk_n, atk_text, atk_alive = self._apply_legion_losses(attacker, atk_id, atk_legion, atk_losses)
        def_n, def_text, def_alive = self._apply_legion_losses(defender, def_id, def_legion, def_losses)

        atk_name = atk_legion.get("name", atk_id)
        def_name = def_legion.get("name", def_id)

        # Stato prima dello scontro: serve per annotare gli eventuali passaggi
        # (una legione può uscire dalla battaglia demoralizzata o promossa).
        atk_status_before = tc.resolve_status(self._legion_condition(attacker, atk_legion))
        def_status_before = tc.resolve_status(self._legion_condition(defender, def_legion))

        def registra_esito(vincitrice: Optional[Dict[str, Any]], perdente: Optional[Dict[str, Any]]) -> None:
            """Applica fatica, morale e vittorie alle legioni sopravvissute."""
            for legione, entita, vinto, prima in (
                (vincitrice, attacker if vincitrice is atk_legion else defender, True,
                 atk_status_before if vincitrice is atk_legion else def_status_before),
                (perdente, attacker if perdente is atk_legion else defender, False,
                 atk_status_before if perdente is atk_legion else def_status_before),
            ):
                if legione is None:
                    continue
                condizione = self._legion_condition(entita, legione)
                tc.apply_battle(condizione, won=vinto)
                # [DOCTRINE-LAYER] Blitz: chi perde contro una carica pura
                # lascia il doppio del morale. Si legge sulla dottrina di chi
                # VINCE e si applica a chi perde, una volta sola.
                if doc is not None and not vinto and vincitrice is not None:
                    entita_vinc = attacker if vincitrice is atk_legion else defender
                    id_vinc = atk_id if vincitrice is atk_legion else def_id
                    stato = self._doctrine_state(
                        entita_vinc, vincitrice, units_before.get(id_vinc)
                    )
                    if stato is not None:
                        urto = doc.morale_loss_multiplier(
                            stato["strategy_id"], stato["units"],
                            turn=stato["turn"], since_turn=stato["since_turn"],
                        )
                        if urto > 1.0:
                            tc.lose_morale(
                                condizione, tc.BATTLE_MORALE_LOSS * (urto - 1.0)
                            )
                legione["acted_this_turn"] = True
                self._log_condition_change(entita, legione, prima, logs)

        header = (
            f"{self._format_battle_location(terrain)} su {pos}: "
            f"{attacker.value.upper()} '{atk_name}' ({int(atk['strength'])}) contro "
            f"{defender.value.upper()} '{def_name}' ({int(dfn['strength'])})"
        )
        if dfn["defense_bonus"] > 0:
            header += (
                f" — difese: fort. {dfn['fortification_level']}, presidio {dfn['garrison_strength']}"
            )
        logs.append(header + ".")
        self._evento(                                         # [EVENT-CHANNEL]
            ev.BATTAGLIA if ev else "battaglia",
            entita=attacker, pos=pos,
            quantita=int(atk["strength"]) + int(dfn["strength"]),
            dettaglio={
                "attaccante": attacker.value, "difensore": defender.value,
                "forza_attaccante": int(atk["strength"]),
                "forza_difensore": int(dfn["strength"]),
                "terreno": terrain,
                # Basta che una delle due parti abbia artiglieria perché lo
                # scontro suoni come un bombardamento invece che come spade.
                "artiglieria": artiglieria_in_campo,
                # [EXPERIMENT-LOG] Perché lo scontro è finito così: strategia,
                # i due punteggi di compatibilità e la dottrina di ciascuna
                # parte. Il frontend non li usa; li legge lo script batch, che
                # altrimenti dovrebbe rifare il conto da fuori e rischiare di
                # misurare una formula diversa da quella che ha deciso.
                "meteo": self.weather,
                "scontro": {
                    lato.value: self._dettaglio_scontro(lato, legione, breakdown)
                    for lato, legione, breakdown in (
                        (attacker, atk_legion, atk), (defender, def_legion, dfn)
                    )
                },
            },
        )

        if atk_n or def_n:
            logs.append(
                f"   Perdite: {attacker.value.upper()} -{atk_n}"
                f"{f' ({atk_text})' if atk_text else ''} · "
                f"{defender.value.upper()} -{def_n}"
                f"{f' ({def_text})' if def_text else ''}."
            )
            for chi, quante in ((attacker, atk_n), (defender, def_n)):   # [EVENT-CHANNEL]
                if quante:
                    self._evento(
                        ev.PERDITE if ev else "perdite",
                        entita=chi, pos=pos, quantita=quante,
                    )

        # Entrambe annientate: la cella resta contesa.
        if not atk_alive and not def_alive:
            if cell is not None and not cell.is_castle:
                cell.occupation = Occupation.NEUTRAL
            logs.append(f"   Esito: annientamento reciproco su {pos}.")
            return

        if not def_alive:
            if cell is not None:
                cell.occupation = attacker
            logs.append(
                f"   Esito: vince {attacker.value.upper()} — legione '{def_name}' distrutta."
            )
            registra_esito(atk_legion, None)
            return

        if not atk_alive:
            if cell is not None:
                cell.occupation = defender
            logs.append(
                f"   Esito: assalto respinto — legione '{atk_name}' distrutta."
            )
            registra_esito(def_legion, None)
            return

        # Entrambe sopravvivono: decide la forza, chi perde ripiega al proprio castello.
        if atk["strength"] > dfn["strength"]:
            winner, loser_entity = attacker, defender
            winner_legion, loser_legion = atk_legion, def_legion
            loser_name = def_name
        else:
            winner, loser_entity = defender, attacker
            winner_legion, loser_legion = def_legion, atk_legion
            loser_name = atk_name

        if cell is not None:
            cell.occupation = winner

        arretra_corto = False
        own_castle = self.game_map.castle_positions.get(loser_entity)
        if own_castle is not None:
            # Se il castello è già presidiato da un'altra legione ripiega di fianco:
            # il ripiegamento non deve ricreare la sovrapposizione.
            loser_id = def_id if loser_entity == defender else atk_id
            meta = own_castle
            # [DOCTRINE-LAYER] Difesa in Profondità: arretra di una cella verso
            # casa invece di attraversare mezza mappa, e resta a contatto.
            if doc is not None:
                stato = self._doctrine_state(
                    loser_entity, loser_legion, units_before.get(loser_id)
                )
                if stato is not None and doc.retreats_one_cell(
                    stato["strategy_id"], stato["units"],
                    turn=stato["turn"], since_turn=stato["since_turn"],
                ):
                    passo = self._one_step_retreat(loser_entity, pos, own_castle, loser_id)
                    if passo is not None:
                        meta = passo
                        arretra_corto = True
            loser_legion["pos"] = (
                self._free_cell_for(loser_entity, meta, exclude_id=loser_id) or meta
            )
        loser_legion["target"] = None

        # [DOCTRINE-LAYER] Con la Difesa in Profondità la legione non torna a
        # casa: va detto nella stessa riga dell'esito, se no si contraddice.
        if arretra_corto:
            dove = (
                f"cede una cella e tiene il fronte su "
                f"{tuple(loser_legion.get('pos', ()))} (Difesa in Profondità)"
            )
        else:
            dove = "ripiega al proprio castello"
        logs.append(
            f"   Esito: vince {winner.value.upper()} — la legione '{loser_name}' {dove}."
        )
        registra_esito(winner_legion, loser_legion)

    # ──────────────────────────────────────────────────────────
    # ECONOMIA DI FINE ROUND
    # ──────────────────────────────────────────────────────────

    def _advance_round_economy(self) -> List[str]:
        """Accredita i grux delle miniere e fa gestire all'IA la propria economia."""
        logs: List[str] = []
        for entity in (PLAYER, AI):
            if entity == AI and self.debug_ai_kill_switch:
                continue
            mine_terrains = self.game_map.mine_terrains(entity)
            mine_count = len(mine_terrains)
            if mine_count > 0:
                # Riferimento: quanto renderebbero queste stesse miniere senza
                # le bande. Il confronto è col terreno già contato, se no una
                # sola miniera in palude sembrerebbe già in calo.
                full_income = int(round(sum(mine_yield(t) for t in mine_terrains)))
                income = self._mine_income_for_terrains(mine_terrains, entity)
                self.grux_balance[entity] += income
                diminishing_note = ""
                if income < full_income:
                    diminishing_note = f" (rendimenti decrescenti: -{full_income - income})"
                logs.append(
                    f"[Turno {self.game_map.turn}] ⛏ {entity.value.upper()} incassa {income} grux da {mine_count} miniere{diminishing_note}"
                )

        # Il banco cambia merce prima che l'IA faccia la spesa: se no
        # comprerebbe sempre le offerte del giro precedente.
        logs.extend(self._tick_black_market())

        if not self.debug_ai_kill_switch:
            logs.extend(self._auto_manage_ai_economy())
        logs.extend(self._run_player_auto_recruit())
        return logs

    def toggle_debug_ai_kill_switch(self) -> Dict[str, Any]:
        """[DEBUG-MODULE] Pausa/riprende totalmente l'IA.

        Chiamato solo da gamecore/debug_module: rimuovibile insieme a esso.
        """
        self.debug_ai_kill_switch = not self.debug_ai_kill_switch
        status = "ATTIVO" if self.debug_ai_kill_switch else "DISATTIVO"
        log_entry = (
            f"[Turno {self.game_map.turn}] 🧪 DEBUG TEMPORANEO - KILL SWITCH IA {status} "
            f"(rimuovere in produzione)"
        )
        self.battle_log.append(log_entry)
        return {
            "ok": True,
            "enabled": self.debug_ai_kill_switch,
            "message": log_entry,
            "session": self.to_dict(),
        }

    def _rebuild_ai_army_for_difficulty(self) -> Optional[str]:
        """Ricostruisce l'esercito IA col profilo della difficoltà, se la partita non è iniziata.

        Ogni difficoltà ha il proprio costruttore (budget, qualità delle unità,
        strategia iniziale): senza questo, cambiare difficoltà dal selettore
        avrebbe cambiato solo il comportamento, lasciando l'esercito di 'facile'.
        """
        if self.game_map.turn > 1 or self.player_legions or self.ai_legions:
            return None

        ai_data = build_ai_army(
            data=self.data,
            ai_terrain=self.ai_home_terrain,
            weather=self.weather,
            n_units=3,
            budget=STARTING_GRUX,
            seed=self.ai_policy_seed,
            difficulty=self.ai_difficulty,
        )

        self.ai_units = list(ai_data["units"])
        self.ai_strategy_id = ai_data["strategy"]["id"]
        self.ai_strategy_name = ai_data["strategy"]["name"]
        self.ai_army = ai_data["army_vector"]
        self.ai_modified = ai_data["modified_vector"]
        self.ai_troop_status = ai_data["troop_status"]
        self.ai_army_cost = ai_data.get("army_cost", 0)
        self.grux_balance[AI] = ai_data.get("remaining_grux", STARTING_GRUX - self.ai_army_cost)

        # Gli HP del castello dipendono dalla dimensione dell'esercito.
        self.castle_hp_max = self._build_castle_hp_pool()
        self.castle_hp[AI] = self.castle_hp_max[AI]

        composition = ", ".join(
            f"{count} {self.units_map.get(uid, {}).get('name', uid)}"
            for uid, count in sorted(Counter(self.ai_units).items())
        )
        return (
            f"[Turno {self.game_map.turn}] 🏗 L'IA riorganizza l'esercito per la difficoltà "
            f"{self.ai_difficulty.upper()}: {composition} · strategia {self.ai_strategy_name} "
            f"· {self.grux_balance[AI]} grux in cassa"
        )

    def set_ai_difficulty(self, difficulty: str) -> Dict[str, Any]:
        """Aggiorna la difficoltà IA runtime e ricalcola la policy decisionale."""
        if self.state != SessionState.ACTIVE:
            raise ValueError("La partita è terminata.")

        normalized = normalize_ai_difficulty(difficulty)
        if normalized == self.ai_difficulty:
            return {
                "ok": True,
                "message": f"Difficoltà IA già impostata su {normalized}.",
                "state": self.state.value,
                "map": self.game_map.to_dict(),
                "session": self.to_dict(),
            }

        self.ai_difficulty = normalized
        self.ai_policy = build_ai_policy(self.ai_difficulty, seed=None)
        # Le ricerche non ancora avviate prendono il ritmo della nuova
        # difficoltà; quelle in corso o già sbloccate restano come sono, che
        # cambiare i turni sotto una ricerca aperta sarebbe riscrivere il passato.
        research_scale = ab.ai_research_scale(self.ai_difficulty)
        for ability_id, ability_state in self.ability_states[AI].items():
            definition = ab.ABILITIES.get(ability_id)
            if definition is not None and ability_state.started_turn is None:
                ability_state.turns_required = max(
                    4, int(round(definition.turns_required * research_scale))
                )
        # Cambia difficoltà, cambia manovra: le corsie già assegnate alle
        # legioni restano, la dottrina nuova vale dal prossimo obiettivo.
        self.ai_doctrine = ai_doctrine.for_difficulty(self.ai_difficulty, seed=None)
        log_entry = f"[Turno {self.game_map.turn}] ⚙ Sistema: difficoltà IA impostata su {self.ai_difficulty.upper()}"
        self.battle_log.append(log_entry)

        # A partita appena iniziata la difficoltà deve valere davvero: ogni profilo
        # costruisce l'esercito IA a modo suo (budget, qualità unità, strategia).
        # A partita in corso non si può riscrivere il passato: cambia solo la policy.
        rebuild_log = self._rebuild_ai_army_for_difficulty()
        if rebuild_log:
            self.battle_log.append(rebuild_log)

        return {
            "ok": True,
            "message": log_entry,
            "state": self.state.value,
            "map": self.game_map.to_dict(),
            "session": self.to_dict(),
        }

    # Attributi che caratterizzano una strategia come offensiva o difensiva.
    _OFFENSE_ATTRS = ("U1_attack", "U3_mobility", "U7_range_power")
    _DEFENSE_ATTRS = ("U2_defense", "U5_discipline", "U8_support")

    def _update_ai_phase_and_profile(self) -> Optional[str]:
        """Aggiorna la fase interna dell'IA leggendo lo stato reale della partita."""
        updater = getattr(self.ai_policy, "update_phase", None)
        if not callable(updater):
            return None

        stats = self.game_map.to_dict().get("stats", {})
        ai_cells = int(stats.get("ai_cells", 0))
        player_cells = int(stats.get("player_cells", 0))

        was_annihilating = bool(getattr(self.ai_policy, "is_annihilating", lambda: False)())
        phase = updater(
            turn=self.game_map.turn,
            ai_army=len(self.ai_units),
            player_army=len(self.player_units) + sum(
                len(lg.get("units", [])) for lg in self.player_legions.values()
            ),
            ai_cells=ai_cells,
            player_cells=player_cells,
            ai_grux=self.grux_balance[AI],
            player_grux=self.grux_balance[PLAYER],
            player_intruding=bool(self._ai_intruder_legions()),
        )

        now_annihilating = bool(getattr(self.ai_policy, "is_annihilating", lambda: False)())
        if now_annihilating and not was_annihilating:
            # Commutazione irreversibile: il giocatore deve accorgersene.
            for legion in self.ai_legions.values():
                legion["target"] = None
                legion["castle_commit"] = False
            return (
                f"[Turno {self.game_map.turn}] ☠️ L'IA ha raggiunto la superiorità: "
                f"abbandona la difesa e marcia per annientarti."
            )
        return None

    def _maybe_update_ai_strategy(self) -> Optional[str]:
        """Rivaluta la strategia IA sull'esercito corrente, adattandola al player."""
        review = getattr(self.ai_policy, "should_review_strategy", None)
        if not callable(review) or not review(self.game_map.turn):
            return None
        if not self.ai_units:
            return None

        terrain = self.ai_home_terrain
        if self.ai_legions:
            first = next(iter(sorted(self.ai_legions.items())))[1]
            pos = tuple(first.get("pos", ()))
            if len(pos) == 2:
                cell = self.game_map.get_cell(*pos)
                if cell is not None:
                    terrain = cell.terrain

        modified, _ = apply_modifiers(
            army_vector=self.ai_army,
            terrain_name=terrain,
            weather_name=self.weather,
            troop_status_name=self.ai_troop_status,
            modifiers_data=self.data,
        )
        ranking = compute_ranking(
            army_vector=modified,
            strategies_list=self.data["strategies"],
            unit_ids=self.ai_units,
            terrain_name=terrain,
            weather_name=self.weather,
            affinities_data=self.data.get("unit_affinities", {}),
        )
        if not ranking:
            return None

        bias_getter = getattr(self.ai_policy, "strategy_bias", None)
        bias = bias_getter() if callable(bias_getter) else {"offense": 0.5, "defense": 0.5}

        # Fra le strategie compatibili sceglie quella coerente con la postura
        # attuale: difensiva se il player preme, offensiva quando deve chiudere.
        def posture_score(entry: Dict[str, Any]) -> float:
            ideal = entry.get("ideal_attributes", {})
            offense = sum(float(ideal.get(a, 0.0)) for a in self._OFFENSE_ATTRS)
            defense = sum(float(ideal.get(a, 0.0)) for a in self._DEFENSE_ATTRS)
            fit = float(entry.get("compatibility", 0.0)) / 100.0
            return fit + (offense * bias.get("offense", 0.5) + defense * bias.get("defense", 0.5)) * 0.22

        best = max(ranking[: min(5, len(ranking))], key=posture_score)
        if best["id"] == self.ai_strategy_id:
            return None

        previous = self.ai_strategy_name
        self.ai_strategy_id = best["id"]
        self.ai_strategy_name = best["name"]
        profile = getattr(self.ai_policy, "player_profile", "sconosciuto")
        return (
            f"[Turno {self.game_map.turn}] 🧠 IA cambia strategia: {previous} → "
            f"{best['name']} (ti legge come '{profile}')"
        )

    # [DOCTRINE-LAYER] ─────────────────────────────────────────────
    def _ai_legion_situation(
        self, legion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Le quattro letture su cui l'IA decide la dottrina di una legione.

        Volutamente grossolane — "sto per menare le mani", "sto attraversando
        una palude" — il criterio con cui pesarle lo mette la difficoltà.
        """
        pos = tuple(legion.get("pos", ()))
        bersaglio = legion.get("target")
        castello_nemico = self.game_map.castle_positions.get(PLAYER)

        contatto = False
        if len(pos) == 2:
            for altra in self.player_legions.values():
                altra_pos = tuple(altra.get("pos", ()))
                if len(altra_pos) != 2:
                    continue
                if abs(altra_pos[0] - pos[0]) + abs(altra_pos[1] - pos[1]) <= 1:
                    contatto = True
                    break

        duro = False
        if len(pos) == 2:
            for candidata in (pos, tuple(bersaglio) if bersaglio else None):
                if candidata is None or len(candidata) != 2:
                    continue
                cella = self.game_map.get_cell(*candidata)
                if cella is not None and cella.terrain in TERRENI_FATICOSI:
                    duro = True
                    break

        # I profili che leggono come gioca il player ne ricavano una postura:
        # senza portarla fin qui resterebbe una riga di log, perché a combattere
        # va la dottrina scelta legione per legione, non la strategia generale.
        bias_getter = getattr(self.ai_policy, "strategy_bias", None)
        postura = bias_getter() if callable(bias_getter) else None

        stato = tc.resolve_status(self._legion_condition(AI, legion))
        return {
            "assalto": bool(bersaglio) and tuple(bersaglio) == castello_nemico,
            "contatto": contatto,
            "in_ritirata": stato == tc.STATUS_DEMORALIZED,
            "terreno_duro": duro,
            "postura": postura,
        }

    def _log_doctrine_gate_changes(self, logs: List[str]) -> None:
        """Avvisa quando l'effetto di una dottrina si accende o si spegne da solo.

        Basta perdere una cavalleria e la Manovra sui Fianchi smette di valere:
        senza questa riga la legione marcia piano con il pannello che dice
        ancora "Manovra sui Fianchi".
        """
        if doc is None:
            return
        for entity, legioni in ((PLAYER, self.player_legions), (AI, self.ai_legions)):
            for legion in legioni.values():
                unita = legion.get("units") or []
                if not unita:
                    continue
                sid = self._legion_strategy_id(entity, legion)
                if not doc.has_effect(sid):
                    legion.pop("doctrine_gate_ok", None)
                    continue
                aperta = doc.gate_passed(sid, unita)
                prima = legion.get("doctrine_gate_ok")
                legion["doctrine_gate_ok"] = aperta
                if prima is None or prima == aperta:
                    continue
                nome = self.strategies_map.get(sid, {}).get("name", sid)
                if aperta:
                    logs.append(
                        f"[Turno {self.game_map.turn}] 🧭 {entity.value.upper()}: "
                        f"'{legion.get('name')}' ha di nuovo la composizione per "
                        f"{nome} — {doc.effect_text(sid)}."
                    )
                else:
                    logs.append(
                        f"[Turno {self.game_map.turn}] ⚠ {entity.value.upper()}: "
                        f"'{legion.get('name')}' non ha più la composizione per "
                        f"{nome}: l'effetto è spento, servono {doc.gate_text(sid)}."
                    )

    def _ai_doctrine_factors(
        self, legion: Dict[str, Any], candidate: Sequence[str]
    ) -> List[float]:
        """Il fattore di forza che ogni dottrina candidata darebbe a questa legione.

        La strategia moltiplica la forza fra 0.30 e 2.35: senza questo conto
        l'IA sorteggiava anche quelle incompatibili e si dimezzava da sola.
        """
        pos = tuple(legion.get("pos", ()))
        cella = self.game_map.get_cell(*pos) if len(pos) == 2 else None
        terreno = cella.terrain if cella is not None else self.ai_home_terrain
        unita = list(legion.get("units") or [])
        modificato, _ = apply_modifiers(
            army_vector=aggregate_army(unita, self.data["units"]),
            terrain_name=terreno,
            weather_name=self.weather,
            troop_status_name=self._legion_troop_status(AI, legion),
            modifiers_data=self.data,
        )
        return [
            self._strategy_factor(AI, terreno, modificato, sid, unita) for sid in candidate
        ]

    def _update_ai_legion_doctrines(self, logs: List[str]) -> None:
        """Ogni legione IA sceglie la propria dottrina, con l'attrito di tutti.

        `doctrine_sharpness()` dice quanto darle retta: a 0 è un sorteggio fra
        le candidate praticabili, salendo si stringe su quella giusta per il
        momento.
        """
        if doc is None:
            return

        rng = self.doctrine_rng
        try:
            getter = getattr(self.ai_policy, "doctrine_sharpness", None)
            durezza = float(getter()) if callable(getter) else 0.0
        except (TypeError, ValueError):
            durezza = 0.0

        for legion in self.ai_legions.values():
            unita = legion.get("units") or []
            if not unita:
                continue
            ultimo = legion.get("doctrine_changed_turn")
            if not doc.can_change(self.game_map.turn, ultimo):
                continue

            candidate = [
                sid for sid in doc.ai_candidates(unita) if sid in self.strategies_map
            ]
            if not candidate:
                continue

            # Le dottrine che indebolirebbero troppo la legione escono di lista
            # prima del sorteggio, se no a difficoltà bassa — dove i punteggi
            # non vengono nemmeno guardati — l'IA si dimezza da sola. Quanto
            # stringe il filtro lo decide la stessa manopola della scelta.
            fattori = self._ai_doctrine_factors(legion, candidate)
            tenuti = doc.ai_viable(candidate, fattori, doc.ai_factor_floor(durezza))
            candidate = [candidate[i] for i in tenuti]
            fattori = [fattori[i] for i in tenuti]

            punteggi = doc.ai_scores(
                unita, candidate, self._ai_legion_situation(legion), fattori
            )
            scelta = candidate[self._indice_pesato(punteggi, durezza, rng)]
            if scelta == self._legion_strategy_id(AI, legion):
                # Fissa comunque la dottrina sulla legione: finché resta sul
                # ripiego alla strategia generale, un cambio di quella gliela
                # cambierebbe di colpo, senza attesa e senza ritardo.
                legion.setdefault("strategy_id", scelta)
                continue

            legion["strategy_id"] = scelta
            self._mark_doctrine_change(legion)
            nome = self.strategies_map.get(scelta, {}).get("name", scelta)
            logs.append(
                f"[Turno {self.game_map.turn}] 🧭 La legione IA "
                f"'{legion.get('name')}' adotta {nome}."
            )

    def _auto_manage_ai_economy(self) -> List[str]:
        """L'IA piazza miniere disponibili e recluta automaticamente se può permetterselo."""
        logs: List[str] = []

        phase_log = self._update_ai_phase_and_profile()
        if phase_log:
            logs.append(phase_log)

        strategy_log = self._maybe_update_ai_strategy()
        if strategy_log:
            logs.append(strategy_log)

        # I grux messi da parte per la ricerca non li tocca nessuna spesa
        # difensiva. Prima li proteggeva solo il reclutamento: l'anello del
        # castello se li mangiava prima che la ricerca partisse, e l'incubo —
        # l'unico profilo che la vuole a ogni turno — era anche l'unico a
        # sbloccare meno abilità del difficile.
        risparmio_ricerca = self._ai_research_savings()
        logs.extend(self._reinforce_ai_castle_ring(risparmio_ricerca))
        logs.extend(self._garrison_ai_castle(risparmio_ricerca))
        if self.ai_policy.should_start_research(self.game_map.turn):
            # La riga di log la scrive già `_start_ability_research`: qui non
            # va rimessa nella lista, o comparirebbe due volte nel registro.
            self._run_ai_research()

        logs.extend(self._run_ai_black_market())

        ai_slots = self._available_mine_slots(AI)
        attempts = self.ai_policy.mine_attempts(ai_slots, self.game_map.turn)
        while attempts > 0 and ai_slots > 0:
            placed = self._place_best_ai_mine()
            if not placed:
                break
            logs.append(placed)
            ai_slots -= 1
            attempts -= 1

        # Fortificazione IA: la cadenza appartiene al profilo di difficoltà.
        if self.grux_balance[AI] >= self.base_fortification_cost:
            gate_getter = getattr(self.ai_policy, "fortify_turn_gate", None)
            fortify_turn_gate = max(1, int(gate_getter())) if callable(gate_getter) else 3
            if self.game_map.turn % fortify_turn_gate == 0:
                fort_log = self._place_best_ai_fortification()
                if fort_log:
                    logs.append(fort_log)

        # L'IA mette da parte i grux della ricerca che ha in programma, se no
        # non ci arriverebbe mai: alle difficoltà alte recluta ogni turno che
        # può e la cassa non supera il prezzo di un'abilità.
        recruit_budget = max(0, self.grux_balance[AI] - self._ai_research_savings())

        can_recruit_now = self._can_recruit_now(AI)
        if can_recruit_now and self.ai_policy.should_recruit(grux_balance=recruit_budget, turn=self.game_map.turn):
            affordable_units = [unit for unit in self.data["units"] if self._recruit_cost(AI, unit["id"]) <= recruit_budget]
            scelta = self._scegli_recluta_ia(affordable_units)
            if scelta is not None:
                self._recruit_unit(AI, scelta, auto=True)

        return logs

    def _recruit_cooldown_for(self, entity: Occupation) -> int:
        """Cooldown reclutamento dell'entità, abilità comprese (minimo 1 turno)."""
        # [ABILITY-EFFECTS] Industria Bellica accorcia l'attesa fra due reclute.
        bonus = ab.recruit_cooldown_bonus(self._unlocked_abilities(entity))
        return max(1, self.recruit_cooldown_turns + bonus)

    def _recruit_cost(self, entity: Occupation, unit_id: str) -> int:
        """Prezzo di una recluta per questa entità, abilità comprese."""
        base = int(self.unit_costs[unit_id])
        # [ABILITY-EFFECTS] Industria Bellica abbassa il listino delle reclute.
        factor = self._ability_economy_factor(entity, ab.ECO_RECRUIT_COST)
        if factor == 1.0:
            return base
        return max(5, int(round(base * factor)))

    def _can_recruit_now(self, entity: Occupation) -> bool:
        """True se l'entità può reclutare in questo turno (cooldown anti-spam)."""
        last_turn = self.last_recruit_turn.get(entity)
        if last_turn is None:
            return True
        return (self.game_map.turn - last_turn) >= self._recruit_cooldown_for(entity)

    def _ai_front_terrain(self) -> str:
        """Il terreno dove l'IA combatte davvero, non quello di partenza.

        È la cella della sua legione più grossa: se il fronte è in palude non
        ha senso comprare cavalleria, e prima invece succedeva.
        """
        fronte = self._fronte_ia()
        if fronte is not None:
            cella = self.game_map.get_cell(*fronte[0])
            if cella is not None:
                return cella.terrain
        return self.ai_home_terrain

    def _unit_terrain_affinity(self, unit_id: str, terrain: str) -> float:
        """Quanto quella truppa rende su quel terreno (0 = pessimo, 1 = ideale).

        Legge la stessa tabella che il gioco usa per il giocatore, così l'IA
        ragiona sui numeri veri e non su una copia divergente.
        """
        tabella = self.data.get("unit_affinities") or {}
        default = float(tabella.get("default_affinity", 0.5))
        voce = (tabella.get("environment_affinity") or {}).get(unit_id) or {}
        terreni = voce.get("terrain") or {}
        try:
            return float(terreni.get(terrain, default))
        except (TypeError, ValueError):
            return default

    def _ai_marginal_strategy_gain(
        self,
        attrs: Dict[str, float],
        terrain: str,
        compat_base: Optional[float],
        unit_id: Optional[str] = None,
    ) -> Optional[float]:
        """Di quanto questa truppa avvicina l'esercito IA alla sua strategia.

        Il vettore d'esercito è una media, quindi aggiungere un uomo lo sposta
        di 1/(n+1): il conto si fa senza riaggregare tutto. Torna None quando
        non c'è ancora un esercito su cui ragionare.
        """
        if compat_base is None:
            return None
        n = len(self.ai_units)
        base = self.ai_army
        nuovo = {
            chiave: ((base.get(chiave, 0.0) * n) + float(attrs.get(chiave, 0.0))) / (n + 1)
            for chiave in base
        }
        modificato, _ = apply_modifiers(
            army_vector=nuovo,
            terrain_name=terrain,
            weather_name=self.weather,
            troop_status_name=self.ai_troop_status,
            modifiers_data=self.data,
        )
        # [COMPAT-LAYER] L'esercito ipotetico comprende la recluta: il confronto
        # con `compat_base` deve usare la stessa formula su entrambi i lati.
        ipotetico = list(self.ai_units) + ([unit_id] if unit_id else [])
        return self._strategy_compatibility(
            AI, modificato, terrain=terrain, unit_ids=ipotetico
        ) - compat_base

    def _scegli_recluta_ia(self, candidate: List[Dict[str, Any]]) -> Optional[str]:
        """Quale truppa arruola l'IA fra quelle che può permettersi.

        Tre criteri, tutti sullo stato reale della partita:
          · quanto la truppa rende sul terreno del fronte, col meteo di adesso;
          · quanto avvicina l'esercito alla strategia che l'IA sta usando;
          · quanta ne ha già — il freno che le impedisce di comprare
            ottanta arcieri di fila.
        Poi il profilo di difficoltà decide quanto darle retta: `easy` tira a
        sorte fra le candidate, `nightmare` prende quasi sempre la migliore.
        """
        if not candidate:
            return None

        terreno = self._ai_front_terrain()
        conteggi = Counter(self.ai_units)
        totale = max(1, len(self.ai_units))
        costi = {unit["id"]: max(1, self._recruit_cost(AI, unit["id"])) for unit in candidate}
        costo_medio = sum(costi.values()) / len(costi)

        # La compatibilità di partenza si calcola una volta sola: le candidate
        # si confrontano su quanto la spostano.
        compat_base: Optional[float] = None
        if self.ai_units:
            base_mod, _ = apply_modifiers(
                army_vector=self.ai_army,
                terrain_name=terreno,
                weather_name=self.weather,
                troop_status_name=self.ai_troop_status,
                modifiers_data=self.data,
            )
            compat_base = self._strategy_compatibility(
                AI, base_mod, terrain=terreno, unit_ids=self.ai_units
            )

        guadagni = [
            self._ai_marginal_strategy_gain(
                unit["attributes"], terreno, compat_base, unit["id"]
            )
            for unit in candidate
        ]
        validi = [g for g in guadagni if g is not None]
        minimo = min(validi) if validi else 0.0
        massimo = max(validi) if validi else 0.0
        larghezza = massimo - minimo

        punteggi: List[float] = []
        for unit, guadagno in zip(candidate, guadagni):
            uid = unit["id"]

            # Resa in campo: il meteo è già dentro `_unit_battle_value`, il
            # terreno lo aggiunge l'affinità (0.6× al peggio, 1.4× al meglio).
            valore = self._unit_battle_value(uid) * (
                0.6 + (0.8 * self._unit_terrain_affinity(uid, terreno))
            )

            # Strategia: punteggio relativo, così non serve tarare una scala.
            if guadagno is None or larghezza <= 1e-9:
                fattore_strategia = 1.0
            else:
                quota = (guadagno - minimo) / larghezza
                fattore_strategia = 1.0 + (RECRUIT_BANDA_STRATEGIA * ((quota * 2.0) - 1.0))

            # Varietà: più ne ha, meno ne vuole.
            fattore_varieta = max(
                0.15, 1.0 - (RECRUIT_PESO_VARIETA * (conteggi.get(uid, 0) / totale))
            )

            relativo = costi[uid] / costo_medio
            punteggi.append(
                (valore * fattore_strategia * fattore_varieta)
                / (relativo ** RECRUIT_ESPONENTE_PREZZO)
            )

        return candidate[self._indice_recluta_ia(punteggi)]["id"]

    def _indice_recluta_ia(self, punteggi: List[float]) -> int:
        """Quale candidata esce, data la decisione del profilo di difficoltà.

        `recruit_sharpness()` è l'unica leva: 0 sorteggia alla pari (l'IA
        facile mette insieme un'armata varia ma senza criterio), i numeri alti
        concentrano la scelta sulla migliore.
        """
        getter = getattr(self.ai_policy, "recruit_sharpness", None)
        try:
            durezza = float(getter()) if callable(getter) else RECRUIT_SHARPNESS_DEFAULT
        except (TypeError, ValueError):
            durezza = RECRUIT_SHARPNESS_DEFAULT

        rng = getattr(self.ai_policy, "rng", None) or random
        return self._indice_pesato(punteggi, durezza, rng)

    def _indice_pesato(self, punteggi: List[float], durezza: float, rng) -> int:
        """Sorteggio pesato: a durezza 0 tutti uguali, salendo vince la migliore.

        Lo usano il reclutamento e la scelta della dottrina, che hanno la
        stessa forma: una lista di punteggi e un profilo di difficoltà che
        decide quanto dargli retta.
        """
        if not punteggi:
            return 0
        if durezza <= 0.0:
            return rng.randrange(len(punteggi))

        migliore = max(punteggi)
        if migliore <= 0.0:
            return rng.randrange(len(punteggi))
        pesi = [max(0.0, p / migliore) ** durezza for p in punteggi]
        if sum(pesi) <= 0.0:
            return rng.randrange(len(punteggi))
        return rng.choices(range(len(punteggi)), weights=pesi, k=1)[0]

    def _entity_units(self, entity: Occupation) -> List[str]:
        return self.player_units if entity == PLAYER else self.ai_units

    def _available_garrisons(self, entity: Occupation) -> int:
        """Unità distaccabili: sempre almeno una legione resta con l'armata principale."""
        units = self._entity_units(entity)
        return max(0, len(units) - 1)

    @staticmethod
    def _order_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _rifiuta_destinazione_impraticabile(self, pos: Tuple[int, int]) -> None:
        """Alza un errore se la meta è su un terreno che nessuno può calpestare.

        Senza questo la legione accettava l'ordine e poi ripiegava in silenzio
        sulla cella più vicina: dal pannello sembrava che ignorasse il comando.
        """
        cella = self.game_map.get_cell(*pos)
        if cella is not None and cella.terrain in TERRENI_INVALICABILI:
            raise ValueError(
                f"Le legioni non attraversano il {cella.terrain.lower()}: "
                f"la cella {pos} non può essere una destinazione. "
                f"Punta un guado o la sponda."
            )

    def _cella_attraversabile(self, pos: Tuple[int, int]) -> bool:
        """Una legione può passare di qui?

        Fuori mappa no, e nemmeno sui terreni invalicabili: è il controllo che
        rende i guadi del fiume delle vere strettoie invece che decorazione.
        """
        cella = self.game_map.get_cell(*pos)
        return cella is not None and cella.terrain not in TERRENI_INVALICABILI

    def _meta_praticabile(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """La meta più vicina a `pos` su cui si possa davvero mettere piede.

        L'IA sceglie i suoi obiettivi con la dottrina, che ragiona di corsie e
        di distanze e ogni tanto punta una cella d'acqua. Invece di rifiutare
        l'ordine — l'IA non ha nessuno a cui dirlo — lo si sposta sulla sponda
        più vicina.
        """
        pos = tuple(pos)
        if self._cella_attraversabile(pos):
            return pos
        visitate = {pos}
        coda = deque([pos])
        while coda:
            r, c = coda.popleft()
            for vicino in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if vicino in visitate:
                    continue
                if self.game_map.get_cell(*vicino) is None:
                    continue
                if self._cella_attraversabile(vicino):
                    return vicino
                visitate.add(vicino)
                coda.append(vicino)
        return pos

    def execute_turn(self) -> dict:
        """Avanza il turno di tutte le legioni e risolve i conflitti."""
        if self.state != SessionState.ACTIVE:
            raise ValueError("Partita terminata.")

        logs = []

        # [CARAVAN] Prima di tutto il resto: i rinforzi arrivati stanotte
        # combattono già oggi. Metterlo dopo il movimento avrebbe fatto
        # incontrare al giocatore una legione più debole di quella vera.
        self._muovi_carovane(logs)

        # 1. Movimento Legioni Player
        for legion_id, legion in list(self.player_legions.items()):
            # Ordine di cattura d'area: la meta di questo turno esce dalla coda
            # delle caselle che restano da prendere.
            self._refresh_capture_area(legion, logs)

            target = legion.get("target")
            target_pos = tuple(target) if target is not None else None
            current_pos = tuple(legion.get("pos", ()))
            if target_pos is None or len(current_pos) != 2 or len(target_pos) != 2:
                continue
            if current_pos == target_pos:
                continue

            # Terreno difficile: la legione può essere ancora in marcia dal turno scorso.
            if self._legion_is_movement_blocked(PLAYER, legion_id, legion, logs):
                continue

            next_pos, resolution = self._next_legion_step(
                current_pos, target_pos, PLAYER, legion_id,
                origin=self._march_origin(legion, current_pos, target_pos),
            )
            self._log_legion_fallback(legion, "PLAYER", target_pos, next_pos, resolution, logs)
            if next_pos == current_pos:
                continue
            legion["pos"] = next_pos
            self._register_legion_move_cost(PLAYER, legion_id, legion, current_pos, next_pos)

            row, col = next_pos
            cell = self.game_map.get_cell(row, col)
            # Terreno attraversato: la fatica viene contata a fine turno.
            legion["marched_terrain"] = cell.terrain if cell is not None else "Pianura"
            if cell and getattr(cell, "occupation", None) != PLAYER:
                cell.occupation = PLAYER
                logs.append(f"⚔️ La legione PLAYER '{legion['name']}' conquista la cella ({row},{col}).")
                self._evento(                                 # [EVENT-CHANNEL]
                    ev.CELLA_CONQUISTATA if ev else "cella_conquistata",
                    entita=PLAYER, pos=(row, col),
                    dettaglio={"legione": legion.get("name")},
                )

            self._apply_legion_castle_assault(PLAYER, legion, current_pos, next_pos, logs)
            if tuple(legion.get("pos", ())) != next_pos:
                continue

            self._resolve_legion_clash_if_any(next_pos, logs, attacker=PLAYER)
            if self.state != SessionState.ACTIVE:
                break

        # 2. Movimento Legioni IA
        ai_moved = False
        ai_skipped = False
        # [DEBUG-MODULE] Gate del kill switch nel ciclo vivo dell'IA.
        if self.state == SessionState.ACTIVE and self.debug_ai_kill_switch:
            # Kill switch: l'IA è completamente congelata. Il controllo va qui,
            # nel ciclo vivo: quello storico stava in `_ai_turn`, cioè nel vecchio
            # sistema a singola armata ormai irraggiungibile, quindi le legioni
            # continuavano a muoversi, conquistare e assaltare a switch attivo.
            ai_skipped = True
            logs.append(
                f"[Turno {self.game_map.turn}] 🧪 IA CONGELATA (kill switch attivo): "
                f"nessun movimento, nessuna costruzione, nessuna recluta."
            )
        elif self.state == SessionState.ACTIVE:
            self._ensure_ai_legions_initialized()
            # Rete di sicurezza: assorbe qualsiasi disallineamento legione/esercito IA
            # maturato fuori dai punti di mutazione noti (recluta, perdite).
            self._sync_ai_legion_units()
            # [DOCTRINE-LAYER] Prima di muovere: ogni legione decide con
            # che dottrina affronta questo turno.
            self._update_ai_legion_doctrines(logs)
            if self.ai_policy.should_skip_turn(self.game_map.turn):
                ai_skipped = True
                logs.append(f"[Turno {self.game_map.turn}] IA esita e mantiene la posizione.")
            else:
                # Nemico dentro casa: la dottrina sospende manovre e attese.
                under_threat = bool(self._ai_intruder_legions())
                for legion_index, (legion_id, legion) in enumerate(list(self.ai_legions.items())):
                    current_pos = tuple(legion.get("pos", ()))
                    if len(current_pos) != 2:
                        continue

                    # Stesso vincolo del player: nessuna corsia preferenziale per l'IA.
                    if self._legion_is_movement_blocked(AI, legion_id, legion, logs):
                        continue

                    target_pos = self._pick_ai_legion_target(legion)
                    if target_pos is None:
                        continue

                    # La dottrina trasforma l'obiettivo in manovra: corsia,
                    # aggiramento, giro largo o sosta.
                    plan = self._ai_doctrine_plan(
                        legion, legion_index, current_pos, target_pos, under_threat
                    )
                    if plan.hold:
                        ai_moved = True   # è una scelta, non immobilismo
                        logs.append(
                            f"🤖 La legione IA '{legion['name']}' resta in {plan.reason} "
                            f"su {current_pos}."
                        )
                        continue
                    if plan.target is not None:
                        target_pos = tuple(plan.target)
                    # La dottrina ragiona di corsie e distanze, non di terreno:
                    # ogni tanto punta una cella di fiume. Si sposta sulla sponda.
                    target_pos = self._meta_praticabile(target_pos)

                    lock_until = int(legion.get("target_lock_until", 0) or 0)
                    existing_target_raw = legion.get("target")
                    existing_target: Tuple[int, int] | Tuple[()] = ()
                    if isinstance(existing_target_raw, (list, tuple)) and len(existing_target_raw) == 2:
                        existing_target = (int(existing_target_raw[0]), int(existing_target_raw[1]))

                    if len(existing_target) == 2 and self.game_map.turn <= lock_until:
                        target_pos = existing_target
                    else:
                        legion["target"] = target_pos
                        legion["target_lock_until"] = self.game_map.turn + self._ai_legion_target_lock_turns()

                    next_pos, resolution = self._next_legion_step(
                        current_pos, target_pos, AI, legion_id, lane_col=plan.lane_col,
                        origin=self._march_origin(legion, current_pos, target_pos),
                    )
                    self._log_legion_fallback(legion, "IA", target_pos, next_pos, resolution, logs)
                    if next_pos == current_pos:
                        if resolution["blocked_by"]:
                            # Non c'è nemmeno un ripiego utile: sblocca il target
                            # invece di restare incollata al lock a fissare il muro.
                            legion["target_lock_until"] = 0
                        continue

                    ai_moved = True
                    legion["pos"] = next_pos
                    self._register_legion_move_cost(AI, legion_id, legion, current_pos, next_pos)

                    row, col = next_pos
                    cell = self.game_map.get_cell(row, col)
                    # Stessa regola del player: la fatica si conta a fine turno.
                    legion["marched_terrain"] = cell.terrain if cell is not None else "Pianura"
                    if cell and getattr(cell, "occupation", None) != AI:
                        cell.occupation = AI
                        logs.append(f"🤖 La legione IA '{legion['name']}' conquista la cella ({row},{col}).")
                        self._evento(                         # [EVENT-CHANNEL]
                            ev.CELLA_CONQUISTATA if ev else "cella_conquistata",
                            entita=AI, pos=(row, col),
                            dettaglio={"legione": legion.get("name")},
                        )

                    self._apply_legion_castle_assault(AI, legion, current_pos, next_pos, logs)
                    if tuple(legion.get("pos", ())) != next_pos:
                        continue

                    self._resolve_legion_clash_if_any(next_pos, logs, attacker=AI)
                    if self.state != SessionState.ACTIVE:
                        break

        if self.state == SessionState.ACTIVE and not ai_moved and not ai_skipped:
            logs.append("🤖 L'IA resta in attesa strategica.")

        # 3. Aggiorna economia (miniere e reclute)
        if self.state == SessionState.ACTIVE:
            econ_logs = self._advance_round_economy()
            if econ_logs:
                logs.extend(econ_logs)

        # 4. Stanchezza, morale e gradi: una sola passata a fine turno, così
        #    ogni legione paga la marcia o incassa il riposo esattamente una volta.
        if self.state == SessionState.ACTIVE:
            self._advance_troop_conditions(logs)
            # [DOCTRINE-LAYER] Le perdite del turno sono contate: se una
            # legione è scesa sotto la composizione richiesta va detto.
            self._log_doctrine_gate_changes(logs)
            self._advance_weather(logs)

        self._prune_legion_movement_states()

        # [EVENT-CHANNEL] Dopo reclute e ripartizioni: le legioni hanno
        # adesso la forza definitiva di questo turno.
        self._annuncia_armate()

        self.game_map.turn += 1
        self.battle_log.extend(logs)

        return {
            "ok": True,
            "message": "Turno eseguito.",
            "session": self.to_dict(),
            "logs": logs
        }

    def _empty_army_vector(self) -> Dict[str, float]:
        return {
            "U1_attack": 0.0,
            "U2_defense": 0.0,
            "U3_mobility": 0.0,
            "U4_stealth": 0.0,
            "U5_discipline": 0.0,
            "U6_terrain_adapt": 0.0,
            "U7_range_power": 0.0,
            "U8_support": 0.0,
        }

    def _recompute_entity_army_vector(self, entity: Occupation) -> None:
        """Riallinea il vettore di forza di `entity` alle truppe che ha in riserva.

        `player_army` / `ai_army` descrivono la RISERVA, non tutto ciò che
        l'entità possiede: chi esce — in legione, in carovana, in presidio —
        non deve più pesarci. Il costo dell'esercito qui non si tocca: quello
        è la spesa storica in truppe, e il pannello lo mostra come tale.
        """
        units = self._entity_units(entity)
        if units:
            army_vector = aggregate_army(units, self.data["units"])
            home_terrain = self.player_home_terrain if entity == PLAYER else self.ai_home_terrain
            troop_status = self.player_troop_status if entity == PLAYER else self.ai_troop_status
            modified, _ = apply_modifiers(
                army_vector=army_vector,
                terrain_name=home_terrain,
                weather_name=self.weather,
                troop_status_name=troop_status,
                modifiers_data=self.data,
            )
        else:
            # `aggregate_army` rifiuta la lista vuota: riserva a zero,
            # vettore a zero, e niente modificatori da applicare.
            army_vector = self._empty_army_vector()
            modified = army_vector.copy()

        if entity == PLAYER:
            self.player_army = army_vector
            self.player_modified = modified
        else:
            self.ai_army = army_vector
            self.ai_modified = modified

    def _recompute_entity_army_state(self, entity: Occupation) -> None:
        """Vettore di forza e costo insieme: la strada delle perdite e dei presidi."""
        self._recompute_entity_army_vector(entity)
        army_cost = sum(self.unit_costs.get(unit_id, 0) for unit_id in self._entity_units(entity))
        if entity == PLAYER:
            self.player_army_cost = army_cost
        else:
            self.ai_army_cost = army_cost

    def _calculate_losses_for_battle(
        self,
        units_before: int,
        own_strength: float,
        enemy_strength: float,
        *,
        fortification_level: int = 0,
        garrison_strength: int = 0,
    ) -> int:
        """Regola perdite condivisa: attrito progressivo, meno wipe istantanei."""
        if units_before <= 0:
            return 0

        if enemy_strength <= 0:
            return 0

        if own_strength <= 0:
            return units_before

        weaker_side = own_strength <= enemy_strength

        if weaker_side:
            disadvantage = 1.0 - min(1.0, own_strength / max(1.0, enemy_strength))
            loss_ratio = (
                0.30
                + (0.34 * disadvantage)
                + (0.03 * fortification_level)
                + (0.02 * garrison_strength)
            )
            loss_ratio = min(0.88, max(0.24, loss_ratio))
        else:
            pressure_ratio = min(1.0, enemy_strength / max(1.0, own_strength))
            loss_ratio = (
                0.05
                + (0.24 * pressure_ratio)
                + (0.02 * fortification_level)
                + (0.015 * garrison_strength)
            )
            loss_ratio = min(0.52, max(0.03, loss_ratio))

        losses = int(round(units_before * loss_ratio))

        catastrophic_gap = own_strength < (enemy_strength * 0.32)
        if units_before > 1:
            losses = max(1, losses)
            if catastrophic_gap and weaker_side:
                losses = min(units_before, losses)
            else:
                losses = min(units_before - 1, losses)
        else:
            losses = min(units_before, losses)
        return losses

    def _unit_battle_value(self, unit_id: str) -> float:
        """Valore base di una singola legione/unità in combattimento.

        Include le condizioni ambientali correnti: l'artiglieria sotto la
        pioggia vale meno, gli assassini di notte valgono di più. Passando da
        qui l'effetto arriva ovunque si conti una truppa — legioni in campo,
        presidi, difesa del castello, assalti, IA compresa — senza doverlo
        ripetere in ogni formula.
        """
        attrs = self.units_map[unit_id]["attributes"]
        weighted = sum(attrs[key] * weight for key, weight in UNIT_BATTLE_WEIGHTS.items())
        return weighted * self._weather_unit_factor(unit_id)

    def _weather_unit_factor(self, unit_id: str) -> float:
        """Moltiplicatore meteo della singola unità, con cache per condizione.

        `_unit_battle_value` viene chiamato migliaia di volte per turno e il
        fattore dipende solo da (condizioni, unità): si calcola una volta per
        combinazione e le condizioni cambiano ogni 20+ turni.
        """
        cache = getattr(self, "_weather_unit_cache", None)
        if cache is None:
            cache = self._weather_unit_cache = {}

        key = (self.weather, unit_id)
        if key not in cache:
            cache[key] = wc.unit_weather_factor(
                self.units_map[unit_id]["attributes"],
                self.weather,
                self.data,
                UNIT_BATTLE_WEIGHTS,
            )
        return cache[key]

    def _garrison_unit_defense_value(self, unit_id: str, terrain: str) -> float:
        """Valore difensivo di una unità distaccata, modulato dal terreno corrente."""
        base_value = self._unit_battle_value(unit_id)
        attrs = self.units_map[unit_id]["attributes"]

        terrain_lower = terrain.lower()
        terrain_factor = 1.0 + (attrs["U6_terrain_adapt"] * 0.18)

        if terrain_lower == "foresta":
            terrain_factor += (attrs["U4_stealth"] * 0.08) + (attrs["U3_mobility"] * 0.04)
        elif terrain_lower == "palude":
            terrain_factor += (attrs["U6_terrain_adapt"] * 0.08) + (attrs["U3_mobility"] * 0.05)
        elif terrain_lower == "montagna":
            terrain_factor += (attrs["U2_defense"] * 0.08) + (attrs["U5_discipline"] * 0.04)
        elif terrain_lower == "pianura":
            terrain_factor += (attrs["U1_attack"] * 0.05) + (attrs["U7_range_power"] * 0.03)
        elif terrain_lower == "fiume":
            terrain_factor += (attrs["U3_mobility"] * 0.08) + (attrs["U6_terrain_adapt"] * 0.08)

        return base_value * terrain_factor

    def _army_composition(self, entity: Occupation) -> Dict[str, int]:
        """Composizione esercito per tipo unità (id -> conteggio)."""
        unit_ids = self.player_units if entity == PLAYER else self.ai_units
        return dict(Counter(unit_ids))

    def _format_composition(self, entity: Occupation) -> str:
        """Stringa leggibile della composizione in legioni."""
        composition = self._army_composition(entity)
        if not composition:
            return "nessuna legione"

        parts: List[str] = []
        for unit_id, count in sorted(composition.items(), key=lambda item: (-item[1], item[0])):
            unit_name = self.units_map.get(unit_id, {}).get("name", unit_id)
            parts.append(f"{count} legioni {unit_name}")
        return ", ".join(parts)

    def _format_battle_location(self, terrain: str) -> str:
        """Rende il tipo battaglia con preposizione naturale in base al terreno."""
        prepositions = {
            "foresta": "nella",
            "palude": "nella",
            "montagna": "sulla",
            "pianura": "sulla",
            "fiume": "sul",
        }
        terrain_lower = terrain.lower()
        preposition = prepositions.get(terrain_lower, "su")
        return f"⚔ Battaglia {preposition} {terrain_lower}"

    def _strategy_factor(
        self,
        entity: Occupation,
        terrain: str,
        modified_vector: Dict[str, float],
        strategy_id: Optional[str] = None,
        unit_ids: Optional[Sequence[str]] = None,
    ) -> float:
        """Fattore tattico legato alla qualità della manovra scelta rispetto all'esercito corrente."""
        return self._factor_from_compatibility(
            self._strategy_compatibility(
                entity, modified_vector, strategy_id, terrain=terrain, unit_ids=unit_ids
            )
        )

    def _factor_from_compatibility(self, compatibility: float) -> float:
        """La curva che trasforma la compatibilità nel moltiplicatore di forza.

        Impatto strategico intenzionalmente forte: strategia affine =>
        moltiplicatore molto alto, strategia disallineata => malus severo, con
        soglie critiche che premiano o puniscono le scelte estreme.
        """
        base_factor = 0.38 + ((compatibility ** 2.6) * 1.92)

        critical_bonus = 0.0
        if compatibility >= 0.88:
            critical_bonus += 0.22
        elif compatibility >= 0.78:
            critical_bonus += 0.12

        critical_malus = 0.0
        if compatibility <= 0.22:
            critical_malus += 0.22
        elif compatibility <= 0.32:
            critical_malus += 0.12

        factor = base_factor + critical_bonus - critical_malus
        return max(0.30, min(2.35, factor))

    def _strategy_scores(
        self,
        entity: Occupation,
        modified_vector: Dict[str, float],
        strategy_id: Optional[str] = None,
        *,
        terrain: Optional[str] = None,
        unit_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """I due punteggi di compatibilità [0..1] fra esercito modificato e strategia.

        `combat_compatibility` è quello che entra nel moltiplicatore di forza,
        `advisory_score` quello che l'Advisor mostra al giocatore. Differiscono
        solo per il peso dell'affinità unità-ambiente: a peso 1.0 coincidono.

        [COMPAT-LAYER] Senza il layer, o senza `unit_ids`, sono entrambi la
        distanza euclidea pura di sempre.
        """
        if strategy_id is None:
            strategy_id = self.player_strategy_id if entity == PLAYER else self.ai_strategy_id
        strategy = next((s for s in self.data["strategies"] if s["id"] == strategy_id), None)
        if strategy is None:
            return {"combat_compatibility": 0.5, "advisory_score": 0.5,
                    "strategy_id": strategy_id, "distance": None}

        distance = euclidean_distance(modified_vector, strategy["ideal_attributes"])
        grezza = max(0.0, min(1.0, 1.0 - (distance / (8 ** 0.5))))

        combat, advisory = grezza, grezza
        if cmp_layer is not None and unit_ids:
            combat, advisory = cmp_layer.scores(
                distance=distance,
                unit_ids=unit_ids,
                strategy_id=strategy_id,
                terrain=terrain,
                weather=self.weather,
                affinities=self.data.get("unit_affinities", {}),
            )
        return {
            "combat_compatibility": combat,
            "advisory_score": advisory,
            "strategy_id": strategy_id,
            "distance": distance,
        }

    def _strategy_compatibility(
        self,
        entity: Occupation,
        modified_vector: Dict[str, float],
        strategy_id: Optional[str] = None,
        *,
        terrain: Optional[str] = None,
        unit_ids: Optional[Sequence[str]] = None,
    ) -> float:
        """Compatibilità di combattimento [0..1] tra esercito modificato e strategia.

        `strategy_id` permette di valutare la strategia di una singola legione
        invece di quella globale dello schieramento.
        """
        return self._strategy_scores(
            entity, modified_vector, strategy_id, terrain=terrain, unit_ids=unit_ids
        )["combat_compatibility"]

    def _current_army_terrain(self, entity: Occupation) -> str:
        """Terreno di riferimento della riserva: quello scelto allo schieramento.

        La riserva non sta su una casella — aspetta al castello. Chi va in
        campo è la legione, e il suo terreno lo calcola `_legion_advisor_payload`
        dalla cella su cui si trova.
        """
        return self.player_home_terrain if entity == PLAYER else self.ai_home_terrain

    # ──────────────────────────────────────────────────────────
    # CONDIZIONE TRUPPE (stanchezza / morale / grado)
    # ──────────────────────────────────────────────────────────

    def _initial_troop_status(self, entity: Occupation) -> str:
        """Stato di riferimento dello schieramento, scelto prima della battaglia."""
        chosen = self.player_troop_status if entity == PLAYER else self.ai_troop_status
        return chosen if chosen in tc.ALL_STATUSES else tc.STATUS_FRESH

    def _legion_condition(self, entity: Occupation, legion: Dict[str, Any]) -> Dict[str, Any]:
        return tc.ensure_condition(legion, self._initial_troop_status(entity))

    def _legion_troop_status(self, entity: Occupation, legion: Dict[str, Any]) -> str:
        """Stato truppe di una legione: è il metro usato in combattimento."""
        return tc.resolve_status(self._legion_condition(entity, legion))

    def _distance_from_home(self, entity: Occupation, pos: Tuple[int, int]) -> int:
        """Distanza dal proprio castello, in celle."""
        castle = self.game_map.castle_positions.get(entity)
        if castle is None or len(pos) != 2:
            return 0
        return abs(int(pos[0]) - int(castle[0])) + abs(int(pos[1]) - int(castle[1]))

    def _log_condition_change(
        self,
        entity: Occupation,
        legion: Dict[str, Any],
        before: str,
        logs: List[str],
    ) -> None:
        """Annota il passaggio di stato, che altrimenti resterebbe invisibile."""
        after = tc.resolve_status(self._legion_condition(entity, legion))
        if after == before:
            return
        condition = self._legion_condition(entity, legion)
        icons = {
            tc.STATUS_FRESH: "🌿",
            tc.STATUS_TIRED: "😓",
            tc.STATUS_DEMORALIZED: "💔",
            tc.STATUS_VETERAN: "🎖",
        }
        extra = ""
        if after == tc.STATUS_VETERAN and before != tc.STATUS_VETERAN and condition.get("veteran"):
            extra = f" — {condition.get('victories', 0)} vittorie sul campo"
        logs.append(
            f"[Turno {self.game_map.turn}] {icons.get(after, '•')} {entity.value.upper()}: "
            f"legione '{legion.get('name', '?')}' {before} → {after}{extra}"
        )

    # ──────────────────────────────────────────────────────────
    # METEO E CICLO GIORNO/NOTTE
    # ──────────────────────────────────────────────────────────

    def _refresh_weather_key(self) -> None:
        """Riallinea la chiave passata all'engine ai due assi correnti."""
        self.weather = wc.combined_key(self.day_cycle, self.weather_base)

    def _advance_weather(self, logs: List[str]) -> None:
        """Fa scorrere ciclo e meteo sui loro due orologi indipendenti.

        Cambiano di rado apposta: sotto i 20 turni diventerebbero rumore e
        renderebbero imprevedibile ogni pianificazione a lungo termine.
        """
        self.turns_to_weather_change -= 1
        if self.turns_to_weather_change > 0:
            return

        self.day_cycle, self.weather_base = wc.advance(
            self.day_cycle, self.weather_base, self.weather_rng
        )
        self.turns_to_weather_change = wc.next_change_delay(self.weather_rng)
        self._refresh_weather_key()
        info = self.weather_state()
        effetti = " · ".join(info["effects"])
        logs.append(
            f"[Turno {self.game_map.turn}] {info['emoji']} Condizioni: "
            f"{info['label']} — {effetti}"
        )

        # Chi ci guadagna e chi ci perde davvero, in numeri: senza questa riga
        # il cambio di condizioni restava una scritta senza conseguenze visibili.
        movers = self._weather_unit_effects(limit=4)
        if movers:
            dettaglio = ", ".join(
                f"{row['unit_name']} {row['percent']:+d}%" for row in movers
            )
            logs.append(f"[Turno {self.game_map.turn}] Effetto sulle truppe: {dettaglio}")

    def weather_state(self) -> Dict[str, Any]:
        """Payload dell'indicatore meteo (emoji, colori, effetti, countdown)."""
        state = wc.describe(
            self.day_cycle,
            self.weather_base,
            changes_in=self.turns_to_weather_change,
        )
        # Chi guadagna e chi perde adesso, in chiaro: il giocatore deve poter
        # decidere quale legione muovere senza andare a leggere il JSON.
        state["unit_effects"] = self._weather_unit_effects()
        return state

    def _weather_unit_effects(self, limit: int = 0) -> List[Dict[str, Any]]:
        """Effetto delle condizioni correnti su ogni tipo di unità."""
        return wc.unit_effects(
            self.data.get("units", []),
            self.weather,
            self.data,
            UNIT_BATTLE_WEIGHTS,
            limit=limit,
        )

    def _advance_troop_conditions(self, logs: List[str]) -> None:
        """Chiude il turno per ogni legione: marcia, riposo, cambio di stato.

        Chi si è mosso paga la fatica del terreno attraversato; chi ha
        combattuto ha già pagato nello scontro; chi è rimasto fermo recupera,
        tanto meno quanto è lontano da casa.
        """
        for entity, legions in ((PLAYER, self.player_legions), (AI, self.ai_legions)):
            for legion in list(legions.values()):
                condition = self._legion_condition(entity, legion)
                before = tc.resolve_status(condition)

                terrain = legion.pop("marched_terrain", None)
                traversing = legion.pop("traversing_terrain", None)
                acted = bool(legion.pop("acted_this_turn", False))

                if terrain is not None:
                    tc.apply_march(condition, terrain)
                elif traversing is not None:
                    tc.apply_traversal(condition, traversing)
                elif not acted:
                    tc.apply_rest(
                        condition,
                        self._distance_from_home(entity, tuple(legion.get("pos", ()))),
                    )

                # Il morale risale piano in ogni turno senza combattimenti,
                # anche marciando: si perde per le sconfitte, non per la strada.
                if not acted:
                    tc.apply_morale_drift(condition)

                self._log_condition_change(entity, legion, before, logs)

    def _merge_legion_into_reserve(self, entity: Occupation, legion: Dict[str, Any]) -> None:
        """Le truppe richiamate si mescolano alla riserva, condizione compresa."""
        condition = self._legion_condition(entity, legion)
        reserve = self._entity_units(entity)
        tc.merge_into_pool(
            self.reserve_condition[entity],
            len(reserve),
            condition,
            len(legion.get("units", [])),
        )

    def _legions_with_strategy(self, entity: Occupation) -> Dict[str, Dict[str, Any]]:
        """Legioni per il payload, con strategia risolta e nome leggibile.

        Il frontend deve poter mostrare la strategia di ogni legione senza
        rifare il ripiego sulla strategia globale.
        """
        source = self.player_legions if entity == PLAYER else self.ai_legions
        out: Dict[str, Dict[str, Any]] = {}
        for legion_id, legion in source.items():
            strategy_id = self._legion_strategy_id(entity, legion)
            condition = self._legion_condition(entity, legion)
            out[legion_id] = {
                **legion,
                "strategy_id": strategy_id,
                "strategy_name": self.strategies_map.get(strategy_id, {}).get(
                    "name", strategy_id
                ),
                "troop_status": tc.resolve_status(condition),
                "condition": tc.describe(condition),
                # [DOCTRINE-LAYER] Perché l'effetto non si vede: o manca la
                # composizione, o deve ancora scattare il turno.
                "doctrine": self._doctrine_payload(entity, legion, strategy_id),
            }
        return out

    # [DOCTRINE-LAYER] ─────────────────────────────────────────────
    def _doctrine_payload(
        self, entity: Occupation, legion: Dict[str, Any], strategy_id: str
    ) -> Optional[Dict[str, Any]]:
        """Stato della dottrina di una legione, per l'interfaccia."""
        if doc is None:
            return None
        unita = legion.get("units") or []
        turno = self.game_map.turn
        da = legion.get("doctrine_since_turn")
        return {
            "has_effect": doc.has_effect(strategy_id),
            "effect_text": doc.effect_text(strategy_id),
            "requirement_text": doc.gate_text(strategy_id),
            "gate_ok": doc.gate_passed(strategy_id, unita),
            "active": doc.is_active(strategy_id, unita, turn=turno, since_turn=da),
            "since_turn": da,
            "turns_before_change": doc.turns_before_change(
                turno, legion.get("doctrine_changed_turn")
            ),
        }

    # [EXPERIMENT-LOG] Fotografia di un lato dello scontro, per il log tecnico.
    # Non entra in nessuna formula: legge numeri che il motore ha già calcolato.
    def _dettaglio_scontro(
        self,
        entity: Occupation,
        legion: Dict[str, Any],
        breakdown: Dict[str, Any],
    ) -> Dict[str, Any]:
        strategy_id = breakdown.get("strategy_id") or self._legion_strategy_id(entity, legion)
        dottrina = self._doctrine_payload(entity, legion, strategy_id)
        condizione = self._legion_condition(entity, legion)
        return {
            "strategia": strategy_id,
            "strategia_nome": self.strategies_map.get(strategy_id, {}).get("name", strategy_id),
            "compat_combat": breakdown.get("combat_compatibility"),
            "compat_advisor": breakdown.get("advisory_score"),
            "fattore_strategia": breakdown.get("strategy_factor"),
            "unita": len(legion.get("units") or []),
            "stato_truppe": tc.resolve_status(condizione),
            "morale": condizione.get("morale"),
            "fatica": condizione.get("fatigue"),
            "fortificazione": breakdown.get("fortification_level"),
            "presidio": breakdown.get("garrison_strength"),
            "bonus_difesa": breakdown.get("defense_bonus"),
            "fattore_marcia": breakdown.get("movement_factor"),
            # [DOCTRINE-LAYER] Quale effetto è acceso, e se non lo è cosa manca.
            "dottrina_attiva": bool(dottrina and dottrina.get("active")),
            "dottrina_effetto": (dottrina or {}).get("effect_text"),
            "dottrina_requisito_ok": (dottrina or {}).get("gate_ok"),
            "dottrina_dal_turno": (dottrina or {}).get("since_turn"),
        }

    def _legion_strategy_id(self, entity: Occupation, legion: Dict[str, Any]) -> str:
        """Strategia di una legione, con ripiego su quella generale dello schieramento.

        Le legioni nate prima che le strategie diventassero individuali non
        hanno il campo: per loro vale ancora quella globale.
        """
        default = self.player_strategy_id if entity == PLAYER else self.ai_strategy_id
        strategy_id = legion.get("strategy_id") or default
        return strategy_id if strategy_id in self.strategies_map else default

    # [DOCTRINE-LAYER] Punto unico da cui il motore chiede se una dottrina è
    # accesa: porta e attrito si leggono in un posto solo.
    def _doctrine_state(
        self,
        entity: Occupation,
        legion: Optional[Dict[str, Any]],
        units: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Dottrina attiva di una legione, o None se il layer non c'è.

        `units` serve agli scontri: la porta va guardata sulla composizione con
        cui la legione è ENTRATA in battaglia. Guardandola dopo, una dottrina
        che serve a incassare le perdite si spegne proprio per via delle
        perdite e non scatta mai.
        """
        if doc is None or not legion:
            return None
        return {
            "strategy_id": self._legion_strategy_id(entity, legion),
            "units": legion.get("units") or [] if units is None else units,
            "turn": self.game_map.turn,
            "since_turn": legion.get("doctrine_since_turn"),
        }

    def _mark_doctrine_change(self, legion: Dict[str, Any]) -> None:
        """Registra il cambio di dottrina su due orologi distinti.

        `doctrine_since_turn` fa partire l'effetto dal turno dopo;
        `doctrine_changed_turn` fa scattare l'attesa prima del prossimo cambio.
        Una legione appena formata ha solo il primo: darle una dottrina subito
        dopo averla creata è il gesto normale, non un ripensamento.
        """
        legion["doctrine_since_turn"] = self.game_map.turn
        legion["doctrine_changed_turn"] = self.game_map.turn

    def set_legion_strategy(self, legion_id: str, strategy_id: str) -> Dict[str, Any]:
        """Assegna una strategia a una singola legione player."""
        if self.state != SessionState.ACTIVE:
            raise ValueError("La partita è terminata.")

        legion = self._get_player_legion_or_raise(legion_id)
        strategy = self.strategies_map.get(strategy_id)
        if strategy is None:
            raise ValueError("Strategia non valida.")

        # [DOCTRINE-LAYER] Le strategie si cambiano in partita, ma non a ogni
        # turno: senza attesa una legione sola le userebbe tutte a rotazione
        # e ogni effetto sarebbe sempre acceso.
        if doc is not None:
            ultimo = legion.get("doctrine_changed_turn")
            cambia = strategy_id != legion.get("strategy_id")
            if cambia and not doc.can_change(self.game_map.turn, ultimo):
                mancano = doc.turns_before_change(self.game_map.turn, ultimo)
                raise ValueError(
                    f"La legione '{legion['name']}' ha cambiato dottrina da poco: "
                    f"può cambiare di nuovo fra {mancano} turn{'o' if mancano == 1 else 'i'}."
                )
            # Anche riconfermare la dottrina ereditata fa partire l'orologio,
            # se no resta un cambio gratis in tasca da spendere quando serve.
            if cambia or ultimo is None:
                self._mark_doctrine_change(legion)

        legion["strategy_id"] = strategy_id

        pos = tuple(legion.get("pos", ()))
        cell = self.game_map.get_cell(*pos) if len(pos) == 2 else None
        terrain = cell.terrain if cell is not None else self.player_home_terrain
        breakdown = self._legion_battle_strength(
            PLAYER, legion, terrain,
            defending=False, cell=cell, enemy_strength=0.0, movement_key=None,
        )

        # [DOCTRINE-LAYER] La porta è invisibile: senza questa coda il log dice
        # "adotta Manovra sui Fianchi" anche a chi non ha un cavallo.
        nota_dottrina = ""
        if doc is not None:
            unita = legion.get("units") or []
            nota_dottrina = " — " + doc.status_text(strategy_id, unita)
            if doc.gate_passed(strategy_id, unita) and doc.has_effect(strategy_id):
                nota_dottrina += f" (dal turno {self.game_map.turn + doc.ACTIVATION_DELAY_TURNS})"

        log_entry = (
            f"[Turno {self.game_map.turn}] 🎯 PLAYER: legione '{legion['name']}' adotta "
            f"{strategy['name']} → forza {int(round(breakdown['strength']))} su {terrain}"
            f"{nota_dottrina}"
        )
        self.battle_log.append(log_entry)
        return {
            "ok": True,
            "message": log_entry,
            "legion_id": legion_id,
            "strategy_id": strategy_id,
            "strategy_name": strategy["name"],
            "strength": int(round(breakdown["strength"])),
            "strategy_factor": round(breakdown["strategy_factor"], 3),
            "state": self.state.value,
            "session": self.to_dict(),
        }

    def set_player_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Aggiorna la strategia corrente del player e logga snapshot forza attuale."""
        if self.state != SessionState.ACTIVE:
            raise ValueError("La partita è terminata.")

        strategy = self.strategies_map.get(strategy_id)
        if strategy is None:
            raise ValueError("Strategia non valida.")

        self.player_strategy_id = strategy_id
        terrain = self._current_army_terrain(PLAYER)
        breakdown = self._strength_breakdown(PLAYER, terrain)
        strength_now = breakdown["effective_strength"]
        reserve_units = len(self.player_units)

        # Questa è la strategia GENERALE, che vale per la riserva e per le
        # legioni che nasceranno: non per quelle già in campo, che hanno la
        # propria. Con tutte le truppe in legione la riserva è vuota e la
        # vecchia riga diceva "forza attuale 0", che sembrava un guasto.
        if reserve_units:
            dettaglio = f"forza riserva {strength_now} su {terrain} ({reserve_units} unità)"
        else:
            dettaglio = "riserva vuota: varrà per le prossime legioni"

        log_entry = (
            f"[Turno {self.game_map.turn}] 🎯 PLAYER imposta strategia generale "
            f"{strategy['name']} → {dettaglio}"
        )
        self.battle_log.append(log_entry)
        log_strength_debug(
            "strategy_change_strength_snapshot",
            {
                "debug_notice": "DEBUG TEMPORANEO - rimuovere cartella debug in produzione",
                "turn": self.game_map.turn,
                "entity": PLAYER.value,
                "strategy_id": strategy_id,
                "strategy_name": strategy["name"],
                "terrain": terrain,
                "strength_breakdown": breakdown,
            },
        )

        return {
            "ok": True,
            "message": log_entry,
            "state": self.state.value,
            "map": self.game_map.to_dict(),
            "session": self.to_dict(),
        }

    def get_in_game_advisor(self) -> Dict[str, Any]:
        """Report advisor in-battle: uno per la riserva, uno per ogni legione in campo.

        Ogni legione ha la propria composizione, il proprio terreno e la propria
        strategia, quindi il consiglio va calcolato su di lei: un solo report
        globale descriveva un esercito che sul campo non esiste più.
        """
        if self.state != SessionState.ACTIVE:
            raise ValueError("La partita è terminata.")

        terrain_name = self._current_army_terrain(PLAYER)
        reserve_status = tc.resolve_status(self.reserve_condition[PLAYER])
        if self.player_units:
            payload = build_in_game_advisor_payload(
                data=self.data,
                turn=self.game_map.turn,
                player_units=list(self.player_units),
                player_army=dict(self.player_army),
                player_strategy_id=self.player_strategy_id,
                troop_status_name=reserve_status,
                terrain_name=terrain_name,
                weather_name=self.weather,
                # [ABILITY-EFFECTS] Industria dello Spionaggio: via il rumore.
                perfect=self._ability_flag(PLAYER, ab.FLAG_SPY_NETWORK),
            )
        else:
            # Riserva vuota: tutte le truppe sono in campo. Su un vettore a
            # zero la classifica delle strategie non vuol dire niente, quindi
            # il report lo dice e si ferma — come già fa per le legioni
            # rimaste senza truppe.
            payload = {
                "turn": self.game_map.turn,
                "terrain_name": terrain_name,
                "weather_name": self.weather,
                "troop_status_name": reserve_status,
                "empty": True,
                "ranking": [],
            }
        payload["troop_condition"] = tc.describe(self.reserve_condition[PLAYER])
        payload["scope"] = "reserve"
        payload["legion_id"] = None
        payload["legion_name"] = "Riserva nel castello"
        payload["current_strategy_id"] = self.player_strategy_id
        payload["current_strategy_name"] = self.strategies_map.get(
            self.player_strategy_id, {}
        ).get("name", self.player_strategy_id)
        payload["units_count"] = len(self.player_units)

        payload["legions"] = [
            self._legion_advisor_payload(legion_id, legion)
            for legion_id, legion in self.player_legions.items()
        ]
        return payload

    def get_enemy_intel(self) -> Dict[str, Any]:
        """[ABILITY-EFFECTS] Dossier sull'IA, aperto dall'Industria dello Spionaggio.

        Non ripete quello che il pannello mostra già a tutti — truppe, grux,
        legioni sono in chiaro nel payload da sempre. Qui c'è solo ciò che il
        gioco non dice mai: quanto la strategia che l'IA sta usando le si addice
        davvero, dove è scoperta, e cosa ha in cantiere.
        """
        if self.state != SessionState.ACTIVE:
            raise ValueError("La partita è terminata.")

        if not self._ability_flag(PLAYER, ab.FLAG_SPY_NETWORK):
            return {
                "available": False,
                "required_ability": ab.ABILITIES[ab.SPY_INDUSTRY_ID].name,
            }

        terrain_name = self._current_army_terrain(AI)
        troop_status = tc.resolve_status(self.reserve_condition[AI])
        modified, warnings = apply_modifiers(
            army_vector=self.ai_army,
            terrain_name=terrain_name,
            weather_name=self.weather,
            troop_status_name=troop_status,
            modifiers_data=self.data,
        )
        # Classifica VERA: nessun rumore. È il senso dell'abilità.
        ranking = compute_ranking(
            army_vector=modified,
            strategies_list=self.data["strategies"],
            unit_ids=list(self.ai_units),
            terrain_name=terrain_name,
            weather_name=self.weather,
            affinities_data=self.data.get("unit_affinities", {}),
        )
        for indice, voce in enumerate(ranking):
            voce["rank"] = indice + 1

        in_uso = next(
            (voce for voce in ranking if voce["id"] == self.ai_strategy_id),
            None,
        )

        return {
            "available": True,
            "turn": self.game_map.turn,
            "difficulty": self.ai_difficulty,
            "strategy": {
                "id": self.ai_strategy_id,
                "name": self.ai_strategy_name,
                "compatibility": round(float(in_uso["compatibility"]), 1) if in_uso else None,
                "rank": in_uso["rank"] if in_uso else None,
                "out_of": len(ranking),
                "description": (in_uso or {}).get("description", ""),
            },
            "best": ranking[0] if ranking else None,
            "worst": ranking[-1] if ranking else None,
            "ranking": ranking,
            # Gli attributi segnati CRITICAL che l'esercito IA non soddisfa: è
            # letteralmente dove conviene colpirlo.
            "weaknesses": warnings,
            "army": {
                "units_count": len(self.ai_units),
                "composition": self._etichetta_unita(self.ai_units),
                "strength": int(round(self._effective_army_strength(AI, terrain_name))),
                "terrain_name": terrain_name,
                "troop_status_name": troop_status,
                "in_marcia": self.ai_convoys.unita_in_marcia(),
            },
            "research": self._dossier_ricerche(),
        }

    def _dossier_ricerche(self) -> Dict[str, Any]:
        """[ABILITY-EFFECTS] Cosa l'IA ha ricercato, cosa sta ricercando, cosa vuole."""
        stati = self.ability_states[AI]
        turno = self.game_map.turn

        in_corso_id = ab.researching_id(stati, turno)
        in_corso = None
        if in_corso_id and in_corso_id in stati:
            stato = stati[in_corso_id]
            in_corso = {
                "name": stato.name,
                "turns_remaining": stato.turns_remaining(turno),
            }

        prossima_id = ab.ai_next_research(
            self.ai_difficulty, stati, turno, self.grux_balance[AI]
        )

        return {
            "unlocked": [
                stati[aid].name for aid in ab.unlocked_ids(stati, turno) if aid in stati
            ],
            "in_progress": in_corso,
            "next_planned": (
                stati[prossima_id].name
                if prossima_id and prossima_id in stati else None
            ),
        }

    def _legion_advisor_payload(self, legion_id: str, legion: Dict[str, Any]) -> Dict[str, Any]:
        """Report advisor calcolato sulla singola legione."""
        unit_ids = list(legion.get("units", []))
        pos = tuple(legion.get("pos", ()))
        cell = self.game_map.get_cell(*pos) if len(pos) == 2 else None
        terrain_name = cell.terrain if cell is not None else self.player_home_terrain
        strategy_id = self._legion_strategy_id(PLAYER, legion)
        condition = self._legion_condition(PLAYER, legion)
        troop_status = tc.resolve_status(condition)

        if not unit_ids:
            return {
                "scope": "legion",
                "legion_id": legion_id,
                "legion_name": legion.get("name", legion_id),
                "legion_type": legion.get("legion_type", LEGION_TYPE_ARMY),
                "legion_type_label": LEGION_TYPE_LABELS.get(
                    legion.get("legion_type", LEGION_TYPE_ARMY), ""
                ),
                "pos": list(pos) if len(pos) == 2 else None,
                "terrain_name": terrain_name,
                "units_count": 0,
                "current_strategy_id": strategy_id,
                "current_strategy_name": self.strategies_map.get(strategy_id, {}).get(
                    "name", strategy_id
                ),
                "troop_status_name": troop_status,
                "troop_condition": tc.describe(condition),
                "empty": True,
                "ranking": [],
            }

        payload = build_in_game_advisor_payload(
            data=self.data,
            turn=self.game_map.turn,
            player_units=unit_ids,
            player_army=aggregate_army(unit_ids, self.data["units"]),
            player_strategy_id=strategy_id,
            troop_status_name=troop_status,
            terrain_name=terrain_name,
            weather_name=self.weather,
            perfect=self._ability_flag(PLAYER, ab.FLAG_SPY_NETWORK),  # [ABILITY-EFFECTS]
        )
        breakdown = self._legion_battle_strength(
            PLAYER, legion, terrain_name,
            defending=False, cell=cell, enemy_strength=0.0, movement_key=None,
        )
        payload.update({
            "scope": "legion",
            "legion_id": legion_id,
            "legion_name": legion.get("name", legion_id),
            "legion_type": legion.get("legion_type", LEGION_TYPE_ARMY),
            "legion_type_label": LEGION_TYPE_LABELS.get(
                legion.get("legion_type", LEGION_TYPE_ARMY), ""
            ),
            "pos": list(pos) if len(pos) == 2 else None,
            "units_count": len(unit_ids),
            # [DOCTRINE-LAYER] Anche il pannello consigli deve poter dire
            # se l'effetto della dottrina è acceso o cosa gli manca.
            "doctrine": self._doctrine_payload(PLAYER, legion, strategy_id),
            "current_strategy_id": strategy_id,
            "current_strategy_name": self.strategies_map.get(strategy_id, {}).get(
                "name", strategy_id
            ),
            "current_strength": int(round(breakdown["strength"])),
            "current_strategy_factor": round(breakdown["strategy_factor"], 3),
            # [COMPAT-LAYER] I due punteggi della strategia in uso, affiancati:
            # il log sperimentale li legge da qui senza rifare il conto.
            "current_combat_compatibility": round(breakdown["combat_compatibility"], 4),
            "current_advisory_score": round(breakdown["advisory_score"], 4),
            "troop_status_name": troop_status,
            "troop_condition": tc.describe(condition),
            "empty": False,
        })
        return payload

    def _available_mine_slots(self, entity: Occupation) -> int:
        """Slot miniera disponibili in base alle celle controllate."""
        return available_mine_slots(
            controlled_cells=self.game_map.count_occupied(entity),
            existing_mines=self.game_map.count_mines(entity),
        )

    # [EVENT-CHANNEL] Punto unico di emissione: gli agganci sparsi nel motore
    # sono tutti una riga sola che passa di qui. Non ritorna niente e non può
    # sollevare, così un evento malformato non porta giù un turno.
    def _evento(
        self,
        tipo: str,
        *,
        entita: Optional[Occupation] = None,
        pos: Optional[Tuple[int, int]] = None,
        quantita: Optional[float] = None,
        dettaglio: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.event_log is None:
            return
        self.event_log.emetti(
            tipo,
            turno=self.game_map.turn,
            entita=entita.value if entita is not None else None,
            pos=pos,
            quantita=quantita,
            dettaglio=dettaglio,
        )

    def _ricalcola_esercito_ia(self) -> None:
        """[CARAVAN] Riallinea i vettori di forza IA a `ai_units`.

        Serve quando le truppe escono dall'esercito per mettersi in viaggio:
        finché marciano non devono pesare nel calcolo della forza.
        """
        self._recompute_entity_army_vector(AI)

    def _fronte_ia(self) -> Optional[Tuple[Tuple[int, int], str]]:
        """[CARAVAN] Dove sta il grosso dell'esercito IA: (posizione, nome).

        La legione più numerosa, non la più vicina: è lì che i rinforzi
        servono, ed è lì che il giocatore deve aspettarseli.
        """
        if not self.ai_legions:
            return None
        legione = max(
            self.ai_legions.values(),
            key=lambda lg: len(lg.get("units") or []),
        )
        pos = legione.get("pos")
        if not pos or len(pos) < 2:
            return None
        return (int(pos[0]), int(pos[1])), str(legione.get("name") or "")

    def _instrada_rinforzi(self, unita: List[str]) -> None:
        """[CARAVAN] Porta al raduno le truppe appena entrate nell'esercito IA.

        `unita` sono già dentro `ai_units` — ce le hanno messe il reclutamento o
        il mercato nero. Qui escono: aspetteranno al castello che si formi una
        carovana, e fino ad allora non contano in battaglia. Prima di questo
        modulo comparivano invece dentro la legione all'istante, ovunque fosse
        (misurato: 98% dei casi, mediana 8 caselle, massimo 18).

        Senza nessuna legione in campo non c'è un fronte da raggiungere: le
        truppe restano nell'esercito, e saranno loro a formare la prossima
        legione al castello.
        """
        if not unita or not self.ai_legions:
            self._sync_ai_legion_units()
            return

        for unit_id in unita:
            if unit_id in self.ai_units:
                self.ai_units.remove(unit_id)
        self._ricalcola_esercito_ia()
        self._sync_ai_legion_units()
        self.ai_convoys.raduna(unita, turno=self.game_map.turn)

    def _muovi_carovane(self, logs: List[str]) -> None:
        """[CARAVAN] Consegna le carovane arrivate e fa partire il raduno pronto."""
        self._consegna_carovane(logs)
        self._parti_carovana(logs)

    def _consegna_carovane(self, logs: List[str]) -> None:
        """[CARAVAN] Le carovane giunte a destinazione versano le truppe."""
        giunte = self.ai_convoys.arrivate(self.game_map.turn)
        if not giunte:
            return

        for carovana in giunte:
            self.ai_units.extend(carovana.unita)
        self._ricalcola_esercito_ia()
        self._sync_ai_legion_units()

        for carovana in giunte:
            fronte = self._fronte_ia()
            quante = len(carovana.unita)
            etichetta = self._etichetta_unita(carovana.unita)
            if fronte is None:
                # La legione a cui erano dirette non c'è più: le truppe
                # rientrano nell'esercito e formeranno la prossima.
                logs.append(
                    f"[Turno {self.game_map.turn}] 🚚 IA: la carovana di {quante} "
                    f"({etichetta}) trova il fronte sciolto e rientra al castello"
                )
                nome, forza = carovana.legione, 0
            else:
                nome = fronte[1]
                forza = max(
                    (len(lg.get("units") or []) for lg in self.ai_legions.values()),
                    default=0,
                )
                logs.append(
                    f"[Turno {self.game_map.turn}] 🚚 IA: rinforzi arrivati al fronte — "
                    f"{quante} ({etichetta}) raggiungono '{nome}', "
                    f"che sale a {forza} unità"
                )
            pos = fronte[0] if fronte else carovana.destinazione
            self._evento(                                 # [EVENT-CHANNEL]
                ev.CAROVANA_ARRIVATA if ev else "carovana_arrivata",
                entita=AI, pos=pos,
                quantita=len(carovana.unita),
                dettaglio={
                    "legione": nome,
                    "forza_legione": forza,
                    "unita": etichetta,
                },
            )

    def _parti_carovana(self, logs: List[str]) -> None:
        """[CARAVAN] Il raduno al castello si mette in marcia, se è ora."""
        fronte = self._fronte_ia()
        castello = self.game_map.castle_positions.get(AI)

        # Nessun fronte da raggiungere (legioni annientate): le truppe radunate
        # rientrano nell'esercito invece di restare in un limbo al castello —
        # da lì `_ensure_ai_legions_initialized` ci formerà la legione nuova.
        if fronte is None or castello is None:
            rientrate = self.ai_convoys.svuota_raduno()
            if rientrate:
                self.ai_units.extend(rientrate)
                self._ricalcola_esercito_ia()
                self._sync_ai_legion_units()
            return

        if not self.ai_convoys.raduno_pronto(self.game_map.turn):
            return

        destinazione, nome_legione = fronte
        carovana = self.ai_convoys.parti(
            turno=self.game_map.turn,
            origine=castello,
            destinazione=destinazione,
            legione=nome_legione,
        )
        if carovana is None:
            # Fronte a distanza zero: la legione è a casa, niente da percorrere.
            rientrate = self.ai_convoys.svuota_raduno()
            if rientrate:
                self.ai_units.extend(rientrate)
                self._ricalcola_esercito_ia()
                self._sync_ai_legion_units()
            return

        etichetta = self._etichetta_unita(carovana.unita)
        logs.append(
            f"[Turno {self.game_map.turn}] 🚚 IA: carovana di {len(carovana.unita)} "
            f"({etichetta}) parte dal castello verso '{nome_legione}' — "
            f"{carovana.distanza} caselle, arrivo previsto al turno "
            f"{carovana.arrivo_turno}"
        )
        self._evento(                                     # [EVENT-CHANNEL]
            ev.CAROVANA_PARTITA if ev else "carovana_partita",
            entita=AI, pos=destinazione,
            quantita=len(carovana.unita),
            dettaglio={
                "legione": nome_legione,
                "arrivo_turno": carovana.arrivo_turno,
                "turni_mancanti": carovana.turni_mancanti(self.game_map.turn),
                "distanza": carovana.distanza,
                "unita": etichetta,
            },
        )

    def _etichetta_unita(self, unita: Sequence[str]) -> str:
        """[CARAVAN] '2 Fanteria Pesante, 1 Arcieri' da una lista di id."""
        conteggio = Counter(unita)
        pezzi = []
        for unit_id, quante in conteggio.most_common():
            nome = self.units_map.get(unit_id, {}).get("name", unit_id)
            pezzi.append(f"{quante}× {nome}" if quante > 1 else nome)
        return ", ".join(pezzi)

    def _annuncia_armate(self) -> None:
        """[EVENT-CHANNEL] Segnala le legioni che sono diventate armate.

        `LEGIONE_CREATA` racconta la nascita, e per l'IA la nascita è
        sempre un pugno di uomini: le sue legioni vengono formate vuote e
        le riempie `_sync_ai_legion_units` mano a mano che arrivano le
        reclute. Misurato su 12 partite: nascono con 1-6 unità e arrivano
        a 13-147. Chi ascolta gli eventi per marcare l'arrivo di una forza
        grossa ha bisogno di questo momento, non di quello della nascita.

        Ogni legione lo annuncia una volta sola: l'id resta segnato, così
        una che ondeggia attorno alla soglia non lo ripete a ogni turno.
        """
        if self.event_log is None:
            return
        soglia = ev.SOGLIA_ARMATA if ev is not None else 10
        for entita, legioni in ((PLAYER, self.player_legions), (AI, self.ai_legions)):
            for legion_id, legione in legioni.items():
                if legion_id in self._armate_annunciate:
                    continue
                forza = len(legione.get("units") or [])
                if forza < soglia:
                    continue
                self._armate_annunciate.add(legion_id)
                self._evento(
                    ev.ARMATA_SCHIERATA if ev else "armata_schierata",
                    entita=entita, pos=legione.get("pos"),
                    quantita=forza,
                    dettaglio={"nome": legione.get("name")},
                )

    @staticmethod
    def _ha_artiglieria(*gruppi: Optional[Sequence[str]]) -> bool:
        """[EVENT-CHANNEL] C'è artiglieria fra queste truppe?

        Lo decide il motore, che sa cosa contengono le legioni: chi ascolta gli
        eventi non deve andarselo a ricostruire.
        """
        unita = ev.UNITA_ARTIGLIERIA if ev is not None else "artillery"
        for gruppo in gruppi:
            if gruppo and unita in gruppo:
                return True
        return False

    def _ability_state(self, entity: Occupation, ability_id: str) -> Optional[Any]:
        return self.ability_states.get(entity, {}).get(ability_id)

    def _is_ability_unlocked(self, entity: Occupation, ability_id: str) -> bool:
        ability = self._ability_state(entity, ability_id)
        if ability is None:
            return False
        return ability.is_unlocked(self.game_map.turn)

    def _start_ability_research(self, entity: Occupation, ability_id: str) -> Optional[str]:
        """Avvia una ricerca pagandola. Ritorna la riga di log, o None se non si può.

        Il controllo di disponibilità (una ricerca alla volta, prerequisiti,
        turno minimo, esclusività) sta in `abilities.availability`: qui si
        aggiunge solo il conto in grux, che è l'unica cosa che il catalogo non
        può sapere da solo.
        """
        states = self.ability_states.get(entity, {})
        ability = states.get(ability_id)
        if ability is None:
            return None

        can_start, _ = ab.availability(ability_id, states, self.game_map.turn)
        if not can_start:
            return None

        cost = int(getattr(ability, "grux_cost", 0))
        if self.grux_balance[entity] < cost:
            return None

        self.grux_balance[entity] -= cost
        ability.start(self.game_map.turn)
        side = entity.value.upper()
        price_note = f", {cost} grux" if cost > 0 else ""
        log_entry = (
            f"[Turno {self.game_map.turn}] ⭐ {side} avvia ricerca Abilità: {ability.name} "
            f"({ability.turns_required} turni{price_note})"
        )
        self.battle_log.append(log_entry)
        return log_entry

    def _ai_research_savings(self) -> int:
        """Grux che l'IA tiene da parte per la prossima ricerca in programma.

        Zero mentre sta già ricercando, e zero finché l'esercito non arriva a
        due reparti pieni: prima vengono le truppe. Risparmiando dal primo turno
        l'incubo restava a una legione per mezza partita e arrivava al castello
        60 turni più tardi, pur finendo col triplo delle truppe. La soglia è un
        compromesso misurato: a 4 unità sblocca 5.2 abilità in 155 turni, a 18
        ne sblocca 3 in 122, a 12 ne sblocca 4 in 125.
        """
        states = self.ability_states.get(AI, {})
        turn = self.game_map.turn
        if ab.researching_id(states, turn) is not None:
            return 0

        if len(self.ai_units) < AI_RESEARCH_MIN_ARMY:
            return 0

        plan = ab.AI_RESEARCH_PLANS.get(self.ai_difficulty.lower()) or ab.AI_RESEARCH_PLANS["normal"]
        for ability_id in plan:
            state = states.get(ability_id)
            if state is None:
                continue
            can_start, _ = ab.availability(ability_id, states, turn)
            if can_start:
                return int(state.grux_cost) + ab.AI_RESEARCH_RESERVE
        return 0

    def _run_ai_research(self) -> Optional[str]:
        """L'IA sceglie e avvia la prossima ricerca del suo piano.

        Stesse regole del giocatore: una alla volta, pagata, con prerequisiti
        e turno minimo. Cambia solo l'ordine delle priorità, che dipende dalla
        difficoltà ed è scritto in `abilities.AI_RESEARCH_PLANS`.
        """
        states = self.ability_states.get(AI, {})
        ability_id = ab.ai_next_research(
            self.ai_difficulty, states, self.game_map.turn, self.grux_balance[AI]
        )
        if ability_id is None:
            return None
        return self._start_ability_research(AI, ability_id)

    # ── Effetti delle abilità ──────────────────────────────────────
    # Punto unico di lettura: il resto del motore chiede "quanto vale" o
    # "posso farlo", e non sa quali abilità esistano.

    def _unlocked_abilities(self, entity: Occupation) -> Tuple[str, ...]:
        """Abilità attive di un'entità in questo momento.

        Volutamente senza cache: il risultato può cambiare a metà turno (il
        pannello di debug sblocca tutto retrodatando la ricerca) e una lista
        di undici id si ricalcola in niente. Chi conta unità in serie chiede
        una volta questa tupla e poi la riusa.
        """
        return ab.unlocked_ids(self.ability_states.get(entity, {}), self.game_map.turn)

    def _ability_flag(self, entity: Occupation, flag: str) -> bool:
        return ab.has_flag(self._unlocked_abilities(entity), flag)

    def _ability_economy_factor(self, entity: Occupation, key: str) -> float:
        return ab.economy_factor(self._unlocked_abilities(entity), key)

    def _ability_value_of(self, entity: Occupation, context: str):
        """Funzione `unit_id -> valore` con dentro i bonus di stile del contesto.

        Serve dove il valore delle unità viene sommato da codice che non deve
        sapere niente di abilità (il layer di bilanciamento, per esempio).
        Quando non c'è nessun bonus in gioco restituisce la funzione originale,
        così il caso normale non paga niente.
        """
        unlocked = self._unlocked_abilities(entity)
        if not ab.has_unit_effects(unlocked, context):
            return self._unit_battle_value
        return lambda unit_id: self._unit_battle_value(unit_id) * ab.unit_factor(
            unlocked, unit_id, context
        )

    def research_player_ability(self, ability_id: str = DOMAIN_ENGINEERING_ID) -> Dict[str, Any]:
        """Avvia la ricerca di una abilità lato player."""
        if self.state != SessionState.ACTIVE:
            raise ValueError("La partita è terminata.")

        states = self.ability_states.get(PLAYER, {})
        ability = states.get(ability_id)
        if ability is None:
            raise ValueError("Abilità sconosciuta.")

        can_start, reason = ab.availability(ability_id, states, self.game_map.turn)
        if not can_start:
            raise ValueError(reason)

        cost = int(getattr(ability, "grux_cost", 0))
        if self.grux_balance[PLAYER] < cost:
            raise ValueError(
                f"Grux insufficienti per la ricerca: servono {cost}, "
                f"disponibili {self.grux_balance[PLAYER]}"
            )

        log_entry = self._start_ability_research(PLAYER, ability_id)
        return {
            "ok": True,
            "message": log_entry,
            "state": self.state.value,
            "map": self.game_map.to_dict(),
            "session": self.to_dict(),
        }

    # ── Mercato Nero ───────────────────────────────────────────────

    def _black_market_open(self, entity: Occupation) -> bool:
        return self._ability_flag(entity, ab.FLAG_BLACK_MARKET)

    def _tick_black_market(self) -> List[str]:
        """Aggiorna i banchi di chi ha l'abilità. Chiusi non fanno nulla."""
        logs: List[str] = []
        for entity in (PLAYER, AI):
            if entity == AI and self.debug_ai_kill_switch:
                continue
            if not self._black_market_open(entity):
                continue
            # Il listino di riferimento è quello che questa entità pagherebbe
            # davvero, non il prezzo base: con Industria Bellica sbloccata uno
            # sconto del 20% sul listino pieno poteva essere un affare PEGGIORE
            # del reclutamento normale, e il banco l'avrebbe comunque scritto
            # come risparmio.
            reference_prices = {
                unit_id: self._recruit_cost(entity, unit_id) for unit_id in self.unit_costs
            }
            line = self.black_market[entity].tick(
                self.game_map.turn, reference_prices, self.units_map, self.black_market_rng
            )
            # Il cambio banco dell'IA non è affar suo: resta fuori dal registro.
            if line and entity == PLAYER:
                logs.append(line)
        return logs

    def _absorb_market_block(self, entity: Occupation, unit_id: str, quantity: int) -> None:
        """Fa entrare in riserva un blocco comprato al banco.

        Stessa strada di una recluta normale — riserva, vettore esercito,
        diluizione della stanchezza — solo moltiplicata per la quantità e
        senza toccare `last_recruit_turn`: il mercato nero non conosce
        cooldown, ed è il motivo per cui esiste.
        """
        tc.dilute_pool(self.reserve_condition[entity], len(self._entity_units(entity)))

        if entity == PLAYER:
            self.player_units.extend([unit_id] * quantity)
            self.player_army = aggregate_army(self.player_units, self.data["units"])
            self.player_modified, _ = apply_modifiers(
                army_vector=self.player_army,
                terrain_name=self.player_home_terrain,
                weather_name=self.weather,
                troop_status_name=self.player_troop_status,
                modifiers_data=self.data,
            )
        else:
            self.ai_units.extend([unit_id] * quantity)
            self.ai_army = aggregate_army(self.ai_units, self.data["units"])
            self.ai_modified, _ = apply_modifiers(
                army_vector=self.ai_army,
                terrain_name=self.ai_home_terrain,
                weather_name=self.weather,
                troop_status_name=self.ai_troop_status,
                modifiers_data=self.data,
            )

    def buy_black_market_offer(self, offer_id: str) -> Dict[str, Any]:
        """Compra un blocco al banco per il giocatore."""
        if self.state != SessionState.ACTIVE:
            raise ValueError("La partita è terminata.")
        if not self._black_market_open(PLAYER):
            raise ValueError("Il Mercato Nero non è ancora sbloccato.")

        market = self.black_market[PLAYER]
        offer = market.get_offer(offer_id)
        if offer is None:
            raise ValueError("Offerta non più al banco.")
        if self.grux_balance[PLAYER] < offer.total_price:
            raise ValueError(
                f"Grux insufficienti: servono {offer.total_price}, "
                f"disponibili {self.grux_balance[PLAYER]}"
            )

        offer = market.take(offer_id, self.game_map.turn)
        self.grux_balance[PLAYER] -= offer.total_price
        self.player_army_cost += offer.total_price
        self._absorb_market_block(PLAYER, offer.unit_id, offer.quantity)

        saving = max(0, offer.list_total - offer.total_price)
        log_entry = (
            f"[Turno {self.game_map.turn}] 🕯 PLAYER compra al Mercato Nero: "
            f"{offer.unit_name} ×{offer.quantity} per {offer.total_price} grux "
            f"(-{offer.discount_pct}%, {saving} risparmiati)"
        )
        self.battle_log.append(log_entry)
        return {
            "ok": True,
            "message": log_entry,
            "state": self.state.value,
            "map": self.game_map.to_dict(),
            "session": self.to_dict(),
        }

    def _run_ai_black_market(self) -> List[str]:
        """L'IA passa al banco: prende l'affare che le fa risparmiare di più.

        Tiene da parte la stessa riserva che tiene per le ricerche, se no
        svuoterebbe la cassa in offerte e smetterebbe di fortificare.
        """
        if not self._black_market_open(AI):
            return []

        market = self.black_market[AI]
        # Non tocca i grux già promessi alla ricerca: senza questo l'IA metteva
        # da parte per l'abilità e poi si comprava le truppe con quei soldi,
        # rimandando la ricerca all'infinito.
        held_back = max(ab.AI_RESEARCH_RESERVE, self._ai_research_savings())
        budget = self.grux_balance[AI] - held_back
        if budget <= 0:
            return []

        offer = market.best_affordable(self.game_map.turn, budget)
        if offer is None:
            return []

        offer = market.take(offer.offer_id, self.game_map.turn)
        self.grux_balance[AI] -= offer.total_price
        self.ai_army_cost += offer.total_price
        self._absorb_market_block(AI, offer.unit_id, offer.quantity)
        # [CARAVAN] Anche il blocco comprato al banco marcia: erano 2-5 unità
        # che comparivano al fronte tutte insieme, nel turno dell'acquisto.
        self._instrada_rinforzi([offer.unit_id] * offer.quantity)

        return [
            f"[Turno {self.game_map.turn}] 🕯 IA compra al Mercato Nero: "
            f"{offer.unit_name} ×{offer.quantity} ({offer.total_price} grux)"
        ]

    def _get_player_legion_or_raise(self, legion_id: str) -> Dict[str, Any]:
        legion = self.player_legions.get(legion_id)
        if legion is None:
            raise ValueError(f"Legione non trovata: {legion_id}")
        return legion

    def _require_legion_type(
        self,
        legion: Dict[str, Any],
        required_type: str,
        action_label: str,
        entity: Occupation = PLAYER,
    ) -> None:
        """La legione deve essere del tipo richiesto per eseguire questa costruzione."""
        legion_type = legion.get("legion_type", LEGION_TYPE_ARMY)
        if legion_type == required_type:
            return

        # [ABILITY-EFFECTS] Costruzione Caotica: i ruoli non contano più,
        # qualsiasi legione può costruire qualsiasi cosa.
        if self._ability_flag(entity, ab.FLAG_BUILD_ANY_LEGION):
            return

        legion_name = legion.get("name", "legione")
        raise ValueError(
            f"Non puoi costruire {action_label}: la legione '{legion_name}' è di tipo "
            f"'{LEGION_TYPE_LABELS.get(legion_type, legion_type)}'. Serve una legione "
            f"'{LEGION_TYPE_LABELS[required_type]}'."
        )

    def _resolve_build_cell(
        self,
        legion: Dict[str, Any],
        target: Optional[Tuple[int, int]],
        action_label: str,
    ) -> Tuple[int, int]:
        """Cella su cui costruire: quella della legione, o una a distanza.

        [ABILITY-EFFECTS] Costruire lontano dalla legione è il senso di
        Costruzione Territoriale: senza l'abilità si lavora solo sotto i propri
        piedi, con l'abilità su qualunque cella del dominio.
        """
        own_pos = tuple(legion.get("pos", ()))
        if target is None:
            return own_pos

        row, col = int(target[0]), int(target[1])
        if (row, col) == own_pos:
            return own_pos

        if not self._ability_flag(PLAYER, ab.FLAG_BUILD_ANYWHERE):
            raise ValueError(
                f"Puoi costruire {action_label} solo dove si trova la legione "
                f"'{legion.get('name', 'legione')}'. Serve l'abilità "
                f"'{ab.ABILITIES[ab.DOMAIN_ENGINEERING_ID].name}' per lavorare a distanza."
            )

        cell = self.game_map.get_cell(row, col)
        if cell is None:
            raise ValueError("Cella fuori dalla mappa.")
        if cell.occupation != PLAYER:
            raise ValueError(
                f"Puoi costruire {action_label} solo su celle che controlli: "
                f"({row},{col}) non è tua."
            )
        return (row, col)

    def place_mine(self, legion_id: str, target: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """Piazza una miniera con una legione Mineraria.

        Sulla cella della legione, oppure — con Costruzione Territoriale — su
        una qualsiasi cella controllata.
        """
        if self.state != SessionState.ACTIVE:
            raise ValueError("La partita è terminata.")

        legion = self._get_player_legion_or_raise(legion_id)
        self._require_legion_type(legion, LEGION_TYPE_MINING, "una Miniera")
        if self._available_mine_slots(PLAYER) <= 0:
            raise ValueError(
                f"Non hai slot miniera disponibili. Serve più territorio controllato "
                f"(1 slot ogni {MINE_TILES_PER_SLOT} celle)."
            )

        row, col = self._resolve_build_cell(legion, target, "una miniera")
        cell = self.game_map.place_mine(PLAYER, row, col)
        remote = "" if (row, col) == tuple(legion.get("pos", ())) else " (cantiere a distanza)"
        log_entry = (
            f"[Turno {self.game_map.turn}] ⛏ PLAYER costruisce una miniera su ({row},{col}) "
            f"con la legione '{legion['name']}'{remote}"
        )
        self.battle_log.append(log_entry)
        self._evento(                                         # [EVENT-CHANNEL]
            ev.MINIERA if ev else "miniera", entita=PLAYER, pos=(row, col),
        )
        return {
            "ok": True,
            "message": log_entry,
            "cell": cell.to_dict(),
            "state": self.state.value,
            "map": self.game_map.to_dict(),
            "player_grux": self.grux_balance[PLAYER],
        }

    def _garrison_capacity(self, cell: Optional[Any]) -> int:
        """Quanti presidi regge una cella.

        Il castello ha il suo tetto fisso; una cella normale ne ospita uno,
        più uno per ogni livello di fortificazione, fino al tetto assoluto.
        """
        if cell is None:
            return 0
        if cell.is_castle:
            return CASTLE_MAX_GARRISON_UNITS
        capacity = (
            CELL_GARRISON_BASE_CAPACITY
            + int(cell.fortification_level) * CELL_GARRISON_PER_FORT_LEVEL
        )
        return min(capacity, CELL_MAX_GARRISON_UNITS)

    def _garrison_full_error(self, cell: Optional[Any]) -> Optional[str]:
        """Messaggio di rifiuto se la cella non regge un altro presidio, altrimenti None."""
        if cell is None:
            return None
        capacity = self._garrison_capacity(cell)
        if len(cell.garrison_unit_ids) < capacity:
            return None

        if cell.is_castle:
            return (
                f"Il castello ha già il massimo di presidi ({capacity}): "
                f"oltre questo limite sarebbe imprendibile."
            )
        if capacity < CELL_MAX_GARRISON_UNITS:
            return (
                f"Questa cella regge {capacity} presidi al suo livello di fortificazione: "
                f"costruisci una fortificazione per ospitarne un altro."
            )
        return f"Questa cella ha già il massimo di presidi ({capacity})."

    def _detach_unit_from_legion_to_garrison(
        self,
        legion: Dict[str, Any],
        cell_pos: Tuple[int, int],
        unit_id: Optional[str],
    ) -> Dict[str, Any]:
        """Distacca una truppa dalla legione stessa (non dalla riserva) per presidiare la cella."""
        units = legion.get("units", [])
        if len(units) < 2:
            raise ValueError(
                "Non puoi lasciare un presidio: la legione deve avere almeno 2 truppe "
                "(ne resta sempre almeno 1 attiva)."
            )

        selected_unit_id = unit_id
        if selected_unit_id is None:
            sorted_units = sorted(units, key=lambda uid: self._unit_battle_value(uid))
            selected_unit_id = sorted_units[0]
        if selected_unit_id not in units:
            raise ValueError("La truppa selezionata non è presente in questa legione.")

        cell = self.game_map.get_cell(*cell_pos)
        if cell is None:
            raise ValueError("Cella presidio non valida.")
        full = self._garrison_full_error(cell)
        if full:
            raise ValueError(full)

        units.remove(selected_unit_id)
        cell.garrison_unit_ids.append(selected_unit_id)
        cell.garrison_strength = max(cell.garrison_strength, len(cell.garrison_unit_ids))

        unit_name = self.units_map.get(selected_unit_id, {}).get("name", selected_unit_id)
        return {"unit_id": selected_unit_id, "unit_name": unit_name}

    def place_garrison(self, legion_id: str, unit_id: Optional[str] = None) -> Dict[str, Any]:
        """Stacca una truppa dalla legione (che deve averne almeno 2) per presidiare la cella dove si trova."""
        if self.state != SessionState.ACTIVE:
            raise ValueError("La partita è terminata.")

        legion = self._get_player_legion_or_raise(legion_id)
        pos = tuple(legion["pos"])
        cell = self.game_map.get_cell(*pos)
        if cell is None or cell.occupation != PLAYER:
            raise ValueError("La cella della legione non è controllata dal PLAYER.")

        detach_result = self._detach_unit_from_legion_to_garrison(legion, pos, unit_id)

        row, col = pos
        log_entry = (
            f"[Turno {self.game_map.turn}] 🛡 PLAYER piazza un presidio su ({row},{col}) "
            f"con la legione '{legion['name']}' — Distaccata: {detach_result['unit_name']}"
        )
        self.battle_log.append(log_entry)
        self._evento(                                         # [EVENT-CHANNEL]
            ev.PRESIDIO if ev else "presidio", entita=PLAYER, pos=(row, col),
            quantita=cell.garrison_strength,
            dettaglio={"unita": detach_result.get("unit_name")},
        )
        return {
            "ok": True,
            "message": log_entry,
            "cell": cell.to_dict(),
            "state": self.state.value,
            "map": self.game_map.to_dict(),
        }

    def _fortification_cost(self, current_level: int, entity: Occupation = PLAYER) -> int:
        """Costo fortificazione con crescita forte sullo stack della stessa cella."""
        cost = self.base_fortification_cost * (1 + (current_level * 1.7))
        # [ABILITY-EFFECTS] Trinceramento Rapido: palizzate prefabbricate.
        cost *= self._ability_economy_factor(entity, ab.ECO_FORTIFICATION_COST)
        return max(5, int(round(cost)))

    def place_fortification(
        self, legion_id: str, target: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """Piazza una fortificazione con una legione Costruzione, con costo crescente.

        Sulla cella della legione, oppure — con Costruzione Territoriale — su
        una qualsiasi cella controllata.
        """
        if self.state != SessionState.ACTIVE:
            raise ValueError("La partita è terminata.")

        legion = self._get_player_legion_or_raise(legion_id)
        # [DOCTRINE-LAYER] Schermo e Fuoco: la legione alza il muro da sé,
        # senza genieri, e non paga i primi livelli. Resta un ordine del
        # giocatore: da sola non costruisce niente.
        stato = self._doctrine_state(PLAYER, legion) if doc is not None else None
        cantiere = stato is not None and doc.fortification_relief(
            stato["strategy_id"], stato["units"],
            turn=stato["turn"], since_turn=stato["since_turn"],
            current_level=0,
        ) is not None
        if not cantiere:
            self._require_legion_type(legion, LEGION_TYPE_CONSTRUCTION, "una Fortificazione")

        row, col = self._resolve_build_cell(legion, target, "una fortificazione")
        cell = self.game_map.get_cell(row, col)
        if cell is None:
            raise ValueError("Cella fuori dalla mappa.")
        if cell.occupation != PLAYER:
            raise ValueError("Puoi fortificare solo celle controllate dal PLAYER.")

        # Il castello è fortificabile, ma con un tetto suo: ogni livello riduce
        # i danni da assedio, quindi senza limite diventerebbe imprendibile.
        is_own_castle = bool(cell.is_castle)
        max_level = CASTLE_MAX_FORTIFICATION_LEVEL if is_own_castle else None
        if is_own_castle and cell.fortification_level >= CASTLE_MAX_FORTIFICATION_LEVEL:
            raise ValueError(
                f"Il castello è già fortificato al massimo "
                f"(livello {CASTLE_MAX_FORTIFICATION_LEVEL}): "
                f"riduce i danni da assedio del "
                f"{int(CASTLE_MAX_FORTIFICATION_LEVEL * CASTLE_FORT_DAMAGE_REDUCTION_PER_LEVEL * 100)}%."
            )

        current_level = cell.fortification_level
        cost = self._fortification_cost(current_level)
        # [DOCTRINE-LAYER] I primi livelli di Schermo e Fuoco sono gratis; dal
        # terzo in poi, e sul proprio castello, si paga come tutti.
        gratis = False
        if cantiere:
            sconto = doc.fortification_relief(
                stato["strategy_id"], stato["units"],
                turn=stato["turn"], since_turn=stato["since_turn"],
                current_level=current_level, is_castle=is_own_castle,
            )
            gratis = bool(sconto and sconto.get("free"))
        if gratis:
            cost = 0
        if self.grux_balance[PLAYER] < cost:
            raise ValueError(f"Grux insufficienti per fortificare: servono {cost}, disponibili {self.grux_balance[PLAYER]}")

        self.grux_balance[PLAYER] -= cost
        cell = self.game_map.place_fortification(PLAYER, row, col, max_level=max_level)
        next_cost = self._fortification_cost(cell.fortification_level)

        if is_own_castle:
            riduzione = int(cell.fortification_level * CASTLE_FORT_DAMAGE_REDUCTION_PER_LEVEL * 100)
            log_entry = (
                f"[Turno {self.game_map.turn}] 🏰 PLAYER fortifica il CASTELLO con la legione "
                f"'{legion['name']}' → livello {cell.fortification_level}/"
                f"{CASTLE_MAX_FORTIFICATION_LEVEL} (costo {cost} grux, "
                f"-{riduzione}% danni da assedio)"
            )
        else:
            remote = "" if (row, col) == tuple(legion.get("pos", ())) else " (cantiere a distanza)"
            spesa = "gratis, posizioni preparate" if gratis else f"costo {cost} grux"
            log_entry = (
                f"[Turno {self.game_map.turn}] 🧱 PLAYER fortifica ({row},{col}) con la legione "
                f"'{legion['name']}' → livello {cell.fortification_level} ({spesa}){remote}"
            )
        self.battle_log.append(log_entry)
        self._evento(                                         # [EVENT-CHANNEL]
            ev.FORTIFICAZIONE if ev else "fortificazione",
            entita=PLAYER, pos=(row, col), quantita=cell.fortification_level,
            dettaglio={"castello": bool(is_own_castle)},
        )
        return {
            "ok": True,
            "message": log_entry,
            "cell": cell.to_dict(),
            "state": self.state.value,
            "map": self.game_map.to_dict(),
            "player_grux": self.grux_balance[PLAYER],
            "next_fortification_cost": next_cost,
        }

    def _place_best_ai_mine(self) -> Optional[str]:
        """L'IA piazza una miniera sulla miglior cella controllata disponibile."""
        ai_can_build_anywhere = self._is_ability_unlocked(AI, DOMAIN_ENGINEERING_ID)
        ai_build_positions = set(self._active_legion_positions(AI))

        best_cell = None
        best_score = float("-inf")
        for row in self.game_map.grid:
            for cell in row:
                if cell.occupation != AI or cell.is_castle or cell.is_mine or cell.terrain == "Fiume":
                    continue
                if not ai_can_build_anywhere and (cell.row, cell.col) not in ai_build_positions:
                    continue
                # Stessa tabella che poi paga: se no l'IA scaverebbe in
                # palude senza sapere che rende la metà della montagna.
                score = 2.0 if cell.is_strategic else 0.0
                score += mine_yield(cell.terrain) / MINE_YIELD_PER_ROUND
                if score > best_score:
                    best_score = score
                    best_cell = cell

        if best_cell is None:
            return None

        self.game_map.place_mine(AI, best_cell.row, best_cell.col)
        self._evento(                                         # [EVENT-CHANNEL]
            ev.MINIERA if ev else "miniera", entita=AI, pos=(best_cell.row, best_cell.col),
        )
        return f"[Turno {self.game_map.turn}] ⛏ IA costruisce una miniera su ({best_cell.row},{best_cell.col})"

    def _ai_castle_ring_cells(self) -> List[Any]:
        """Celle immediatamente adiacenti al castello IA e da essa controllate."""
        castle_pos = self.game_map.castle_positions.get(AI)
        if castle_pos is None:
            return []

        ring: List[Any] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cell = self.game_map.get_cell(castle_pos[0] + dr, castle_pos[1] + dc)
            if cell is None or cell.is_castle:
                continue
            if cell.occupation != AI:
                continue
            ring.append(cell)
        return ring

    def _garrison_ai_castle(self, reserve: int = 0) -> List[str]:
        """L'IA presidia il proprio castello, come può fare il giocatore.

        Le legioni non difendono più il castello: senza questo, solo il player
        avrebbe potuto guarnirlo. Quante truppe schierare è una manopola per
        difficoltà (`castle_garrison_target`); `reserve` è quanto non può
        spendere perché destinato ad altro.
        """
        target = int(getattr(self.ai_policy, "castle_garrison_target", 0) or 0)
        if target <= 0:
            return []

        castle_pos = self.game_map.castle_positions.get(AI)
        cell = self.game_map.get_cell(*castle_pos) if castle_pos else None
        if cell is None:
            return []

        target = min(target, self._garrison_capacity(cell))
        if len(cell.garrison_unit_ids) >= target:
            return []

        disponibili = self.grux_balance[AI] - max(0, int(reserve))
        affordable = [
            unit for unit in self.data["units"]
            if self.unit_costs[unit["id"]] <= disponibili
        ]
        if not affordable:
            return []

        best_unit = max(
            affordable,
            key=lambda unit: self._garrison_unit_defense_value(unit["id"], cell.terrain),
        )
        cost = self.unit_costs[best_unit["id"]]
        self.grux_balance[AI] -= cost
        cell.garrison_unit_ids.append(best_unit["id"])
        cell.garrison_strength = max(cell.garrison_strength, len(cell.garrison_unit_ids))
        self._evento(                                         # [EVENT-CHANNEL]
            ev.PRESIDIO if ev else "presidio", entita=AI, pos=(cell.row, cell.col),
            quantita=cell.garrison_strength,
            dettaglio={"unita": best_unit.get("name"), "castello": True},
        )
        return [
            f"[Turno {self.game_map.turn}] 🛡 IA arruola {best_unit['name']} a presidio "
            f"del CASTELLO — {cost} grux ({len(cell.garrison_unit_ids)}/{target})"
        ]

    def _reinforce_ai_castle_ring(self, extra_reserve: int = 0) -> List[str]:
        """Trincera l'IA sulle caselle adiacenti al proprio castello.

        Fortifica l'anello e vi compra truppe di presidio con il budget extra:
        sono unità acquistate apposta, quindi NON tolgono forza alla legione
        in campo (`ai_units` resta intatto). `extra_reserve` è quanto deve
        restare in cassa per altro, oltre alla riserva del piano.
        """
        plan_getter = getattr(self.ai_policy, "castle_ring_plan", None)
        if not callable(plan_getter):
            return []

        plan = plan_getter()
        # Il piano dell'anello non può sfondare il tetto generale delle celle.
        max_fort = min(int(plan.get("max_fort_level", 0)), GameMap.MAX_FORTIFICATION_LEVEL)
        target_garrison = int(plan.get("target_garrison", 0))
        reserve = int(plan.get("reserve_grux", 0)) + max(0, int(extra_reserve))

        logs: List[str] = []
        ring = self._ai_castle_ring_cells()
        if not ring:
            return logs

        # 1. Fortificazioni sull'anello, dalla cella più scoperta.
        for cell in sorted(ring, key=lambda c: c.fortification_level):
            if cell.fortification_level >= max_fort:
                continue
            cost = self._fortification_cost(int(cell.fortification_level), AI)
            if self.grux_balance[AI] < cost + reserve:
                break
            self.grux_balance[AI] -= cost
            placed = self.game_map.place_fortification(AI, cell.row, cell.col)
            self._evento(                                     # [EVENT-CHANNEL]
                ev.FORTIFICAZIONE if ev else "fortificazione",
                entita=AI, pos=(placed.row, placed.col),
                quantita=placed.fortification_level,
            )
            logs.append(
                f"[Turno {self.game_map.turn}] 🧱 IA blinda l'anello del castello "
                f"({placed.row},{placed.col}) → livello {placed.fortification_level} "
                f"(costo {cost} grux)"
            )
            break   # una costruzione per turno, non svuota le casse in un colpo

        # 2. Presidi comprati apposta per l'anello, entro la capienza della cella.
        for cell in sorted(ring, key=lambda c: len(c.garrison_unit_ids)):
            capacity = min(target_garrison, self._garrison_capacity(cell))
            if len(cell.garrison_unit_ids) >= capacity:
                continue

            affordable = [
                unit for unit in self.data["units"]
                if self.unit_costs[unit["id"]] + reserve <= self.grux_balance[AI]
            ]
            if not affordable:
                break

            best_unit = max(
                affordable,
                key=lambda unit: self._garrison_unit_defense_value(unit["id"], cell.terrain),
            )
            cost = self.unit_costs[best_unit["id"]]
            self.grux_balance[AI] -= cost
            cell.garrison_unit_ids.append(best_unit["id"])
            cell.garrison_strength = max(cell.garrison_strength, len(cell.garrison_unit_ids))
            self._evento(                                     # [EVENT-CHANNEL]
                ev.PRESIDIO if ev else "presidio", entita=AI, pos=(cell.row, cell.col),
                quantita=cell.garrison_strength,
                dettaglio={"unita": best_unit.get("name"), "anello": True},
            )
            logs.append(
                f"[Turno {self.game_map.turn}] 🛡 IA arruola {best_unit['name']} a presidio "
                f"di ({cell.row},{cell.col}) — {cost} grux, anello del castello"
            )
            break   # un presidio per turno

        return logs

    def _place_best_ai_fortification(self) -> Optional[str]:
        """L'IA fortifica la miglior cella controllata disponibile, rispettando costo e vincoli abilità."""
        ai_can_build_anywhere = self._is_ability_unlocked(AI, DOMAIN_ENGINEERING_ID)
        ai_build_positions = set(self._active_legion_positions(AI))

        player_anchor: Optional[Tuple[int, int]] = None
        if self.player_legions:
            first_player_legion = next(iter(self.player_legions.values()))
            pos = tuple(first_player_legion.get("pos", ()))
            if len(pos) == 2:
                player_anchor = (int(pos[0]), int(pos[1]))

        best: Optional[Tuple[float, Any, int]] = None
        for row in self.game_map.grid:
            for cell in row:
                if cell.occupation != AI:
                    continue
                # Il castello ora è fortificabile e riduce i danni da assedio:
                # l'IA deve poterlo fare come il player, o l'asimmetria falserebbe
                # tutto il bilanciamento delle difficoltà. Valgono per lei gli
                # stessi tetti: una cella al massimo non è più un bersaglio.
                cell_cap = (
                    CASTLE_MAX_FORTIFICATION_LEVEL if cell.is_castle
                    else GameMap.MAX_FORTIFICATION_LEVEL
                )
                if cell.fortification_level >= cell_cap:
                    continue
                if not ai_can_build_anywhere and (cell.row, cell.col) not in ai_build_positions:
                    continue

                current_level = int(cell.fortification_level)
                cost = self._fortification_cost(current_level, AI)
                if self.grux_balance[AI] < cost:
                    continue

                score = 0.0
                if cell.is_castle:
                    # Difendere il proprio castello è la priorità: è la condizione
                    # di sconfitta, e i livelli disponibili sono solo 4.
                    score += 4.0
                if cell.is_strategic:
                    score += 2.4
                if cell.is_mine:
                    score += 1.7
                score += max(0.0, 1.2 - (current_level * 0.45))

                if player_anchor is not None:
                    score += max(0.0, 1.4 - (self._order_distance((cell.row, cell.col), player_anchor) * 0.18))

                if best is None or score > best[0]:
                    best = (score, cell, cost)

        if best is None:
            return None

        _, cell, cost = best
        self.grux_balance[AI] -= cost
        max_level = CASTLE_MAX_FORTIFICATION_LEVEL if cell.is_castle else None
        placed = self.game_map.place_fortification(AI, cell.row, cell.col, max_level=max_level)
        # [EVENT-CHANNEL] Prima del bivio: il ramo del castello esce con un
        # return anticipato, e più sotto l'evento non sarebbe mai scattato per
        # la fortificazione più importante che l'IA costruisce.
        self._evento(
            ev.FORTIFICAZIONE if ev else "fortificazione",
            entita=AI, pos=(placed.row, placed.col),
            quantita=placed.fortification_level,
            dettaglio={"castello": bool(placed.is_castle)},
        )
        if placed.is_castle:
            riduzione = int(placed.fortification_level * CASTLE_FORT_DAMAGE_REDUCTION_PER_LEVEL * 100)
            return (
                f"[Turno {self.game_map.turn}] 🏰 IA fortifica il CASTELLO → livello "
                f"{placed.fortification_level}/{CASTLE_MAX_FORTIFICATION_LEVEL} "
                f"(costo {cost} grux, -{riduzione}% danni da assedio)"
            )
        return (
            f"[Turno {self.game_map.turn}] 🧱 IA fortifica ({placed.row},{placed.col}) "
            f"→ livello {placed.fortification_level} (costo {cost} grux)"
        )

    def recruit_player_unit(self, unit_id: str) -> Dict[str, Any]:
        """Compra una unità per il giocatore e aggiorna l'esercito."""
        log_entry = self._recruit_unit(PLAYER, unit_id, auto=False)
        return {
            "ok": True,
            "message": log_entry,
            "state": self.state.value,
            "map": self.game_map.to_dict(),
            "session": self.to_dict(),
        }

    def start_player_auto_recruit(self, unit_id: str, turns: int) -> Dict[str, Any]:
        """Avvia il piano di autoreclutamento del player per un numero di turni."""
        if self.state != SessionState.ACTIVE:
            raise ValueError("La partita è terminata.")
        if unit_id not in self.unit_costs:
            raise ValueError(f"Unità sconosciuta: {unit_id}")

        turns_value = int(turns)
        if turns_value <= 0:
            raise ValueError("I turni di autoreclutamento devono essere almeno 1.")
        if turns_value > 40:
            raise ValueError("I turni di autoreclutamento non possono superare 40.")

        unit_name = self.units_map.get(unit_id, {}).get("name", unit_id)
        self.player_auto_recruit.update(
            {
                "enabled": True,
                "unit_id": unit_id,
                "unit_name": unit_name,
                "turns_total": turns_value,
                "turns_remaining": turns_value,
                "attempted_turns": 0,
                "successful_recruits": 0,
                "funds_wait": 0,
                "last_result": "scheduled",
                "last_reason": None,
            }
        )

        log_entry = (
            f"[Turno {self.game_map.turn}] 🤖 PLAYER avvia autoreclutamento: "
            f"{unit_name} per {turns_value} turni"
        )
        self.battle_log.append(log_entry)
        return {
            "ok": True,
            "message": log_entry,
            "state": self.state.value,
            "map": self.game_map.to_dict(),
            "session": self.to_dict(),
        }

    def stop_player_auto_recruit(self, reason: str = "manual") -> Dict[str, Any]:
        """Ferma il piano di autoreclutamento del player."""
        was_enabled = bool(self.player_auto_recruit.get("enabled"))
        unit_name = self.player_auto_recruit.get("unit_name") or "unità"

        self.player_auto_recruit["enabled"] = False
        self.player_auto_recruit["turns_remaining"] = 0
        self.player_auto_recruit["funds_wait"] = 0
        self.player_auto_recruit["last_result"] = "stopped"

        if was_enabled:
            reason_label = "manuale" if reason == "manual" else reason
            log_entry = (
                f"[Turno {self.game_map.turn}] 🤖 PLAYER ferma autoreclutamento "
                f"({unit_name}) - motivo: {reason_label}"
            )
            self.battle_log.append(log_entry)
            message = log_entry
        else:
            message = "Autoreclutamento non attivo."

        return {
            "ok": True,
            "message": message,
            "state": self.state.value,
            "map": self.game_map.to_dict(),
            "session": self.to_dict(),
        }

    def _run_player_auto_recruit(self) -> List[str]:
        """Esegue una iterazione del piano di autoreclutamento player al termine del round."""
        logs: List[str] = []
        if self.state != SessionState.ACTIVE:
            return logs

        if not self.player_auto_recruit.get("enabled"):
            return logs

        unit_id = self.player_auto_recruit.get("unit_id")
        unit_name = self.player_auto_recruit.get("unit_name") or (unit_id or "unità")
        turns_remaining = int(self.player_auto_recruit.get("turns_remaining") or 0)

        if not unit_id or unit_id not in self.unit_costs:
            self.player_auto_recruit["enabled"] = False
            self.player_auto_recruit["last_result"] = "invalid_unit"
            logs.append(
                f"[Turno {self.game_map.turn}] 🤖 Autoreclutamento interrotto: unità non valida"
            )
            return logs

        if turns_remaining <= 0:
            self.player_auto_recruit["enabled"] = False
            self.player_auto_recruit["last_result"] = "completed"
            logs.append(
                f"[Turno {self.game_map.turn}] 🤖 Autoreclutamento completato ({unit_name})"
            )
            return logs

        self.player_auto_recruit["attempted_turns"] = int(self.player_auto_recruit.get("attempted_turns") or 0) + 1

        # Il motivo va calcolato PRIMA del tentativo: `_recruit_unit` in modalità
        # automatica restituisce None sia per cooldown sia per grux, e il log
        # finiva per dire "cooldown o grux insufficienti" anche con le casse piene.
        block_kind, block_reason = self._recruit_block(PLAYER, unit_id)

        # Le casse vuote non consumano il piano. Prima sì: un piano da dieci
        # turni avviato senza grux se li bruciava tutti senza arruolare niente,
        # e alla fine si dichiarava pure "completato". Il cooldown invece resta
        # a carico del piano — è il suo ritmo, ed è quello che il grafico
        # previsionale promette al giocatore quando lo imposta.
        if block_kind == "funds":
            attesa = int(self.player_auto_recruit.get("funds_wait") or 0) + 1
            self.player_auto_recruit["funds_wait"] = attesa
            self.player_auto_recruit["last_result"] = "skipped"
            self.player_auto_recruit["last_reason"] = block_reason
            if attesa >= AUTO_RECRUIT_MAX_WAIT:
                self.player_auto_recruit["enabled"] = False
                self.player_auto_recruit["last_result"] = "out_of_funds"
                logs.append(
                    f"[Turno {self.game_map.turn}] 🤖 Autoreclutamento annullato "
                    f"({unit_name}): casse vuote da {attesa} turni. "
                    f"Restavano {turns_remaining} turni di piano, non consumati."
                )
            else:
                logs.append(
                    f"[Turno {self.game_map.turn}] 🤖 Autoreclutamento in attesa di "
                    f"fondi ({attesa}/{AUTO_RECRUIT_MAX_WAIT}) — {block_reason}. "
                    f"Il piano non consuma turni: ne restano {turns_remaining}."
                )
            return logs

        self.player_auto_recruit["turns_remaining"] = turns_remaining - 1
        self.player_auto_recruit["funds_wait"] = 0

        recruit_log = None if block_kind else self._recruit_unit(PLAYER, unit_id, auto=True)

        if recruit_log:
            self.player_auto_recruit["successful_recruits"] = int(self.player_auto_recruit.get("successful_recruits") or 0) + 1
            self.player_auto_recruit["last_result"] = "success"
            self.player_auto_recruit["last_reason"] = None
            logs.append(
                f"[Turno {self.game_map.turn}] 🤖 Autoreclutamento riuscito: {unit_name}"
            )
        else:
            self.player_auto_recruit["last_result"] = "skipped"
            self.player_auto_recruit["last_reason"] = block_reason or "motivo sconosciuto"
            logs.append(
                f"[Turno {self.game_map.turn}] 🤖 Autoreclutamento in pausa ({unit_name}): "
                f"{block_reason or 'motivo sconosciuto'}"
            )

        if int(self.player_auto_recruit.get("turns_remaining") or 0) <= 0:
            self.player_auto_recruit["enabled"] = False
            # Un piano che non ha arruolato nessuno non è "completato": dirlo
            # lasciava il giocatore a chiedersi dove fossero finite le reclute.
            fatte = int(self.player_auto_recruit.get("successful_recruits") or 0)
            if fatte > 0:
                self.player_auto_recruit["last_result"] = "completed"
                quante = "1 arruolata" if fatte == 1 else f"{fatte} arruolate"
                logs.append(
                    f"[Turno {self.game_map.turn}] 🤖 Piano autoreclutamento "
                    f"completato ({unit_name}): {quante}."
                )
            else:
                self.player_auto_recruit["last_result"] = "blocked"
                motivo = self.player_auto_recruit.get("last_reason") or "motivo sconosciuto"
                logs.append(
                    f"[Turno {self.game_map.turn}] 🤖 Piano autoreclutamento terminato "
                    f"({unit_name}) senza arruolare nulla: {motivo}"
                )

        return logs

    def _recruit_block(
        self, entity: Occupation, unit_id: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Cosa impedisce il reclutamento adesso: (codice, spiegazione).

        Serve all'autoreclutamento per dire nel log cosa lo sta fermando davvero:
        il tentativo automatico fallisce in silenzio e non distingue i due casi.
        Il codice serve a distinguerli anche nel comportamento: il cooldown è
        il ritmo del piano, le casse vuote sono un ostacolo esterno.
        """
        if unit_id not in self.unit_costs:
            return "unknown_unit", f"unità sconosciuta ({unit_id})"

        if not self._can_recruit_now(entity):
            last_turn = self.last_recruit_turn.get(entity)
            turns_passed = 0 if last_turn is None else (self.game_map.turn - last_turn)
            remaining = max(0, self._recruit_cooldown_for(entity) - turns_passed)
            return "cooldown", f"reclutamento in cooldown, ancora {remaining} turno/i"

        cost = self._recruit_cost(entity, unit_id)
        balance = self.grux_balance.get(entity, 0)
        if balance < cost:
            return "funds", f"grux insufficienti: servono {cost}, disponibili {balance}"

        return None, None

    def _recruit_unit(self, entity: Occupation, unit_id: str, auto: bool) -> Optional[str]:
        """Recluta una unità, scala il costo e ricalcola il vettore esercito."""
        if self.state != SessionState.ACTIVE:
            if auto:
                return None
            raise ValueError("La partita è terminata.")

        if unit_id not in self.unit_costs:
            raise ValueError(f"Unità sconosciuta: {unit_id}")

        if not self._can_recruit_now(entity):
            last_turn = self.last_recruit_turn.get(entity)
            turns_passed = 0 if last_turn is None else (self.game_map.turn - last_turn)
            remaining = max(0, self._recruit_cooldown_for(entity) - turns_passed)
            if auto:
                return None
            raise ValueError(
                f"Reclutamento in cooldown: attendi ancora {remaining} turno/i prima di reclutare di nuovo."
            )

        cost = self._recruit_cost(entity, unit_id)
        if self.grux_balance[entity] < cost:
            if auto:
                return None
            raise ValueError(f"Grux insufficienti: servono {cost}, disponibili {self.grux_balance[entity]}")

        home_terrain = self.player_home_terrain if entity == PLAYER else self.ai_home_terrain
        before_breakdown = self._strength_breakdown(entity, home_terrain)

        self.grux_balance[entity] -= cost

        # La recluta entra riposata e abbassa la stanchezza media della riserva:
        # è il modo lento ma legittimo di rimettere in sesto un esercito logoro.
        tc.dilute_pool(self.reserve_condition[entity], len(self._entity_units(entity)))

        if entity == PLAYER:
            self.player_units.append(unit_id)
            self.player_army = aggregate_army(self.player_units, self.data["units"])
            self.player_modified, _ = apply_modifiers(
                army_vector=self.player_army,
                terrain_name=self.player_home_terrain,
                weather_name=self.weather,
                troop_status_name=self.player_troop_status,
                modifiers_data=self.data,
            )
            self.player_army_cost += cost
        else:
            self.ai_units.append(unit_id)
            self.ai_army = aggregate_army(self.ai_units, self.data["units"])
            self.ai_modified, _ = apply_modifiers(
                army_vector=self.ai_army,
                terrain_name=self.ai_home_terrain,
                weather_name=self.weather,
                troop_status_name=self.ai_troop_status,
                modifiers_data=self.data,
            )
            self.ai_army_cost += cost

        side = entity.value.upper()
        unit_name = self.units_map.get(unit_id, {}).get("name", unit_id)
        log_entry = f"[Turno {self.game_map.turn}] 💰 {side} recluta {unit_name} per {cost} grux"
        self.battle_log.append(log_entry)

        # [CARAVAN] La recluta IA deve raggiungere le legioni in campo, e ci
        # deve arrivare marciando. Prima qui c'era un `_sync_ai_legion_units()`
        # diretto, e la truppa compariva dentro la legione anche a diciotto
        # caselle di distanza, nello stesso turno in cui veniva arruolata.
        if entity == AI:
            self._instrada_rinforzi([unit_id])
        self.last_recruit_turn[entity] = self.game_map.turn

        after_breakdown = self._strength_breakdown(entity, home_terrain)
        log_strength_debug(
            "recruit_strength_delta",
            {
                "debug_notice": "DEBUG TEMPORANEO - rimuovere cartella debug in produzione",
                "turn": self.game_map.turn,
                "entity": entity.value,
                "unit_id": unit_id,
                "unit_cost": cost,
                "grux_balance_after": self.grux_balance[entity],
                "before": before_breakdown,
                "after": after_breakdown,
                "delta_effective_strength": round(
                    after_breakdown["effective_strength"] - before_breakdown["effective_strength"],
                    4,
                ),
            },
        )
        return log_entry

    # ──────────────────────────────────────────────────────────
    # FORZA IN COMBATTIMENTO (privato)
    # ──────────────────────────────────────────────────────────

    def _effective_army_strength(self, entity: Occupation, terrain: str) -> float:
        """Forza effettiva dell'armata sul terreno corrente, penalizzata se ha staccato troppi presidi."""
        return self._strength_breakdown(entity, terrain)["effective_strength"]

    def _strength_breakdown(self, entity: Occupation, terrain: str) -> Dict[str, Any]:
        """Breakdown DEBUG della forza usata in combattimento."""
        army_vector = self.player_army if entity == PLAYER else self.ai_army
        troop_status = self.player_troop_status if entity == PLAYER else self.ai_troop_status
        modified, warnings = apply_modifiers(
            army_vector=army_vector,
            terrain_name=terrain,
            weather_name=self.weather,
            troop_status_name=troop_status,
            modifiers_data=self.data,
        )

        composition = self._army_composition(entity)
        unit_power_rows: List[Dict[str, Any]] = []
        base_total = 0.0
        stack_bonus_total = 0.0
        total_legions = 0
        for unit_id, count in composition.items():
            value = self._unit_battle_value(unit_id)
            base_part = value * count
            # Bonus stack controllato: utile ma meno esplosivo in late game.
            stack_bonus = value * 0.22 * ((count - 1) ** 1.08) if count > 1 else 0.0
            stack_bonus = min(stack_bonus, base_part * 0.55)
            unit_power_rows.append(
                {
                    "unit_id": unit_id,
                    "unit_name": self.units_map.get(unit_id, {}).get("name", unit_id),
                    "count": count,
                    "unit_value": round(value, 3),
                    "base_part": round(base_part, 3),
                    "stack_bonus": round(stack_bonus, 3),
                }
            )
            base_total += base_part
            stack_bonus_total += stack_bonus
            total_legions += count

        # [COMPAT-LAYER] Il breakdown porta entrambi i punteggi: è la finestra
        # da cui si legge la distanza fra consiglio e battaglia.
        esercito = list(self.player_units if entity == PLAYER else self.ai_units)
        punteggi = self._strategy_scores(entity, modified, terrain=terrain, unit_ids=esercito)
        strategy_compatibility = punteggi["combat_compatibility"]
        strategy_factor = self._factor_from_compatibility(strategy_compatibility)
        detached = self.game_map.count_garrisons(entity)
        detach_penalty = max(0.7, 1.0 - (max(0, detached) * 0.06))

        # Fattore contesto ricavato dai modificatori su attributi: centro su 1.0
        context_factor = max(0.75, min(1.25, (sum(modified.values()) / 4.6)))

        base_strength = base_total + stack_bonus_total
        effective_strength = base_strength * strategy_factor * context_factor * detach_penalty
        return {
            "entity": entity.value,
            "terrain": terrain,
            "weather": self.weather,
            "troop_status": troop_status,
            "composition": composition,
            "composition_text": self._format_composition(entity),
            "legions_total": total_legions,
            "unit_power_rows": unit_power_rows,
            "detached_garrisons_over_base": max(0, detached),
            "detach_penalty": round(detach_penalty, 4),
            "strategy_factor": round(strategy_factor, 4),
            "strategy_compatibility": round(strategy_compatibility, 4),
            # [COMPAT-LAYER] Nomi espliciti, come chiede la specifica: quello
            # che la battaglia usa e quello che l'Advisor mostra.
            "combat_compatibility": round(punteggi["combat_compatibility"], 4),
            "advisory_score": round(punteggi["advisory_score"], 4),
            "compat_calibration": cmp_layer.describe() if cmp_layer is not None else None,
            "context_factor": round(context_factor, 4),
            "base_strength": round(base_strength, 4),
            "effective_strength": int(round(effective_strength)),
            "modified_vector": {k: round(v, 4) for k, v in modified.items()},
            "modifier_warnings": warnings,
        }

    # ──────────────────────────────────────────────────────────
    # SERIALIZZAZIONE
    # ──────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Stato completo della sessione (per l'API)."""
        return {
            "state":   self.state.value,
            "winner":  self.winner,
            "weather": self.weather,
            # Stato ambientale completo per l'indicatore: i due assi separati,
            # emoji, colori, effetti in chiaro e quanti turni mancano al cambio.
            "weather_state": self.weather_state(),
            # [SETUP-RULE] Cosa è stato simulato e cosa vale davvero in campo.
            # Il frontend lo usa per non promettere condizioni che non avrà, il
            # log sperimentale per distinguere le due cose.
            "simulation_setup": self.simulation_setup,
            "battle_conditions": {
                "weather": self.weather,
                "troop_status": self.player_troop_status,
                "home_terrain": self.player_home_terrain,
                "chosen_by_player": False,
            },
            "player": {
                "units":         self.player_units,
                "strategy_id":   self.player_strategy_id,
                "strategy_name": self.strategies_map.get(self.player_strategy_id, {}).get("name", self.player_strategy_id),
                "army":          self.player_army,
                "modified":      self.player_modified,
                # Stato della riserva nel castello: non è più il valore fisso
                # scelto a inizio partita (che restava "n/d" se non sceglievi),
                # ma la condizione viva delle truppe ferme in castello.
                "troop_status":  tc.resolve_status(self.reserve_condition[PLAYER]),
                "troop_condition": tc.describe(self.reserve_condition[PLAYER]),
                "initial_troop_status": self.player_troop_status,
                "legions":       self._legions_with_strategy(PLAYER),
                "castle": {
                    "hp": self.castle_hp.get(PLAYER, self.castle_hp_max.get(PLAYER, CASTLE_BASE_HP)),
                    "max_hp": self.castle_hp_max.get(PLAYER, CASTLE_BASE_HP),
                },
                "auto_recruit": {
                    "enabled": bool(self.player_auto_recruit.get("enabled")),
                    "unit_id": self.player_auto_recruit.get("unit_id"),
                    "unit_name": self.player_auto_recruit.get("unit_name"),
                    "turns_total": int(self.player_auto_recruit.get("turns_total") or 0),
                    "turns_remaining": int(self.player_auto_recruit.get("turns_remaining") or 0),
                    "attempted_turns": int(self.player_auto_recruit.get("attempted_turns") or 0),
                    "successful_recruits": int(self.player_auto_recruit.get("successful_recruits") or 0),
                    "last_result": self.player_auto_recruit.get("last_result") or "inactive",
                    "last_reason": self.player_auto_recruit.get("last_reason"),
                },
                "available_garrisons": self._available_garrisons(PLAYER),
                "grux_balance":  self.grux_balance[PLAYER],
                "army_cost":     self.player_army_cost,
                "available_mine_slots": self._available_mine_slots(PLAYER),
                "mine_income": self._mine_income_for(PLAYER),
                "fortification_base_cost": self.base_fortification_cost,
                "movement": self.movement_system.export_entity_state(PLAYER),
                "abilities": ab.states_payload(
                    self.ability_states[PLAYER], self.game_map.turn, self.grux_balance[PLAYER]
                ),
                "ability_paths": ab.path_order(),
                # Cosa concedono adesso le abilità di costruzione. La UI accende
                # le celle da qui invece di ragionare sugli id delle abilità.
                "build_rules": {
                    "anywhere": self._ability_flag(PLAYER, ab.FLAG_BUILD_ANYWHERE),
                    "any_legion": self._ability_flag(PLAYER, ab.FLAG_BUILD_ANY_LEGION),
                },
                # [ABILITY-EFFECTS] Con l'Industria dello Spionaggio i numeri
                # mostrati sono quelli veri: la UI toglie gli avvisi di
                # approssimazione e apre il dossier sul nemico.
                "intel": {
                    "accurate": self._ability_flag(PLAYER, ab.FLAG_SPY_NETWORK),
                    "ability_name": ab.ABILITIES[ab.SPY_INDUSTRY_ID].name,
                },
                "black_market": self.black_market[PLAYER].to_dict(
                    self.game_map.turn, self._black_market_open(PLAYER)
                ),
                "recruit_cooldown_turns": self._recruit_cooldown_for(PLAYER),
                # Prezzi effettivi di reclutamento: il listino di /config non
                # sa niente delle abilità, e senza questi il menu mostrerebbe
                # un prezzo diverso da quello che poi viene scalato.
                "recruit_costs": {
                    unit_id: self._recruit_cost(PLAYER, unit_id) for unit_id in self.unit_costs
                },
                "unit_costs":    {unit_id: self.unit_costs[unit_id] for unit_id in self.player_units},
            },
            "ai": {
                "units":         self.ai_units,
                "strategy_id":   self.ai_strategy_id,
                "strategy_name": self.ai_strategy_name,
                "difficulty":    self.ai_difficulty,
                "difficulty_labels": get_ai_difficulty_labels(),
                "legions":       self._legions_with_strategy(AI),
                "castle": {
                    "hp": self.castle_hp.get(AI, self.castle_hp_max.get(AI, CASTLE_BASE_HP)),
                    "max_hp": self.castle_hp_max.get(AI, CASTLE_BASE_HP),
                },
                "army":          self.ai_army,
                "modified":      self.ai_modified,
                "troop_status":  tc.resolve_status(self.reserve_condition[AI]),
                "troop_condition": tc.describe(self.reserve_condition[AI]),
                "initial_troop_status": self.ai_troop_status,
                "available_garrisons": self._available_garrisons(AI),
                "grux_balance":  self.grux_balance[AI],
                "army_cost":     self.ai_army_cost,
                "available_mine_slots": self._available_mine_slots(AI),
                "mine_income": self._mine_income_for(AI),
                "movement": self.movement_system.export_entity_state(AI),
                "abilities": ab.states_payload(
                    self.ability_states[AI], self.game_map.turn, self.grux_balance[AI]
                ),
                "black_market": self.black_market[AI].to_dict(
                    self.game_map.turn, self._black_market_open(AI)
                ),
                "unit_costs":    {unit_id: self.unit_costs[unit_id] for unit_id in self.ai_units},
                # [CARAVAN] Rinforzi in marcia, esposti finché viaggiano: sono
                # truppe dirette all'avversario del giocatore, e devono restare
                # sotto gli occhi invece di affidarsi a un avviso di passaggio.
                "convoys":       self.ai_convoys.to_payload(self.game_map.turn),
            },
            "movement": self.movement_system.export_config(),
            "map":        self.game_map.to_dict(),
            "battle_log": self.battle_log,
            # [EVENT-CHANNEL] Cosa è successo e dove, per gli effetti del
            # frontend. Lista a id crescente: chi la consuma tiene da parte
            # l'ultimo id già mostrato e riproduce solo i più recenti.
            "events": self.event_log.to_list() if self.event_log is not None else [],
            # [ENDGAME-STATS] Numeri per la schermata di fine partita. Il tempo
            # sta qui e non nel browser perché deve sopravvivere a un ricarico
            # della pagina: la sessione vive nel server, la scheda no.
            "stats": {
                "elapsed_seconds": max(0, int(time.time() - self.started_at)),
                "troops_lost": {
                    "player": int(self.troops_lost.get(PLAYER, 0)),
                    "ai": int(self.troops_lost.get(AI, 0)),
                },
            },
            # [DEBUG-MODULE] Letto dal pannello debug per mostrare lo stato dello
            # switch. Rimuovibile con il modulo: nessun'altra parte lo consuma.
            "debug": {
                "ai_kill_switch_active": self.debug_ai_kill_switch,
            },
        }

"""
War Advisor - FastAPI Backend
API per il sistema di raccomandazione strategica per wargame
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import os
import sys
from typing import List, Dict, Any, Optional

from engine import (
    load_data,
    aggregate_army,
    apply_modifiers,
    compute_ranking,
    get_available_units,
    get_available_terrains,
    get_available_weather,
    get_available_troop_status
)

from gamecore.session import GameSession, SessionState
from gamecore.gameflow import start_game_session
from gamecore.economy import STARTING_GRUX, calculate_army_cost, get_unit_costs

if getattr(sys, 'frozen', False):
    APP_BASE_DIR = sys._MEIPASS
else:
    APP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRONTEND_DIR = os.path.join(APP_BASE_DIR, "frontend")

# Inizializza FastAPI
app = FastAPI(
    title="War Advisor API",
    description="Sistema di raccomandazione strategica per wargame",
    version="1.0.0"
)

# CORS - Consenti richieste dal frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def disable_static_cache(request: Request, call_next):
    """Impedisce al browser di tenere in cache frontend e pagine di gioco.

    Senza questo il browser serviva JS/CSS vecchi dopo ogni modifica e serviva
    aprire una scheda in incognito (o incrementare a mano un `?v=` negli URL)
    per vedere i cambiamenti.
    """
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static") or path in ("/", "/battle"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        # StaticFiles usa ETag/Last-Modified per rispondere 304: vanno tolti,
        # altrimenti il browser riuserebbe comunque la copia locale.
        # (MutableHeaders non ha .pop(): serve del con controllo di presenza.)
        for header in ("etag", "last-modified"):
            if header in response.headers:
                del response.headers[header]
    return response


# Monta la cartella statica per il frontend
app.mount("/static", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

# [AUDIO-MODULE] I suoni stanno in assets/, fuori da frontend/: senza questo
# mount il browser non li vedrebbe. Nel pacchetto .exe serve anche
# --add-data "assets;assets", se no la cartella non finisce nel bundle.
ASSETS_DIR = os.path.join(APP_BASE_DIR, "assets")
if os.path.isdir(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

# Carica i dati all'avvio
try:
    DATA = load_data()
except Exception as e:
    raise RuntimeError(f"Errore nel caricamento dei dati: {e}")

UNIT_COSTS = get_unit_costs(DATA["units"])

# Sessione di gioco attiva (una sola partita alla volta)
_active_session: Optional[GameSession] = None


# ==================== MODELLI PYDANTIC ====================

class ConfigResponse(BaseModel):
    """Risposta per l'endpoint GET /config"""
    units: List[Dict[str, Any]] = Field(..., description="Lista di unità disponibili")
    strategies: List[Dict[str, Any]] = Field(..., description="Lista di strategie disponibili")
    terrains: List[Dict[str, Any]] = Field(..., description="Lista di terreni disponibili")
    weather: List[Dict[str, Any]] = Field(..., description="Lista di condizioni meteo disponibili")
    troop_status: List[Dict[str, Any]] = Field(..., description="Lista di stati truppe disponibili")
    budget_grux: int = Field(..., description="Tesoreria iniziale con cui si compone l'esercito")


class CalculateRequest(BaseModel):
    """Richiesta per l'endpoint POST /calculate"""
    units: List[str] = Field(..., description="Lista di ID unità selezionate", min_items=1)
    terrain: str = Field(..., description="Nome del terreno")
    weather: Optional[str] = Field(None, description="Nome della condizione meteo")
    troop_status: Optional[str] = Field(None, description="Stato delle truppe")


class StrategyResult(BaseModel):
    """Risultato di una strategia nel ranking"""
    id: str
    name: str
    description: str
    distance: float
    compatibility: float
    ideal_attributes: Dict[str, float]


class CalculateResponse(BaseModel):
    """Risposta per l'endpoint POST /calculate"""
    army_profile: Dict[str, float] = Field(..., description="Vettore dell'esercito originale")
    modified_profile: Dict[str, float] = Field(..., description="Vettore dell'esercito modificato dai modificatori")
    terrain_id: str
    terrain_name: str
    weather_name: Optional[str] = Field(None, description="Condizione meteo applicata")
    troop_status_name: Optional[str] = Field(None, description="Stato truppe applicato")
    budget_grux: int = Field(..., description="Budget iniziale disponibile")
    selected_units_cost: int = Field(..., description="Costo totale delle unità selezionate")
    remaining_grux: int = Field(..., description="Bilancio residuo in grux")
    critical_warnings: List[str] = Field(default_factory=list, description="Warning per attributi CRITICAL non soddisfatti")
    ranking: List[StrategyResult] = Field(..., description="Ranking delle strategie (da migliore a peggiore)")
    top_strategy: StrategyResult = Field(..., description="Strategia consigliata (migliore)")


# ==================== ENDPOINT ====================

@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """
    Endpoint GET /config
    Ritorna le liste di unità, terreni, meteo e stati truppe disponibili per popolare i dropdown del frontend.
    """
    try:
        units_by_id = {unit["id"]: unit for unit in DATA["units"]}
        units = [
            {
                **unit,
                "attributes": units_by_id.get(unit["id"], {}).get("attributes", unit.get("attributes", {})),
                "cost_grux": UNIT_COSTS[unit["id"]],
            }
            for unit in get_available_units(DATA)
        ]
        terrains = get_available_terrains(DATA)
        strategies = [
            {
                "id": strategy["id"],
                "name": strategy["name"],
                "description": strategy["description"],
            }
            for strategy in DATA["strategies"]
        ]
        weather = get_available_weather(DATA)
        troop_status = get_available_troop_status(DATA)
        return ConfigResponse(
            units=units,
            strategies=strategies,
            terrains=terrains,
            weather=weather,
            troop_status=troop_status,
            # La schermata iniziale non deve tenersi una copia del numero:
            # il budget lo decide l'economia e basta.
            budget_grux=STARTING_GRUX,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calculate", response_model=CalculateResponse)
async def calculate(request: CalculateRequest):
    """
    Endpoint POST /calculate
    Calcola il profilo dell'esercito e ritorna il ranking delle strategie.
    
    Input:
    - units: Lista di ID unità selezionate
    - terrain: Nome del terreno
    - weather: (Opzionale) Nome della condizione meteo
    - troop_status: (Opzionale) Stato delle truppe
    
    Output:
    - Profilo esercito originale
    - Profilo esercito modificato dai modificatori
    - Ranking completo delle strategie
    - Strategia consigliata (top 1)
    """
    try:
        # Valida l'input
        if not request.units:
            raise ValueError("Deve essere selezionata almeno un'unità")

        selected_units_cost = calculate_army_cost(request.units, UNIT_COSTS)
        if selected_units_cost > STARTING_GRUX:
            raise ValueError(
                f"Esercito troppo costoso: {selected_units_cost} grux su {STARTING_GRUX} disponibili"
            )
        
        # Calcola il vettore dell'esercito originale
        army_profile = aggregate_army(request.units, DATA["units"])
        
        # Applica i modificatori (terreno, meteo, stato truppe)
        modified_profile, critical_warnings = apply_modifiers(
            army_vector=army_profile,
            terrain_name=request.terrain,
            weather_name=request.weather,
            troop_status_name=request.troop_status,
            modifiers_data=DATA
        )
        
        # Calcola il ranking delle strategie (con affinità bidirezionali)
        ranking_data = compute_ranking(
            army_vector=modified_profile,
            strategies_list=DATA["strategies"],
            unit_ids=request.units,
            terrain_name=request.terrain,
            weather_name=request.weather,
            affinities_data=DATA.get("unit_affinities", {})
        )
        
        # Prepara il ranking come lista di StrategyResult
        ranking = [
            StrategyResult(
                id=s["id"],
                name=s["name"],
                description=s["description"],
                distance=s["distance"],
                compatibility=s["compatibility"],
                ideal_attributes=s["ideal_attributes"]
            )
            for s in ranking_data
        ]
        
        # Top strategy (migliore)
        top_strategy = ranking[0] if ranking else None
        
        return CalculateResponse(
            army_profile=army_profile,
            modified_profile=modified_profile,
            terrain_id=request.terrain.lower().replace(" ", "_"),
            terrain_name=request.terrain,
            weather_name=request.weather,
            troop_status_name=request.troop_status,
            budget_grux=STARTING_GRUX,
            selected_units_cost=selected_units_cost,
            remaining_grux=STARTING_GRUX - selected_units_cost,
            critical_warnings=critical_warnings,
            ranking=ranking,
            top_strategy=top_strategy
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== MODELLI PYDANTIC — GIOCO ====================

class ConfirmRequest(BaseModel):
    """Richiesta POST /game/confirm — il giocatore ha scelto la sua strategia."""
    units: List[str]             = Field(..., description="ID unità del giocatore", min_items=1)
    terrain: str                 = Field(..., description="Terreno scelto")
    weather: Optional[str]       = Field(None, description="Condizione meteo")
    troop_status: Optional[str]  = Field(None, description="Stato truppe del giocatore")
    strategy_id: str             = Field(..., description="ID strategia confermata dal giocatore")
    army_profile: Dict[str, float]      = Field(..., description="Vettore esercito grezzo (da /calculate)")
    modified_profile: Dict[str, float]  = Field(..., description="Vettore esercito modificato (da /calculate)")
    map_seed: Optional[int]      = Field(None, description="Seed opzionale per la mappa (None = casuale)")
    ai_difficulty: Optional[str] = Field(
        None,
        description="ID difficoltà IA all'avvio (easy, normal, hard, nightmare). "
                    "Determina anche come l'IA costruisce il proprio esercito iniziale.",
    )


class GarrisonRequest(BaseModel):
    """Richiesta per piazzare presidio con una legione (che deve avere almeno 2 truppe)."""
    legion_id: str = Field(..., description="ID della legione che lascia il presidio")
    unit_id: Optional[str] = Field(None, description="ID unità da distaccare")


class LegionBuildRequest(BaseModel):
    """Richiesta per costruire (miniera/fortificazione) con una legione specifica."""
    legion_id: str = Field(..., description="ID della legione che esegue la costruzione")
    target: Optional[tuple[int, int]] = Field(
        None,
        description="Cella [row, col] su cui costruire. Assente = la cella della legione. "
                    "Una cella diversa richiede l'abilità Costruzione Territoriale.",
    )


class RecruitRequest(BaseModel):
    """Richiesta per reclutare una unità."""
    unit_id: str = Field(..., description="ID unità da comprare")


class AutoRecruitRequest(BaseModel):
    """Richiesta per avviare autoreclutamento player."""
    unit_id: str = Field(..., description="ID unità da autoreclutare")
    turns: int = Field(..., description="Numero turni piano autoreclutamento", ge=1, le=40)


class StrategyChangeRequest(BaseModel):
    """Richiesta per cambiare strategia del player durante la battaglia."""
    strategy_id: str = Field(..., description="ID strategia da impostare")


class LegionStrategyRequest(BaseModel):
    """Richiesta per assegnare una strategia a una singola legione."""
    legion_id: str   = Field(..., description="ID della legione da aggiornare")
    strategy_id: str = Field(..., description="ID strategia da impostare su quella legione")


class AIDifficultyRequest(BaseModel):
    """Richiesta per cambiare la difficoltà runtime dell'IA."""
    difficulty: str = Field(..., description="ID difficoltà IA (es: easy, normal)")


class AbilityResearchRequest(BaseModel):
    """Richiesta per avviare ricerca di una abilità specifica."""
    ability_id: str = Field(..., description="ID abilità da ricercare")


class BlackMarketBuyRequest(BaseModel):
    """Richiesta per comprare un blocco di truppe al Mercato Nero."""
    offer_id: str = Field(..., description="ID dell'offerta esposta al banco")


class CreateLegionRequest(BaseModel):
    """Richiesta per creare una nuova legione del player."""
    name: str = Field(..., description="Nome della legione")
    units: Dict[str, int] = Field(..., description="Dizionario di id_unità -> quantità da prelevare")
    target: Optional[tuple[int, int]] = Field(None, description="Destinazione opzionale [row, col]")
    capture_area: Optional[List[tuple[int, int]]] = Field(
        None,
        description="Caselle da conquistare in sequenza. Alternativa a `target`: "
                    "o l'una o l'altra, mai entrambe.",
    )
    legion_type: str = Field(
        "army", description="Tipo legione: 'army' (Esercito), 'mining' (Mineraria) o 'construction' (Costruzione)"
    )


class RecallLegionRequest(BaseModel):
    """Richiesta per richiamare una legione del player in riserva."""
    legion_id: str = Field(..., description="ID della legione da richiamare")


class RetargetLegionRequest(BaseModel):
    """Richiesta per assegnare una nuova destinazione a una legione del player."""
    legion_id: str = Field(..., description="ID della legione da ridirigere")
    target: tuple[int, int] = Field(..., description="Nuova destinazione [row, col]")


class LegionCaptureAreaRequest(BaseModel):
    """Richiesta per (ri)assegnare un ordine di cattura d'area a una legione."""
    legion_id: str = Field(..., description="ID della legione")
    capture_area: List[tuple[int, int]] = Field(..., description="Caselle da conquistare")



# ==================== ENDPOINT GIOCO ====================

@app.post("/game/confirm")
async def game_confirm(request: ConfirmRequest):
    """
    POST /game/confirm

    Chiamato quando il giocatore ha confermato la sua strategia nel frontend.

    Azioni:
      1. Valida il terreno.
      2. Sorteggia le condizioni di partenza e costruisce l'esercito dell'IA.
      3. Crea la GameSession con mappa procedurale.
      4. Ritorna lo stato iniziale della partita (mappa + info eserciti).

    [SETUP-RULE] Terreno, meteo e stato truppe della richiesta descrivono la
    SIMULAZIONE fatta nella schermata iniziale: non sono condizioni che il
    giocatore possa portarsi in partita. Il meteo si sorteggia, le truppe
    partono fresche e il terreno di casa è quello della cella del castello.
    """
    global _active_session

    try:
        started = start_game_session(
            data=DATA,
            player_units=request.units,
            terrain=request.terrain,
            weather=request.weather,
            troop_status=request.troop_status,
            strategy_id=request.strategy_id,
            army_profile=request.army_profile,
            modified_profile=request.modified_profile,
            map_seed=request.map_seed,
            ai_difficulty=request.ai_difficulty,
        )

        _active_session = started["session"]
        session_dict = _active_session.to_dict()
        session_dict["message"] = started["message"]
        # Il seed effettivo, anche quando la richiesta non ne portava uno:
        # senza, una partita interessante non si può più rigiocare.
        session_dict["map_seed"] = started["map_seed"]
        return session_dict

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/execute-turn")
async def game_execute_turn():
    """Esegue l'avanzamento del turno per il sistema a legioni."""
    if _active_session is None:
        raise HTTPException(status_code=400, detail="Nessuna partita attiva.")

    try:
        return _active_session.execute_turn()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/legions/create")
async def game_create_legion(request: CreateLegionRequest):
    """Crea una nuova legione dal castello del player."""
    if _active_session is None:
        raise HTTPException(status_code=400, detail="Nessuna partita attiva.")

    try:
        return _active_session.create_player_legion(
            name=request.name,
            units_dict=request.units,
            target=request.target,
            legion_type=request.legion_type,
            capture_area=request.capture_area,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/legions/recall")
async def game_recall_legion(request: RecallLegionRequest):
    """Richiama una legione del player: le unità tornano subito in riserva."""
    if _active_session is None:
        raise HTTPException(status_code=400, detail="Nessuna partita attiva.")

    try:
        return _active_session.recall_player_legion(request.legion_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/legions/retarget")
async def game_retarget_legion(request: RetargetLegionRequest):
    """Assegna una nuova destinazione a una legione del player già in campo."""
    if _active_session is None:
        raise HTTPException(status_code=400, detail="Nessuna partita attiva.")

    try:
        return _active_session.retarget_player_legion(request.legion_id, request.target)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/legions/capture-area")
async def game_legion_capture_area(request: LegionCaptureAreaRequest):
    """Assegna un nuovo ordine di cattura d'area a una legione del player."""
    if _active_session is None:
        raise HTTPException(status_code=400, detail="Nessuna partita attiva.")

    try:
        return _active_session.set_legion_capture_area(
            legion_id=request.legion_id,
            cells=request.capture_area,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/game/state")
async def game_state():
    """
    GET /game/state

    Ritorna lo stato completo della sessione di gioco corrente
    (mappa, eserciti, log battaglie, turno, ecc.).
    """
    if _active_session is None:
        raise HTTPException(status_code=404, detail="Nessuna partita attiva.")
    return _active_session.to_dict()


@app.get("/game/in-game-advisor")
async def game_in_game_advisor():
    """Ritorna il report strategico in-battle con affidabilità intenzionalmente limitata."""
    if _active_session is None:
        raise HTTPException(status_code=404, detail="Nessuna partita attiva.")

    try:
        return _active_session.get_in_game_advisor()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/game/enemy-intel")
async def game_enemy_intel():
    """[ABILITY-EFFECTS] Dossier sull'IA: aperto solo dall'Industria dello Spionaggio.

    Senza la ricerca risponde comunque 200, con `available: false`: è la UI a
    decidere cosa mostrare, e un 403 avrebbe costretto il frontend a trattare
    come errore una condizione normalissima di inizio partita.
    """
    if _active_session is None:
        raise HTTPException(status_code=404, detail="Nessuna partita attiva.")

    try:
        return _active_session.get_enemy_intel()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/place-mine")
async def game_place_mine(request: LegionBuildRequest):
    """Piazza una miniera con una legione Mineraria, sulla cella dove si trova."""
    if _active_session is None:
        raise HTTPException(status_code=400, detail="Nessuna partita attiva.")

    try:
        return _active_session.place_mine(request.legion_id, request.target)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/place-garrison-here")
async def game_place_garrison_here(request: GarrisonRequest):
    """Stacca una truppa dalla legione indicata (min. 2 truppe) per presidiare la sua cella attuale."""
    if _active_session is None:
        raise HTTPException(status_code=400, detail="Nessuna partita attiva.")

    try:
        return _active_session.place_garrison(request.legion_id, unit_id=request.unit_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/place-fortification")
async def game_place_fortification(request: LegionBuildRequest):
    """Piazza una fortificazione con una legione Costruzione, sulla cella dove si trova."""
    if _active_session is None:
        raise HTTPException(status_code=400, detail="Nessuna partita attiva.")

    try:
        return _active_session.place_fortification(request.legion_id, request.target)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/recruit")
async def game_recruit(request: RecruitRequest):
    """Compra una nuova unità spendendo grux dalla tesoreria globale."""
    if _active_session is None:
        raise HTTPException(status_code=400, detail="Nessuna partita attiva.")

    try:
        return _active_session.recruit_player_unit(request.unit_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/auto-recruit/start")
async def game_auto_recruit_start(request: AutoRecruitRequest):
    """Avvia un piano di autoreclutamento per il player."""
    if _active_session is None:
        raise HTTPException(status_code=400, detail="Nessuna partita attiva.")

    try:
        return _active_session.start_player_auto_recruit(request.unit_id, request.turns)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/auto-recruit/stop")
async def game_auto_recruit_stop():
    """Ferma il piano di autoreclutamento del player."""
    if _active_session is None:
        raise HTTPException(status_code=400, detail="Nessuna partita attiva.")

    try:
        return _active_session.stop_player_auto_recruit(reason="manual")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/research-ability")
async def game_research_ability(request: AbilityResearchRequest):
    """Avvia la ricerca abilità del player."""
    if _active_session is None:
        raise HTTPException(status_code=400, detail="Nessuna partita attiva.")

    try:
        return _active_session.research_player_ability(request.ability_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/black-market/buy")
async def game_black_market_buy(request: BlackMarketBuyRequest):
    """Compra un blocco di truppe al Mercato Nero."""
    if _active_session is None:
        raise HTTPException(status_code=400, detail="Nessuna partita attiva.")

    try:
        return _active_session.buy_black_market_offer(request.offer_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/set-strategy")
async def game_set_strategy(request: StrategyChangeRequest):
    """Aggiorna la strategia attiva del player durante la partita."""
    if _active_session is None:
        raise HTTPException(status_code=400, detail="Nessuna partita attiva.")

    try:
        return _active_session.set_player_strategy(request.strategy_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/legions/set-strategy")
async def game_set_legion_strategy(request: LegionStrategyRequest):
    """Assegna una strategia a una singola legione: ognuna combatte con la propria."""
    if _active_session is None:
        raise HTTPException(status_code=400, detail="Nessuna partita attiva.")

    try:
        return _active_session.set_legion_strategy(request.legion_id, request.strategy_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/set-ai-difficulty")
async def game_set_ai_difficulty(request: AIDifficultyRequest):
    """Aggiorna la difficoltà runtime dell'IA per la sessione attiva."""
    if _active_session is None:
        raise HTTPException(status_code=400, detail="Nessuna partita attiva.")

    try:
        return _active_session.set_ai_difficulty(request.difficulty)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── [DEBUG-MODULE] Aggancio del modulo debug ───────────────────────
# Unico punto di contatto lato server: registra le rotte /game/debug/*.
# Il try/except rende la cartella cancellabile senza toccare questo file
# (il server parte comunque, il modulo semplicemente non si registra).
# Istruzioni complete: gamecore/debug_module/README.md
try:
    from gamecore.debug_module import build_debug_router

    app.include_router(build_debug_router(lambda: _active_session))
except ImportError:
    pass
# ── [DEBUG-MODULE] fine aggancio ───────────────────────────────────


@app.delete("/game/reset")
async def game_reset():
    """
    DELETE /game/reset

    Termina la sessione corrente e la azzera,
    permettendo di iniziare una nuova partita.
    """
    global _active_session
    _active_session = None
    return {"message": "Sessione azzerata. Puoi iniziare una nuova partita."}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Radice: serve il file index.html direttamente"""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/battle", response_class=HTMLResponse)
async def battle_page():
    """Serve la pagina dedicata alla battaglia."""
    battle_path = os.path.join(FRONTEND_DIR, "battle.html")

    with open(battle_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

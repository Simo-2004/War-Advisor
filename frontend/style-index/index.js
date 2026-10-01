/* ══════════════════════════════════════════════════════════════════
   War Advisor — Simulatore (pagina iniziale)

   Stessa logica di prima, spostata fuori da index.html: carica la config,
   raccoglie la scelta del giocatore, chiede il calcolo al backend, disegna
   il verdetto e prepara il passaggio alla battaglia.

   Le uniche aggiunte sono di presentazione: lo stato della pagina (setup /
   caricamento / pronto) che pilota testata e segnaposto, la barra della
   tesoreria e i numeri che salgono invece di comparire.
   ══════════════════════════════════════════════════════════════════ */

const ATTRIBUTE_NAMES = {
    'U1_attack': 'Attacco',
    'U2_defense': 'Difesa',
    'U3_mobility': 'Mobilità',
    'U4_stealth': 'Furtività',
    'U5_discipline': 'Disciplina',
    'U6_terrain_adapt': 'Adatt. Terreno',
    'U7_range_power': 'Potenza a Distanza',
    'U8_support': 'Supporto'
};

const API = 'http://127.0.0.1:8000';

let radarChart = null;

/* Quante ne prendo di ciascun tipo: `{ archers: 2, pikemen: 1 }`.
   Prima si poteva prendere una sola truppa per tipo, quindi bastava una
   casella accesa o spenta. Adesso si sceglie la quantità, e `selectedUnits`
   diventa la lista distesa — `['archers', 'archers', 'pikemen']` — che è
   la forma che il backend si aspetta da sempre. */
let quantita = {};
let selectedUnits = [];
let allUnits = [];
let allTerrains = [];
let allWeather = [];
let allTroopStatus = [];
let latestCalculation = null;
/* Il budget arriva da /config: il numero vero sta nell'economia, non qui.
   Questo è solo il valore da mostrare finché la risposta non torna. */
let startingBudget = 0;

window.addEventListener('DOMContentLoaded', init);

async function init() {
    document.getElementById('terrainSelect').addEventListener('change', updateButtonState);
    document.getElementById('weatherSelect').addEventListener('change', updateButtonState);
    document.getElementById('statusSelect').addEventListener('change', updateButtonState);
    await loadData();
    updateSelectedUnits();
}

// ── Caricamento dati ────────────────────────────────────────────────

async function loadData() {
    try {
        const response = await fetch(`${API}/config`);
        if (!response.ok) throw new Error('Errore nel caricamento dei dati');

        const data = await response.json();
        allUnits = data.units;
        if (Number.isFinite(data.budget_grux)) startingBudget = data.budget_grux;
        allTerrains = data.terrains;
        allWeather = data.weather;
        allTroopStatus = data.troop_status;

        populateUnitsSelector();
        fillSelect('terrainSelect', allTerrains, '-- Seleziona un terreno --');
        fillSelect('weatherSelect', allWeather, '-- Nessuna --');
        fillSelect('statusSelect', allTroopStatus, '-- Nessuno --');
    } catch (error) {
        showError('Errore nel caricamento dei dati: ' + error.message);
    }
}

function populateUnitsSelector() {
    const container = document.getElementById('unitsContainer');
    container.innerHTML = '';

    allUnits.forEach(unit => {
        // Non più un <label> con la casella: adesso di ogni tipo si sceglie
        // quante prenderne. Il corpo della carta resta cliccabile e vale +1,
        // che è il gesto che si fa nove volte su dieci; i due bottoni servono
        // per correggere.
        const card = document.createElement('div');
        card.className = 'unit-card';
        card.dataset.unit = unit.id;
        card.innerHTML = `
            <div class="unit-info">
                <div class="unit-name">${unit.name}</div>
                <div class="unit-desc">${unit.description}</div>
            </div>
            <div class="unit-cost">${unit.cost_grux} grux</div>
            <div class="unit-stepper">
                <button type="button" class="step step-less"
                        aria-label="Una ${unit.name} in meno">−</button>
                <span class="step-count">0</span>
                <button type="button" class="step step-more"
                        aria-label="Una ${unit.name} in più">+</button>
            </div>
        `;

        card.querySelector('.step-less').addEventListener('click', event => {
            event.stopPropagation();
            cambiaQuantita(unit.id, -1);
        });
        card.querySelector('.step-more').addEventListener('click', event => {
            event.stopPropagation();
            cambiaQuantita(unit.id, +1);
        });
        card.addEventListener('click', () => cambiaQuantita(unit.id, +1));

        container.appendChild(card);
    });
}

/** Aggiunge o toglie una truppa di quel tipo, senza mai sforare il budget. */
function cambiaQuantita(unitId, delta) {
    const unit = allUnits.find(item => item.id === unitId);
    if (!unit) return;

    const adesso = quantita[unitId] || 0;
    const dopo = adesso + delta;
    if (dopo < 0) return;
    // Il tetto è la tesoreria e nient'altro: nessun limite inventato per
    // tipo. Chi vuole cinque arcieri e nient'altro deve poterlo fare.
    if (delta > 0 && getSelectedUnitsCost() + unit.cost_grux > startingBudget) {
        lampeggiaTesoreria();
        return;
    }

    if (dopo === 0) delete quantita[unitId];
    else quantita[unitId] = dopo;

    updateSelectedUnits();
}

/** Il budget dice di no: la barra fa un guizzo invece di restare muta. */
function lampeggiaTesoreria() {
    const box = document.getElementById('budgetBox');
    box.classList.remove('is-negato');
    void box.offsetWidth;                       // riavvia l'animazione
    box.classList.add('is-negato');
}

function fillSelect(id, items, placeholder) {
    const select = document.getElementById(id);
    select.innerHTML = `<option value="">${placeholder}</option>`;
    items.forEach(item => {
        const option = document.createElement('option');
        option.value = item.name;
        option.textContent = item.name;
        select.appendChild(option);
    });
}

// ── Selezione e tesoreria ───────────────────────────────────────────

function updateSelectedUnits() {
    // La lista distesa: una voce per ogni soldato, ripetuta quante volte
    // serve. È la forma che il backend legge, e conta anche le ripetizioni
    // nella media dell'esercito.
    selectedUnits = [];
    allUnits.forEach(unit => {
        const quante = quantita[unit.id] || 0;
        for (let i = 0; i < quante; i++) selectedUnits.push(unit.id);
    });

    const spesa = getSelectedUnitsCost();
    const residuo = startingBudget - spesa;

    document.querySelectorAll('#unitsContainer .unit-card').forEach(card => {
        const id = card.dataset.unit;
        const quante = quantita[id] || 0;
        const unit = allUnits.find(item => item.id === id);
        const costo = unit ? unit.cost_grux : 0;

        card.classList.toggle('is-picked', quante > 0);
        card.querySelector('.step-count').textContent = quante;
        card.querySelector('.step-less').disabled = quante === 0;
        // Il "+" si spegne quando quella truppa non ci sta più nel budget:
        // meglio un bottone visibilmente morto che un clic che non fa nulla.
        card.querySelector('.step-more').disabled = costo > residuo;
    });

    const box = document.getElementById('selectedUnitsBox');
    const tipi = Object.keys(quantita).length;
    document.getElementById('selectedCount').textContent = tipi === 0
        ? 'Unità selezionate: 0'
        : `Unità selezionate: ${selectedUnits.length} su ${tipi} ` +
          `tip${tipi === 1 ? 'o' : 'i'}`;
    renderSelectedChips();
    box.classList.toggle('active', selectedUnits.length > 0);

    const budget = document.getElementById('budgetBox');
    document.getElementById('budgetFigures').innerHTML =
        `<strong>${residuo}</strong> / ${startingBudget} grux · spesi ${spesa}`;
    // Oltre budget la barra resta piena: serve a far vedere che è finita,
    // non a misurare di quanto si è sforato.
    document.getElementById('budgetFill').style.width =
        `${Math.min(100, (spesa / Math.max(1, startingBudget)) * 100)}%`;
    budget.classList.toggle('over-budget', residuo < 0);

    const hint = document.getElementById('budgetHint');
    if (hint) hint.textContent = `${startingBudget} grux di budget`;

    updateButtonState();
}

/** Chi hai arruolato, in chiaro: "3× Arcieri · 2× Picchieri". */
function renderSelectedChips() {
    const zona = document.getElementById('selectedChips');
    if (!zona) return;
    zona.innerHTML = '';

    allUnits.forEach(unit => {
        const quante = quantita[unit.id] || 0;
        if (quante === 0) return;
        const chip = document.createElement('span');
        chip.className = 'troop-chip';
        chip.innerHTML = `<b>${quante}×</b> ${unit.name}`;
        chip.title = `${quante * unit.cost_grux} grux`;
        zona.appendChild(chip);
    });
}

function getSelectedUnitsCost() {
    return allUnits.reduce((total, unit) => {
        return total + (quantita[unit.id] || 0) * unit.cost_grux;
    }, 0);
}

function updateButtonState() {
    const terrain = document.getElementById('terrainSelect').value;
    const pronto = selectedUnits.length > 0
        && terrain !== ''
        && getSelectedUnitsCost() <= startingBudget;

    document.getElementById('calculateBtn').disabled = !pronto;
    updatePhase(pronto ? 'field' : 'army');
}

/** Accende i passi nella testata: dove sei e cosa hai già fatto. */
function updatePhase(phase) {
    if (phase !== 'done' && latestCalculation) phase = 'done';

    const stato = {
        army:  ['is-active', '', ''],
        field: ['is-done', 'is-active', ''],
        done:  ['is-done', 'is-done', 'is-active'],
    }[phase] || ['is-active', '', ''];

    document.querySelectorAll('.hero-steps li').forEach((li, i) => {
        li.className = stato[i] || '';
    });
}

// ── Calcolo ─────────────────────────────────────────────────────────

async function calculateStrategy() {
    const errorBox = document.getElementById('errorBox');
    errorBox.classList.remove('show');
    document.getElementById('loading').classList.add('show');
    document.getElementById('resultsArea').classList.remove('active');

    const terrain = document.getElementById('terrainSelect').value;
    const weather = document.getElementById('weatherSelect').value;
    const troop_status = document.getElementById('statusSelect').value;

    try {
        const response = await fetch(`${API}/calculate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                units: selectedUnits,
                terrain: terrain,
                weather: weather || null,
                troop_status: troop_status || null
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Errore nella richiesta');
        }

        const data = await response.json();
        latestCalculation = data;
        renderResults(data);
        saveSimulationForBattle(data, terrain, weather || null, troop_status || null);
    } catch (error) {
        showError(error.message);
    } finally {
        document.getElementById('loading').classList.remove('show');
    }
}

function renderResults(data) {
    const top = data.top_strategy;
    document.getElementById('topStrategyName').textContent = top.name;
    countUp('topCompatibility', top.compatibility);
    document.getElementById('topDistance').textContent = `Distanza ${top.distance.toFixed(4)}`;
    document.getElementById('topDescription').textContent = top.description || '';

    if (data.ranking && data.ranking.length >= 2) {
        const second = data.ranking[1];
        document.getElementById('secondStrategyName').textContent = second.name;
        countUp('secondCompatibility', second.compatibility);
        document.getElementById('secondDistance').textContent = `Distanza ${second.distance.toFixed(4)}`;
        document.getElementById('secondDescription').textContent = second.description || '';
    }

    if (data.ranking && data.ranking.length > 0) {
        const worst = data.ranking[data.ranking.length - 1];
        document.getElementById('worstStrategyName').textContent = worst.name;
        document.getElementById('worstCompatibility').textContent = `${worst.compatibility.toFixed(1)}%`;
        document.getElementById('worstDistance').textContent = `Distanza ${worst.distance.toFixed(4)}`;
        document.getElementById('worstDescription').textContent = worst.description || '';
    }

    const warnings = document.getElementById('warningsArea');
    if (data.critical_warnings && data.critical_warnings.length > 0) {
        warnings.innerHTML = '<strong>⚠️ Avvisi CRITICAL:</strong><br>'
            + data.critical_warnings.join('<br>');
        warnings.style.display = 'block';
    } else {
        warnings.style.display = 'none';
    }

    updateChart(data.army_profile, data.modified_profile, data.top_strategy.ideal_attributes);

    document.getElementById('resultsArea').classList.add('active');
    document.getElementById('launchArea').classList.add('active');
    // [SETUP-RULE] Terreno, meteo e stato truppe qui sopra servono a simulare
    // lo scenario: in partita non si scelgono, altrimenti basterebbe
    // selezionare le condizioni più comode. Va detto qui, dove si preme il
    // bottone.
    document.getElementById('launchSummary').innerHTML =
        `Strategia consigliata: <strong>${top.name}</strong>`
        + ` · spesa ${data.selected_units_cost} grux, residuo ${data.remaining_grux}.`
        + `<br><span class="launch-note">In battaglia porti solo l'esercito:`
        + ` il meteo viene sorteggiato e le truppe partono sempre fresche.</span>`;
    updatePhase('done');
}

/** Percentuale che sale invece di comparire: il verdetto si fa guardare.
 *
 *  Il valore giusto viene scritto SUBITO, prima di animare: se i frame non
 *  arrivano — scheda in secondo piano, animazioni disattivate dal sistema —
 *  resta a schermo il numero vero e non uno zero. Un'animazione può saltare,
 *  un dato sbagliato no.
 */
function countUp(id, value) {
    const el = document.getElementById(id);
    const finale = `${value.toFixed(1)}%`;
    el.textContent = finale;

    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    const durata = 700;
    const inizio = performance.now();
    // Il primo passo va chiesto a rAF, non eseguito qui: chiamandolo subito
    // scriverebbe comunque lo zero iniziale sopra il valore buono, e senza
    // fotogrammi successivi resterebbe quello.
    requestAnimationFrame(function step(ora) {
        const t = Math.min(1, (ora - inizio) / durata);
        el.textContent = t < 1
            ? `${(value * (1 - Math.pow(1 - t, 3))).toFixed(1)}%`
            : finale;
        if (t < 1) requestAnimationFrame(step);
    });

    // Rete di sicurezza: se i frame si fermano a metà corsa, il numero
    // rimarrebbe congelato a un valore parziale.
    setTimeout(() => { el.textContent = finale; }, durata + 150);
}

// ── Passaggio alla battaglia ────────────────────────────────────────

function saveSimulationForBattle(calculation, terrain, weather, troopStatus) {
    const payload = {
        units: [...selectedUnits],
        terrain: terrain,
        weather: weather,
        troop_status: troopStatus,
        strategy_id: calculation.top_strategy.id,
        army_profile: calculation.army_profile,
        modified_profile: calculation.modified_profile,
        map_seed: null,
        summary: {
            top_strategy_name: calculation.top_strategy.name,
            top_strategy_compatibility: calculation.top_strategy.compatibility,
            budget_grux: calculation.budget_grux,
            selected_units_cost: calculation.selected_units_cost,
            remaining_grux: calculation.remaining_grux
        }
    };
    sessionStorage.setItem('warAdvisorBattleSetup', JSON.stringify(payload));
}

function goToBattlePage() {
    if (!latestCalculation) {
        showError('Prima esegui una simulazione strategica.');
        return;
    }
    // Percorso relativo, non l'indirizzo assoluto del backend: aprendo la
    // pagina da `localhost` il salto a `127.0.0.1` cambiava origine e il
    // sessionStorage con la simulazione restava indietro, così la battaglia
    // si apriva vuota. Le chiamate API continuano ad andare a `API`.
    window.location.href = '/battle';
}

// ── Grafico ed errori ───────────────────────────────────────────────

function updateChart(originalArmy, modifiedArmy, topStrategy) {
    const ctx = document.getElementById('radarChart').getContext('2d');
    const keys = Object.keys(originalArmy);

    if (radarChart) radarChart.destroy();

    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: keys.map(k => ATTRIBUTE_NAMES[k] || k),
            datasets: [
                {
                    label: 'Il Tuo Esercito',
                    data: keys.map(k => originalArmy[k]),
                    backgroundColor: 'rgba(233, 69, 96, 0.15)',
                    borderColor: '#e94560',
                    pointBackgroundColor: '#e94560',
                    borderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: 'Con Modificatori Terreno',
                    data: keys.map(k => modifiedArmy[k]),
                    backgroundColor: 'rgba(125, 95, 212, 0.15)',
                    borderColor: '#7d5fd4',
                    pointBackgroundColor: '#7d5fd4',
                    borderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: 'Strategia Ideale',
                    data: keys.map(k => topStrategy[k]),
                    backgroundColor: 'rgba(22, 163, 74, 0.15)',
                    borderColor: '#16a34a',
                    pointBackgroundColor: '#16a34a',
                    borderDash: [5, 5],
                    borderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { font: { size: 11, weight: '600' }, padding: 14, boxWidth: 14 }
                }
            },
            scales: {
                r: {
                    min: 0,
                    max: 1,
                    ticks: { stepSize: 0.2, font: { size: 10 } },
                    pointLabels: { font: { size: 11, weight: '500' } },
                    angleLines: { color: 'rgba(0,0,0,0.1)' },
                    grid: { color: 'rgba(0,0,0,0.05)' }
                }
            }
        }
    });
}

function showError(msg) {
    const box = document.getElementById('errorBox');
    box.textContent = '❌ ' + msg;
    box.classList.add('show');
    document.getElementById('loading').classList.remove('show');
}

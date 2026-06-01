const canvas = document.getElementById('simCanvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const statsEl = document.getElementById('stats');
const networkCanvas = document.getElementById('networkCanvas');
const networkCtx = networkCanvas.getContext('2d');
const toggleButton = document.getElementById('toggleButton');
const stepButton = document.getElementById('stepButton');
const saveButton = document.getElementById('saveButton');
const loadButton = document.getElementById('loadButton');
const chartCtx = document.getElementById('populationChart').getContext('2d');
const killButton = document.getElementById('killButton');

let running = true;
let lastState = null;
let lastTickTime = performance.now();
let neuralNetPulsesEnabled = true;

//charts
const populationChart = new Chart(chartCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Plants',
                data: [],
            },
            {
                label: 'Herbivores',
                data: [],
            },
            {
                label: 'Predators',
                data: [],
            }
        ]
    },
    options: {
        responsive: true,
        animation: false,
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'World Time'
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Population'
                },
                beginAtZero: true
            }
        }
    }
});

// --- DOM ELEMENTS FOR SETTINGS ---
const simView = document.getElementById('simView');
const settingsView = document.getElementById('settingsView');
const settingsButton = document.getElementById('settingsButton');
const backButton = document.getElementById('backButton');
const restartSettingsButton = document.getElementById('restartSettingsButton');
const inputsConfig = [
    // Globals Setup
    { input: 'worldSpeedInput', val: 'worldSpeedValue', isFloat: true, fixed: 2 },
    { input: 'maxSpeedInput', val: 'maxSpeedValue', isFloat: false },
    { input: 'maxAngularVelocityInput', val: 'maxAngularVelocityValue', isFloat: true, fixed: 1 },
    { input: 'globalMutationRateInput', val: 'globalMutationRateValue', isFloat: true, fixed: 3 },
    { input: 'globalMutationStrengthInput', val: 'globalMutationStrengthValue', isFloat: true, fixed: 2 },
    { input: 'weightStdNewNeuronsInput', val: 'weightStdNewNeuronsValue', isFloat: true, fixed: 2 },
    { input: 'startingHerbivoreInput', val: 'startingHerbivoreValue', isFloat: false },
    { input: 'startingPredatorInput', val: 'startingPredatorValue', isFloat: false },
    { input: 'startingPlantInput', val: 'startingPlantValue', isFloat: false },
    
    // Plant Params
    { input: 'maxPlantInput', val: 'maxPlantValue', isFloat: false },
    { input: 'plantNutritionValueInput', val: 'plantNutritionValueValue', isFloat: true, fixed: 2 },
    { input: 'plantRegrowthPowerInput', val: 'plantRegrowthPowerValue', isFloat: true, fixed: 1 },
    
    // Predator Settings & Resilience
    { input: 'maxPredatorInput', val: 'maxPredatorValue', isFloat: false },
    { input: 'predatorAvgGestationInput', val: 'predatorAvgGestationValue', isFloat: true, fixed: 1 },
    { input: 'predatorGestationStdInput', val: 'predatorGestationStdValue', isFloat: true, fixed: 1 },
    { input: 'predatorMinReproductionSatietyInput', val: 'predatorMinReproductionSatietyValue', isFloat: true, fixed: 1 },
    { input: 'predatorReproductionLossInput', val: 'predatorReproductionLossValue', isFloat: true, fixed: 2 },
    { input: 'predatorEatPercentThresholdInput', val: 'predatorEatPercentThresholdValue', isFloat: false },
    { input: 'predatorFOVInput', val: 'predatorFOVValue', isFloat: true, fixed: 2 },
    { input: 'predatorVisionRangeInput', val: 'predatorVisionRangeValue', isFloat: false },
    { input: 'predatorAvgAgeInput', val: 'predatorAvgAgeValue', isFloat: true, fixed: 1 },
    { input: 'predatorAgeStdInput', val: 'predatorAgeStdValue', isFloat: true, fixed: 1 },
    { input: 'predatorMinAgeReproductionInput', val: 'predatorMinAgeReproductionValue', isFloat: true, fixed: 1 },
    { input: 'predatorsResurrectAfterHerbivoresReachInput', val: 'predatorsResurrectAfterHerbivoresReachValue', isFloat: false },
    { input: 'predatorResurrectionCountInput', val: 'predatorResurrectionCountValue', isFloat: false },
    { input: 'predatorResurrectionRecentCountInput', val: 'predatorResurrectionRecentCountValue', isFloat: false },
    { input: 'predatorResurrectionRandomCountInput', val: 'predatorResurrectionRandomCountValue', isFloat: false },
    
    // Herbivore Parameters
    { input: 'maxHerbivoreInput', val: 'maxHerbivoreValue', isFloat: false },
    { input: 'herbivoreSatietyLossInput', val: 'herbivoreSatietyLossValue', isFloat: true, fixed: 3 },
    { input: 'herbivoreMaxSatietyInput', val: 'herbivoreMaxSatietyValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreAvgGestationInput', val: 'herbivoreAvgGestationValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreGestationStdInput', val: 'herbivoreGestationStdValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreMinReproductionSatietyInput', val: 'herbivoreMinReproductionSatietyValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreReproductionLossInput', val: 'herbivoreReproductionLossValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreEatPercentThresholdInput', val: 'herbivoreEatPercentThresholdValue', isFloat: false },
    { input: 'herbivoreFOVInput', val: 'herbivoreFOVValue', isFloat: true, fixed: 2 },
    { input: 'herbivoreVisionRangeInput', val: 'herbivoreVisionRangeValue', isFloat: false },
    { input: 'herbivoreAvgAgeInput', val: 'herbivoreAvgAgeValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreAgeStdInput', val: 'herbivoreAgeStdValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreMinAgeReproductionInput', val: 'herbivoreMinAgeReproductionValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreNutritionValueInput', val: 'herbivoreNutritionValueValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreResurrectionCountInput', val: 'herbivoreResurrectionCountValue', isFloat: false },
    { input: 'herbivoreResurrectionRandomCountInput', val: 'herbivoreResurrectionRandomCountValue', isFloat: false },
    { input: 'herbivoreResurrectionRecentCountInput', val: 'herbivoreResurrectionRecentCountValue', isFloat: false }
];

inputsConfig.forEach(cfg => {
    const el = document.getElementById(cfg.input);
    const valEl = document.getElementById(cfg.val);
    if (el && valEl) {
        el.addEventListener('input', (e) => {
            const parsed = cfg.isFloat ? parseFloat(e.target.value) : parseInt(e.target.value, 10);
            valEl.textContent = cfg.isFloat ? parsed.toFixed(cfg.fixed) : parsed;
        });
    }
});

async function fetchState() {
    const response = await fetch('/state');
    return response.json();
}

async function fetchChartData() {
    const response = await fetch('/chart');
    return response.json();
}

async function stepWorld(dt) {
    const response = await fetch('/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dt }),
    });
    if (!response.ok) {
        const text = await response.text();
        throw new Error(`Step failed: ${response.status} ${text}`);
    }
    return response.json();
}

/**
 * Maps a normalised distance_angle pair into canvas-space x/y offset from animal centre.
 * norm_dist:  1 - dist/vision_range  (so 1 = right on top, 0 = at edge)
 * norm_angle: relative angle / half_fov, in [-1, 1]
 */
function visionVecToCanvas(normDist, normAngle, heading, visionRange, halfFov, scaleX, scaleY) {
    const actualDist  = (1 - normDist) * visionRange;
    const actualAngle = heading + normAngle * halfFov;
    return {
        dx: Math.cos(actualAngle) * actualDist * scaleX,
        dy: Math.sin(actualAngle) * actualDist * scaleY,
    };
}
 
/**
 * Draws the vision overlay for the selected animal using the new nn_distances_angles format.
 *
 * nn_distances_angles layout:
 *   [dist_plant, angle_plant, dist_conspecific, angle_conspecific, dist_predator, angle_predator]
 *   dist  = 1 - d/vision_range, or -1 if not detected
 *   angle = relative angle / half_fov in [-1, 1], or 0 if not detected
 *
 * Draws:
 *   1. Transparent grey FOV cone (full visible area)
 *   2. Green line  → nearest plant        (if detected)
 *   3. Blue line   → nearest conspecific  (if detected)
 *   4. Red line    → nearest predator     (if detected)
 *
 * @param {object} animal  - animal object from state
 * @param {number} scaleX  - world-to-canvas x scale
 * @param {number} scaleY  - world-to-canvas y scale
 */

function drawVisionOverlay(animal, scaleX, scaleY) {
    const data = animal.nn_distances_angles;
 
    const fov         = animal.fov;
    const visionRange = animal.vision_range;
    const heading     = animal.face_direction;
    const halfFov     = fov / 2;
 
    const cx = animal.x * scaleX;
    const cy = animal.y * scaleY;
    const visionPx = visionRange * scaleX;

    ctx.save();
 
    // ── 1. FOV cone ───────────────────────────────────────────────────────────
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, visionPx, heading - halfFov, heading + halfFov);
    ctx.closePath();
    ctx.fillStyle   = 'rgba(200, 200, 210, 0.18)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(160, 160, 180, 0.35)';
    ctx.lineWidth   = 1;
    ctx.stroke();
 
    // ── 2. Detection lines ────────────────────────────────────────────────────
    let targets;
    
    if (animal.species === 'herbivore') {
        // Herbivores: see plants (food), herbivores (conspecifics), predators
        targets = [
            { normDist: data[0], normAngle: data[1], color: '#3ecf60', label: 'food'       },
            { normDist: data[2], normAngle: data[3], color: '#5a9ff5', label: 'conspecific'  },
            { normDist: data[4], normAngle: data[5], color: '#e84545', label: 'predator'     },
        ];
    } else if (animal.species === 'predator') {
        // Predators: see herbivores (food), predators (conspecifics), no predators of predators
        targets = [
            { normDist: data[0], normAngle: data[1], color: '#3ecf60', label: 'food'       },  // herbivores as food
            { normDist: data[2], normAngle: data[3], color: '#5a9ff5', label: 'conspecific'  },  // other predators
            // No third target for predators since nothing eats them
        ];
    }
 
    targets.forEach(({ normDist, normAngle, color }) => {
        if (normDist <= 0.0001) return; // nothing detected for this type
 
        const { dx, dy } = visionVecToCanvas(
            normDist, normAngle, heading, visionRange, halfFov, scaleX, scaleY
        );
 
        const tx = cx + dx;
        const ty = cy + dy;
 
        // Line from animal to detected object
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(tx, ty);
        ctx.strokeStyle = color;
        ctx.lineWidth   = 2;
        ctx.globalAlpha = 0.85;
        ctx.stroke();
 
        // Small dot at the detected position
        ctx.beginPath();
        ctx.arc(tx, ty, 4, 0, Math.PI * 2);
        ctx.fillStyle   = color;
        ctx.globalAlpha = 0.9;
        ctx.fill();
    });
 
    ctx.restore(); // resets globalAlpha and all other state
}

function drawState(state) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#eef3f7';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#495057';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, canvas.width, canvas.height);

    const worldWidth = state.world?.width || canvas.width;
    const worldHeight = state.world?.height || canvas.height;

    const scaleX = canvas.width / worldWidth;
    const scaleY = canvas.height / worldHeight;

    const selectedNN = state.selected;

    function screenX(x) {
        return x * scaleX;
    }

    function screenY(y) {
        return y * scaleY;
    }

    if (Array.isArray(state.plants)) {
        ctx.fillStyle = '#2f8f3a';
        state.plants.forEach((plant) => {
            ctx.beginPath();
            ctx.arc(screenX(plant.x), screenY(plant.y), 5, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    // --- Vision overlay (drawn before animals so animals appear on top) ---
    if (state.selected) {
        if (state.selected.species === 'herbivore') {
            drawVisionOverlay(state.selected, scaleX, scaleY);
            drawLiveNeuralNetwork(state.selected);
            updateStatsPanel(formatStats(state.selected)); 
        } else if (state.selected.species === 'predator') {
            drawVisionOverlay(state.selected, scaleX, scaleY);
            drawLiveNeuralNetwork(state.selected);
            updateStatsPanel(formatStats(state.selected));
        }
    } else {
        drawLiveNeuralNetwork(null);
        updateStatsPanel('Click a herbivore or predator to view stats.');
    }


    if (Array.isArray(state.herbivores)) {
        state.herbivores.forEach((herbivore) => {
            ctx.beginPath();
            ctx.arc(screenX(herbivore.x), screenY(herbivore.y), 6, 0, Math.PI * 2);
            ctx.fillStyle = `rgb(${herbivore.red}, ${herbivore.green}, ${herbivore.blue})`;
            ctx.fill();
            if (typeof herbivore.angle === 'number') {
                const dirX = Math.cos(herbivore.angle) * 10;
                const dirY = Math.sin(herbivore.angle) * 10;
                ctx.strokeStyle = '#0f4bb5';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(screenX(herbivore.x), screenY(herbivore.y));
                ctx.lineTo(screenX(herbivore.x + dirX), screenY(herbivore.y + dirY));
                ctx.stroke();
            }
        });
    }

    if (Array.isArray(state.predators)) {
        state.predators.forEach((predator) => {

            const x = screenX(predator.x);
            const y = screenY(predator.y);

            ctx.save();

            // Move origin to predator
            ctx.translate(x, y);

            // Rotate to face movement direction
            ctx.rotate(predator.angle);

            // Draw triangle
            ctx.beginPath();

            // Front point
            ctx.moveTo(12, 0);

            // Back bottom
            ctx.lineTo(-8, 6);

            // Back top
            ctx.lineTo(-8, -6);

            ctx.closePath();

            ctx.fillStyle = `rgb(${predator.red}, ${predator.green}, ${predator.blue})`;
            ctx.fill();

            ctx.restore();
        });
    }

    if (state.selected) {
         // draw a circle around the selected animal
            const pos = { x: state.selected.x, y: state.selected.y };
            ctx.strokeStyle = '#111111';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(screenX(pos.x), screenY(pos.y), state.selected.species === 'predator' ? 12 : 10, 0, Math.PI * 2);
            ctx.stroke();
    }

    if (typeof state.world?.time === 'number') {
        statusEl.textContent = `Time: ${state.world.time.toFixed(2)}s | Plants: ${state.plants?.length || 0} | Herbivores: ${state.herbivores?.length || 0} | Predators: ${state.predators?.length || 0}`;
    } else {
        statusEl.textContent = `Plants: ${state.plants?.length || 0} | Herbivores: ${state.herbivores?.length || 0} | Predators: ${state.predators?.length || 0}`;
    }

}

function getCanvasPos(event) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: ((event.clientX - rect.left) / rect.width) * canvas.width,
        y: ((event.clientY - rect.top) / rect.height) * canvas.height,
    };
}

function worldFromCanvas(screenX, screenY, worldWidth, worldHeight) {
    return {
        x: (screenX / canvas.width) * worldWidth,
        y: (screenY / canvas.height) * worldHeight,
    };
}

let chart_last_time = -1;

async function updateChart() {
    try {
        const data = await fetchChartData();
        if (data.world_time <= chart_last_time) {
            return; // simulation is paused and world time is the same.
        }
        populationChart.data.labels.push(data.world_time.toFixed(1));

        populationChart.data.datasets[0].data.push(data.current_plant);
        populationChart.data.datasets[1].data.push(data.current_herbivore);
        populationChart.data.datasets[2].data.push(data.current_predator);

        // keep only latest n points
        const maxPoints = 150;

        if (populationChart.data.labels.length > maxPoints) {
            populationChart.data.labels.shift();

            populationChart.data.datasets.forEach(dataset => {
                dataset.data.shift();
            });
        }
        
        chart_last_time = data.world_time;
        populationChart.update();

    } catch (err) {
        console.error('Chart update failed:', err);
    }
}

function formatStats(stats) {
    const speciesName = stats.species.charAt(0).toUpperCase() + stats.species.slice(1);
    return `
        <div class="stats-title">${speciesName} #${stats.id} [Gen ${stats.generation}]</div>
        <div class="stats-grid">
            <div class="stat-card">
                <span class="label">Position</span>
                <span class="value">X:${stats.x.toFixed(1)} Y:${stats.y.toFixed(1)}</span>
            </div>
            <div class="stat-card">
                <span class="label">Age</span>
                <span class="value">${stats.age.toFixed(1)}s</span>
            </div>
            <div class="stat-card">
                <span class="label">Speed</span>
                <span class="value">${stats.speed.toFixed(2)}</span>
            </div>
            <div class="stat-card">
                <span class="label">Satiety</span>
                <span class="value">${stats.satiety.toFixed(2)}</span>
            </div>
            <div class="stat-card">
                <span class="label">Fitness Rating</span>
                <span class="value">${stats.fitness.toFixed(3)}</span>
            </div>
            <div class="stat-card">
                <span class="label">Total Offspring</span>
                <span class="value">${stats.offspring_count}</span>
            </div>
            <div class="stat-card" style="grid-column: span 2;">
                <span class="label">Reproduction</span>
                <span class="value">${(stats.reproduction_progress * 100).toFixed(0)}%</span>
            </div>
        </div>
    `;
}

function updateStatsPanel(message) {
    // If it's the raw default text string, wrap it inside our placeholder visual stylings
    if (message.trim().startsWith("Click a herbivore")) {
        statsEl.innerHTML = `<div class="empty-stats-msg">${message}</div>`;
    } else {
        statsEl.innerHTML = message;
    }
}

function drawLiveNeuralNetwork(nn) {
    if (!nn) {
        networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
        return;
    }

    // --- FIX: Dynamic resolution synchronization ---
    // If the CSS size doesn't match the rendering bitmap size, sync them instantly
    if (networkCanvas.width !== networkCanvas.clientWidth || networkCanvas.height !== networkCanvas.clientHeight) {
        networkCanvas.width = networkCanvas.clientWidth;
        networkCanvas.height = networkCanvas.clientHeight;
    }

    const w = networkCanvas.width;
    const h = networkCanvas.height;
    
    // Adjusted padding to protect labels from clipping at taller aspect ratios
    const padding = 30; 
    const nodeRadius = 6;
    const t = performance.now() / 1000;

    networkCtx.clearRect(0, 0, w, h);
    networkCtx.fillStyle = '#f5f7fb';
    networkCtx.fillRect(0, 0, w, h);

    const layers = [
        { values: nn.inputs,       x: 60 },
        { values: nn.hidden_dim_1, x: w * 0.33 },
        { values: nn.hidden_dim_2, x: w * 0.66 },
        { values: nn.output,       x: w - 60 },
    ];

    const weightMatrices = nn.weights ? [
        nn.weights.input_to_hidden1,
        nn.weights.hidden1_to_hidden2,
        nn.weights.hidden2_to_output,
    ] : null;

    const getY = (layerIndex, i) => {
        const layer = layers[layerIndex];
        // Adjusted to leave extra space at the bottom for the labels
        const spacing = (h - (padding * 2.5)) / Math.max(layer.values.length - 1, 1);
        return padding + i * spacing;
    };

    // --- Draw connections ---
    for (let l = 0; l < layers.length - 1; l++) {
        const from = layers[l];
        const to   = layers[l + 1];

        for (let i = 0; i < from.values.length; i++) {
            for (let j = 0; j < to.values.length; j++) {

                const x1 = from.x;
                const y1 = getY(l, i);
                const x2 = to.x;
                const y2 = getY(l + 1, j);

                let edgeColor = 'rgba(120,120,140,0.12)';
                let edgeWidth = 0.5;
                let signal    = 0;
                let absSignal = 0;

                if (weightMatrices && weightMatrices[l]) {
                    const weight = weightMatrices[l][j]?.[i];
                    if (weight !== undefined) {
                        signal    = weight * from.values[i];
                        absSignal = Math.abs(signal);

                        edgeWidth = 0.5 + Math.min(absSignal, 1.0) * 3.5;

                        const alpha = 0.08 + Math.min(absSignal, 1.0) * 0.75;
                        edgeColor = signal > 0
                            ? `rgba(40, 200, 80, ${alpha})`
                            : signal < 0
                                ? `rgba(220, 55, 55, ${alpha})`
                                : `rgba(120, 120, 140, 0.08)`;
                    }
                }

                // Static edge
                networkCtx.beginPath();
                networkCtx.moveTo(x1, y1);
                networkCtx.lineTo(x2, y2);
                networkCtx.strokeStyle = edgeColor;
                networkCtx.lineWidth   = edgeWidth;
                networkCtx.stroke();

                // Travelling pulse (optional)
                if (typeof neuralNetPulsesEnabled !== 'undefined' && neuralNetPulsesEnabled && absSignal >= 0.05) {
                    const speed       = 0.3 + Math.min(absSignal, 1.0) * 1.2;
                    const phaseOffset = ((l * 97 + i * 31 + j * 13) % 100) / 100;
                    const pulsePos    = (t * speed + phaseOffset) % 1.0;

                    const px = x1 + (x2 - x1) * pulsePos;
                    const py = y1 + (y2 - y1) * pulsePos;

                    const pulseRadius = 1.5 + Math.min(absSignal, 1.0) * 2.5;
                    const pulseAlpha  = 0.5 + Math.min(absSignal, 1.0) * 0.5;
                    const pulseColor  = signal > 0
                        ? `rgba(80, 240, 120,`
                        : `rgba(255, 80, 80,`;

                    const glow = networkCtx.createRadialGradient(px, py, 0, px, py, pulseRadius * 3);
                    glow.addColorStop(0, `${pulseColor}${pulseAlpha})`);
                    glow.addColorStop(1, `${pulseColor}0)`);
                    networkCtx.beginPath();
                    networkCtx.arc(px, py, pulseRadius * 3, 0, Math.PI * 2);
                    networkCtx.fillStyle = glow;
                    networkCtx.fill();

                    networkCtx.beginPath();
                    networkCtx.arc(px, py, pulseRadius, 0, Math.PI * 2);
                    networkCtx.fillStyle = `${pulseColor}${pulseAlpha})`;
                    networkCtx.fill();
                }
            }
        }
    }

    // --- Draw nodes ---
    for (let l = 0; l < layers.length; l++) {
        const layer = layers[l];
        for (let i = 0; i < layer.values.length; i++) {
            const v     = layer.values[i];
            const n     = Math.max(-1, Math.min(1, v));
            const red   = Math.round(Math.max(0, -n) * 255);
            const green = Math.round(Math.max(0, n) * 255);
            const alpha = 0.3 + Math.abs(n) * 0.7;
            const x     = layer.x;
            const y     = getY(l, i);

            // Glow
            networkCtx.beginPath();
            networkCtx.arc(x, y, nodeRadius + Math.abs(n) * 4, 0, Math.PI * 2);
            networkCtx.fillStyle = `rgba(${red},${green},80,${alpha * 0.4})`;
            networkCtx.fill();

            // Core
            networkCtx.beginPath();
            networkCtx.arc(x, y, nodeRadius, 0, Math.PI * 2);
            networkCtx.fillStyle = `rgba(${red},${green},80,${alpha})`;
            networkCtx.fill();
        }
    }

    // --- Labels ---
    networkCtx.fillStyle = '#111827';
    networkCtx.font = '12px sans-serif';
    networkCtx.textAlign = 'center';
    ['Input', 'Hidden 1', 'Hidden 2', 'Output'].forEach((label, i) => {
        networkCtx.fillText(label, layers[i].x, h - 8);
    });
}

// --- VIEW CONTROLLER FUNCTIONS ---
async function openSettings() {
    stopLoop(); // Pause the simulation loop
    
    // Fetch live backend parameters before rendering the menu layout
    await syncSlidersWithBackend();

    simView.style.display = 'none';
    settingsView.style.display = 'block';
}

// Function to fetch active configurations and apply them to DOM element positions
async function syncSlidersWithBackend() {
    try {
        const response = await fetch('/settings');
        if (!response.ok) throw new Error('Could not pull live configurations from server.');
        const settings = await response.json();

        const dataMap = {
            'worldSpeedInput': settings.world_speed_multiplier,
            'maxSpeedInput': settings.max_speed,
            'maxAngularVelocityInput': settings.max_angular_velocity,
            'globalMutationRateInput': settings.global_mutation_rate,
            'globalMutationStrengthInput': settings.global_mutation_strength,
            'weightStdNewNeuronsInput': settings.weight_std_for_new_neurons,
            'startingHerbivoreInput': settings.starting_herbivore,
            'startingPredatorInput': settings.starting_predator,
            'startingPlantInput': settings.starting_plant,
            'maxPlantInput': settings.max_plant,
            'plantNutritionValueInput': settings.plant_nutrition_value,
            'plantRegrowthPowerInput': settings.plant_regrowth_power,
            'maxPredatorInput': settings.max_predator,
            'predatorAvgGestationInput': settings.predator_avg_gestation_time,
            'predatorGestationStdInput': settings.predator_gestation_time_std_dev,
            'predatorMinReproductionSatietyInput': settings.predator_reproduction_minimum_satiety,
            'predatorReproductionLossInput': settings.predator_reproduction_satiety_loss,
            'predatorEatPercentThresholdInput': Math.round(settings.predator_max_percent_satiety_to_eat * 100),
            'predatorFOVInput': settings.predator_FOV,
            'predatorVisionRangeInput': settings.predator_vision_range,
            'predatorAvgAgeInput': settings.predator_avg_age,
            'predatorAgeStdInput': settings.predator_age_std_dev,
            'predatorMinAgeReproductionInput': settings.predator_min_age_to_reproduce,
            'predatorsResurrectAfterHerbivoresReachInput': settings.predators_resurrect_after_herbivores_reach,
            'predatorResurrectionCountInput': settings.predator_resurrection_count,
            'predatorResurrectionRecentCountInput': settings.predator_resurrection_recent_count,
            'predatorResurrectionRandomCountInput': settings.predator_resurrection_random_count,
            'maxHerbivoreInput': settings.max_herbivore,
            'herbivoreSatietyLossInput': settings.herbivore_satiety_loss_factor,
            'herbivoreMaxSatietyInput': settings.herbivore_max_satiety,
            'herbivoreAvgGestationInput': settings.herbivore_avg_gestation_time,
            'herbivoreGestationStdInput': settings.herbivore_gestation_time_std_dev,
            'herbivoreMinReproductionSatietyInput': settings.herbivore_reproduction_minimum_satiety,
            'herbivoreReproductionLossInput': settings.herbivore_reproduction_satiety_loss,
            'herbivoreEatPercentThresholdInput': Math.round(settings.herbivore_max_percent_satiety_to_eat * 100),
            'herbivoreFOVInput': settings.herbivore_FOV,
            'herbivoreVisionRangeInput': settings.herbivore_vision_range,
            'herbivoreAvgAgeInput': settings.herbivore_avg_age,
            'herbivoreAgeStdInput': settings.herbivore_age_std_dev,
            'herbivoreMinAgeReproductionInput': settings.herbivore_min_age_to_reproduce,
            'herbivoreNutritionValueInput': settings.herbivore_nutrition_value,
            'herbivoreResurrectionCountInput': settings.herbivore_resurrection_count,
            'herbivoreResurrectionRandomCountInput': settings.herbivore_resurrection_random_count,
            'herbivoreResurrectionRecentCountInput': settings.herbivore_resurrection_recent_count
        };

        inputsConfig.forEach(cfg => {
            const inputEl = document.getElementById(cfg.input);
            const valueEl = document.getElementById(cfg.val);
            const currentVal = dataMap[cfg.input];

            if (inputEl && valueEl && currentVal !== undefined) {
                inputEl.value = currentVal;
                valueEl.textContent = cfg.isFloat ? currentVal.toFixed(cfg.fixed) : currentVal;
            }
        });
    } catch (err) {
        console.error('Failed to sync settings overlay layout data:', err);
    }
}

function closeSettings() {
    settingsView.style.display = 'none';
    simView.style.display = 'block';
    startLoop(); // Resumes and unpauses the simulation
}

// --- EVENT LISTENERS ---
settingsButton.addEventListener('click', openSettings);
backButton.addEventListener('click', closeSettings);

restartSettingsButton.addEventListener('click', async () => {
    const getVal = (id, isFloat) => {
        const val = document.getElementById(id).value;
        return isFloat ? parseFloat(val) : parseInt(val, 10);
    };

    const dataToSend = {
        world_speed_multiplier: getVal('worldSpeedInput', true),
        max_speed: getVal('maxSpeedInput', true),
        max_angular_velocity: getVal('maxAngularVelocityInput', true),
        global_mutation_rate: getVal('globalMutationRateInput', true),
        global_mutation_strength: getVal('globalMutationStrengthInput', true),
        weight_std_for_new_neurons: getVal('weightStdNewNeuronsInput', true),
        starting_herbivore: getVal('startingHerbivoreInput', false),
        starting_predator: getVal('startingPredatorInput', false),
        starting_plant: getVal('startingPlantInput', false),
        
        max_plant: getVal('maxPlantInput', false),
        plant_size: 6, // Managed implicitly by index layout
        plant_nutrition_value: getVal('plantNutritionValueInput', true),
        plant_regrowth_power: getVal('plantRegrowthPowerInput', true),
        
        max_predator: getVal('maxPredatorInput', false),
        predator_avg_gestation_time: getVal('predatorAvgGestationInput', true),
        predator_gestation_time_std_dev: getVal('predatorGestationStdInput', true),
        predator_reproduction_minimum_satiety: getVal('predatorMinReproductionSatietyInput', true),
        predator_reproduction_satiety_loss: getVal('predatorReproductionLossInput', true),
        predator_max_percent_satiety_to_eat: getVal('predatorEatPercentThresholdInput', false) / 100,
        predator_FOV: getVal('predatorFOVInput', true),
        predator_vision_range: getVal('predatorVisionRangeInput', false),
        predator_avg_age: getVal('predatorAvgAgeInput', true),
        predator_age_std_dev: getVal('predatorAgeStdInput', true),
        predator_min_age_to_reproduce: getVal('predatorMinAgeReproductionInput', true),
        predators_resurrect_after_herbivores_reach: getVal('predatorsResurrectAfterHerbivoresReachInput', false),
        predator_resurrection_count: getVal('predatorResurrectionCountInput', false),
        predator_resurrection_recent_count: getVal('predatorResurrectionRecentCountInput', false),
        predator_resurrection_random_count: getVal('predatorResurrectionRandomCountInput', false),
        
        max_herbivore: getVal('maxHerbivoreInput', false),
        herbivore_satiety_loss_factor: getVal('herbivoreSatietyLossInput', true),
        herbivore_max_satiety: getVal('herbivoreMaxSatietyInput', true),
        herbivore_avg_gestation_time: getVal('herbivoreAvgGestationInput', true),
        herbivore_gestation_time_std_dev: getVal('herbivoreGestationStdInput', true),
        herbivore_reproduction_minimum_satiety: getVal('herbivoreMinReproductionSatietyInput', true),
        herbivore_reproduction_satiety_loss: getVal('herbivoreReproductionLossInput', true),
        herbivore_max_percent_satiety_to_eat: getVal('herbivoreEatPercentThresholdInput', false) / 100,
        herbivore_FOV: getVal('herbivoreFOVInput', true),
        herbivore_vision_range: getVal('herbivoreVisionRangeInput', false),
        herbivore_avg_age: getVal('herbivoreAvgAgeInput', true),
        herbivore_age_std_dev: getVal('herbivoreAgeStdInput', true),
        herbivore_min_age_to_reproduce: getVal('herbivoreMinAgeReproductionInput', true),
        herbivore_nutrition_value: getVal('herbivoreNutritionValueInput', true),
        herbivore_resurrection_count: getVal('herbivoreResurrectionCountInput', false),
        herbivore_resurrection_random_count: getVal('herbivoreResurrectionRandomCountInput', false),
        herbivore_resurrection_recent_count: getVal('herbivoreResurrectionRecentCountInput', false)
    };

    try {
        const response = await fetch('/restart_simulation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(dataToSend)
        });

        if (!response.ok) throw new Error(`Status error: ${response.statusText}`);
        const result = await response.json();
        console.log(result.message);

        chart_last_time = -1;
        populationChart.data.labels = [];
        populationChart.data.datasets.forEach(d => d.data = []);
        populationChart.update();

        selectedAnimal = null;
        updateStatsPanel('Click a herbivore or predator to view stats.');
        networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);

        closeSettings();
    } catch (error) {
        console.error('Failed to commit configurations via endpoint sequence:', error);
        alert('Could not update active running simulation profiles.');
    }
});

canvas.addEventListener('click', async (event) => {
    if (!lastState) return;

    const worldWidth = lastState.world?.width || canvas.width;
    const worldHeight = lastState.world?.height || canvas.height;
    const canvasPos = getCanvasPos(event);
    const worldPos = worldFromCanvas(canvasPos.x, canvasPos.y, worldWidth, worldHeight);

    const candidates = [];
    const captureDistance = 12;

    (lastState.herbivores || []).forEach((herbivore) => {
        const dx = herbivore.x - worldPos.x;
        const dy = herbivore.y - worldPos.y;
        candidates.push({ species: 'herbivore', id: herbivore.id, generation: herbivore.generation, distance: Math.hypot(dx, dy), x: herbivore.x, y: herbivore.y });
    });
    (lastState.predators || []).forEach((predator) => {
        const dx = predator.x - worldPos.x;
        const dy = predator.y - worldPos.y;
        candidates.push({ species: 'predator', id: predator.id, generation: predator.generation, distance: Math.hypot(dx, dy), x: predator.x, y: predator.y });
    });

    candidates.sort((a, b) => a.distance - b.distance);
    const selection = candidates.find((item) => item.distance <= captureDistance);

    if (!selection) {
        // click empty space: deselect
        sendSelection(null, null);

        drawState(lastState);
        return;
    }

    // Toggle deselect when clicking the already selected animal
    if (lastState.selected && lastState.selected.species === selection.species && lastState.selected.id === selection.id) {
        sendSelection(null, null);
        drawState(lastState);
        return;
    }
    console.log('selected species and id:',selection.species, selection.id);
    sendSelection(selection.species, selection.id); // inform backend about selection so it can prepare detailed stats (like NN activations) for this animal
    // Fetch full stats and update panel when ready 
    const temp_state = await fetchState();
    updateStatsPanel(formatStats(temp_state.selected));
    drawState(temp_state);
});

async function sendSelection(species, id) {
    await fetch('/select_animal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            species,
            id
        }),
    });
}

async function tick() {
    const now = performance.now();
    const dt = (now - lastTickTime) / 1000;
    lastTickTime = now;

    try {
        const state = await stepWorld(dt);

        lastState = state;
        drawState(state);

    } catch (error) {
        statusEl.textContent = 'Error updating simulation — retrying...';
        console.error('Tick error:', error);
    }
}


async function loop() {
    while (running) {
        await tick();
    }
}

function startLoop() {
    if (running) return;

    running = true;

    lastTickTime = performance.now();

    toggleButton.textContent = 'Pause';

    loop();
}

function stopLoop() {
    running = false;
    toggleButton.textContent = 'Start';
}

toggleButton.addEventListener('click', () => {
    if (running) {
        stopLoop();
    } else {
        startLoop();
    }
});

stepButton.addEventListener('click', async () => {
    if (running) {
        stopLoop();
    }
    try {
        const state = await stepWorld(0.04);
        lastState = state;
        drawState(state);
    } catch (error) {
        statusEl.textContent = 'Error during step';
        console.error('Step error:', error);
    }
});

saveButton.addEventListener('click', () => {
    window.location.href = '/save';
});

// Load — hidden file input, triggered by button click
const loadInput = document.getElementById('loadInput'); // hidden <input type="file">
document.getElementById('loadButton').addEventListener('click', () => {
    loadInput.click();
});
loadInput.addEventListener('change', async () => {
    const file = loadInput.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    await fetch('/load', { method: 'POST', body: formData });
    location.reload();
});

killButton.addEventListener('click', debugKillSelectedAnimal)
async function debugKillSelectedAnimal() {
    try {
        const response = await fetch('/debug_kill_selected', {
            method: 'POST'
        });

        if (!response.ok) {
            console.error('Failed to kill selected animal');
            return;
        }

        console.log('Selected animal killed');
    } catch (err) {
        console.error('Request failed:', err);
    }
}

window.addEventListener('load', async () => {
    try {
        const state = await fetchState();
        lastState = state;
        drawState(state);
        updateStatsPanel('Click a herbivore or predator to view stats.');
        startLoop();
        setInterval(updateChart, 3000);
    } catch (error) {
        statusEl.textContent = 'Unable to load simulation state.';
        console.error(error);
    }
});

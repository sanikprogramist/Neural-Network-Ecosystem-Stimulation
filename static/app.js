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

let running = true;
let lastState = null;
let lastTickTime = performance.now();

// --- DOM ELEMENTS FOR SETTINGS ---
const simView = document.getElementById('simView');
const settingsView = document.getElementById('settingsView');
const settingsButton = document.getElementById('settingsButton');
const backButton = document.getElementById('backButton');
const restartSettingsButton = document.getElementById('restartSettingsButton');


const inputsConfig = [
    { input: 'worldSpeedInput', val: 'worldSpeedValue', isFloat: true, fixed: 1 },
    { input: 'globalMutationRateInput', val: 'globalMutationRateValue', isFloat: true, fixed: 3 },
    { input: 'globalMutationStrengthInput', val: 'globalMutationStrengthValue', isFloat: true, fixed: 2 },
    { input: 'maxPlantInput', val: 'maxPlantValue', isFloat: false },
    { input: 'plantSizeInput', val: 'plantSizeValue', isFloat: true, fixed: 1 },
    { input: 'plantNutritionValueInput', val: 'plantNutritionValueValue', isFloat: true, fixed: 2 },
    { input: 'plantRegrowthPowerInput', val: 'plantRegrowthPowerValue', isFloat: true, fixed: 1 },
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
    { input: 'herbivoreMinAgeReproductionInput', val: 'herbivoreMinAgeReproductionValue', isFloat: true, fixed: 1 }
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
    const targets = [
        { normDist: data[0], normAngle: data[1], color: '#3ecf60', label: 'food'       },
        { normDist: data[2], normAngle: data[3], color: '#5a9ff5', label: 'conspecific'  },
        { normDist: data[4], normAngle: data[5], color: '#e84545', label: 'predator'     },
    ];
 
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
            console.log('Vision overlay for predators not implemented yet');
            console.log('Neural Network visualization for predators not implemented yet');
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
    return `
        <strong>${stats.species.charAt(0).toUpperCase() + stats.species.slice(1)} #${stats.id}</strong><br>
        Position: (${stats.x.toFixed(1)}, ${stats.y.toFixed(1)})<br>
        Age: ${stats.age.toFixed(1)}<br>
        Speed: ${stats.speed.toFixed(2)}<br>
        Satiety: ${stats.satiety.toFixed(2)}<br>
        Generation: ${stats.generation}<br>
        Fitness: ${stats.fitness.toFixed(3)}<br>
        Offspring: ${stats.offspring_count}<br>
        Reproduction: ${(stats.reproduction_progress * 100).toFixed(0)}%
    `;
}

function updateStatsPanel(message) {
    statsEl.innerHTML = message;
}

let neuralNetPulsesEnabled = true;
 
function drawLiveNeuralNetwork(nn) {
    if (!nn || !nn.hidden_dim_1 || !nn.output) {
        networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
        return;
    }
 
    const w = networkCanvas.width;
    const h = networkCanvas.height;
    const padding = 20;
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
        const spacing = (h - 2 * padding) / Math.max(layer.values.length - 1, 1);
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
                        // signal = weight * activation of source node
                        signal    = weight * from.values[i];
                        absSignal = Math.abs(signal);
 
                        // thickness 0.5..4 with signal strength
                        edgeWidth = 0.5 + Math.min(absSignal, 1.0) * 3.5;
 
                        // green positive, red negative, alpha fades weak connections
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
                if (neuralNetPulsesEnabled && absSignal >= 0.05) {
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
 
                    // Soft glow
                    const glow = networkCtx.createRadialGradient(px, py, 0, px, py, pulseRadius * 3);
                    glow.addColorStop(0, `${pulseColor}${pulseAlpha})`);
                    glow.addColorStop(1, `${pulseColor}0)`);
                    networkCtx.beginPath();
                    networkCtx.arc(px, py, pulseRadius * 3, 0, Math.PI * 2);
                    networkCtx.fillStyle = glow;
                    networkCtx.fill();
 
                    // Hard core
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
    networkCtx.font = '12px Arial';
    networkCtx.textAlign = 'center';
    ['Input', 'Hidden 1', 'Hidden 2', 'Output'].forEach((label, i) => {
        networkCtx.fillText(label, layers[i].x, h - 5);
    });
}

function updateStatsPanel(message) {
    statsEl.innerHTML = message;
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
        if (!response.ok) throw new Error('Could not retrieve ecosystem configurations.');
        
        const settings = await response.json();

        // Object mapping your configuration key names to backend payload JSON keys
        const dataMap = {
            'worldSpeedInput': settings.world_speed_multiplier,
            'globalMutationRateInput': settings.global_mutation_rate,
            'globalMutationStrengthInput': settings.global_mutation_strength,
            'maxPlantInput': settings.max_plant,
            'plantSizeInput': settings.plant_size,
            'plantNutritionValueInput': settings.plant_nutrition_value,
            'plantRegrowthPowerInput': settings.plant_regrowth_power,
            'maxHerbivoreInput': settings.max_herbivore,
            'herbivoreSatietyLossInput': settings.herbivore_satiety_loss_factor,
            'herbivoreMaxSatietyInput': settings.herbivore_max_satiety,
            'herbivoreAvgGestationInput': settings.herbivore_avg_gestation_time,
            'herbivoreGestationStdInput': settings.herbivore_gestation_time_std_dev,
            'herbivoreMinReproductionSatietyInput': settings.herbivore_reproduction_minimum_satiety,
            'herbivoreReproductionLossInput': settings.herbivore_reproduction_satiety_loss,
            // Multiply ratio (e.g. 0.75) by 100 to sync cleanly back to integer slider scale (75)
            'herbivoreEatPercentThresholdInput': Math.round(settings.herbivore_max_percent_satiety_to_eat * 100),
            'herbivoreFOVInput': settings.herbivore_FOV,
            'herbivoreVisionRangeInput': settings.herbivore_vision_range,
            'herbivoreAvgAgeInput': settings.herbivore_avg_age,
            'herbivoreAgeStdInput': settings.herbivore_age_std_dev,
            'herbivoreMinAgeReproductionInput': settings.herbivore_min_age_to_reproduce
        };

        // Populate values dynamically across your predefined tracking configurations
        inputsConfig.forEach(cfg => {
            const inputEl = document.getElementById(cfg.input);
            const valueEl = document.getElementById(cfg.val);
            const liveBackendValue = dataMap[cfg.input];

            if (inputEl && valueEl && liveBackendValue !== undefined) {
                // Update the visual handle position
                inputEl.value = liveBackendValue;
                
                // Format the displayed numerical text context correctly
                if (cfg.isFloat) {
                    valueEl.textContent = liveBackendValue.toFixed(cfg.fixed);
                } else {
                    valueEl.textContent = liveBackendValue;
                }
            }
        });

    } catch (error) {
        console.error('Failed to sync settings pane with running simulation:', error);
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
    // Helper function to extract cleanly parsed input DOM data
    const getVal = (id, isFloat) => {
        const value = document.getElementById(id).value;
        return isFloat ? parseFloat(value) : parseInt(value, 10);
    };

    const dataToSend = {
        world_speed_multiplier: getVal('worldSpeedInput', true),
        global_mutation_rate: getVal('globalMutationRateInput', true),
        global_mutation_strength: getVal('globalMutationStrengthInput', true),
        max_plant: getVal('maxPlantInput', false),
        plant_size: getVal('plantSizeInput', true),
        plant_nutrition_value: getVal('plantNutritionValueInput', true),
        plant_regrowth_power: getVal('plantRegrowthPowerInput', true),
        max_herbivore: getVal('maxHerbivoreInput', false),
        herbivore_satiety_loss_factor: getVal('herbivoreSatietyLossInput', true),
        herbivore_max_satiety: getVal('herbivoreMaxSatietyInput', true),
        herbivore_avg_gestation_time: getVal('herbivoreAvgGestationInput', true),
        herbivore_gestation_time_std_dev: getVal('herbivoreGestationStdInput', true),
        herbivore_reproduction_minimum_satiety: getVal('herbivoreMinReproductionSatietyInput', true),
        herbivore_reproduction_satiety_loss: getVal('herbivoreReproductionLossInput', true),
        // Convert slider percent (e.g. 75) back to float proportion (0.75) for the Python backend
        herbivore_max_percent_satiety_to_eat: getVal('herbivoreEatPercentThresholdInput', false) / 100,
        herbivore_FOV: getVal('herbivoreFOVInput', true),
        herbivore_vision_range: getVal('herbivoreVisionRangeInput', false),
        herbivore_avg_age: getVal('herbivoreAvgAgeInput', true),
        herbivore_age_std_dev: getVal('herbivoreAgeStdInput', true),
        herbivore_min_age_to_reproduce: getVal('herbivoreMinAgeReproductionInput', true)
    };

    try {
        const response = await fetch('/restart_simulation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(dataToSend)
        });

        if (!response.ok) {
            throw new Error(`Failed to restart: ${response.statusText}`);
        }

        const result = await response.json();
        console.log(result.message);

        // Reset tracking states for the live population visualization chart
        chart_last_time = -1;
        populationChart.data.labels = [];
        populationChart.data.datasets.forEach(dataset => dataset.data = []);
        populationChart.update();

        // Close overlay pane context and automatically unpause
        closeSettings();

    } catch (error) {
        console.error('Error while sending restart configurations:', error);
        alert('Could not restart simulation with current parameters.');
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

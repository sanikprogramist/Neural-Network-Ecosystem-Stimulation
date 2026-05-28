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
//const dt = 0.04;
//let selectedAnimal = null;

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
        { normDist: data[0], normAngle: data[1], color: '#3ecf60', label: 'plant'       },
        { normDist: data[2], normAngle: data[3], color: '#5a9ff5', label: 'conspecific'  },
        { normDist: data[4], normAngle: data[5], color: '#e84545', label: 'predator'     },
    ];
 
    targets.forEach(({ normDist, normAngle, color }) => {
        if (normDist < 0) return; // nothing detected for this type
 
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
            ctx.arc(screenX(plant.x), screenY(plant.y), 4, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    // --- Vision overlay (drawn before animals so animals appear on top) ---
    if (state.selected) {
        if (state.selected.species === 'herbivore') {
            if (state.selected && state.selected.nn_distances_angles) {
                drawVisionOverlay(state.selected, scaleX, scaleY);
                drawLiveNeuralNetwork(state.selected);
                updateStatsPanel(formatStats(state.selected)); 
            } // there are two selection logics going on and i need to fix it later
        } else if (state.selected.species === 'predator') {
            console.log('Vision overlay for predators not implemented yet');
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
        Offspring: ${stats.offspring_count}<br>
        Reproduction: ${(stats.reproduction_progress * 100).toFixed(0)}%
    `;
}

function updateStatsPanel(message) {
    statsEl.innerHTML = message;
}

let neuralNetPulsesEnabled = false;
 
// Call once after DOM loaded to wire up the toggle button
function initNeuralNetControls() {
    const btn = document.getElementById('pulseToggleButton');
    if (btn) {
        btn.textContent = 'Pulses: OFF';
        btn.addEventListener('click', () => {
            neuralNetPulsesEnabled = !neuralNetPulsesEnabled;
            btn.textContent = neuralNetPulsesEnabled ? 'Pulses: ON' : 'Pulses: OFF';
        });
    }
}
 
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

function drawLiveNeuralNetwork_stable(nn) {
    if (!nn || !nn.hidden_dim_1 || !nn.output) {
        networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
        return;
    }

    const w = networkCanvas.width;
    const h = networkCanvas.height;
    const padding = 20;

    networkCtx.clearRect(0, 0, w, h);
    networkCtx.fillStyle = '#f5f7fb';
    networkCtx.fillRect(0, 0, w, h);

    // --- layer sizes from live brain ---
    const input = nn.inputs;
    const h1 = nn.hidden_dim_1;
    const h2 = nn.hidden_dim_2;
    const out = nn.output;

    const layers = [
        { values: input, x: 60 },
        { values: h1, x: w * 0.33 },
        { values: h2, x: w * 0.66 },
        { values: out, x: w - 60 }
    ];

    const nodeRadius = 6;

    const getY = (layerIndex, i) => {
        const layer = layers[layerIndex];
        const spacing = (h - 2 * padding) / Math.max(layer.values.length - 1, 1);
        return padding + i * spacing;
    };

    // --- draw connections as faint structure (no weights now) ---
    for (let l = 0; l < layers.length - 1; l++) {
        const from = layers[l];
        const to = layers[l + 1];

        for (let i = 0; i < from.values.length; i++) {
            for (let j = 0; j < to.values.length; j++) {
                networkCtx.beginPath();
                networkCtx.moveTo(from.x, getY(l, i));
                networkCtx.lineTo(to.x, getY(l + 1, j));
                networkCtx.strokeStyle = 'rgba(120,120,140,0.15)';
                networkCtx.lineWidth = 1;
                networkCtx.stroke();
            }
        }
    }

    // --- draw nodes with activation glow ---
    const drawLayer = (layerIndex) => {
        const layer = layers[layerIndex];

        for (let i = 0; i < layer.values.length; i++) {
            const v = layer.values[i];

            // normalize activation (-1..1)
            const n = Math.max(-1, Math.min(1, v));

            const red = Math.round(Math.max(0, -n) * 255);
            const green = Math.round(Math.max(0, n) * 255);
            const alpha = 0.3 + Math.abs(n) * 0.7;

            const x = layer.x;
            const y = getY(layerIndex, i);

            // glow
            networkCtx.beginPath();
            networkCtx.arc(x, y, nodeRadius + Math.abs(n) * 4, 0, Math.PI * 2);
            networkCtx.fillStyle = `rgba(${red},${green},80,${alpha * 0.4})`;
            networkCtx.fill();

            // core node
            networkCtx.beginPath();
            networkCtx.arc(x, y, nodeRadius, 0, Math.PI * 2);
            networkCtx.fillStyle = `rgba(${red},${green},80,${alpha})`;
            networkCtx.fill();
        }
    };

    for (let i = 0; i < layers.length; i++) {
        drawLayer(i);
    }

    // labels
    networkCtx.fillStyle = '#111827';
    networkCtx.font = '12px Arial';
    networkCtx.textAlign = 'center';

    const labels = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];
    const xs = [60, w * 0.33, w * 0.66, w - 60];

    for (let i = 0; i < labels.length; i++) {
        networkCtx.fillText(labels[i], xs[i], h - 5);
    }
}

function updateStatsPanel(message) {
    statsEl.innerHTML = message;
}

async function fetchAnimalStats(species, id) {
    const response = await fetch(`/animal/${species}/${id}`);
    if (!response.ok) {
        throw new Error(`Unable to load stats for ${species} ${id}`);
    }
    return response.json();
}

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
        // selectedAnimal = null;
        sendSelection(null, null);

        //drawNeuralNetwork(null);
        drawState(lastState);
        return;
    }

    // Toggle deselect when clicking the already selected animal
    if (lastState.selected && lastState.selected.species === selection.species && lastState.selected.id === selection.id) {
        // selectedAnimal = null;
        sendSelection(null, null);
        //drawNeuralNetwork(null);
        drawState(lastState);
        return;
    }
    // Select new animal and initialize panel from current state; fetch detailed stats in background
    //const instant = (selection.species === 'predator' ? (lastState.predators || []) : (lastState.herbivores || [])).find((a) => a.id === selection.id);
    //selectedAnimal = {
    //    species: selection.species,
    //    id: selection.id,
    //    generation: instant?.generation,
    //};

    sendSelection(selection.species, selection.id); // inform backend about selection so it can prepare detailed stats (like NN activations) for this animal
    // Fetch full stats and update panel when ready 
    const temp_state = await fetchState();
    //selectedAnimal.generation = temp_state.selected.generation;
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

saveButton.addEventListener('click', async () => {
    fetch('/save', { method: 'POST' })
});

loadButton.addEventListener('click', async () => {
    fetch('/load', { method: 'POST' }).then(() => location.reload());
});

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

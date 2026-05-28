const canvas = document.getElementById('simCanvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const statsEl = document.getElementById('stats');
const networkCanvas = document.getElementById('networkCanvas');
const networkCtx = networkCanvas.getContext('2d');
const toggleButton = document.getElementById('toggleButton');
const stepButton = document.getElementById('stepButton');
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
    console.log('drawState called');
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
            } // there are two selection logics going on and i need to fix it later
            //if (selectedNN) {
            //    drawLiveNeuralNetwork(selectedNN);
            //}   
        } else if (state.selected.species === 'predator') {
            console.log('Vision overlay for predators not implemented yet');
        }
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

    if (state.selected) {
        updateStatsPanel(formatStats(state.selected)); 
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

function drawNeuralNetwork(weights) {
    console.log('drawNeuralNetwork called with weights:', weights);
    if (!weights) {
        networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
        return;
    }

    const w = networkCanvas.width;
    const h = networkCanvas.height;
    const padding = 20;
    
    networkCtx.clearRect(0, 0, w, h);
    networkCtx.fillStyle = '#f5f7fb';
    networkCtx.fillRect(0, 0, w, h);

    const input_dim = weights.input_dim;
    const hidden1_dim = weights.hidden1_dim;
    const hidden2_dim = weights.hidden2_dim;
    const output_dim = weights.output_dim;

    // Layer positions (x coordinates)
    const layerX = [
        padding + 40,
        padding + 40 + (w - 2*padding - 80) * 0.33,
        padding + 40 + (w - 2*padding - 80) * 0.66,
        w - padding - 40
    ];

    // Calculate node positions
    const layers = [
        { dim: input_dim, x: layerX[0] },
        { dim: hidden1_dim, x: layerX[1] },
        { dim: hidden2_dim, x: layerX[2] },
        { dim: output_dim, x: layerX[3] }
    ];

    const nodeRadius = 5;

    // Helper to get node y position
    const getNodeY = (layer, nodeIdx) => {
        const layer_data = layers[layer];
        const spacing = (h - 2*padding) / Math.max(layer_data.dim - 1, 1);
        return padding + nodeIdx * spacing;
    };

    // Draw connections with weights as thickness
    const drawConnections = (weights_matrix, from_layer, to_layer) => {
        const from_dim = layers[from_layer].dim;
        const to_dim = layers[to_layer].dim;

        for (let to = 0; to < to_dim; to++) {
            for (let from = 0; from < from_dim; from++) {
                const weight = weights_matrix[to][from];
                const normalized = Math.max(-1, Math.min(1, weight));
                const red = Math.round(Math.max(0, -normalized) * 255);
                const green = Math.round(Math.max(0, normalized) * 255);
                const alpha = Math.abs(weight) > 0.01 ? Math.min(1, Math.abs(weight)) : 0; // hide very weak connections
                const thickness = Math.abs(weight) * 6;
                const color = `rgba(${red}, ${green}, 0, ${alpha})`;

                const x1 = layers[from_layer].x;
                const y1 = getNodeY(from_layer, from);
                const x2 = layers[to_layer].x;
                const y2 = getNodeY(to_layer, to);

                networkCtx.strokeStyle = color;
                networkCtx.lineWidth = thickness;
                networkCtx.beginPath();
                networkCtx.moveTo(x1, y1);
                networkCtx.lineTo(x2, y2);
                networkCtx.stroke();
            }
        }
    };

    // Draw connections for each layer pair
    drawConnections(weights.input_to_hidden1, 0, 1);
    drawConnections(weights.hidden1_to_hidden2, 1, 2);
    drawConnections(weights.hidden2_to_output, 2, 3);

    // Draw nodes
    for (let layer = 0; layer < 4; layer++) {
        networkCtx.fillStyle = '#1f6ecb';
        for (let node = 0; node < layers[layer].dim; node++) {
            const x = layers[layer].x;
            const y = getNodeY(layer, node);
            networkCtx.beginPath();
            networkCtx.arc(x, y, nodeRadius, 0, Math.PI * 2);
            networkCtx.fill();
        }
    }

    // Draw layer labels
    networkCtx.fillStyle = '#111827';
    networkCtx.font = '12px Arial';
    networkCtx.textAlign = 'center';
    const labels = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];
    for (let i = 0; i < 4; i++) {
        networkCtx.fillText(labels[i], layers[i].x, h - 5);
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
        updateStatsPanel('Click a herbivore or predator to view stats.');
        //drawNeuralNetwork(null);
        drawState(lastState);
        return;
    }

    // Toggle deselect when clicking the already selected animal
    if (lastState.selected && lastState.selected.species === selection.species && lastState.selected.id === selection.id) {
        // selectedAnimal = null;
        sendSelection(null, null);
        updateStatsPanel('Click a herbivore or predator to view stats.');
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

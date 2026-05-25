const canvas = document.getElementById('simCanvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const statsEl = document.getElementById('stats');
const networkCanvas = document.getElementById('networkCanvas');
const networkCtx = networkCanvas.getContext('2d');
const toggleButton = document.getElementById('toggleButton');
const stepButton = document.getElementById('stepButton');

let running = true;
let lastState = null;
let prevState = null;
let intervalId = null;
const stepIntervalMs = 120;
const dt = 0.04;
let selectedAnimal = null;

async function fetchState() {
    const response = await fetch('/state');
    return response.json();
}

async function stepWorld() {
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

    if (Array.isArray(state.herbivores)) {
        ctx.fillStyle = '#1f6ecb';
        state.herbivores.forEach((herbivore) => {
            ctx.beginPath();
            ctx.arc(screenX(herbivore.x), screenY(herbivore.y), 6, 0, Math.PI * 2);
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
        ctx.fillStyle = '#d43f3f';
        state.predators.forEach((predator) => {
            ctx.beginPath();
            ctx.arc(screenX(predator.x), screenY(predator.y), 8, 0, Math.PI * 2);
            ctx.fill();
            if (typeof predator.angle === 'number') {
                const dirX = Math.cos(predator.angle) * 12;
                const dirY = Math.sin(predator.angle) * 12;
                ctx.strokeStyle = '#8b1f1f';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(screenX(predator.x), screenY(predator.y));
                ctx.lineTo(screenX(predator.x + dirX), screenY(predator.y + dirY));
                ctx.stroke();
            }
        });
    }

    if (selectedAnimal) {
        // Detect if the selected animal died or was replaced since the last tick
        const prevList = prevState ? (selectedAnimal.species === 'predator' ? prevState.predators : prevState.herbivores) : null;
        const prevAnimal = (prevList || []).find((item) => item.id === selectedAnimal.id);

        const currentList = selectedAnimal.species === 'predator' ? state.predators : state.herbivores;
        const currentAnimal = (currentList || []).find((item) => item.id === selectedAnimal.id);

        if (prevAnimal) {
            // If it existed previously but no longer exists, or generation changed, clear selection
            if (!currentAnimal || (currentAnimal.generation !== undefined && prevAnimal.generation !== undefined && currentAnimal.generation !== prevAnimal.generation)) {
                selectedAnimal = null;
                updateStatsPanel('Click a herbivore or predator to view stats.');
                drawNeuralNetwork(null);
            }
        }

        // Draw selection circle only if still selected and present
        if (selectedAnimal && currentAnimal) {
            const pos = { x: currentAnimal.x, y: currentAnimal.y };
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(screenX(pos.x), screenY(pos.y), selectedAnimal.species === 'predator' ? 12 : 10, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    if (typeof state.world?.time === 'number') {
        statusEl.textContent = `Time: ${state.world.time.toFixed(2)}s | Plants: ${state.plants?.length || 0} | Herbivores: ${state.herbivores?.length || 0} | Predators: ${state.predators?.length || 0}`;
    } else {
        statusEl.textContent = `Plants: ${state.plants?.length || 0} | Herbivores: ${state.herbivores?.length || 0} | Predators: ${state.predators?.length || 0}`;
    }

    // Note: stats panel will only be updated when full stats are fetched from the server
    // This avoids briefly showing instant local state that duplicates the detailed fetch results.
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

function formatStats(stats) {
    let brainArchitecture = '';
    if (stats.hidden_dim_1 !== undefined && stats.hidden_dim_1 !== null) {
        brainArchitecture = `<br><strong>Brain Architecture:</strong><br>Input Layer: ${stats.species === 'herbivore' ? 23 : 13}<br>Hidden Layer 1: ${stats.hidden_dim_1}<br>Hidden Layer 2: ${stats.hidden_dim_2}<br>Output Layer: 2`;
    }
    return `
        <strong>${stats.species.charAt(0).toUpperCase() + stats.species.slice(1)} #${stats.id}</strong><br>
        Position: (${stats.x.toFixed(1)}, ${stats.y.toFixed(1)})<br>
        Age: ${stats.age.toFixed(1)}<br>
        Speed: ${stats.speed.toFixed(2)}<br>
        Satiety: ${stats.satiety.toFixed(2)}<br>
        Generation: ${stats.generation}<br>
        Fitness: ${stats.fitness.toFixed(2)}<br>
        Reproduction: ${(stats.reproduction_progress * 100).toFixed(0)}%${brainArchitecture}
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
                const strength = Math.abs(weight);
                const color = weight > 0 ? 'rgba(0, 200, 0,' : 'rgba(200, 0, 0,';
                const alpha = Math.min(strength / 2, 0.8);
                const thickness = Math.max(0.5, strength * 3);

                const x1 = layers[from_layer].x;
                const y1 = getNodeY(from_layer, from);
                const x2 = layers[to_layer].x;
                const y2 = getNodeY(to_layer, to);

                networkCtx.strokeStyle = color + alpha + ')';
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
        candidates.push({ species: 'herbivore', id: herbivore.id, distance: Math.hypot(dx, dy), x: herbivore.x, y: herbivore.y });
    });
    (lastState.predators || []).forEach((predator) => {
        const dx = predator.x - worldPos.x;
        const dy = predator.y - worldPos.y;
        candidates.push({ species: 'predator', id: predator.id, distance: Math.hypot(dx, dy), x: predator.x, y: predator.y });
    });

    candidates.sort((a, b) => a.distance - b.distance);
    const selection = candidates.find((item) => item.distance <= captureDistance);

    if (!selection) {
        // click empty space: deselect
        selectedAnimal = null;
        updateStatsPanel('Click a herbivore or predator to view stats.');
        drawNeuralNetwork(null);
        drawState(lastState);
        return;
    }

    // Toggle deselect when clicking the already selected animal
    if (selectedAnimal && selectedAnimal.species === selection.species && selectedAnimal.id === selection.id) {
        selectedAnimal = null;
        updateStatsPanel('Click a herbivore or predator to view stats.');
        drawNeuralNetwork(null);
        drawState(lastState);
        return;
    }

    // Select new animal and initialize panel from current state; fetch detailed stats in background
    selectedAnimal = { species: selection.species, id: selection.id };
    // Fetch full stats (may include extra fields) and update panel when ready
    console.log(`Fetching full stats for ${selection.species} ${selection.id}`);
    fetchAnimalStats(selection.species, selection.id)
        .then((stats) => {
            console.log('Received full stats:', stats);
            updateStatsPanel(formatStats(stats));
            drawNeuralNetwork(stats.network_weights);
        })
        .catch((err) => {
            console.warn('Could not fetch full stats:', err);
        });
    drawState(lastState);
});

async function tick() {
    try {
        const state = await stepWorld();
        // keep previous state for death detection
        prevState = lastState;
        lastState = state;
        drawState(state);
    } catch (error) {
        // Surface the error but keep the simulation loop running.
        statusEl.textContent = 'Error updating simulation — retrying...';
        console.error('Tick error:', error);
        // Do not stop the loop; leave lastState as-is so UI remains interactive.
    }
}

function startLoop() {
    if (intervalId !== null) return;
    intervalId = setInterval(tick, stepIntervalMs);
    toggleButton.textContent = 'Pause';
    running = true;
}

function stopLoop() {
    if (intervalId !== null) {
        clearInterval(intervalId);
        intervalId = null;
    }
    toggleButton.textContent = 'Start';
    running = false;
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
    await tick();
});

window.addEventListener('load', async () => {
    try {
        const state = await fetchState();
        lastState = state;
        drawState(state);
        updateStatsPanel('Click a herbivore or predator to view stats.');
        startLoop();
    } catch (error) {
        statusEl.textContent = 'Unable to load simulation state.';
        console.error(error);
    }
});

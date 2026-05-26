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
let lastTickTime = performance.now();
//const dt = 0.04;
let selectedAnimal = null;

async function fetchState() {
    const response = await fetch('/state');
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

    //if (Array.isArray(state.predators)) {
    //    state.predators.forEach((predator) => {
    //          ctx.beginPath();
    //        ctx.arc(screenX(predator.x), screenY(predator.y), 8, 0, Math.PI * 2);
    //        ctx.fillStyle = `rgb(${predator.red}, ${predator.green}, ${predator.blue})`;
    //        ctx.fill();
    //        if (typeof predator.angle === 'number') {
    //            const dirX = Math.cos(predator.angle) * 12;
    //            const dirY = Math.sin(predator.angle) * 12;
    //            ctx.strokeStyle = '#8b1f1f';
    //            ctx.lineWidth = 2;
    //            ctx.beginPath();
    //            ctx.moveTo(screenX(predator.x), screenY(predator.y));
    //            ctx.lineTo(screenX(predator.x + dirX), screenY(predator.y + dirY));
    //            ctx.stroke();
    //        }
    //    });
    //}


    if (selectedAnimal) {
        const currentList = selectedAnimal.species === 'predator' ? state.predators : state.herbivores;
        const currentAnimal = (currentList || []).find((item) => item.id === selectedAnimal.id);

        if (!currentAnimal) {
            selectedAnimal = null;
            updateStatsPanel('Click a herbivore or predator to view stats.');
            drawNeuralNetwork(null);
        } else if (selectedAnimal.generation !== undefined && currentAnimal.generation !== selectedAnimal.generation) {
            selectedAnimal = null;
            updateStatsPanel('Click a herbivore or predator to view stats.');
            drawNeuralNetwork(null);
        } else {
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

    if (selectedAnimal) {
        const list = selectedAnimal.species === 'predator' ? state.predators : state.herbivores;
        const current = (list || []).find((a) => a.id === selectedAnimal.id);
        if (current) {
            updateStatsPanel(formatStats(current));
        }
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

function formatStats(stats) {
    return `
        <strong>${stats.species.charAt(0).toUpperCase() + stats.species.slice(1)} #${stats.id}</strong><br>
        Position: (${stats.x.toFixed(1)}, ${stats.y.toFixed(1)})<br>
        Age: ${stats.age.toFixed(1)}<br>
        Speed: ${stats.speed.toFixed(2)}<br>
        Satiety: ${stats.satiety.toFixed(2)}<br>
        Generation: ${stats.generation}<br>
        Fitness: ${stats.fitness.toFixed(2)}<br>
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
    const instant = (selection.species === 'predator' ? (lastState.predators || []) : (lastState.herbivores || [])).find((a) => a.id === selection.id);
    selectedAnimal = {
        species: selection.species,
        id: selection.id,
        generation: instant?.generation,
    };
    // Fetch full stats (may include extra fields) and update panel when ready
    console.log(`Fetching full stats for ${selection.species} ${selection.id}`);
    fetchAnimalStats(selection.species, selection.id)
        .then((stats) => {
            console.log('Received full stats:', stats);
            selectedAnimal.generation = stats.generation;
            updateStatsPanel(formatStats(stats));
            drawNeuralNetwork(stats.network_weights);
        })
        .catch((err) => {
            console.warn('Could not fetch full stats:', err);
        });
    drawState(lastState);
});

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
    } catch (error) {
        statusEl.textContent = 'Unable to load simulation state.';
        console.error(error);
    }
});

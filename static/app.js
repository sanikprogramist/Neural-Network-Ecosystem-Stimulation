// This script draws the simulation in the browser and lets the page control it.
// It talks to the server for the current simulation state, draws the scene,
// and handles clicks, start/pause, and one-step updates.

// Grab the important page elements from the HTML by their IDs.
const canvas = document.getElementById('simCanvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const statsEl = document.getElementById('stats');
const toggleButton = document.getElementById('toggleButton');
const stepButton = document.getElementById('stepButton');

// Runtime state for this page.
let running = true;         // whether the simulation should auto-advance
let lastState = null;       // the most recent world state received from the server
let intervalId = null;      // timer reference used for the auto-update loop
const stepIntervalMs = 120; // how often to advance the world, in milliseconds
const dt = 0.04;            // simulation timestep sent to the server each step
let selectedAnimal = null;  // which animal the user clicked on, if any

// Fetch the initial simulation state from the server.
async function fetchState() {
    const response = await fetch('/state');
    return response.json();
}

// Ask the server to advance the simulation by dt and return the next state.
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

// Draw the current world state on the canvas.
function drawState(state) {
    // Erase the previous frame and fill the background.
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#eef3f7';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#495057';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, canvas.width, canvas.height);

    // Use the world size from the server if available.
    const worldWidth = state.world?.width || canvas.width;
    const worldHeight = state.world?.height || canvas.height;

    const scaleX = canvas.width / worldWidth;
    const scaleY = canvas.height / worldHeight;

    // Helper functions to convert world coordinates to canvas pixels.
    function screenX(x) {
        return x * scaleX;
    }

    function screenY(y) {
        return y * scaleY;
    }

    // Draw plants as green dots.
    if (Array.isArray(state.plants)) {
        ctx.fillStyle = '#2f8f3a';
        state.plants.forEach((plant) => {
            ctx.beginPath();
            ctx.arc(screenX(plant.x), screenY(plant.y), 4, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    // Draw herbivores as blue circles with a direction line.
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

    // Draw predators as red circles with a direction line.
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

    // If the user has selected an animal, draw a highlight around it.
    if (selectedAnimal) {
        const currentList = selectedAnimal.species === 'predator' ? state.predators : state.herbivores;
        const currentAnimal = (currentList || []).find((item) => item.id === selectedAnimal.id);
        if (currentAnimal) {
            const pos = { x: currentAnimal.x, y: currentAnimal.y };
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(screenX(pos.x), screenY(pos.y), selectedAnimal.species === 'predator' ? 12 : 10, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    // Update the status bar text with time and counts.
    if (typeof state.world?.time === 'number') {
        statusEl.textContent = `Time: ${state.world.time.toFixed(2)}s | Plants: ${state.plants?.length || 0} | Herbivores: ${state.herbivores?.length || 0} | Predators: ${state.predators?.length || 0}`;
    } else {
        statusEl.textContent = `Plants: ${state.plants?.length || 0} | Herbivores: ${state.herbivores?.length || 0} | Predators: ${state.predators?.length || 0}`;
    }

    // If an animal is selected, refresh the stats panel using the latest state.
    if (selectedAnimal) {
        console.log('Hello from JavaScript');
        const list = selectedAnimal.species === 'predator' ? state.predators : state.herbivores;
        const current = (list || []).find((a) => a.id === selectedAnimal.id);
        if (current) {
            updateStatsPanel(formatStats(current));
        } else {
            updateStatsPanel('Selected animal is no longer alive.');
            selectedAnimal = null;
        }
    }
}

// Convert a mouse click event into the canvas coordinate system.
function getCanvasPos(event) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: ((event.clientX - rect.left) / rect.width) * canvas.width,
        y: ((event.clientY - rect.top) / rect.height) * canvas.height,
    };
}

// Convert canvas coordinates back into world coordinates.
function worldFromCanvas(screenX, screenY, worldWidth, worldHeight) {
    return {
        x: (screenX / canvas.width) * worldWidth,
        y: (screenY / canvas.height) * worldHeight,
    };
}

// Build the HTML text shown for a selected animal's stats.
function formatStats(stats) {
    return `
        <strong>${stats.species.charAt(0).toUpperCase() + stats.species.slice(1)} #${stats.id}</strong><br>
        Position: (${stats.x.toFixed(1)}, ${stats.y.toFixed(1)})<br>
        Age: ${stats.age.toFixed(1)}<br>
        Speed: ${stats.speed.toFixed(2)}<br>
        Satiety: ${stats.satiety.toFixed(2)}<br>
        Generation: ${stats.generation}<br>
        Fitness: ${stats.fitness.toFixed(2)}<br>
        Reproduction: ${(stats.reproduction_progress * 100).toFixed(0)}%<br>
    `;
}

// Update the stats panel HTML content.
function updateStatsPanel(message) {
    statsEl.innerHTML = message;
}

// Ask the server for more detailed stats for a specific animal.
async function fetchAnimalStats(species, id) {
    const response = await fetch(`/animal/${species}/${id}`);
    if (!response.ok) {
        throw new Error(`Unable to load stats for ${species} ${id}`);
    }
    return response.json();
}

// Handle clicks on the canvas to select or deselect animals.
canvas.addEventListener('click', async (event) => {
    if (!lastState) return;

    const worldWidth = lastState.world?.width || canvas.width;
    const worldHeight = lastState.world?.height || canvas.height;
    const canvasPos = getCanvasPos(event);
    const worldPos = worldFromCanvas(canvasPos.x, canvasPos.y, worldWidth, worldHeight);

    const candidates = [];
    const captureDistance = 12; // how close the click must be to an animal to count

    // Add herbivores as possible clicked animals.
    (lastState.herbivores || []).forEach((herbivore) => {
        const dx = herbivore.x - worldPos.x;
        const dy = herbivore.y - worldPos.y;
        candidates.push({ species: 'herbivore', id: herbivore.id, distance: Math.hypot(dx, dy), x: herbivore.x, y: herbivore.y });
    });

    // Add predators as possible clicked animals.
    (lastState.predators || []).forEach((predator) => {
        const dx = predator.x - worldPos.x;
        const dy = predator.y - worldPos.y;
        candidates.push({ species: 'predator', id: predator.id, distance: Math.hypot(dx, dy), x: predator.x, y: predator.y });
    });

    candidates.sort((a, b) => a.distance - b.distance);
    const selection = candidates.find((item) => item.distance <= captureDistance);

    if (!selection) {
        // Clicking empty space removes the current selection.
        selectedAnimal = null;
        updateStatsPanel('Click a herbivore or predator to view stats.');
        drawState(lastState);
        return;
    }

    // If the clicked animal is already selected, deselect it.
    if (selectedAnimal && selectedAnimal.species === selection.species && selectedAnimal.id === selection.id) {
        selectedAnimal = null;
        updateStatsPanel('Click a herbivore or predator to view stats.');
        drawState(lastState);
        return;
    }

    // Otherwise, select the clicked animal.
    selectedAnimal = { species: selection.species, id: selection.id };
    const list = selection.species === 'predator' ? (lastState.predators || []) : (lastState.herbivores || []);
    const instant = list.find((a) => a.id === selection.id);
    if (instant) {
        updateStatsPanel(formatStats(instant));
    } else {
        updateStatsPanel('Selected animal data not available in current state.');
    }

    // Also request more detailed stats from the server and update when ready.
    fetchAnimalStats(selection.species, selection.id)
        .then((stats) => updateStatsPanel(formatStats(stats)))
        .catch((err) => {
            console.warn('Could not fetch full stats:', err);
        });
    drawState(lastState);
});

// One simulation tick: move the world forward and redraw.
async function tick() {
    try {
        const state = await stepWorld();
        lastState = state;
        drawState(state);
    } catch (error) {
        statusEl.textContent = 'Error updating simulation — retrying...';
        console.error('Tick error:', error);
    }
}

// Start the automatic simulation loop.
function startLoop() {
    if (intervalId !== null) return;
    intervalId = setInterval(tick, stepIntervalMs);
    toggleButton.textContent = 'Pause';
    running = true;
}

// Stop the automatic simulation loop.
function stopLoop() {
    if (intervalId !== null) {
        clearInterval(intervalId);
        intervalId = null;
    }
    toggleButton.textContent = 'Start';
    running = false;
}

// Wire the start/pause button.
toggleButton.addEventListener('click', () => {
    if (running) {
        stopLoop();
    } else {
        startLoop();
    }
});

// Wire the manual step button: stop automatic mode and advance once.
stepButton.addEventListener('click', async () => {
    if (running) {
        stopLoop();
    }
    await tick();
});

// When the page loads, fetch the initial state and begin the loop.
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

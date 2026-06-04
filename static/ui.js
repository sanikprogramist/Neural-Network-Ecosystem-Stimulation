import * as api from './api.js';
import * as charts from './charts.js';
import * as settings from './settings.js';
import { drawState } from './viz.js';

let running = false;
let lastState = null;
let lastTickTime = performance.now();

export function initUI() {
    const canvas = document.getElementById('simCanvas');
    const networkCanvas = document.getElementById('networkCanvas');
    const statusEl = document.getElementById('status');
    const statsEl = document.getElementById('stats');
    const toggleButton = document.getElementById('toggleButton');
    const stepButton = document.getElementById('stepButton');
    const saveButton = document.getElementById('saveButton');
    const loadInput = document.getElementById('loadInput');
    const loadButton = document.getElementById('loadButton');
    const killButton = document.getElementById('killButton');
    const saveBrainButton = document.getElementById('saveBrainButton');
    const loadBrainButton = document.getElementById('loadBrainButton');
    const loadBrainInput = document.getElementById('loadBrainInput');
    const downloadChartDataButton = document.getElementById('downloadChartDataButton');
    const settingsButton = document.getElementById('settingsButton');
    const backButton = document.getElementById('backButton');
    const restartSettingsButton = document.getElementById('restartSettingsButton');

    charts.initCharts(document.getElementById('populationChart').getContext('2d'), document.getElementById('agePyramidChart').getContext('2d'));

    async function tick() {
        const now = performance.now();
        const dt = (now - lastTickTime) / 1000;
        lastTickTime = now;
        try {
            const state = await api.stepWorld(dt);
            lastState = state;
            drawState(state, canvas, networkCanvas, statsEl);
            updateBrainButtonStates(lastState);
        } catch (err) {
            statusEl.textContent = 'Error updating simulation — retrying...';
            console.error('Tick error:', err);
        }
    }

    async function loop() {
        while (running) await tick();
    }

    function startLoop() {
        if (running) return;
        running = true;
        lastTickTime = performance.now();
        toggleButton.textContent = 'Pause';
        charts.startChartUpdates();
        loop();
    }

    function stopLoop() {
        running = false;
        toggleButton.textContent = 'Start';
        charts.stopChartUpdates();
    }

    toggleButton.addEventListener('click', () => { running ? stopLoop() : startLoop(); });

    stepButton.addEventListener('click', async () => {
        if (running) stopLoop();
        try {
            const state = await api.stepWorld(0.04);
            lastState = state;
            drawState(state, canvas, networkCanvas, statsEl);
            updateBrainButtonStates(lastState);
        } catch (error) {
            statusEl.textContent = 'Error during step';
            console.error('Step error:', error);
        }
    });

    saveButton.addEventListener('click', () => api.saveSimulation());

    document.getElementById('loadButton').addEventListener('click', () => loadInput.click());
    loadInput.addEventListener('change', async () => {
        const file = loadInput.files[0];
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        await api.uploadLoadForm(formData);
        location.reload();
    });

    killButton.addEventListener('click', async () => { await api.debugKillSelected(); });

    saveBrainButton.addEventListener('click', () => api.saveBrain());

    loadBrainButton.addEventListener('click', () => loadBrainInput.click());
    loadBrainInput.addEventListener('change', async () => {
        const file = loadBrainInput.files[0];
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        try {
            const result = await api.uploadLoadBrainForm(formData);
            if (result.success) {
                statusEl.textContent = result.message;
                // Reload state to show updated brain
                const state = await api.fetchState();
                lastState = state;
                drawState(state, canvas, networkCanvas, statsEl);
                updateBrainButtonStates(lastState);
            } else {
                statusEl.textContent = 'Error: ' + result.error;
            }
        } catch (error) {
            statusEl.textContent = 'Error loading brain: ' + error.message;
            console.error('Brain load error:', error);
        }
        // Reset the input so the same file can be selected again
        loadBrainInput.value = '';
    });

    function updateBrainButtonStates(state) {
        const hasSelection = state && state.selected && state.selected.species && state.selected.id !== undefined;
        saveBrainButton.disabled = !hasSelection;
        loadBrainButton.disabled = !hasSelection;
    }

    canvas.addEventListener('click', async (event) => {
        if (!lastState) return;
        const rect = canvas.getBoundingClientRect();
        const canvasPos = { x: ((event.clientX - rect.left) / rect.width) * canvas.width, y: ((event.clientY - rect.top) / rect.height) * canvas.height };
        const worldWidth = lastState.world?.width || canvas.width;
        const worldHeight = lastState.world?.height || canvas.height;
        const worldPos = { x: (canvasPos.x / canvas.width) * worldWidth, y: (canvasPos.y / canvas.height) * worldHeight };

        const candidates = [];
        const captureDistance = 12;
        (lastState.herbivores || []).forEach(h => { const dx = h.x - worldPos.x; const dy = h.y - worldPos.y; candidates.push({ species: 'herbivore', id: h.id, generation: h.generation, distance: Math.hypot(dx, dy) }); });
        (lastState.predators || []).forEach(p => { const dx = p.x - worldPos.x; const dy = p.y - worldPos.y; candidates.push({ species: 'predator', id: p.id, generation: p.generation, distance: Math.hypot(dx, dy) }); });

        candidates.sort((a, b) => a.distance - b.distance);
        const selection = candidates.find(item => item.distance <= captureDistance);
        if (!selection) {
            await api.sendSelection(null, null);
            drawState(lastState, canvas, networkCanvas, statsEl);
            updateBrainButtonStates(lastState);
            return;
        }

        if (lastState.selected && lastState.selected.species === selection.species && lastState.selected.id === selection.id) {
            await api.sendSelection(null, null);
            drawState(lastState, canvas, networkCanvas, statsEl);
            updateBrainButtonStates(lastState);
            return;
        }

        await api.sendSelection(selection.species, selection.id);
        if (!running) {
            const temp = await api.stepWorld(0.001);
            drawState(temp, canvas, networkCanvas, statsEl);
            updateBrainButtonStates(temp);
        }
    });

    settingsButton.addEventListener('click', async () => {
        stopLoop();
        await settings.syncSlidersWithBackend();
        document.getElementById('simView').style.display = 'none';
        document.getElementById('settingsView').style.display = 'block';
    });

    backButton.addEventListener('click', () => {
        document.getElementById('settingsView').style.display = 'none';
        document.getElementById('simView').style.display = 'block';
        startLoop();
    });

    restartSettingsButton.addEventListener('click', async () => {
        try {
            await settings.commitSettingsFromDOM();
            charts.resetPopulationChart();
            charts.resetChartHistory();
            await api.sendSelection(null, null);
            document.getElementById('networkCanvas').getContext('2d').clearRect(0,0, document.getElementById('networkCanvas').width, document.getElementById('networkCanvas').height);
            document.getElementById('settingsView').style.display = 'none';
            document.getElementById('simView').style.display = 'block';
            settings.updateWorldSettings();
            startLoop();
        } catch (err) {
            console.error('Failed to commit settings config:', err);
            alert('Could not update simulation settings.');
        }
    });

    downloadChartDataButton.addEventListener('click', () => {
        charts.downloadChartHistoryAsCsv();
    });

    // initial load
    window.addEventListener('load', async () => {
        try {
            settings.updateWorldSettings();
            const state = await api.fetchState();
            lastState = state;
            drawState(state, canvas, networkCanvas, statsEl);
            updateBrainButtonStates(lastState);
            document.getElementById('stats').innerHTML = '<div class="empty-stats-msg">Click an animal to view its stats and neural network brain.</div>';
        } catch (error) {
            statusEl.textContent = 'Unable to load simulation state.';
            console.error(error);
        }
    });
}

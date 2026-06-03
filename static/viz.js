import { drawLiveNeuralNetwork } from './network.js';
import { worldSettings} from './settings.js'

export function visionVecToCanvas(normDist, normAngle, heading, visionRange, halfFov, scaleX, scaleY) {
    const actualDist = (1 - normDist) * visionRange;
    const actualAngle = heading + normAngle * halfFov;
    return {
        dx: Math.cos(actualAngle) * actualDist * scaleX,
        dy: Math.sin(actualAngle) * actualDist * scaleY,
    };
}

export function drawVisionOverlay(ctx, animal, scaleX, scaleY) {
    const data = animal.nn_distances_angles;
    const fov = animal.fov;
    const visionRange = animal.vision_range;
    const heading = animal.face_direction;
    const halfFov = fov / 2;
    const cx = animal.x * scaleX;
    const cy = animal.y * scaleY;
    const visionPx = visionRange * scaleX;

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, visionPx, heading - halfFov, heading + halfFov);
    ctx.closePath();
    ctx.fillStyle = 'rgba(200, 200, 210, 0.18)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(160, 160, 180, 0.35)';
    ctx.lineWidth = 1;
    ctx.stroke();

    let targets;
    if (animal.species === 'herbivore') {
        targets = [
            { normDist: data[0], normAngle: data[1], color: '#3ecf60' },
            { normDist: data[2], normAngle: data[3], color: '#5a9ff5' },
            { normDist: data[4], normAngle: data[5], color: '#e84545' },
        ];
    } else if (animal.species === 'predator') {
        targets = [
            { normDist: data[0], normAngle: data[1], color: '#3ecf60' },
            { normDist: data[2], normAngle: data[3], color: '#5a9ff5' },
        ];
    }

    targets.forEach(({ normDist, normAngle, color }) => {
        if (normDist <= 0.0001) return;
        const { dx, dy } = visionVecToCanvas(normDist, normAngle, heading, visionRange, halfFov, scaleX, scaleY);
        const tx = cx + dx;
        const ty = cy + dy;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(tx, ty);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.85;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(tx, ty, 4, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.9;
        ctx.fill();
    });

    ctx.restore();
}

export function formatStats(stats) {
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

export function updateStatsPanel(statsEl, message) {
    if (typeof message === 'string' && message.trim().startsWith('Click an animal')) {
        statsEl.innerHTML = `<div class="empty-stats-msg">${message}</div>`;
    } else if (typeof message === 'string') {
        statsEl.innerHTML = message;
    } else {
        statsEl.innerHTML = formatStats(message);
    }
}

export function drawState(state, canvas, networkCanvas, statsEl) {
    const ctx = canvas.getContext('2d');
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
    const scaleFactor = (scaleX + scaleY) / 2;

    const screenX = x => x * scaleX;
    const screenY = y => y * scaleY;


    if (Array.isArray(state.plants)) {
        ctx.fillStyle = '#2f8f3a';
        state.plants.forEach((plant) => {
            ctx.beginPath();
            ctx.arc(screenX(plant.x), screenY(plant.y), worldSettings.plant_size, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    if (state.selected) {
        drawVisionOverlay(ctx, state.selected, scaleX, scaleY);
        drawLiveNeuralNetwork(state.selected, networkCanvas);
        updateStatsPanel(statsEl, state.selected);
    } else {
        drawLiveNeuralNetwork(null, networkCanvas);
        updateStatsPanel(statsEl, 'Click an animal to view its stats and neural network brain.');
    }

    if (Array.isArray(state.herbivores)) {
        state.herbivores.forEach((herbivore) => {
            const hx = screenX(herbivore.x);
            const hy = screenY(herbivore.y);
            ctx.beginPath();
            ctx.arc(hx, hy, scaleFactor*worldSettings.herbivore_size, 0, Math.PI * 2);
            ctx.fillStyle = `rgb(${herbivore.red}, ${herbivore.green}, ${herbivore.blue})`;
            ctx.fill();
            if (typeof herbivore.angle === 'number') {
                const dirLen = Math.max(8, scaleFactor * worldSettings.herbivore_size * 2.5);
                const dirX = Math.cos(herbivore.angle) * dirLen;
                const dirY = Math.sin(herbivore.angle) * dirLen;
                ctx.strokeStyle = '#0f4bb5';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(hx, hy);
                ctx.lineTo(hx + dirX, hy + dirY);
                ctx.stroke();
            }
        });
    }

    if (Array.isArray(state.predators)) {
        state.predators.forEach((predator) => {
            const x = screenX(predator.x);
            const y = screenY(predator.y);
            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(predator.angle);
            const front = Math.max(8, worldSettings.predator_size * 2.4);
            const backX = -Math.max(5, worldSettings.predator_size * 1.6);
            const backY = Math.max(4, worldSettings.predator_size * 1.6);
            ctx.beginPath();
            ctx.moveTo(front, 0);
            ctx.lineTo(backX, backY);
            ctx.lineTo(backX, -backY);
            ctx.closePath();
            ctx.fillStyle = `rgb(${predator.red}, ${predator.green}, ${predator.blue})`;
            ctx.fill();
            ctx.restore();
            if (typeof predator.angle === 'number') {
                const dirLen = Math.max(8, scaleFactor * worldSettings.predator_size * 2.5);
                const dirX = Math.cos(predator.angle) * dirLen;
                const dirY = Math.sin(predator.angle) * dirLen;
                ctx.strokeStyle = '#870707';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(x + dirX, y + dirY);
                ctx.stroke();
            }
        });
    }

    if (state.selected) {
        const pos = { x: state.selected.x, y: state.selected.y };
        ctx.strokeStyle = '#111111';
        ctx.lineWidth = 2;
        const selRadius = state.selected.species === 'predator' ? worldSettings.predator_size * 2.2 : worldSettings.herbivore_size * 2.2;
        ctx.beginPath();
        ctx.arc(screenX(pos.x), screenY(pos.y), Math.max(8, selRadius), 0, Math.PI * 2);
        ctx.stroke();
    }

    if (typeof state.world?.time === 'number') {
        const statusEl = document.getElementById('status');
        statusEl.textContent = `Time: ${state.world.time.toFixed(2)}s | Plants: ${state.plants?.length || 0} | Herbivores: ${state.herbivores?.length || 0} | Predators: ${state.predators?.length || 0}`;
    }
}

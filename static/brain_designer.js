import * as settings from './settings.js';

const SPECIES_INPUT_COUNTS = {
    herbivore: 9,
    predator: 7,
};

const OUTPUT_LABELS = ['speed', 'angle'];
const MAX_HIDDEN_LAYERS = 10;
const NODE_RADIUS = 18;
const CANVAS_PADDING = 60;

const herbivore_input_tooltips = [
    'Input 1 - distance to closest food object (0-1). 1 means food is very close.',
    'Input 2 - angle to closest food object (-1-1). 1 means food is furthest right relative to facing direction.',
    'Input 3 - distance to closest conspecific (0-1). 1 means conspecific is very close.',
    'Input 4 - angle to closest conspecific (-1-1). 1 means conspecific is furthest right relative to facing direction.',
    'Input 5 - distance to closest predator (0-1). 1 means predator is very close.',
    'Input 6 - angle to closest predator (-1-1). 1 means predator is furthest right relative to facing direction.',
    'Input 7 - hunger (0-1). 1 means animal is extremely hungry.',
    'Input 8 - age (0-1). 1 means animal is about to die of old age.',
    'Input 9 - gestation (0-1). 1 means current pregnancy is about to be completed.'
]

const predator_input_tooltips = [
    'Input 1 - distance to closest food object (0-1). 1 means food is very close.',
    'Input 2 - angle to closest food object (-1-1). 1 means food is furthest right relative to facing direction.',
    'Input 3 - distance to closest conspecific (0-1). 1 means conspecific is very close.',
    'Input 4 - angle to closest conspecific (-1-1). 1 means conspecific is furthest right relative to facing direction.',
    'Input 5 - hunger (0-1). 1 means animal is extremely hungry.',
    'Input 6 - age (0-1). 1 means animal is about to die of old age.',
    'Input 7 - gestation (0-1). 1 means current pregnancy is about to be completed.'
]

export function initBrainDesigner({
    brainDesignerView,
    brainDesignerButton,
    brainBackButton,
    addHiddenLayerButton,
    hiddenLayersList,
    brainSpeciesSelect,
    brainColourPicker,
    brainDesignerCanvas,
    brainDesignerTooltip,
    spawnCountInput,
    spawnWithBrainButton,
    statusEl,
    simView,
    settingsView,
    startLoop,
    stopLoop,
}) {
    let currentSpecies = brainSpeciesSelect.value || 'herbivore';
    let hiddenDims = [];
    const ctx = brainDesignerCanvas.getContext('2d');
    let inputTooltipTargets = [];
    // weight storage: weights[layerIndex][fromIndex][toIndex]
    let weights = [];
    let biases = [];
    let hoveredConnection = null;
    let hoveredBias = null;
    let selectedConnection = null;
    let selectedBias = null;
    // inline input element for editing weights or biases (fixed to viewport so it appears at mouse)
    const weightInput = document.createElement('input');
    weightInput.type = 'number';
    weightInput.step = 'any';
    weightInput.style.position = 'fixed';
    weightInput.style.display = 'none';
    weightInput.style.zIndex = 10000;
    weightInput.style.width = '120px';
    weightInput.style.padding = '6px 8px';
    weightInput.style.borderRadius = '8px';
    weightInput.style.border = '1px solid #cbd5e1';
    document.body.appendChild(weightInput);

    function getHiddenDimLimits() {
        const minHidden = settings.worldSettings?.min_hidden_dim_size ?? 1;
        const maxHidden = settings.worldSettings?.max_hidden_dim_size ?? 10;
        return { minHidden, maxHidden };
    }

    function getInputCount() {
        return SPECIES_INPUT_COUNTS[currentSpecies] ?? 0;
    }

    function resetTopology() {
        hiddenDims = [];
        weights = [];
        biases = [];
        hoveredConnection = null;
        hoveredBias = null;
        selectedConnection = null;
        selectedBias = null;
    }

    function open() {
        if (simView) simView.style.display = 'none';
        if (settingsView) settingsView.style.display = 'none';
        brainDesignerView.style.display = 'block';
        drawDesigner();
    }

    function close() {
        brainDesignerView.style.display = 'none';
        if (simView) simView.style.display = 'block';
        if (startLoop) startLoop();
    }

    function drawDesigner() {
        drawTopologyCanvas();
        renderInputAnnotations();
        renderHiddenLayerControls();
    }

    function drawTopologyCanvas() {
        const width = brainDesignerCanvas.width;
        const height = brainDesignerCanvas.height;
        ctx.clearRect(0, 0, width, height);

        const layers = [getInputCount(), ...hiddenDims, OUTPUT_LABELS.length];
        ensureWeightsForLayers(layers);
        const layerCount = layers.length;
        const horizontalStep = (width - CANVAS_PADDING * 2) / Math.max(layerCount - 1, 1);

        const layerPositions = layers.map((nodeCount, layerIndex) => {
            const x = CANVAS_PADDING + layerIndex * horizontalStep;
            const yStep = nodeCount > 1 ? (height - CANVAS_PADDING * 2) / (nodeCount - 1) : 0;
            const nodes = Array.from({ length: nodeCount }, (_, idx) => ({
                x,
                y: CANVAS_PADDING + idx * yStep,
            }));
            return nodes;
        });

        // draw weighted connections
        for (let layerIndex = 0; layerIndex < layerPositions.length - 1; layerIndex += 1) {
            const fromLayer = layerPositions[layerIndex];
            const toLayer = layerPositions[layerIndex + 1];
            const matrix = weights[layerIndex] || [];
            fromLayer.forEach((fromNode, fromIdx) => {
                toLayer.forEach((toNode, toIdx) => {
                    const w = (matrix[fromIdx] && typeof matrix[fromIdx][toIdx] === 'number') ? matrix[fromIdx][toIdx] : 0;
                    const absw = Math.min(1.0, Math.abs(w) / 5.0);
                    const lineWidth = w === 0 ? 0.5 : 1 + absw * 6;
                    const color = w === 0 ? '#cbd5e1' : (w > 0 ? '#16a34a' : '#ef4444');
                    const isHovered = hoveredConnection && hoveredConnection.layer === layerIndex && hoveredConnection.from === fromIdx && hoveredConnection.to === toIdx;
                    const isSelected = selectedConnection && selectedConnection.layer === layerIndex && selectedConnection.from === fromIdx && selectedConnection.to === toIdx;

                    // highlight background for hover/selection
                    if (isHovered || isSelected) {
                        ctx.beginPath();
                        ctx.moveTo(fromNode.x + NODE_RADIUS, fromNode.y);
                        ctx.lineTo(toNode.x - NODE_RADIUS, toNode.y);
                        ctx.strokeStyle = isSelected ? 'rgba(250,204,21,0.18)' : 'rgba(99,102,241,0.12)';
                        ctx.lineWidth = lineWidth + 6;
                        ctx.stroke();
                    }

                    // main colored line
                    ctx.beginPath();
                    ctx.moveTo(fromNode.x + NODE_RADIUS, fromNode.y);
                    ctx.lineTo(toNode.x - NODE_RADIUS, toNode.y);
                    ctx.strokeStyle = color;
                    ctx.lineWidth = lineWidth;
                    ctx.stroke();
                });
            });
        }

        inputTooltipTargets = [];
        layerPositions.forEach((layerNodes, layerIndex) => {
            const isInputLayer = layerIndex === 0;
            const isOutputLayer = layerIndex === layerPositions.length - 1;
            layerNodes.forEach((node, nodeIndex) => {
                const biasLayerIndex = layerIndex - 1;
                const isBiasNode = biasLayerIndex >= 0;
                const isSelectedBias = selectedBias && selectedBias.layer === biasLayerIndex && selectedBias.neuron === nodeIndex;
                const isHoveredBias = hoveredBias && hoveredBias.layer === biasLayerIndex && hoveredBias.neuron === nodeIndex;

                ctx.beginPath();
                ctx.fillStyle = isInputLayer ? '#93c5fd' : isOutputLayer ? '#fde68a' : '#a5b4fc';
                ctx.strokeStyle = isSelectedBias ? '#f59e0b' : isHoveredBias ? '#6366f1' : '#475569';
                ctx.lineWidth = isSelectedBias ? 4 : isHoveredBias ? 3 : 2;
                ctx.arc(node.x, node.y, NODE_RADIUS, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

                ctx.fillStyle = '#0f172a';
                ctx.font = '14px Inter, sans-serif';
                ctx.textBaseline = 'middle';
                if (isInputLayer) {
                    ctx.textAlign = 'right';
                    ctx.fillText(`In ${nodeIndex + 1}`, node.x - NODE_RADIUS - 12, node.y);
                    const tooltipText = (currentSpecies === 'herbivore'
                        ? herbivore_input_tooltips[nodeIndex]
                        : predator_input_tooltips[nodeIndex]) || `Input neuron ${nodeIndex + 1}`;
                    inputTooltipTargets.push({ x: node.x, y: node.y, label: tooltipText, index: nodeIndex });
                } else if (isOutputLayer) {
                    ctx.textAlign = 'center';
                    ctx.fillText(OUTPUT_LABELS[nodeIndex], node.x, node.y);
                } else {
                    ctx.textAlign = 'center';
                    ctx.fillText(`${nodeIndex + 1}`, node.x, node.y);
                }
            });
        });

        ctx.fillStyle = '#334155';
        ctx.font = '14px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(currentSpecies === 'herbivore' ? 'Herbivore Brain' : 'Predator Brain', width / 2, 24);
    }

    function ensureWeightsForLayers(layers) {
        // layers is array of node counts per layer
        // ensure weights and biases arrays match the topology
        while (weights.length < layers.length - 1) weights.push([]);
        while (weights.length > layers.length - 1) weights.pop();
        while (biases.length < layers.length - 1) biases.push([]);
        while (biases.length > layers.length - 1) biases.pop();

        for (let li = 0; li < layers.length - 1; li += 1) {
            const fromN = layers[li];
            const toN = layers[li + 1];
            const mat = weights[li] || [];
            // adjust rows
            while (mat.length < fromN) mat.push([]);
            while (mat.length > fromN) mat.pop();
            for (let fi = 0; fi < fromN; fi += 1) {
                const row = mat[fi] || [];
                while (row.length < toN) row.push((Math.random() - 0.5) * 0.6);
                while (row.length > toN) row.pop();
                mat[fi] = row;
            }
            weights[li] = mat;

            const biasVec = biases[li] || [];
            while (biasVec.length < toN) biasVec.push((Math.random() - 0.5) * 0.6);
            while (biasVec.length > toN) biasVec.pop();
            biases[li] = biasVec;
        }
    }

    function renderInputAnnotations() {
        const count = getInputCount();
        for (let i = 1; i <= count; i += 1) {
            const chip = document.createElement('span');
            chip.textContent = `Input ${i}`;
            const tip = (currentSpecies === 'herbivore' ? herbivore_input_tooltips[i - 1] : predator_input_tooltips[i - 1]) || `Input ${i}`;
            chip.title = tip;
            chip.style.background = '#f1f5f9';
            chip.style.color = '#0f172a';
            chip.style.padding = '8px 10px';
            chip.style.borderRadius = '999px';
            chip.style.fontSize = '13px';
            chip.style.border = '1px solid #cbd5e1';
        }
    }

    function renderHiddenLayerControls() {
        hiddenLayersList.innerHTML = '';
        if (hiddenDims.length === 0) {
            const message = document.createElement('span');
            message.textContent = 'No hidden layers. Add one to begin designing topology.';
            message.style.color = '#475569';
            message.style.fontSize = '13px';
            hiddenLayersList.appendChild(message);
            updateHiddenButtons();
            return;
        }

        hiddenDims.forEach((count, index) => {
            const container = document.createElement('div');
            container.style.display = 'flex';
            container.style.gap = '6px';
            container.style.alignItems = 'center';
            container.style.background = '#f8fafc';
            container.style.border = '1px solid #e2e8f0';
            container.style.padding = '8px 10px';
            container.style.borderRadius = '12px';

            const label = document.createElement('span');
            label.textContent = `Layer ${index + 1}: ${count} neuron${count === 1 ? '' : 's'}`;
            label.style.fontSize = '13px';
            label.style.fontWeight = '600';
            label.style.color = '#0f172a';

            const decrease = document.createElement('button');
            decrease.textContent = '-';
            decrease.style.width = '28px';
            decrease.style.height = '28px';
            decrease.style.background = '#f8fafc';
            decrease.style.color = '#dc2626';
            decrease.style.border = '1px solid #e2e8f0';
            decrease.style.borderRadius = '8px';
            decrease.disabled = count <= getHiddenDimLimits().minHidden;
            decrease.addEventListener('click', () => {
                adjustHiddenLayer(index, -1);
            });

            const increase = document.createElement('button');
            increase.textContent = '+';
            increase.style.width = '28px';
            increase.style.height = '28px';
            increase.style.background = '#f8fafc';
            increase.style.color = '#16a34a';
            increase.style.border = '1px solid #e2e8f0';
            increase.style.borderRadius = '8px';
            increase.disabled = count >= getHiddenDimLimits().maxHidden;
            increase.addEventListener('click', () => {
                adjustHiddenLayer(index, 1);
            });

            const remove = document.createElement('button');
            remove.textContent = 'Remove';
            remove.style.background = '#ef4444';
            remove.style.color = 'white';
            remove.style.border = 'none';
            remove.style.borderRadius = '8px';
            remove.style.padding = '8px 10px';
            remove.addEventListener('click', () => {
                removeHiddenLayer(index);
            });

            container.append(label, decrease, increase, remove);
            hiddenLayersList.appendChild(container);
        });
        updateHiddenButtons();
    }

    function updateHiddenButtons() {
        addHiddenLayerButton.disabled = hiddenDims.length >= MAX_HIDDEN_LAYERS;
    }

    function adjustHiddenLayer(index, delta) {
        const { minHidden, maxHidden } = getHiddenDimLimits();
        hiddenDims[index] = Math.min(maxHidden, Math.max(minHidden, hiddenDims[index] + delta));
        drawDesigner();
    }

    function addHiddenLayer() {
        const { minHidden } = getHiddenDimLimits();
        if (hiddenDims.length >= MAX_HIDDEN_LAYERS) return;
        hiddenDims.push(minHidden);
        drawDesigner();
    }

    function removeHiddenLayer(index = hiddenDims.length - 1) {
        if (hiddenDims.length === 0) return;
        hiddenDims.splice(index, 1);
        drawDesigner();
    }

    function setTooltip(text, x, y) {
        if (!brainDesignerTooltip) return;
        brainDesignerTooltip.textContent = text;
        brainDesignerTooltip.style.display = 'block';
        brainDesignerTooltip.style.left = `${x + 18}px`;
        brainDesignerTooltip.style.top = `${y + 12}px`;
    }

    function hideTooltip() {
        if (!brainDesignerTooltip) return;
        brainDesignerTooltip.style.display = 'none';
    }

    function handleCanvasPointer(event) {
        if (!brainDesignerTooltip) return;
        const rect = brainDesignerCanvas.getBoundingClientRect();
        const x = ((event.clientX - rect.left) / rect.width) * brainDesignerCanvas.width;
        const y = ((event.clientY - rect.top) / rect.height) * brainDesignerCanvas.height;
        const target = inputTooltipTargets.find(node => Math.hypot(node.x - x, node.y - y) <= NODE_RADIUS + 4);
        if (target) {
            setTooltip(target.label, event.clientX - rect.left, event.clientY - rect.top);
            brainDesignerCanvas.style.cursor = 'pointer';
        } else {
            hideTooltip();
            brainDesignerCanvas.style.cursor = 'default';
        }

        const layers = [getInputCount(), ...hiddenDims, OUTPUT_LABELS.length];
        const horizontalStep = (brainDesignerCanvas.width - CANVAS_PADDING * 2) / Math.max(layers.length - 1, 1);
        const layerPositions = layers.map((nodeCount, layerIndex) => {
            const xPos = CANVAS_PADDING + layerIndex * horizontalStep;
            const yStep = nodeCount > 1 ? (brainDesignerCanvas.height - CANVAS_PADDING * 2) / (nodeCount - 1) : 0;
            return Array.from({ length: nodeCount }, (_, idx) => ({ x: xPos, y: CANVAS_PADDING + idx * yStep }));
        });

        let nearestBiasCandidate = null;
        let nearestBiasDist = Infinity;
        layerPositions.forEach((layerNodes, layerIndex) => {
            if (layerIndex === 0) return;
            const biasLayerIndex = layerIndex - 1;
            layerNodes.forEach((node, nodeIndex) => {
                const d = Math.hypot(node.x - x, node.y - y);
                if (d < nearestBiasDist) {
                    nearestBiasDist = d;
                    nearestBiasCandidate = { layer: biasLayerIndex, neuron: nodeIndex, x: node.x, y: node.y, dist: d, layerIndex };
                }
            });
        });

        let nearestConnection = null;
        let nearestConnectionDist = Infinity;
        for (let li = 0; li < layerPositions.length - 1; li += 1) {
            const fromLayer = layerPositions[li];
            const toLayer = layerPositions[li + 1];
            for (let fi = 0; fi < fromLayer.length; fi += 1) {
                for (let tj = 0; tj < toLayer.length; tj += 1) {
                    const p1 = { x: fromLayer[fi].x + NODE_RADIUS, y: fromLayer[fi].y };
                    const p2 = { x: toLayer[tj].x - NODE_RADIUS, y: toLayer[tj].y };
                    const d = pointToSegmentDistance(x, y, p1.x, p1.y, p2.x, p2.y);
                    if (d < nearestConnectionDist) {
                        nearestConnectionDist = d;
                        nearestConnection = { layer: li, from: fi, to: tj, dist: d, p1, p2 };
                    }
                }
            }
        }

        const hoverThreshold = 6; // canvas units
        const biasThreshold = NODE_RADIUS + 6;
        if (nearestBiasCandidate && nearestBiasCandidate.dist <= biasThreshold && nearestBiasCandidate.dist <= nearestConnectionDist) {
            if (!hoveredBias || hoveredBias.layer !== nearestBiasCandidate.layer || hoveredBias.neuron !== nearestBiasCandidate.neuron) {
                hoveredBias = { layer: nearestBiasCandidate.layer, neuron: nearestBiasCandidate.neuron };
                hoveredConnection = null;
                setTooltip(`Bias ${nearestBiasCandidate.neuron + 1} on ${nearestBiasCandidate.layer === biases.length - 1 ? 'output' : 'hidden'} layer`, event.clientX - rect.left, event.clientY - rect.top);
                brainDesignerCanvas.style.cursor = 'pointer';
                drawDesigner();
            }
        } else {
            if (hoveredBias) {
                hoveredBias = null;
                drawDesigner();
            }
            if (nearestConnection && nearestConnection.dist <= hoverThreshold) {
                if (!hoveredConnection || hoveredConnection.layer !== nearestConnection.layer || hoveredConnection.from !== nearestConnection.from || hoveredConnection.to !== nearestConnection.to) {
                    hoveredConnection = { layer: nearestConnection.layer, from: nearestConnection.from, to: nearestConnection.to };
                    drawDesigner();
                }
            } else if (hoveredConnection) {
                hoveredConnection = null;
                drawDesigner();
            }
        }
    }

    brainDesignerCanvas.addEventListener('mousemove', handleCanvasPointer);
    brainDesignerCanvas.addEventListener('mouseleave', () => {
        hideTooltip();
        brainDesignerCanvas.style.cursor = 'default';
        hoveredConnection = null;
        drawDesigner();
    });

    brainDesignerCanvas.addEventListener('click', (event) => {
        const rect = brainDesignerCanvas.getBoundingClientRect();
        if (hoveredBias) {
            selectedBias = { ...hoveredBias };
            selectedConnection = null;
            weightInput.placeholder = 'bias';
            weightInput.style.left = `${event.clientX + 8}px`;
            weightInput.style.top = `${event.clientY + 8}px`;
            const current = biases[selectedBias.layer][selectedBias.neuron];
            weightInput.value = String(current);
            weightInput.style.display = 'block';
            weightInput.focus();
            weightInput.select();
            drawDesigner();
        } else if (hoveredConnection) {
            selectedConnection = { ...hoveredConnection };
            selectedBias = null;
            weightInput.placeholder = 'weight';
            weightInput.style.left = `${event.clientX + 8}px`;
            weightInput.style.top = `${event.clientY + 8}px`;
            const current = weights[selectedConnection.layer][selectedConnection.from][selectedConnection.to];
            weightInput.value = String(current);
            weightInput.style.display = 'block';
            weightInput.focus();
            weightInput.select();
            drawDesigner();
        } else {
            if (selectedConnection || selectedBias) {
                selectedConnection = null;
                selectedBias = null;
                weightInput.style.display = 'none';
                drawDesigner();
            }
        }
    });

    weightInput.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter') commitWeightInput();
        else if (ev.key === 'Escape') {
            selectedConnection = null;
            weightInput.style.display = 'none';
            drawDesigner();
        }
    });

    weightInput.addEventListener('blur', () => {
        commitWeightInput();
    });

    function commitWeightInput() {
        if (!selectedConnection && !selectedBias) return;
        const v = parseFloat(weightInput.value);
        if (!Number.isNaN(v)) {
            const clamped = Math.max(-5, Math.min(5, v));
            if (selectedConnection) {
                weights[selectedConnection.layer][selectedConnection.from][selectedConnection.to] = clamped;
            } else if (selectedBias) {
                biases[selectedBias.layer][selectedBias.neuron] = clamped;
            }
        }
        selectedConnection = null;
        selectedBias = null;
        weightInput.style.display = 'none';
        drawDesigner();
    }

    function pointToSegmentDistance(px, py, x1, y1, x2, y2) {
        const A = px - x1;
        const B = py - y1;
        const C = x2 - x1;
        const D = y2 - y1;
        const dot = A * C + B * D;
        const len_sq = C * C + D * D;
        let param = -1;
        if (len_sq !== 0) param = dot / len_sq;
        let xx, yy;
        if (param < 0) {
            xx = x1;
            yy = y1;
        } else if (param > 1) {
            xx = x2;
            yy = y2;
        } else {
            xx = x1 + param * C;
            yy = y1 + param * D;
        }
        const dx = px - xx;
        const dy = py - yy;
        return Math.sqrt(dx * dx + dy * dy);
    }

    function setSpecies(species) {
        currentSpecies = species;
        resetTopology();
        drawDesigner();
    }

    function createBrainTemplate() {
        const template = {
            species: currentSpecies,
            inputs: getInputCount(),
            hiddenDims: [...hiddenDims],
            outputs: OUTPUT_LABELS.length,
            colour: brainColourPicker.value,
            spawnCount: parseInt(spawnCountInput.value, 10) || 1,
            biases: biases.map(layer => [...layer]),
        };
        return template;
    }

    brainDesignerButton.addEventListener('click', () => {
        stopLoop();
        open();
    });

    brainBackButton.addEventListener('click', () => {
        close();
        startLoop();
    });

    addHiddenLayerButton.addEventListener('click', addHiddenLayer);

    brainSpeciesSelect.addEventListener('change', (event) => {
        setSpecies(event.target.value);
    });

    spawnWithBrainButton.addEventListener('click', async () => {
        const template = createBrainTemplate();
        try {
            const response = await fetch('/spawn_with_brain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    species: template.species,
                    hiddenDims: template.hiddenDims,
                    weights: weights,
                    biases: template.biases,
                    spawnCount: template.spawnCount,
                    color: template.colour
                })
            });
            const result = await response.json();
            if (result.success) {
                statusEl.textContent = result.message;
                close();
                startLoop();
            } else {
                statusEl.textContent = 'Error: ' + (result.error || 'Unknown error');
            }
        } catch (err) {
            statusEl.textContent = 'Error spawning animals: ' + err.message;
            console.error('Spawn error:', err);
        }
    });

    drawDesigner();

    return {
        open,
        close,
    };
}

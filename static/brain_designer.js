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

        ctx.strokeStyle = '#cbd5e1';
        ctx.lineWidth = 1.5;
        for (let layerIndex = 0; layerIndex < layerPositions.length - 1; layerIndex += 1) {
            const fromLayer = layerPositions[layerIndex];
            const toLayer = layerPositions[layerIndex + 1];
            fromLayer.forEach(fromNode => {
                toLayer.forEach(toNode => {
                    ctx.beginPath();
                    ctx.moveTo(fromNode.x + NODE_RADIUS, fromNode.y);
                    ctx.lineTo(toNode.x - NODE_RADIUS, toNode.y);
                    ctx.stroke();
                });
            });
        }

        inputTooltipTargets = [];
        layerPositions.forEach((layerNodes, layerIndex) => {
            const isInputLayer = layerIndex === 0;
            const isOutputLayer = layerIndex === layerPositions.length - 1;
            layerNodes.forEach((node, nodeIndex) => {
                ctx.beginPath();
                ctx.fillStyle = isInputLayer ? '#93c5fd' : isOutputLayer ? '#fde68a' : '#a5b4fc';
                ctx.strokeStyle = '#475569';
                ctx.lineWidth = 2;
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
        if (!brainDesignerTooltip || inputTooltipTargets.length === 0) return;
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
    }

    brainDesignerCanvas.addEventListener('mousemove', handleCanvasPointer);
    brainDesignerCanvas.addEventListener('mouseleave', () => {
        hideTooltip();
        brainDesignerCanvas.style.cursor = 'default';
    });

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

    spawnWithBrainButton.addEventListener('click', () => {
        const template = createBrainTemplate();
        statusEl.textContent = `Designed a ${template.species} brain with ${template.hiddenDims.length} hidden layer${template.hiddenDims.length === 1 ? '' : 's'}. Spawn feature coming soon.`;
    });

    drawDesigner();

    return {
        open,
        close,
    };
}

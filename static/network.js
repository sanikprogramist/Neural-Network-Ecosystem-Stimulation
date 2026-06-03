

export function drawLiveNeuralNetwork(nn, networkCanvas, neuralNetPulsesEnabled = true) {
    const networkCtx = networkCanvas.getContext('2d');
    const w = networkCanvas.width = networkCanvas.clientWidth;
    const h = networkCanvas.height = networkCanvas.clientHeight;

    if (!nn) {
        return;
    }

    const padding = 30;
    const nodeRadius = 6;
    const t = performance.now() / 1000;

    networkCtx.clearRect(0, 0, w, h);
    networkCtx.fillStyle = '#f5f7fb';
    networkCtx.fillRect(0, 0, w, h);

    const layers = [];
    layers.push({ label: 'Input', values: nn.inputs || [], x: 60 });

    const hActivations = nn.hidden_layers_activations || [];
    hActivations.forEach((actValues, index) => {
        const progressionFraction = (index + 1) / (hActivations.length + 1);
        const columnX = 60 + progressionFraction * (w - 120);
        layers.push({ label: `Hidden ${index + 1}`, values: actValues, x: columnX });
    });

    layers.push({ label: 'Output', values: nn.output || [], x: w - 60 });

    if (hActivations.length > 0) {
        for (let i = 1; i <= hActivations.length; i++) {
            layers[i].x = 60 + (i / (hActivations.length + 1)) * (w - 120);
        }
    }

    const weightMatrices = [];
    if (nn.weights) {
        let layerIdx = 0;
        while (nn.weights[`layer_${layerIdx}_weights`] !== undefined) {
            weightMatrices.push(nn.weights[`layer_${layerIdx}_weights`]);
            layerIdx++;
        }
        if (nn.weights.out_weights) weightMatrices.push(nn.weights.out_weights);
    }

    const getY = (layerIndex, i) => {
        const layer = layers[layerIndex];
        if (!layer || !layer.values || layer.values.length === 0) return padding;
        if (layer.values.length === 1) return h / 2;
        const spacing = (h - (padding * 2.5)) / (layer.values.length - 1);
        return padding + i * spacing;
    };

    for (let l = 0; l < layers.length - 1; l++) {
        const from = layers[l];
        const to = layers[l + 1];
        if (!from || !to || !from.values || !to.values) continue;
        for (let i = 0; i < from.values.length; i++) {
            for (let j = 0; j < to.values.length; j++) {
                const x1 = from.x;
                const y1 = getY(l, i);
                const x2 = to.x;
                const y2 = getY(l + 1, j);

                let edgeColor = 'rgba(120,120,140,0.12)';
                let edgeWidth = 0.5;
                let signal = 0;
                let absSignal = 0;

                if (weightMatrices && weightMatrices[l] && weightMatrices[l][j]) {
                    const weight = weightMatrices[l][j][i];
                    if (weight !== undefined && weight !== null) {
                        const fromValue = from.values[i] !== undefined ? from.values[i] : 0;
                        signal = weight * fromValue;
                        absSignal = Math.abs(signal);
                        edgeWidth = 0.5 + Math.min(absSignal, 1.0) * 3.5;
                        const alpha = 0.08 + Math.min(absSignal, 1.0) * 0.75;
                        edgeColor = signal > 0
                            ? `rgba(40, 200, 80, ${alpha})`
                            : signal < 0
                                ? `rgba(220, 55, 55, ${alpha})`
                                : `rgba(120, 120, 140, 0.08)`;
                    }
                }

                networkCtx.beginPath();
                networkCtx.moveTo(x1, y1);
                networkCtx.lineTo(x2, y2);
                networkCtx.strokeStyle = edgeColor;
                networkCtx.lineWidth = edgeWidth;
                networkCtx.stroke();

                if (neuralNetPulsesEnabled && absSignal >= 0.05) {
                    const speed = 0.3 + Math.min(absSignal, 1.0) * 1.2;
                    const phaseOffset = ((l * 97 + i * 31 + j * 13) % 100) / 100;
                    const pulsePos = (t * speed + phaseOffset) % 1.0;
                    const px = x1 + (x2 - x1) * pulsePos;
                    const py = y1 + (y2 - y1) * pulsePos;
                    const pulseRadius = 1.5 + Math.min(absSignal, 1.0) * 2.5;
                    const pulseAlpha = 0.5 + Math.min(absSignal, 1.0) * 0.5;
                    const pulseColor = signal > 0 ? `rgba(80, 240, 120,` : `rgba(255, 80, 80,`;

                    const glow = networkCtx.createRadialGradient(px, py, 0, px, py, pulseRadius * 3);
                    glow.addColorStop(0, `${pulseColor}${pulseAlpha})`);
                    glow.addColorStop(1, `${pulseColor}0)`);
                    networkCtx.beginPath();
                    networkCtx.arc(px, py, pulseRadius * 3, 0, Math.PI * 2);
                    networkCtx.fillStyle = glow;
                    networkCtx.fill();

                    networkCtx.beginPath();
                    networkCtx.arc(px, py, pulseRadius, 0, Math.PI * 2);
                    networkCtx.fillStyle = `${pulseColor}${pulseAlpha})`;
                    networkCtx.fill();
                }
            }
        }
    }

    for (let l = 0; l < layers.length; l++) {
        const layer = layers[l];
        if (!layer || !layer.values) continue;
        for (let i = 0; i < layer.values.length; i++) {
            const v = layer.values[i] !== undefined ? layer.values[i] : 0;
            const n = Math.max(-1, Math.min(1, v));
            const red = Math.round(Math.max(0, -n) * 255);
            const green = Math.round(Math.max(0, n) * 255);
            const alpha = 0.3 + Math.abs(n) * 0.7;
            const x = layer.x;
            const y = getY(l, i);

            networkCtx.beginPath();
            networkCtx.arc(x, y, nodeRadius + Math.abs(n) * 4, 0, Math.PI * 2);
            networkCtx.fillStyle = `rgba(${red},${green},80,${alpha * 0.4})`;
            networkCtx.fill();

            networkCtx.beginPath();
            networkCtx.arc(x, y, nodeRadius, 0, Math.PI * 2);
            networkCtx.fillStyle = `rgba(${red},${green},80,${alpha})`;
            networkCtx.fill();
        }
    }

    networkCtx.fillStyle = '#111827';
    networkCtx.font = '12px sans-serif';
    networkCtx.textAlign = 'center';
    layers.forEach((layer) => {
        networkCtx.fillText(layer.label, layer.x, h - 8);
    });
}

import { fetchChartData } from './api.js';

let populationChart = null;
let agePyramidChart = null;
let chartInterval = null;
let chartHistory = [];

function cloneChartData(data) {
    return JSON.parse(JSON.stringify(data || {}));
}

function buildAgePyramid(data) {
    const ageGroups = [0, 20, 40, 60, 80, 100, Infinity];
    const herbivoreCounts = new Array(ageGroups.length - 1).fill(0);
    const predatorCounts = new Array(ageGroups.length - 1).fill(0);

    (data.alive_herbivore_ages || []).forEach(age => {
        for (let i = 0; i < ageGroups.length - 1; i++) {
            if (age >= ageGroups[i] && age < ageGroups[i + 1]) {
                herbivoreCounts[i]++;
                break;
            }
        }
    });

    (data.alive_predator_ages || []).forEach(age => {
        for (let i = 0; i < ageGroups.length - 1; i++) {
            if (age >= ageGroups[i] && age < ageGroups[i + 1]) {
                predatorCounts[i]++;
                break;
            }
        }
    });

    return {
        herbivore: herbivoreCounts.reverse().map(c => -c),
        predator: predatorCounts.reverse(),
    };
}

export function initCharts(popCtx, ageCtx) {
    populationChart = new Chart(popCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Plants', data: [], borderColor: '#6082e7', fill: false },
                { label: 'Herbivores', data: [], borderColor: 'rgba(76, 175, 80, 1)', fill: false },
                { label: 'Predators', data: [], borderColor: '#b34700', fill: false }
            ]
        },
        options: {
            responsive: true,
            animation: false,
            scales: {
                x: { title: { display: true, text: 'World Time' } },
                y: { title: { display: true, text: 'Population' }, beginAtZero: true }
            }
        }
    });

    agePyramidChart = new Chart(ageCtx, {
        type: 'bar',
        data: {
            labels: ['100+', '80-100', '60-80', '40-60', '20-40', '0-20'],
            datasets: [
                { label: 'Herbivores', data: [0, 0, 0, 0, 0, 0], backgroundColor: 'rgba(76, 175, 80, 0.85)', borderColor: 'rgba(76, 175, 80, 1)', borderWidth: 0.5, stack: 'population' },
                { label: 'Predators', data: [0, 0, 0, 0, 0, 0], backgroundColor: '#b34700', borderColor: '#b34700', borderWidth: 0.5, stack: 'population' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            animation: false,
            indexAxis: 'y',
            scales: {
                x: { ticks: { callback: v => Math.abs(v) } },
                y: {}
            }
        }
    });
}

export async function refreshCharts() {
    try {
        const data = await fetchChartData();
        if (!data) return;

        chartHistory.push(cloneChartData(data));

        populationChart.data.labels.push(data.world_time.toFixed(1));
        populationChart.data.datasets[0].data.push(data.current_plant);
        populationChart.data.datasets[1].data.push(data.current_herbivore);
        populationChart.data.datasets[2].data.push(data.current_predator);

        const maxPoints = 150;
        if (populationChart.data.labels.length > maxPoints) {
            populationChart.data.labels.shift();
            populationChart.data.datasets.forEach(d => d.data.shift());
        }
        populationChart.update();

        const pyramid = buildAgePyramid(data);
        agePyramidChart.data.datasets[0].data = pyramid.herbivore;
        agePyramidChart.data.datasets[1].data = pyramid.predator;

        const maxValue = Math.max(...pyramid.herbivore.map(Math.abs), ...pyramid.predator);
        const bound = Math.max(40, Math.ceil(maxValue / 40) * 40);
        agePyramidChart.options.scales.x.max = bound;
        agePyramidChart.options.scales.x.min = -bound;
        agePyramidChart.update();
    } catch (err) {
        console.error('Chart update failed:', err);
    }
}

export function startChartUpdates() {
    if (!chartInterval) {
        chartInterval = setInterval(refreshCharts, 3000);
        refreshCharts();
    }
}

export function stopChartUpdates() {
    if (chartInterval) {
        clearInterval(chartInterval);
        chartInterval = null;
    }
}

export function resetPopulationChart() {
    if (!populationChart) return;
    populationChart.data.labels = [];
    populationChart.data.datasets.forEach(d => d.data = []);
    populationChart.update();
    if (agePyramidChart) {
        agePyramidChart.data.datasets[0].data = [0, 0, 0, 0, 0, 0];
        agePyramidChart.data.datasets[1].data = [0, 0, 0, 0, 0, 0];
        agePyramidChart.update();
    }
}

export function resetChartHistory() {
    chartHistory = [];
}

function escapeCsvValue(value) {
    if (value == null) return '';
    if (typeof value === 'object') value = JSON.stringify(value);
    const stringValue = String(value);
    const escaped = stringValue.replace(/"/g, '""');
    return `"${escaped}"`;
}

export function downloadChartHistoryAsCsv(filename = 'chart_history.csv') {
    if (!chartHistory.length) {
        alert('No chart history has been recorded yet. Start the simulation to accumulate data.');
        return;
    }

    const columns = Object.keys(chartHistory[0]);
    const csvRows = [columns.map(escapeCsvValue).join(',')];

    for (const row of chartHistory) {
        const values = columns.map(key => escapeCsvValue(row[key]));
        csvRows.push(values.join(','));
    }

    const csvContent = csvRows.join('\r\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

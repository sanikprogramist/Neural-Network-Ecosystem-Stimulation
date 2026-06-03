import { fetchChartData } from './api.js';

let populationChart = null;
let agePyramidChart = null;
let popChartInterval = null;
let agePyramidInterval = null;

export function initCharts(popCtx, ageCtx) {
    populationChart = new Chart(popCtx, {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'Plants', data: [], borderColor: '#6082e7', fill: false },
            { label: 'Herbivores', data: [], borderColor: 'rgba(76, 175, 80, 1)', fill: false },
            { label: 'Predators', data: [], borderColor: '#b34700', fill: false }
        ]},
        options: { responsive: true, animation: false, scales: { x: { title: { display: true, text: 'World Time' }}, y: { title: { display: true, text: 'Population' }, beginAtZero: true } } }
    });

    agePyramidChart = new Chart(ageCtx, {
        type: 'bar',
        data: { labels: ['100+', '80-100', '60-80', '40-60', '20-40', '0-20'], datasets: [
            { label: 'Herbivores', data: [0,0,0,0,0,0], backgroundColor: 'rgba(76, 175, 80, 0.85)', borderColor: 'rgba(76, 175, 80, 1)', borderWidth: 0.5, stack: 'population' },
            { label: 'Predators', data: [0,0,0,0,0,0], backgroundColor: '#b34700', borderColor: '#b34700', borderWidth: 0.5, stack: 'population' }
        ]},
        options: { responsive: true, maintainAspectRatio: true, animation: false, indexAxis: 'y', scales: { x: { ticks: { callback: v => Math.abs(v) } }, y: { } } }
    });
}

export async function updatePopulationChart() {
    try {
        const data = await fetchChartData();
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
    } catch (err) {
        console.error('Chart update failed:', err);
    }
}

export async function updateAgePyramid() {
    try {
        const data = await fetchChartData();
        const ageGroups = [0,20,40,60,80,100,Infinity];
        const herbivoreCounts = new Array(ageGroups.length - 1).fill(0);
        (data.alive_herbivore_ages || []).forEach(age => {
            for (let i=0;i<ageGroups.length-1;i++) if (age>=ageGroups[i] && age<ageGroups[i+1]) { herbivoreCounts[i]++; break }
        });
        const predatorCounts = new Array(ageGroups.length - 1).fill(0);
        (data.alive_predator_ages || []).forEach(age => {
            for (let i=0;i<ageGroups.length-1;i++) if (age>=ageGroups[i] && age<ageGroups[i+1]) { predatorCounts[i]++; break }
        });

        const reversedHerbivoreCounts = herbivoreCounts.reverse().map(c => -c);
        const reversedPredatorCounts = predatorCounts.reverse();

        agePyramidChart.data.datasets[0].data = reversedHerbivoreCounts;
        agePyramidChart.data.datasets[1].data = reversedPredatorCounts;

        const maxValue = Math.max(...reversedHerbivoreCounts.map(Math.abs), ...reversedPredatorCounts);
        agePyramidChart.options.scales.x.max = maxValue;
        agePyramidChart.options.scales.x.min = -maxValue;
        agePyramidChart.update();
    } catch (err) {
        console.error('Age pyramid update failed:', err);
    }
}

export function startChartUpdates() {
    if (!popChartInterval) {
        popChartInterval = setInterval(updatePopulationChart, 3000);
        agePyramidInterval = setInterval(updateAgePyramid, 3000);
    }
}

export function stopChartUpdates() {
    if (popChartInterval) {
        clearInterval(popChartInterval);
        clearInterval(agePyramidInterval);
        popChartInterval = null;
        agePyramidInterval = null;
    }
}

export function resetPopulationChart() {
    if (!populationChart) return;
    populationChart.data.labels = [];
    populationChart.data.datasets.forEach(d => d.data = []);
    populationChart.update();
}

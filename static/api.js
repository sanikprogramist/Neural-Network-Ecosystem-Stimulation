// Lightweight API wrappers for backend endpoints
export async function fetchState() {
    const res = await fetch('/state');
    return res.json();
}

export async function fetchChartData() {
    const res = await fetch('/chart');
    return res.json();
}

export async function stepWorld(dt) {
    const res = await fetch('/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dt }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function sendSelection(species, id) {
    await fetch('/select_animal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ species, id }),
    });
}

export async function fetchSettings() {
    const res = await fetch('/settings');
    if (!res.ok) throw new Error('Could not fetch settings');
    return res.json();
}

export async function restartSimulation(data) {
    const res = await fetch('/restart_simulation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function uploadLoadForm(formData) {
    return fetch('/load', { method: 'POST', body: formData });
}

export async function saveSimulation() {
    // navigate to save endpoint
    window.location.href = '/save';
}

export async function debugKillSelected() {
    return fetch('/debug_kill_selected', { method: 'POST' });
}

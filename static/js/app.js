/* ============================================================
   PatchCascade SOC — Main Application Controller
   State management, API integration, simulation loop
   ============================================================ */

// ============================================================
// API Configuration
// ============================================================
const API_BASE = window.location.origin;

// ============================================================
// Global State
// ============================================================
const state = {
    initialized: false,
    running: false,
    autoRunning: false,
    autoRunTimer: null,
    selectedTask: 'easy',
    currentObservation: null,
    turn: 0,
    maxTurns: 50,
    done: false,
    totalReward: 0,
    rewards: [],
    totalCascades: 0,
    lastStepInfo: null,
    gradeResults: null,
};

// ============================================================
// Visualization Instances
// ============================================================
let topology, particles, rewardChart, radarChart;

// ============================================================
// Initialization
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    // Initialize visualizations
    topology = new TopologyRenderer('topology-canvas');
    particles = new ParticleBackground('particles-bg');
    rewardChart = new RewardChart('reward-chart');
    radarChart = new RadarChart('radar-canvas');

    // Start renderers
    particles.start();
    topology.start();

    // Start render loops for charts
    startChartLoop();

    // Speed slider
    const slider = document.getElementById('speed-slider');
    slider.addEventListener('input', (e) => {
        document.getElementById('speed-value').textContent = e.target.value + 'ms';
    });

    // Set initial active tab
    document.getElementById('tab-dashboard').classList.add('active');

    // Check server health
    checkServerHealth();

    showToast('Dashboard ready. Initialize a task to begin.', 'info');
});

function startChartLoop() {
    function loop() {
        rewardChart.render();
        radarChart.render();
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
}

// ============================================================
// Server Health Check
// ============================================================
async function checkServerHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
            setStatus('server-status', 'green');
            setStatus('env-status', 'yellow');
        } else {
            setStatus('server-status', 'red');
        }
    } catch (e) {
        setStatus('server-status', 'red');
        showToast('Server not reachable. Start the FastAPI server first.', 'error');
    }
}

function setStatus(elementId, color) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const dot = el.querySelector('.dot');
    if (!dot) return;
    dot.className = 'dot';
    if (color === 'yellow') dot.classList.add('yellow');
    else if (color === 'red') dot.classList.add('red');
    // green is default (no extra class)
}

// ============================================================
// View Switching
// ============================================================
function switchView(view) {
    document.querySelectorAll('.dashboard-view, .architecture-view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));

    if (view === 'architecture') {
        document.getElementById('view-architecture').classList.add('active');
        document.getElementById('tab-architecture').classList.add('active');
    } else {
        document.getElementById('view-dashboard').classList.add('active');
        document.getElementById('tab-dashboard').classList.add('active');
        // Re-setup canvas sizes after view switch
        setTimeout(() => {
            topology.resize();
            rewardChart._setup();
            radarChart._setup();
        }, 50);
    }
}

// ============================================================
// Task Selection
// ============================================================
function selectTask(task) {
    state.selectedTask = task;
    document.querySelectorAll('#task-pills .pill').forEach(p => {
        p.classList.toggle('active', p.dataset.task === task);
    });
}

// ============================================================
// API Calls
// ============================================================
async function apiCall(endpoint, method = 'GET', body = null) {
    const opts = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body) opts.body = JSON.stringify(body);

    const res = await fetch(`${API_BASE}${endpoint}`, opts);
    if (!res.ok) {
        const err = await res.text();
        throw new Error(`API Error (${res.status}): ${err}`);
    }
    return res.json();
}

// ============================================================
// Reset Environment
// ============================================================
async function resetEnvironment() {
    try {
        // Stop any auto-run
        stopAutoRun();

        const btn = document.getElementById('btn-reset');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Initializing...';

        const data = await apiCall('/reset', 'POST', {
            task_level: state.selectedTask,
            seed: null,
        });

        // Update state
        state.initialized = true;
        state.done = false;
        state.turn = 0;
        state.totalReward = 0;
        state.rewards = [];
        state.totalCascades = 0;
        state.gradeResults = null;
        state.currentObservation = data.observation;

        // Parse max turns from task
        const maxTurnsMap = { easy: 30, medium: 50, hard: 100, incident_response: 60, zero_day: 80 };
        state.maxTurns = maxTurnsMap[state.selectedTask] || 50;

        // Update UI
        updateDashboard(data.observation);
        resetCharts();
        clearFeed();
        addFeedItem('system', `Environment initialized: ${state.selectedTask.toUpperCase()} mode`, null);

        // Enable controls
        document.getElementById('btn-step').disabled = false;
        document.getElementById('btn-autorun').disabled = false;
        document.getElementById('btn-grade').disabled = true;

        // Status indicators
        setStatus('env-status', 'green');
        setStatus('agent-status', 'green');

        // Task badge
        const badge = document.getElementById('topology-task-badge');
        badge.textContent = state.selectedTask.toUpperCase().replace('_', ' ');

        showToast(`${state.selectedTask.toUpperCase()} mode initialized with ${data.observation.nodes.length} nodes`, 'success');
    } catch (e) {
        showToast(`Reset failed: ${e.message}`, 'error');
    } finally {
        const btn = document.getElementById('btn-reset');
        btn.disabled = false;
        btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg> Initialize`;
    }
}

// ============================================================
// Step Environment (Heuristic Agent)
// ============================================================
async function stepEnvironment() {
    if (!state.initialized || state.done) return;

    try {
        // Decide action using built-in heuristic
        const action = computeHeuristicAction(state.currentObservation);

        const data = await apiCall('/step', 'POST', action);

        state.turn++;
        state.currentObservation = data.observation;
        state.totalReward += data.reward;
        state.rewards.push(data.reward);
        state.done = data.done;
        state.lastStepInfo = data.info;

        if (data.info && data.info.total_cascade_failures) {
            state.totalCascades = data.info.total_cascade_failures;
        }

        // Update visualizations
        updateDashboard(data.observation);
        rewardChart.addReward(data.reward);
        addFeedItem(
            data.info && !data.info.valid ? 'error' : (data.reward >= 0 ? 'success' : 'warning'),
            formatAction(action),
            data.reward
        );

        // Check episode end
        if (data.done) {
            onEpisodeEnd(data);
        }
    } catch (e) {
        showToast(`Step failed: ${e.message}`, 'error');
        stopAutoRun();
    }
}

// ============================================================
// Heuristic Agent (Smart rule-based strategy)
// ============================================================
function computeHeuristicAction(obs) {
    if (!obs) return { action_type: 'noop', target: '', reason: 'No observation' };

    const nodes = obs.nodes || [];
    const vulns = obs.vulnerabilities || [];
    const deps = obs.dependencies || [];

    // Build lookup maps
    const nodeMap = {};
    for (const n of nodes) nodeMap[n.hostname] = n;

    // Find nodes that depend on a given node
    const dependentsOf = {};
    for (const d of deps) {
        if (!dependentsOf[d.depends_on]) dependentsOf[d.depends_on] = [];
        dependentsOf[d.depends_on].push(d.node);
    }

    // Priority 1: Resume crashed nodes
    const crashed = nodes.filter(n => n.state === 'crashed');
    if (crashed.length > 0) {
        const target = crashed.sort((a, b) => a.tier - b.tier)[0];
        return {
            action_type: 'resume_service',
            target: target.hostname,
            reason: `Recovering crashed ${target.tier === 1 ? 'CRITICAL' : ''} node`,
        };
    }

    // Priority 2: Patch vulnerabilities (sorted by severity)
    const severityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
    const sortedVulns = [...vulns].sort((a, b) => {
        // Prioritize exploited vulns
        if (a.exploit_in_wild && !b.exploit_in_wild) return -1;
        if (!a.exploit_in_wild && b.exploit_in_wild) return 1;
        return (severityOrder[a.severity] || 3) - (severityOrder[b.severity] || 3);
    });

    for (const vuln of sortedVulns) {
        for (const hostName of vuln.affected_hosts) {
            const node = nodeMap[hostName];
            if (!node) continue;

            // Can patch if ONLINE (tier 2/3) or SUSPENDED (tier 1)
            if (node.tier === 1) {
                if (node.state === 'suspended') {
                    // Check if dependents are suspended/already safe
                    const depNodes = dependentsOf[hostName] || [];
                    const unsafeDeps = depNodes.filter(dh => {
                        const dn = nodeMap[dh];
                        return dn && dn.state === 'online';
                    });

                    if (unsafeDeps.length === 0) {
                        return {
                            action_type: 'apply_patch',
                            target: hostName,
                            cve_id: vuln.cve_id,
                            reason: `Patching ${vuln.severity} vuln ${vuln.cve_id} on suspended T1 node`,
                        };
                    }
                } else if (node.state === 'online') {
                    // Need to suspend dependents first, then this node
                    const depNodes = dependentsOf[hostName] || [];
                    for (const dh of depNodes) {
                        const dn = nodeMap[dh];
                        if (dn && dn.state === 'online') {
                            return {
                                action_type: 'suspend_service',
                                target: dh,
                                reason: `Suspending dependent before patching T1 node ${hostName}`,
                            };
                        }
                    }
                    // All dependents safe, suspend the T1 node
                    return {
                        action_type: 'suspend_service',
                        target: hostName,
                        reason: `Suspending T1 node to prepare for patch ${vuln.cve_id}`,
                    };
                }
            } else {
                // Tier 2/3: can patch if online
                if (node.state === 'online') {
                    return {
                        action_type: 'apply_patch',
                        target: hostName,
                        cve_id: vuln.cve_id,
                        reason: `Patching ${vuln.severity} vuln ${vuln.cve_id}`,
                    };
                }
            }
        }
    }

    // Priority 3: Resume suspended nodes (if no vulns to handle)
    const suspended = nodes.filter(n => n.state === 'suspended');
    if (suspended.length > 0) {
        // Resume highest tier first (most penalty)
        const target = suspended.sort((a, b) => a.tier - b.tier)[0];
        return {
            action_type: 'resume_service',
            target: target.hostname,
            reason: `Resuming suspended node to reduce downtime`,
        };
    }

    return { action_type: 'noop', target: '', reason: 'No actionable items' };
}

function formatAction(action) {
    const type = action.action_type;
    const icons = {
        scan_host: '🔍',
        suspend_service: '⏸️',
        apply_patch: '🔧',
        resume_service: '▶️',
        noop: '⏭️',
    };
    const icon = icons[type] || '❓';

    let text = `<strong>${icon} ${type}</strong>`;
    if (action.target) text += ` → ${action.target}`;
    if (action.cve_id) text += ` (${action.cve_id})`;
    if (action.reason) text += `<br><span style="color: var(--text-muted); font-size: 0.65rem">${action.reason}</span>`;
    return text;
}

// ============================================================
// Auto-Run
// ============================================================
function toggleAutoRun() {
    if (state.autoRunning) {
        stopAutoRun();
    } else {
        startAutoRun();
    }
}

function startAutoRun() {
    if (!state.initialized || state.done) return;
    state.autoRunning = true;

    const btn = document.getElementById('btn-autorun');
    btn.classList.add('running');
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Stop`;

    document.getElementById('btn-step').disabled = true;
    document.getElementById('btn-reset').disabled = true;

    autoStep();
}

function stopAutoRun() {
    state.autoRunning = false;
    if (state.autoRunTimer) {
        clearTimeout(state.autoRunTimer);
        state.autoRunTimer = null;
    }

    const btn = document.getElementById('btn-autorun');
    btn.classList.remove('running');
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/><line x1="19" y1="3" x2="19" y2="21"/></svg> Auto-Run`;

    if (state.initialized && !state.done) {
        document.getElementById('btn-step').disabled = false;
        document.getElementById('btn-reset').disabled = false;
    }
}

async function autoStep() {
    if (!state.autoRunning || state.done) {
        stopAutoRun();
        return;
    }

    await stepEnvironment();

    if (state.autoRunning && !state.done) {
        const speed = parseInt(document.getElementById('speed-slider').value);
        state.autoRunTimer = setTimeout(autoStep, speed);
    } else {
        stopAutoRun();
    }
}

// ============================================================
// Grade Episode
// ============================================================
async function gradeEpisode() {
    try {
        const episodeData = {
            task_level: state.selectedTask,
            rewards: state.rewards,
            total_reward: state.totalReward,
            steps: state.turn,
            success: state.currentObservation && (state.currentObservation.vulnerabilities || []).length === 0,
            cascade_failures: state.totalCascades,
        };

        const data = await apiCall(`/grade/${state.selectedTask}`, 'POST', episodeData);
        state.gradeResults = data;

        // Update radar chart
        radarChart.setValues(
            data.completion_score || 0,
            data.efficiency_score || 0,
            data.safety_score || 0,
            data.strategy_score || 0
        );

        // Update grade display
        document.getElementById('grade-completion').textContent = (data.completion_score || 0).toFixed(3);
        document.getElementById('grade-efficiency').textContent = (data.efficiency_score || 0).toFixed(3);
        document.getElementById('grade-safety').textContent = (data.safety_score || 0).toFixed(3);
        document.getElementById('grade-strategy').textContent = (data.strategy_score || 0).toFixed(3);

        showToast(`Grading complete! Score: ${(data.normalized_score || 0).toFixed(3)}`, 'success');
    } catch (e) {
        showToast(`Grading failed: ${e.message}`, 'warning');
    }
}

// ============================================================
// Episode End
// ============================================================
function onEpisodeEnd(data) {
    stopAutoRun();

    document.getElementById('btn-step').disabled = true;
    document.getElementById('btn-autorun').disabled = true;
    document.getElementById('btn-grade').disabled = false;

    setStatus('agent-status', 'yellow');

    const vulnsLeft = (state.currentObservation.vulnerabilities || []).length;
    const success = vulnsLeft === 0;

    // Show modal
    const modal = document.getElementById('result-modal');
    document.getElementById('modal-icon').textContent = success ? '🏆' : '💀';
    document.getElementById('modal-title').textContent = success ? 'Mission Complete!' : 'Mission Failed';
    document.getElementById('modal-subtitle').textContent = success
        ? 'All vulnerabilities successfully patched!'
        : `${vulnsLeft} vulnerabilities remaining`;

    // Compute normalized score
    const REWARD_MIN = -300;
    const REWARD_MAX = 50;
    const normalizedScore = Math.max(0.001, Math.min(0.999,
        (state.totalReward - REWARD_MIN) / (REWARD_MAX - REWARD_MIN)
    ));

    document.getElementById('modal-stats').innerHTML = `
        <div class="modal-stat">
            <div class="modal-stat__value">${normalizedScore.toFixed(3)}</div>
            <div class="modal-stat__label">SCORE</div>
        </div>
        <div class="modal-stat">
            <div class="modal-stat__value">${state.turn}</div>
            <div class="modal-stat__label">STEPS</div>
        </div>
        <div class="modal-stat">
            <div class="modal-stat__value">${state.totalReward.toFixed(1)}</div>
            <div class="modal-stat__label">TOTAL REWARD</div>
        </div>
        <div class="modal-stat">
            <div class="modal-stat__value">${state.totalCascades}</div>
            <div class="modal-stat__label">CASCADES</div>
        </div>
    `;

    modal.classList.add('visible');

    addFeedItem(
        success ? 'success' : 'error',
        success ? '🏆 <strong>VICTORY!</strong> All vulnerabilities patched' : '💀 <strong>Episode ended</strong>',
        state.totalReward
    );

    // Auto-grade
    setTimeout(() => gradeEpisode(), 500);
}

function closeModal() {
    document.getElementById('result-modal').classList.remove('visible');
}

// ============================================================
// Dashboard Update
// ============================================================
function updateDashboard(obs) {
    if (!obs) return;

    // Turn counter
    document.getElementById('turn-value').textContent = state.turn;
    document.getElementById('turn-max').textContent = state.maxTurns;

    // Metrics
    const REWARD_MIN = -300;
    const REWARD_MAX = 50;
    const score = Math.max(0.001, Math.min(0.999,
        (state.totalReward - REWARD_MIN) / (REWARD_MAX - REWARD_MIN)
    ));

    document.getElementById('val-score').textContent = score.toFixed(3);
    document.getElementById('val-reward').textContent = state.totalReward.toFixed(2);
    document.getElementById('val-vulns').textContent = (obs.vulnerabilities || []).length;
    document.getElementById('val-online').textContent = obs.health ? obs.health.nodes_online : 0;
    document.getElementById('val-total').textContent = obs.health ? obs.health.total_nodes : 0;
    document.getElementById('val-cascades').textContent = state.totalCascades;

    // Trends
    if (state.rewards.length > 0) {
        const lastReward = state.rewards[state.rewards.length - 1];
        const trendReward = document.getElementById('trend-reward');
        trendReward.textContent = (lastReward >= 0 ? '+' : '') + lastReward.toFixed(2);
        trendReward.className = 'metric-trend ' + (lastReward >= 0 ? 'up' : 'down');

        const trendScore = document.getElementById('trend-score');
        trendScore.textContent = score.toFixed(3);
        trendScore.className = 'metric-trend ' + (score > 0.5 ? 'up' : 'down');
    }

    // Topology
    topology.updateData(obs);

    // Vulnerability panel
    updateVulnPanel(obs.vulnerabilities || []);
}

function updateVulnPanel(vulns) {
    const container = document.getElementById('vulns-container');
    if (vulns.length === 0) {
        container.innerHTML = '<div class="feed-empty"><p>All vulnerabilities patched! ✅</p></div>';
        return;
    }

    container.innerHTML = vulns.map(v => `
        <div class="vuln-card">
            <div class="vuln-card__header">
                <span class="vuln-card__cve">${v.cve_id}</span>
                <span class="vuln-card__severity ${v.severity}">${v.severity.toUpperCase()}</span>
            </div>
            <div class="vuln-card__cvss">CVSS: ${v.cvss_score}</div>
            <div class="vuln-card__hosts">
                ${(v.affected_hosts || []).map(h => `<span class="vuln-host-tag">${h}</span>`).join('')}
            </div>
            ${v.exploit_in_wild ? '<div class="vuln-card__exploit">🔥 EXPLOIT IN WILD — 2× penalty</div>' : ''}
        </div>
    `).join('');
}

// ============================================================
// Feed
// ============================================================
function clearFeed() {
    document.getElementById('feed-container').innerHTML = '';
}

function addFeedItem(type, text, reward) {
    const container = document.getElementById('feed-container');

    // Remove empty state
    const empty = container.querySelector('.feed-empty');
    if (empty) empty.remove();

    const item = document.createElement('div');
    item.className = `feed-item ${type}`;

    let rewardHtml = '';
    if (reward !== null && reward !== undefined) {
        const cls = reward >= 0 ? 'positive' : 'negative';
        rewardHtml = `<span class="feed-reward ${cls}">${reward >= 0 ? '+' : ''}${reward.toFixed(2)}</span>`;
    }

    item.innerHTML = `
        <span class="feed-step">${state.turn > 0 ? state.turn : '•'}</span>
        <span class="feed-text">${text}</span>
        ${rewardHtml}
    `;

    container.insertBefore(item, container.firstChild);

    // Limit feed items
    while (container.children.length > 50) {
        container.removeChild(container.lastChild);
    }
}

// ============================================================
// Chart Reset
// ============================================================
function resetCharts() {
    rewardChart.reset();
    radarChart.reset();

    // Reset grade displays
    document.getElementById('grade-completion').textContent = '—';
    document.getElementById('grade-efficiency').textContent = '—';
    document.getElementById('grade-safety').textContent = '—';
    document.getElementById('grade-strategy').textContent = '—';
}

// ============================================================
// Toast Notifications
// ============================================================
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('out');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

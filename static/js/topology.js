/* ============================================================
   PatchCascade SOC — Network Topology Visualization Engine
   Canvas-based force-directed graph with animated states
   ============================================================ */

class TopologyRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.nodes = [];
        this.edges = [];
        this.hoveredNode = null;
        this.animFrame = null;
        this.time = 0;
        this.dpr = window.devicePixelRatio || 1;

        // Physics
        this.repulsion = 8000;
        this.attraction = 0.005;
        this.damping = 0.85;
        this.centerGravity = 0.01;

        // Visuals
        this.nodeRadius = 28;
        this.colors = {
            online: '#00ff88',
            suspended: '#ffaa00',
            patching: '#00bbff',
            crashed: '#ff3355',
            offline: '#4a5a72',
        };

        this._setupCanvas();
        this._bindEvents();
    }

    _setupCanvas() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width * this.dpr;
        this.canvas.height = rect.height * this.dpr;
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
        this.ctx.scale(this.dpr, this.dpr);
        this.width = rect.width;
        this.height = rect.height;
    }

    _bindEvents() {
        window.addEventListener('resize', () => {
            this._setupCanvas();
        });

        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            this.hoveredNode = null;
            for (const node of this.nodes) {
                const dx = mx - node.x;
                const dy = my - node.y;
                if (dx * dx + dy * dy < (this.nodeRadius + 6) * (this.nodeRadius + 6)) {
                    this.hoveredNode = node;
                    break;
                }
            }
            this.canvas.style.cursor = this.hoveredNode ? 'pointer' : 'default';
        });

        this.canvas.addEventListener('mouseleave', () => {
            this.hoveredNode = null;
        });
    }

    updateData(observationData) {
        if (!observationData) return;

        const obsNodes = observationData.nodes || [];
        const obsDeps = observationData.dependencies || [];
        const obsVulns = observationData.vulnerabilities || [];

        // Build vuln host lookup
        const vulnHosts = new Set();
        const exploitHosts = new Set();
        for (const v of obsVulns) {
            for (const h of (v.affected_hosts || [])) {
                vulnHosts.add(h);
                if (v.exploit_in_wild) exploitHosts.add(h);
            }
        }

        // Update or create nodes
        const existingMap = {};
        for (const n of this.nodes) existingMap[n.id] = n;

        const newNodes = [];
        for (let i = 0; i < obsNodes.length; i++) {
            const on = obsNodes[i];
            const existing = existingMap[on.hostname];
            if (existing) {
                existing.state = on.state;
                existing.tier = on.tier;
                existing.services = on.services;
                existing.patchTurns = on.patch_turns_remaining;
                existing.hasVuln = vulnHosts.has(on.hostname);
                existing.hasExploit = exploitHosts.has(on.hostname);
                newNodes.push(existing);
            } else {
                // Position in a circle initially
                const angle = (i / obsNodes.length) * Math.PI * 2 - Math.PI / 2;
                const rx = this.width * 0.3;
                const ry = this.height * 0.3;
                newNodes.push({
                    id: on.hostname,
                    label: on.hostname,
                    state: on.state,
                    tier: on.tier,
                    services: on.services || [],
                    patchTurns: on.patch_turns_remaining,
                    hasVuln: vulnHosts.has(on.hostname),
                    hasExploit: exploitHosts.has(on.hostname),
                    x: this.width / 2 + Math.cos(angle) * rx,
                    y: this.height / 2 + Math.sin(angle) * ry,
                    vx: 0,
                    vy: 0,
                });
            }
        }
        this.nodes = newNodes;

        // Update edges
        this.edges = obsDeps.map(d => ({
            source: d.node,
            target: d.depends_on,
            type: d.dependency_type || 'hard',
        }));
    }

    _applyForces() {
        const nodes = this.nodes;
        const cx = this.width / 2;
        const cy = this.height / 2;

        // Repulsion between all node pairs
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                let dx = nodes[j].x - nodes[i].x;
                let dy = nodes[j].y - nodes[i].y;
                let dist = Math.sqrt(dx * dx + dy * dy) || 1;
                let force = this.repulsion / (dist * dist);
                let fx = (dx / dist) * force;
                let fy = (dy / dist) * force;
                nodes[i].vx -= fx;
                nodes[i].vy -= fy;
                nodes[j].vx += fx;
                nodes[j].vy += fy;
            }
        }

        // Attraction along edges
        for (const edge of this.edges) {
            const src = nodes.find(n => n.id === edge.source);
            const tgt = nodes.find(n => n.id === edge.target);
            if (!src || !tgt) continue;
            let dx = tgt.x - src.x;
            let dy = tgt.y - src.y;
            let dist = Math.sqrt(dx * dx + dy * dy) || 1;
            let force = dist * this.attraction;
            let fx = (dx / dist) * force;
            let fy = (dy / dist) * force;
            src.vx += fx;
            src.vy += fy;
            tgt.vx -= fx;
            tgt.vy -= fy;
        }

        // Center gravity
        for (const node of nodes) {
            node.vx += (cx - node.x) * this.centerGravity;
            node.vy += (cy - node.y) * this.centerGravity;
        }

        // Apply velocity with damping
        for (const node of nodes) {
            node.vx *= this.damping;
            node.vy *= this.damping;
            node.x += node.vx;
            node.y += node.vy;

            // Keep within bounds
            const pad = this.nodeRadius + 10;
            node.x = Math.max(pad, Math.min(this.width - pad, node.x));
            node.y = Math.max(pad, Math.min(this.height - pad, node.y));
        }
    }

    _drawEdge(ctx, edge, t) {
        const src = this.nodes.find(n => n.id === edge.source);
        const tgt = this.nodes.find(n => n.id === edge.target);
        if (!src || !tgt) return;

        const dx = tgt.x - src.x;
        const dy = tgt.y - src.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const nx = dx / dist;
        const ny = dy / dist;

        const x1 = src.x + nx * this.nodeRadius;
        const y1 = src.y + ny * this.nodeRadius;
        const x2 = tgt.x - nx * this.nodeRadius;
        const y2 = tgt.y - ny * this.nodeRadius;

        // Edge line
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);

        if (edge.type === 'hard') {
            ctx.strokeStyle = 'rgba(100, 160, 255, 0.2)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([]);
        } else {
            ctx.strokeStyle = 'rgba(100, 160, 255, 0.1)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // Animated data flow particle
        const progress = ((t * 0.3) % 100) / 100;
        const px = x1 + (x2 - x1) * progress;
        const py = y1 + (y2 - y1) * progress;
        ctx.beginPath();
        ctx.arc(px, py, 2, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 187, 255, 0.5)';
        ctx.fill();

        // Arrow head
        const arrowSize = 6;
        const angle = Math.atan2(y2 - y1, x2 - x1);
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - arrowSize * Math.cos(angle - 0.4), y2 - arrowSize * Math.sin(angle - 0.4));
        ctx.lineTo(x2 - arrowSize * Math.cos(angle + 0.4), y2 - arrowSize * Math.sin(angle + 0.4));
        ctx.closePath();
        ctx.fillStyle = 'rgba(100, 160, 255, 0.3)';
        ctx.fill();
    }

    _drawNode(ctx, node, t) {
        const x = node.x;
        const y = node.y;
        const r = this.nodeRadius;
        const color = this.colors[node.state] || this.colors.offline;
        const isHovered = this.hoveredNode === node;

        // Outer glow
        if (node.state === 'crashed' || node.hasExploit) {
            const pulseR = r + 8 + Math.sin(t * 0.05) * 4;
            const gradient = ctx.createRadialGradient(x, y, r, x, y, pulseR);
            gradient.addColorStop(0, 'rgba(255, 51, 85, 0.15)');
            gradient.addColorStop(1, 'rgba(255, 51, 85, 0)');
            ctx.beginPath();
            ctx.arc(x, y, pulseR, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();
        } else if (node.state === 'patching') {
            const pulseR = r + 6 + Math.sin(t * 0.08) * 3;
            const gradient = ctx.createRadialGradient(x, y, r, x, y, pulseR);
            gradient.addColorStop(0, 'rgba(0, 187, 255, 0.12)');
            gradient.addColorStop(1, 'rgba(0, 187, 255, 0)');
            ctx.beginPath();
            ctx.arc(x, y, pulseR, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();
        }

        // Hovered glow
        if (isHovered) {
            const gradient = ctx.createRadialGradient(x, y, r, x, y, r + 14);
            gradient.addColorStop(0, color.replace(')', ', 0.15)').replace('rgb', 'rgba'));
            gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
            ctx.beginPath();
            ctx.arc(x, y, r + 14, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();
        }

        // Node body
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);

        const bodyGrad = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, 0, x, y, r);
        bodyGrad.addColorStop(0, 'rgba(20, 30, 50, 0.9)');
        bodyGrad.addColorStop(1, 'rgba(10, 15, 25, 0.95)');
        ctx.fillStyle = bodyGrad;
        ctx.fill();

        // Node border
        ctx.strokeStyle = color;
        ctx.lineWidth = isHovered ? 2.5 : 1.8;
        ctx.stroke();

        // Patching spinner
        if (node.state === 'patching') {
            ctx.beginPath();
            const startAngle = (t * 0.06) % (Math.PI * 2);
            ctx.arc(x, y, r + 3, startAngle, startAngle + Math.PI * 1.2);
            ctx.strokeStyle = 'rgba(0, 187, 255, 0.5)';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Tier indicator (corner badge)
        const tierColors = { 1: '#ff3355', 2: '#ffaa00', 3: '#00ff88' };
        const tierLabels = { 1: 'T1', 2: 'T2', 3: 'T3' };
        const tc = tierColors[node.tier] || '#4a5a72';
        const badgeX = x + r * 0.65;
        const badgeY = y - r * 0.65;

        ctx.beginPath();
        ctx.arc(badgeX, badgeY, 9, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(6, 10, 19, 0.9)';
        ctx.fill();
        ctx.strokeStyle = tc;
        ctx.lineWidth = 1.2;
        ctx.stroke();

        ctx.fillStyle = tc;
        ctx.font = '600 7px "Inter", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(tierLabels[node.tier] || '?', badgeX, badgeY + 0.5);

        // Vulnerability indicator
        if (node.hasVuln) {
            const vx = x - r * 0.65;
            const vy = y - r * 0.65;
            ctx.beginPath();
            ctx.arc(vx, vy, 7, 0, Math.PI * 2);
            ctx.fillStyle = node.hasExploit ? 'rgba(255, 51, 85, 0.9)' : 'rgba(255, 170, 0, 0.8)';
            ctx.fill();
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 8px "Inter", sans-serif';
            ctx.fillText('!', vx, vy + 0.5);
        }

        // Node icon (based on hostname pattern)
        let icon = '🖥';
        const h = node.id.toLowerCase();
        if (h.includes('db') || h.includes('data')) icon = '🗄';
        else if (h.includes('web') || h.includes('frontend')) icon = '🌐';
        else if (h.includes('app') || h.includes('api')) icon = '⚙';
        else if (h.includes('auth')) icon = '🔐';
        else if (h.includes('lb') || h.includes('load')) icon = '⚖';
        else if (h.includes('cache') || h.includes('redis')) icon = '💾';
        else if (h.includes('gate') || h.includes('gateway')) icon = '🚪';
        else if (h.includes('monitor') || h.includes('log')) icon = '📊';

        ctx.font = '14px serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(icon, x, y - 2);

        // Hostname label
        const shortName = node.id.length > 14 ? node.id.substring(0, 12) + '…' : node.id;
        ctx.font = '500 8px "Inter", sans-serif';
        ctx.fillStyle = 'rgba(232, 236, 244, 0.8)';
        ctx.fillText(shortName, x, y + r + 12);

        // State label
        ctx.font = '600 7px "Inter", sans-serif';
        ctx.fillStyle = color;
        ctx.fillText(node.state.toUpperCase(), x, y + 13);

        // Hover tooltip
        if (isHovered) {
            this._drawTooltip(ctx, node, x, y);
        }
    }

    _drawTooltip(ctx, node, x, y) {
        const lines = [
            `Host: ${node.id}`,
            `State: ${node.state.toUpperCase()}`,
            `Tier: ${node.tier} (${node.tier === 1 ? 'CRITICAL' : node.tier === 2 ? 'IMPORTANT' : 'STANDARD'})`,
            `Services: ${node.services.join(', ') || 'none'}`,
        ];
        if (node.patchTurns > 0) lines.push(`Patch completes in: ${node.patchTurns} turn(s)`);
        if (node.hasExploit) lines.push('⚠️ ACTIVELY EXPLOITED!');
        else if (node.hasVuln) lines.push('⚠️ Has vulnerability');

        const lineHeight = 15;
        const padding = 10;
        const maxWidth = Math.max(...lines.map(l => ctx.measureText(l).width)) + padding * 2;
        const totalHeight = lines.length * lineHeight + padding * 2;

        let tx = x + this.nodeRadius + 12;
        let ty = y - totalHeight / 2;

        // Keep tooltip on screen
        if (tx + maxWidth > this.width - 10) tx = x - this.nodeRadius - maxWidth - 12;
        if (ty < 10) ty = 10;
        if (ty + totalHeight > this.height - 10) ty = this.height - totalHeight - 10;

        // Background
        ctx.fillStyle = 'rgba(6, 10, 19, 0.95)';
        ctx.strokeStyle = 'rgba(0, 187, 255, 0.3)';
        ctx.lineWidth = 1;
        const tooltipR = 6;
        this._roundRect(ctx, tx, ty, maxWidth, totalHeight, tooltipR);
        ctx.fill();
        ctx.stroke();

        // Text
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.fillStyle = 'rgba(232, 236, 244, 0.85)';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        lines.forEach((line, i) => {
            if (line.startsWith('⚠️')) {
                ctx.fillStyle = '#ff3355';
            } else {
                ctx.fillStyle = 'rgba(232, 236, 244, 0.85)';
            }
            ctx.fillText(line, tx + padding, ty + padding + i * lineHeight);
        });
    }

    _roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.arcTo(x + w, y, x + w, y + r, r);
        ctx.lineTo(x + w, y + h - r);
        ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
        ctx.lineTo(x + r, y + h);
        ctx.arcTo(x, y + h, x, y + h - r, r);
        ctx.lineTo(x, y + r);
        ctx.arcTo(x, y, x + r, y, r);
        ctx.closePath();
    }

    render() {
        const ctx = this.ctx;
        this.time++;

        // Clear
        ctx.clearRect(0, 0, this.width, this.height);

        // Grid pattern background
        ctx.strokeStyle = 'rgba(100, 160, 255, 0.03)';
        ctx.lineWidth = 0.5;
        const gridSize = 40;
        for (let gx = 0; gx < this.width; gx += gridSize) {
            ctx.beginPath();
            ctx.moveTo(gx, 0);
            ctx.lineTo(gx, this.height);
            ctx.stroke();
        }
        for (let gy = 0; gy < this.height; gy += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, gy);
            ctx.lineTo(this.width, gy);
            ctx.stroke();
        }

        // Apply physics
        if (this.nodes.length > 0) {
            this._applyForces();
        }

        // Draw edges
        for (const edge of this.edges) {
            this._drawEdge(ctx, edge, this.time);
        }

        // Draw nodes
        for (const node of this.nodes) {
            this._drawNode(ctx, node, this.time);
        }

        // No nodes message
        if (this.nodes.length === 0) {
            ctx.fillStyle = 'rgba(74, 90, 114, 0.5)';
            ctx.font = '500 14px "Inter", sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Initialize a task to visualize the network', this.width / 2, this.height / 2);
        }

        this.animFrame = requestAnimationFrame(() => this.render());
    }

    start() {
        if (!this.animFrame) {
            this.render();
        }
    }

    stop() {
        if (this.animFrame) {
            cancelAnimationFrame(this.animFrame);
            this.animFrame = null;
        }
    }

    resize() {
        this._setupCanvas();
    }
}

// ============================================================
// Particle Background System
// ============================================================
class ParticleBackground {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        this.connections = [];
        this.animFrame = null;

        this._resize();
        this._createParticles();
        window.addEventListener('resize', () => this._resize());
    }

    _resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        this.width = this.canvas.width;
        this.height = this.canvas.height;
    }

    _createParticles() {
        const count = Math.min(60, Math.floor((this.width * this.height) / 20000));
        this.particles = [];
        for (let i = 0; i < count; i++) {
            this.particles.push({
                x: Math.random() * this.width,
                y: Math.random() * this.height,
                vx: (Math.random() - 0.5) * 0.3,
                vy: (Math.random() - 0.5) * 0.3,
                r: Math.random() * 1.5 + 0.5,
                alpha: Math.random() * 0.3 + 0.1,
            });
        }
    }

    render() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.width, this.height);

        // Update & draw particles
        for (const p of this.particles) {
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0) p.x = this.width;
            if (p.x > this.width) p.x = 0;
            if (p.y < 0) p.y = this.height;
            if (p.y > this.height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 187, 255, ${p.alpha})`;
            ctx.fill();
        }

        // Draw connections
        const maxDist = 150;
        for (let i = 0; i < this.particles.length; i++) {
            for (let j = i + 1; j < this.particles.length; j++) {
                const dx = this.particles[i].x - this.particles[j].x;
                const dy = this.particles[i].y - this.particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < maxDist) {
                    const alpha = (1 - dist / maxDist) * 0.08;
                    ctx.beginPath();
                    ctx.moveTo(this.particles[i].x, this.particles[i].y);
                    ctx.lineTo(this.particles[j].x, this.particles[j].y);
                    ctx.strokeStyle = `rgba(0, 187, 255, ${alpha})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }

        this.animFrame = requestAnimationFrame(() => this.render());
    }

    start() {
        if (!this.animFrame) this.render();
    }
}

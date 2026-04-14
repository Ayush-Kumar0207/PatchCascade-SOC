/* ============================================================
   PatchCascade SOC — Chart Visualization Engine
   Real-time reward timeline + radar grading chart
   ============================================================ */

class RewardChart {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.rewards = [];
        this.cumulativeRewards = [];
        this.maxPoints = 120;
        this.dpr = window.devicePixelRatio || 1;
        this._setup();
        window.addEventListener('resize', () => this._setup());
    }

    _setup() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width * this.dpr;
        this.canvas.height = rect.height * this.dpr;
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
        this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
        this.width = rect.width;
        this.height = rect.height;
    }

    addReward(reward) {
        this.rewards.push(reward);
        const prev = this.cumulativeRewards.length > 0
            ? this.cumulativeRewards[this.cumulativeRewards.length - 1]
            : 0;
        this.cumulativeRewards.push(prev + reward);

        if (this.rewards.length > this.maxPoints) {
            this.rewards.shift();
            this.cumulativeRewards.shift();
        }
    }

    reset() {
        this.rewards = [];
        this.cumulativeRewards = [];
    }

    render() {
        const ctx = this.ctx;
        const w = this.width;
        const h = this.height;
        const padding = { top: 20, right: 16, bottom: 28, left: 50 };

        ctx.clearRect(0, 0, w, h);

        const chartW = w - padding.left - padding.right;
        const chartH = h - padding.top - padding.bottom;

        if (this.rewards.length < 2) {
            ctx.fillStyle = 'rgba(74, 90, 114, 0.4)';
            ctx.font = '500 11px "Inter", sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Reward data will appear here', w / 2, h / 2);
            return;
        }

        // Compute ranges
        const allVals = [...this.rewards, ...this.cumulativeRewards];
        let minVal = Math.min(...allVals);
        let maxVal = Math.max(...allVals);
        if (minVal === maxVal) { minVal -= 1; maxVal += 1; }
        const range = maxVal - minVal;
        const yPad = range * 0.1;
        minVal -= yPad;
        maxVal += yPad;

        const toX = (i) => padding.left + (i / (this.rewards.length - 1)) * chartW;
        const toY = (v) => padding.top + chartH - ((v - minVal) / (maxVal - minVal)) * chartH;

        // Grid lines
        ctx.strokeStyle = 'rgba(100, 160, 255, 0.06)';
        ctx.lineWidth = 0.5;
        const gridCount = 5;
        for (let i = 0; i <= gridCount; i++) {
            const gy = padding.top + (i / gridCount) * chartH;
            ctx.beginPath();
            ctx.moveTo(padding.left, gy);
            ctx.lineTo(w - padding.right, gy);
            ctx.stroke();

            const val = maxVal - (i / gridCount) * (maxVal - minVal);
            ctx.fillStyle = 'rgba(74, 90, 114, 0.6)';
            ctx.font = '500 8px "JetBrains Mono", monospace';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(val.toFixed(1), padding.left - 6, gy);
        }

        // Zero line
        if (minVal < 0 && maxVal > 0) {
            const zeroY = toY(0);
            ctx.beginPath();
            ctx.moveTo(padding.left, zeroY);
            ctx.lineTo(w - padding.right, zeroY);
            ctx.strokeStyle = 'rgba(100, 160, 255, 0.15)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Cumulative reward area + line
        const cumGrad = ctx.createLinearGradient(0, padding.top, 0, padding.top + chartH);
        cumGrad.addColorStop(0, 'rgba(0, 187, 255, 0.12)');
        cumGrad.addColorStop(1, 'rgba(0, 187, 255, 0)');

        ctx.beginPath();
        ctx.moveTo(toX(0), toY(0));
        for (let i = 0; i < this.cumulativeRewards.length; i++) {
            ctx.lineTo(toX(i), toY(this.cumulativeRewards[i]));
        }
        ctx.lineTo(toX(this.cumulativeRewards.length - 1), toY(0));
        ctx.closePath();
        ctx.fillStyle = cumGrad;
        ctx.fill();

        // Cumulative line
        ctx.beginPath();
        for (let i = 0; i < this.cumulativeRewards.length; i++) {
            if (i === 0) ctx.moveTo(toX(i), toY(this.cumulativeRewards[i]));
            else ctx.lineTo(toX(i), toY(this.cumulativeRewards[i]));
        }
        ctx.strokeStyle = 'rgba(0, 187, 255, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Per-step reward bars
        const barW = Math.max(2, chartW / this.rewards.length * 0.6);
        for (let i = 0; i < this.rewards.length; i++) {
            const r = this.rewards[i];
            const x = toX(i) - barW / 2;
            const yZero = toY(0);
            const yVal = toY(r);

            ctx.fillStyle = r >= 0
                ? 'rgba(0, 255, 136, 0.35)'
                : 'rgba(255, 51, 85, 0.35)';
            ctx.fillRect(x, Math.min(yZero, yVal), barW, Math.abs(yVal - yZero));
        }

        // Latest point glow
        if (this.cumulativeRewards.length > 0) {
            const lastI = this.cumulativeRewards.length - 1;
            const lx = toX(lastI);
            const ly = toY(this.cumulativeRewards[lastI]);

            const glowGrad = ctx.createRadialGradient(lx, ly, 0, lx, ly, 12);
            glowGrad.addColorStop(0, 'rgba(0, 187, 255, 0.4)');
            glowGrad.addColorStop(1, 'rgba(0, 187, 255, 0)');
            ctx.beginPath();
            ctx.arc(lx, ly, 12, 0, Math.PI * 2);
            ctx.fillStyle = glowGrad;
            ctx.fill();

            ctx.beginPath();
            ctx.arc(lx, ly, 3, 0, Math.PI * 2);
            ctx.fillStyle = '#00bbff';
            ctx.fill();
        }

        // X-axis label
        ctx.fillStyle = 'rgba(74, 90, 114, 0.5)';
        ctx.font = '500 8px "Inter", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`Step (${this.rewards.length})`, w / 2, h - 4);
    }
}

// ============================================================
// Radar Chart for Multi-Dimensional Grading
// ============================================================
class RadarChart {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.dimensions = ['Completion', 'Efficiency', 'Safety', 'Strategy'];
        this.values = [0, 0, 0, 0];
        this.targetValues = [0, 0, 0, 0];
        this.colors = ['#00ff88', '#00bbff', '#ff3355', '#aa55ff'];
        this.dpr = window.devicePixelRatio || 1;
        this._setup();
        window.addEventListener('resize', () => this._setup());
    }

    _setup() {
        const size = Math.min(
            this.canvas.parentElement.clientWidth,
            160
        );
        this.canvas.width = size * this.dpr;
        this.canvas.height = size * this.dpr;
        this.canvas.style.width = size + 'px';
        this.canvas.style.height = size + 'px';
        this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
        this.size = size;
    }

    setValues(completion, efficiency, safety, strategy) {
        this.targetValues = [completion, efficiency, safety, strategy];
    }

    reset() {
        this.values = [0, 0, 0, 0];
        this.targetValues = [0, 0, 0, 0];
    }

    render() {
        // Animate towards target
        for (let i = 0; i < 4; i++) {
            this.values[i] += (this.targetValues[i] - this.values[i]) * 0.08;
        }

        const ctx = this.ctx;
        const s = this.size;
        const cx = s / 2;
        const cy = s / 2;
        const r = s * 0.38;

        ctx.clearRect(0, 0, s, s);

        const n = this.dimensions.length;
        const angleStep = (Math.PI * 2) / n;
        const startAngle = -Math.PI / 2;

        // Draw concentric rings
        for (let ring = 1; ring <= 4; ring++) {
            const rr = (ring / 4) * r;
            ctx.beginPath();
            for (let i = 0; i <= n; i++) {
                const angle = startAngle + i * angleStep;
                const x = cx + Math.cos(angle) * rr;
                const y = cy + Math.sin(angle) * rr;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.strokeStyle = `rgba(100, 160, 255, ${ring === 4 ? 0.12 : 0.06})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
        }

        // Draw axes
        for (let i = 0; i < n; i++) {
            const angle = startAngle + i * angleStep;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(cx + Math.cos(angle) * r, cy + Math.sin(angle) * r);
            ctx.strokeStyle = 'rgba(100, 160, 255, 0.08)';
            ctx.lineWidth = 0.5;
            ctx.stroke();
        }

        // Draw data polygon
        ctx.beginPath();
        for (let i = 0; i <= n; i++) {
            const idx = i % n;
            const angle = startAngle + idx * angleStep;
            const val = Math.max(0, Math.min(1, this.values[idx]));
            const x = cx + Math.cos(angle) * r * val;
            const y = cy + Math.sin(angle) * r * val;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.closePath();

        // Fill
        const fillGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
        fillGrad.addColorStop(0, 'rgba(0, 187, 255, 0.15)');
        fillGrad.addColorStop(1, 'rgba(0, 187, 255, 0.03)');
        ctx.fillStyle = fillGrad;
        ctx.fill();

        // Stroke
        ctx.strokeStyle = 'rgba(0, 187, 255, 0.5)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Draw data points
        for (let i = 0; i < n; i++) {
            const angle = startAngle + i * angleStep;
            const val = Math.max(0, Math.min(1, this.values[i]));
            const x = cx + Math.cos(angle) * r * val;
            const y = cy + Math.sin(angle) * r * val;

            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fillStyle = this.colors[i];
            ctx.fill();

            // Glow
            const glowGrad = ctx.createRadialGradient(x, y, 0, x, y, 8);
            glowGrad.addColorStop(0, this.colors[i].replace(')', ', 0.3)').replace('rgb', 'rgba').replace('#', ''));
            glowGrad.addColorStop(1, 'rgba(0,0,0,0)');

            // Use hex to rgba conversion
            const hex = this.colors[i];
            const rr = parseInt(hex.slice(1, 3), 16);
            const gg = parseInt(hex.slice(3, 5), 16);
            const bb = parseInt(hex.slice(5, 7), 16);

            const ptGlow = ctx.createRadialGradient(x, y, 0, x, y, 8);
            ptGlow.addColorStop(0, `rgba(${rr}, ${gg}, ${bb}, 0.3)`);
            ptGlow.addColorStop(1, `rgba(${rr}, ${gg}, ${bb}, 0)`);
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fillStyle = ptGlow;
            ctx.fill();
        }

        // Labels
        ctx.font = '600 8px "Inter", sans-serif';
        ctx.textBaseline = 'middle';

        for (let i = 0; i < n; i++) {
            const angle = startAngle + i * angleStep;
            const lx = cx + Math.cos(angle) * (r + 14);
            const ly = cy + Math.sin(angle) * (r + 14);

            ctx.fillStyle = this.colors[i];
            ctx.textAlign = lx < cx ? 'right' : lx > cx ? 'left' : 'center';
            ctx.fillText(this.dimensions[i], lx, ly);
        }
    }
}

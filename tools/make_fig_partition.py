#!/usr/bin/env python3
"""Figure 2: REALISED Dirichlet non-IID partition (alpha=1.0, 6 clients, seed 42).

v3.9 fix: the previous generator drew an INDEPENDENT
np.random.dirichlet sample whose proportions (9%-94% label skew) never
matched the partition the experiments actually used
(28.6%-80.5%). This version loads the cached client assignment and
training labels that the frozen seed-42 run itself used, so the figure
cannot desynchronise from the artefacts.

Run from anywhere: writes into paper/figures/ relative to the repo root.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
# R-FS9-R7 (B11): guarded — the explicit DejaVu registration is only
# needed when the system font path exists; matplotlib bundles DejaVu
# internally, so a different install location must not crash the tool.
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except Exception:
    pass
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Authoritative source: the cached assignment of the frozen run ──
z = np.load(os.path.join(ROOT, "data/cache/phase_data.npz"))
y = z["y_train"]
a = np.load(os.path.join(ROOT, "data/cache/phase_assignment.npz"))["assignment"]
N = len(y)
POOL_RATE = float(y.mean())

N_CLIENTS = 6
CLIENTS = ["Client A", "Client B", "Client C",
           "Client D", "Client E", "Client F"]

rel_size = np.array([(a == k).sum() for k in range(N_CLIENTS)]) / N
flare_rate = np.array([y[a == k].mean() for k in range(N_CLIENTS)])

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)

y_pos = np.arange(N_CLIENTS)[::-1]

# ── (a) relative shard sizes ──
ax = axes[0]
ax.barh(y_pos, rel_size * 100, height=0.62, color="#93C5FD",
        edgecolor="#3B82F6", linewidth=1.0)
ax.set_yticks(y_pos)
ax.set_yticklabels(CLIENTS, fontsize=9)
for yi, v in zip(y_pos, rel_size * 100):
    ax.text(v + 0.8, yi, f"{v:.1f}%", va="center", ha="left", fontsize=9, color="#334155")
ax.set_xlim(0, max(rel_size) * 100 * 1.22)
ax.set_xlabel("Share of pooled training data (%)", fontsize=10.5)
ax.set_title("(a) Realised client shard sizes", fontsize=11, fontweight="bold", loc="left", pad=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color("#9CA3AF")
ax.tick_params(axis='y', length=0)
ax.tick_params(axis='x', colors="#4B5563")
ax.grid(True, axis='x', linestyle='--', alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)

# ── (b) per-client flare rates ──
ax = axes[1]
rates = flare_rate * 100
colors = ["#F59E0B" if r > 60 or r < 35 else "#34D399" for r in rates]
ax.barh(y_pos, rates, height=0.62, color=colors,
        edgecolor=["#B45309" if c == "#F59E0B" else "#059669" for c in colors],
        linewidth=1.0)
ax.set_yticks(y_pos)
ax.set_yticklabels(CLIENTS, fontsize=9)
for yi, v in zip(y_pos, rates):
    ax.text(v + 1.2, yi, f"{v:.1f}%", va="center", ha="left", fontsize=9, color="#334155")
ax.axvline(POOL_RATE * 100, color="#EF4444", linestyle="--", linewidth=1.4, alpha=0.85)
ax.text(POOL_RATE * 100 + 1.0, N_CLIENTS - 0.42, f"pooled rate\n{POOL_RATE*100:.1f}%",
        fontsize=8.5, color="#B91C1C", va="top")
ax.set_xlim(0, 100)
ax.set_xlabel("Local positive-class (flare) rate (%)", fontsize=10.5)
ax.set_title("(b) Realised label skew across clients", fontsize=11, fontweight="bold", loc="left", pad=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color("#9CA3AF")
ax.tick_params(axis='y', length=0)
ax.tick_params(axis='x', colors="#4B5563")
ax.grid(True, axis='x', linestyle='--', alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)

out = os.path.join(ROOT, "paper/figures/fig_partition.png")
fig.savefig(out, dpi=220, facecolor="white")
print("Saved", out)
print("Relative sizes (%):", np.round(rel_size * 100, 1))
print("Flare rates (%):", np.round(flare_rate * 100, 1))

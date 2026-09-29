#!/usr/bin/env python3
"""Figure 2: Illustrative Dirichlet non-IID partition realization (alpha=1.0, 6 clients, seed 42).

Run from anywhere: writes into paper/figures/ relative to the repo root.
Replicates the partition proportions logic of partition_clients.partition_data_dirichlet
on a class-balanced training pool (48.9% flare rate, as in the cleaned SWAN-SF training set).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── Reproduce the partition proportions (deterministic, seed 42) ──
np.random.seed(42)
N_CLIENTS = 6
ALPHA = 1.0
n_classes = 2
proportions = np.random.dirichlet([ALPHA] * N_CLIENTS, size=n_classes)

CLIENTS = ["Client A\nAmericas", "Client B\nEurope", "Client C\nAsia-Pacific",
           "Client D\nSouth Asia", "Client E\nEast Asia", "Client F\nOceania"]

POOL_RATE = 0.4887  # balanced training pool flare rate (global_pos_rate in code)

# Relative shard size: share of the pooled training samples per client
rel_size = (proportions[0] + proportions[1]) / (proportions.sum())
# Per-client flare rate (both class pools equally sized => weighted mix)
flare_rate = proportions[1] / (proportions[0] + proportions[1])

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)

y = np.arange(N_CLIENTS)[::-1]

# ── (a) relative shard sizes ──
ax = axes[0]
bars = ax.barh(y, rel_size * 100, height=0.62, color="#93C5FD",
               edgecolor="#3B82F6", linewidth=1.0)
ax.set_yticks(y)
ax.set_yticklabels(CLIENTS, fontsize=9)
for yi, v in zip(y, rel_size * 100):
    ax.text(v + 0.8, yi, f"{v:.1f}%", va="center", ha="left", fontsize=9, color="#334155")
ax.set_xlim(0, max(rel_size) * 100 * 1.22)
ax.set_xlabel("Share of pooled training data (%)", fontsize=10.5)
ax.set_title("(a) Client shard sizes", fontsize=11, fontweight="bold", loc="left", pad=8)
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
colors = ["#F59E0B" if r > 60 or r < 20 else "#34D399" for r in rates]
bars = ax.barh(y, rates, height=0.62, color=colors,
               edgecolor=["#B45309" if c == "#F59E0B" else "#059669" for c in colors],
               linewidth=1.0)
ax.set_yticks(y)
ax.set_yticklabels(CLIENTS, fontsize=9)
for yi, v in zip(y, rates):
    ax.text(v + 1.2, yi, f"{v:.1f}%", va="center", ha="left", fontsize=9, color="#334155")
ax.axvline(POOL_RATE * 100, color="#EF4444", linestyle="--", linewidth=1.4, alpha=0.85)
ax.text(POOL_RATE * 100 + 1.0, N_CLIENTS - 0.42, "pooled rate\n48.9%",
        fontsize=8.5, color="#B91C1C", va="top")
ax.set_xlim(0, 100)
ax.set_xlabel("Local positive-class (flare) rate (%)", fontsize=10.5)
ax.set_title("(b) Label skew across clients", fontsize=11, fontweight="bold", loc="left", pad=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color("#9CA3AF")
ax.tick_params(axis='y', length=0)
ax.tick_params(axis='x', colors="#4B5563")
ax.grid(True, axis='x', linestyle='--', alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)

import os
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper/figures/fig_partition.png")
fig.savefig(out, dpi=220, facecolor="white")
print("Saved", out)
print("Relative sizes (%):", np.round(rel_size * 100, 1))
print("Flare rates (%):", np.round(flare_rate * 100, 1))

#!/usr/bin/env python3
"""Figure: per-client PR-AUC (local-only vs FedAvg vs FedProx global models).

Run from anywhere: uses the repo root relative to this file.

Regenerated for review-2 (R8): neutral client labels from config.CLIENT_NAMES,
data from the frozen-protocol artefact outputs/results.json (client_level).
Grouped horizontal bars, same palette family as the other paper figures.
"""
import json
import sys
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from config import CLIENT_NAMES  # neutral labels (R8)

with open("outputs/results.json") as f:
    cl = json.load(f)["client_level"]["per_client"]

names = [c.replace("\n", " ") for c in CLIENT_NAMES]
local = [c["local"]["pr_auc"] for c in cl]
fedavg = [c["fedavg"]["pr_auc"] for c in cl]
fedprox = [c["fedprox"]["pr_auc"] for c in cl]

n = len(names)
y = np.arange(n)[::-1]
h = 0.26

fig, ax = plt.subplots(figsize=(9.2, 4.3), constrained_layout=True)

b1 = ax.barh(y + h, local, height=h, color="#93C5FD", edgecolor="#3B82F6",
             linewidth=0.9, label="Local-only model")
b2 = ax.barh(y, fedavg, height=h, color="#FCA5A5", edgecolor="#EF4444",
             linewidth=0.9, label="FedAvg global (diverged)")
b3 = ax.barh(y - h, fedprox, height=h, color="#86EFAC", edgecolor="#059669",
             linewidth=0.9, label="FedProx global")

for yi, v in zip(y + h, local):
    ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=7.6, color="#1D4ED8")
for yi, v in zip(y, fedavg):
    ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=7.6, color="#B91C1C")
for yi, v in zip(y - h, fedprox):
    ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=7.6, color="#047857")

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=9.5)
ax.set_xlim(0, 1.14)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("PR-AUC on per-client 20% shard holdout (within-distribution; see paper caveat)",
              fontsize=10)
ax.set_title("Per-client discrimination: local-only vs federated global models",
             fontsize=11.5, fontweight="bold", loc="left", pad=8)
ax.legend(loc="lower left", fontsize=8.8, frameon=True, framealpha=0.95)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(True, axis='x', linestyle='--', alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)

out = os.path.join(REPO, "paper/figures/fig_clients.png")
fig.savefig(out, dpi=220, facecolor="white")
print("Saved", out)
print("mean local:", round(float(np.mean(local)), 3),
      "mean fedavg:", round(float(np.mean(fedavg)), 3),
      "mean fedprox:", round(float(np.mean(fedprox)), 3))

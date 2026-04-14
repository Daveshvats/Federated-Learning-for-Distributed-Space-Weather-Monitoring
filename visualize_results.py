"""
visualize_results.py
─────────────────────
Generates all figures needed for the ICE2CT-2026 paper.

Figures produced:
  1. Confusion_Matrices.png      — 4 models side by side
  2. ROC_Curves.png              — all models on one axes
  3. FL_Convergence.png          — F1 / Recall per round (FedAvg vs FedProx)
  4. SHAP_Feature_Importance.png — bar chart of mean |SHAP| for XGBoost
  5. Comparison_Table.png        — formatted metrics table
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, List

from config import OUTPUT_DIR, CLIENT_NAMES


os.makedirs(OUTPUT_DIR, exist_ok=True)

# Consistent colour map for the paper
COLORS = {
    "Logistic Regression":      "#4E79A7",
    "XGBoost (Centralised)":    "#F28E2B",
    "FedAvg MLP":               "#59A14F",
    "FedProx MLP":              "#E15759",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(all_results: Dict, y_test: np.ndarray) -> None:
    """4-panel confusion matrix grid. Normalised by true class (row-norm)."""
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, all_results.items()):
        cm = confusion_matrix(y_test, res["preds"], normalize="true")
        sns.heatmap(
            cm, annot=True, fmt=".2f", ax=ax,
            cmap="Blues", vmin=0, vmax=1,
            xticklabels=["No Flare", "Flare"],
            yticklabels=["No Flare", "Flare"],
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 12}
        )
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

    fig.suptitle("Confusion Matrices (row-normalised)", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "Confusion_Matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ROC CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(all_results: Dict, y_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC = 0.50)")

    for name, res in all_results.items():
        fpr, tpr, _ = roc_curve(y_test, res["probs"])
        roc_auc      = auc(fpr, tpr)
        color        = COLORS.get(name, "#888888")
        ax.plot(fpr, tpr, label=f"{name}  (AUC = {roc_auc:.3f})", color=color, linewidth=2)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    path = os.path.join(OUTPUT_DIR, "ROC_Curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FL CONVERGENCE CURVE
# ─────────────────────────────────────────────────────────────────────────────

def plot_fl_convergence(
    fedavg_history:  List[Dict],
    fedprox_history: List[Dict]
) -> None:
    """Show F1 and Recall per communication round for both FL methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for history, label, color in [
        (fedavg_history,  "FedAvg",  COLORS["FedAvg MLP"]),
        (fedprox_history, "FedProx", COLORS["FedProx MLP"]),
    ]:
        rounds  = [h["round"]  for h in history]
        f1s     = [h["f1"]     for h in history]
        recalls = [h["recall"] for h in history]
        ax1.plot(rounds, f1s,     marker="o", label=label, color=color, linewidth=2, markersize=5)
        ax2.plot(rounds, recalls, marker="s", label=label, color=color, linewidth=2, markersize=5)

    for ax, ylabel, title in [
        (ax1, "F1-Score",            "F1-Score vs. Communication Rounds"),
        (ax2, "Recall (True Positive Rate)", "Recall vs. Communication Rounds"),
    ]:
        ax.set_xlabel("Communication Round", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Federated Learning Convergence", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "FL_Convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SHAP FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_importance(
    mean_shap:     np.ndarray,
    feature_names: List[str],
    top_n:         int = 15
) -> None:
    """Horizontal bar chart of top-N features by mean |SHAP|."""
    idx     = np.argsort(mean_shap)[-top_n:]
    vals    = mean_shap[idx]
    labels  = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(labels, vals, color=COLORS["XGBoost (Centralised)"], edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|  (impact on model output)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances\n(XGBoost Centralised — SHAP)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    # Annotate values
    for bar, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "SHAP_Feature_Importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_table(all_results: Dict) -> None:
    """Render a formatted metrics table as a PNG for direct paper inclusion."""
    rows   = []
    header = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    for name, res in all_results.items():
        rows.append([
            name,
            f"{res['accuracy']:.3f}",
            f"{res['precision']:.3f}",
            f"{res['recall']:.3f}",
            f"{res['f1']:.3f}",
            f"{res['roc_auc']:.3f}",
        ])

    fig, ax = plt.subplots(figsize=(12, 1 + 0.6 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=header,
        cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)

    # Style header row
    for j in range(len(header)):
        table[(0, j)].set_facecolor("#2C3E50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Highlight best Recall row
    recalls = [float(r[3]) for r in rows]
    best_row = np.argmax(recalls) + 1
    for j in range(len(header)):
        table[(best_row, j)].set_facecolor("#D5F5E3")

    ax.set_title("Model Comparison (threshold = 0.35, best Recall highlighted)",
                 fontsize=12, fontweight="bold", pad=20)

    path = os.path.join(OUTPUT_DIR, "Comparison_Table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PRINT RESULTS TO CONSOLE
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(all_results: Dict) -> None:
    sep = "=" * 72
    fmt = "{:<30} | {:<6} | {:<6} | {:<7} | {:<6} | {:<7}"
    print(f"\n{sep}")
    print(fmt.format("Model", "Acc", "Prec", "Recall", "F1", "ROC-AUC"))
    print(sep)
    for name, res in all_results.items():
        print(fmt.format(
            name,
            f"{res['accuracy']:.3f}",
            f"{res['precision']:.3f}",
            f"{res['recall']:.3f}",
            f"{res['f1']:.3f}",
            f"{res['roc_auc']:.3f}",
        ))
    print(sep + "\n")

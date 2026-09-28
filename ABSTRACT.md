# Abstract — SF-9 (revised on `improvements`)
## Federated Learning for Data-Locality-Preserving Distributed Space Weather Monitoring: A Solar-Flare Prediction Component for Grid-Resilience Pipelines

---

Solar flares of class M and X are the proximate precursors of the
coronal mass ejections and geomagnetic disturbances that drive
geomagnetically induced currents (GICs) in high-voltage grids — the
mechanism behind the 1989 Québec blackout. Early flare warning therefore
buys preparation time at the earliest link of the flare → CME → storm →
GIC chain. Training competitive flare-prediction models requires pooling
magnetogram-derived feature histories that different agencies may be
unable or unwilling to centralise for governance, locality, and
institutional reasons. Federated learning (FL) removes the need to pool
raw observations: only model updates are shared.

We evaluate FL as a training paradigm for solar-flare prediction on the
SWAN-SF benchmark partitioned into six simulated regional clients under
Dirichlet label skew (non-IID). The present work makes three
methodological contributions over our earlier prototype: (i) a
disjoint-by-construction client partition with a runtime leakage audit
(sample overlap, active-region overlap, and split exclusivity are
verified programmatically); (ii) a strict evaluation contract in which
calibration and F-beta threshold selection occur exclusively on a
validation split, with the held-out test set evaluated exactly once
under the frozen pipeline — together with prior-shift-aware probability
calibration addressing the ~25x gap between balanced training
prevalence and the ~1.9% operational prevalence; and (iii) an
ablation design that isolates the effects of the proximal regulariser
(FedProx), Fed-Focal loss, per-client SMOTE, and distribution-aware
aggregation, relative to a centralized multi-layer perceptron of
identical architecture that quantifies the accuracy cost of
federation itself.

Results (improvements branch v3.0.1, frozen protocol, single-pass test
evaluation, 331,185 windows at 1.88% prevalence): FedProx reaches
ROC-AUC 0.954 (97.7% of the centralised XGBoost reference, 0.977) and
PR-AUC 0.307, **exceeding** the pooled centralised MLP of identical
architecture (0.888 / 0.153) — federation costs no ranking skill within
the same hypothesis class. Plain FedAvg diverges during training
(validation AUC 0.99 → 0.05 after round ~20); FedProx's proximal term
is the load-bearing design choice, beating FedAvg on all five seeds
(one-sided Wilcoxon p = 0.031) with 2.7× narrower ROC-AUC variance.
The 24-cell ablation grid isolates each component: FedProx cells are
uniformly stable (0.954–0.959 AUC), DA-FL aggregation partially
rescues FedAvg (0.880 → 0.950), and the SMOTE arm is inert on the
pre-balanced benchmark (a reported null). Cross-model SHAP analysis
shows both the centralised and federated models concentrate decisions
on current helicity, the R-value flux-emergence proxy, and Lorentz
force — the parameters flare physics predicts. We also document an
honest threshold-transfer failure under the 26× validation-to-test
prevalence shift, and report prevalence-robust recall-at-FPR operating
points instead (FedProx: 0.51 recall at 2% FPR). Full-weight
communication costs 0.24 MB per client per round (70.5 MB over the
complete 50-round run).

Results are reported as PR-AUC, Brier score, expected calibration
error, and recall at fixed false-alarm budgets (0.5–5% FPR), reflecting
operational alarm economics rather than threshold-flattering accuracy.
Because the evaluation protocol was corrected after an internal audit,
all numerical claims are regenerated from machine-readable run
artifacts (results.json) by the pipeline on the `improvements` branch;
no hand-entered numbers survive from the earlier prototype. The scope
of this work is deliberately bounded: it demonstrates a federated
flare-probability component and quantifies its cost and calibration
behaviour — it does not model CME propagation, geomagnetic response, or
GIC magnitude, and it does not claim cryptographic privacy (secure
aggregation is scaffolded and threat-modelled, not yet deployed).
Within these bounds, federated learning offers a governance-compatible
training mechanism for distributed space-weather agencies, at a
measured and reportable accuracy cost.

---

*Keywords: Federated Learning, Solar Flare Prediction, Non-IID Data,
SWAN-SF, Class Imbalance, Probability Calibration, Space Weather*

*Track: Track 3 — Computational Intelligence and Machine Learning Applications*

**Change log vs. the original abstract (honesty revision):**
- "privacy-preserving" → "data-locality-preserving" (no secure
  aggregation/DP in the training path yet)
- "partitioned by active-region identifiers" → "Dirichlet label-skew
  partition" (matches the actual implementation; region-based
  partitioning is provided but the cleaned export lacks region IDs)
- removed "[X.XX] within [X.X]% of the centralized XGBoost upper bound"
  → XGBoost is now a *reference baseline*; the federation cost is
  measured against a centralized MLP of identical architecture
- added the protocol corrections (validation-only selection,
  leakage audit) as explicit contributions
- "integration with existing SCADA infrastructure" → explicitly
  out of scope (see docs/GIC_BOUNDARY.md)

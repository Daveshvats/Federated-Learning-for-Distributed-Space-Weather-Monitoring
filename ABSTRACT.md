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

Results (improvements branch v3.1, frozen protocol, single-pass test
evaluation, 331,185 windows at 1.88% prevalence): FedProx reaches
ROC-AUC 0.954 (97.7% of the centralised XGBoost reference, 0.977) and
PR-AUC 0.307, and **did not underperform** the pooled centralised MLP
of identical architecture (0.888 / 0.153) on this benchmark and
configuration — an observation reported with its caveat: the
federated arm consumed the larger optimisation budget (500 effective
local passes vs ≤30 pooled passes; the budget audit is now a shipped
artefact), and a regularisation-like mechanism is the plausible
reading, not "federation is free of cost". Plain FedAvg diverges
during training (validation AUC 0.99 → 0.05 after round ~20);
FedProx is the **most consistent stabiliser** among the components
studied — exceeding FedAvg's PR-AUC on all five seeds (one-sided
Wilcoxon p = 0.031) with 2.7× narrower ROC-AUC variance — while the
24-cell ablation grid shows distribution-aware aggregation also
rescues FedAvg substantially (best FedAvg cell 0.963 > FedProx's
0.954), so stability is not attributable to the proximal term alone.
The SMOTE arm is inert on the pre-balanced benchmark (a reported
null). Cross-model SHAP analysis shows both the centralised and
federated models concentrate decisions on current helicity, the
R-value flux-emergence proxy, and Lorentz force — the parameters
flare physics predicts. We also document an honest threshold-transfer
failure under the 26× validation-to-test prevalence shift — at
natural prevalence the neural posteriors are currently **worse than
climatology** on Brier and ECE, which we state as an open calibration
problem with a shipped comparison harness (raw / prior-shift /
Platt / isotonic / temperature / frozen-FPR) — and report
prevalence-robust recall-at-FPR operating points as window-level
curve statistics (FedProx: 0.51 recall at 2% FPR). Full-weight
communication costs 0.24 MB per client per round (70.5 MB over the
complete 50-round run; parameter-exchange arithmetic, not measured
latency).

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

**Change log v3.2 (review-2 response, see docs/REVIEW2_RESPONSE.md):**
- "exceeding the pooled centralised MLP / costs no ranking skill" →
  "did not underperform … on this benchmark and configuration" + the
  optimisation-budget confound is stated and audited
- "load-bearing design choice" → "most consistent stabiliser among the
  components studied"; the ablation contradiction (best FedAvg+DA-FL
  cell 0.963 > FedProx 0.954) is stated in the abstract itself
- calibration honesty: neural posteriors worse than climatology
  (Brier/ECE) at natural prevalence; calibration question declared
  open; comparison harness shipped
- recall-at-FPR relabelled as a window-level curve statistic over
  correlated windows
- client labels neutralised (A–F by region; no institution implied)
- communication numbers marked as parameter-exchange arithmetic,
  latency/aggregation/straggler costs explicitly unmeasured

## v3.3 addendum (2026-09-29, executed edition)

Abstract-level updates from the independent re-execution:

- new sentence: "An independent re-execution from the public dataset
  artefacts (byte-verified against the SHA-256 data manifest)
  reproduces every headline metric exactly; in the same session a
  five-arm calibration comparison shows that no validation-fit
  calibrator survives the prior shift, and an untouched-holdout study
  confirms that the proximal-stabilised global model matches
  local-only performance for every simulated client while clearly
  aiding the two smallest."
- the calibration question is no longer "declared open": it is
  answered with a closed negative result (validation-Brier selection
  picks isotonic, the worst arm on test for the neural models)
- untouched-holdout numbers: local 0.983 / FedAvg 0.866 / FedProx
  0.988 mean PR-AUC; FedProx +0.027 / +0.019 on the two smallest
  clients
- frozen-FPR realised test FPR quantified (FedProx 28.2-72.1%
  against 0.5-5% targets), strengthening the threshold-transfer
  honesty

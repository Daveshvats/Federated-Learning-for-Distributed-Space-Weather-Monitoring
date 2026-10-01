# Interpretation Freeze — v3.9 (evidence-locked)

Date: 2026-10-02. This memo locks the interpretation of the BN-transport
diagnostic and the controlled no-BatchNorm experiment BEFORE the paper is
rewritten. Every number below is machine-verified against frozen artefacts
(`outputs/bn_diagnostic.json`, `outputs/nobn_control.json`,
`data/cache/phase_*.pt`, `data/cache/bn_diag_state.pt`,
`data/cache/nobn_*_state.pt`, seed-42 frozen protocol). No number in this
memo may be reinterpreted during the v3.9 rewrite; the paper must conform
to this memo, not the reverse.

## 1. Locked facts

F1. **BN running statistics are never transported.** `get_weights`/
`set_weights` (model.py) move `model.parameters()` only. The server-side
global model's three BatchNorm1d layers have `num_batches_tracked = 0`,
`running_mean = 0`, `running_var = 1` in both saved checkpoints
(phase_fedavg.pt, phase_fedprox.pt) — i.e. they remained at initialisation
for the entire 50-round run, and every server-side evaluation (validation
monitor, checkpoint selection, final test) used those init buffers.

F2. **The shipped FedAvg artefact is the round-20 weights.** The
validation-F1 monitor tied at the saturated all-positive value
2p/(1+p) = 0.6565 (p = 0.4887) for rounds 5–15 and then registered
0.6588 at round 20 — where validation ROC-AUC had already collapsed from
0.993 to 0.541. The strict-`>` best-checkpoint rule therefore restored a
collapse-round state over the healthy round-5 state. FedProx's monitor
tied at 0.6565 from round 5 onward and never collapsed, so its restore
was harmless.

F3. **FedAvg weights never lost discriminative content.** Faithful
re-run (val trajectory matches frozen history within ±0.01 ROC,
best-checkpoint rule restores round 20 exactly). Test ROC-AUC of the
same FedAvg global weights under four BN treatments:

| round | A: shipped (init buffers) | B: recalibrated exact | C: batch stats | D: transported round buffers |
|---|---|---|---|---|
| 5  | 0.950 | 0.786 | 0.946 | 0.885 |
| 20 | 0.575 | 0.893 | 0.951 | 0.931 |
| 50 | 0.098 | 0.881 | 0.953 | 0.935 |

The published 0.575 is the round-20 weights evaluated with init buffers.
The "diverged" round-50 state (shipped val ROC 0.054) carries 0.953 test
ROC under batch statistics. PR-AUC at round 20: 0.199 (A) vs 0.329 (C).

F4. **FedProx's returned artefact is also protocol-dependent.** Its
round-5 weights score test ROC 0.954 (A, init buffers) / 0.757 (B) /
0.951 (C); PR 0.307 / 0.040 / 0.333. Under batch statistics the two
algorithms' artefacts are statistically indistinguishable
(0.951/0.951 ROC; 0.329/0.333 PR).

F5. **No-BN control (identical protocol, BatchNorm1d → Identity,
28,929 params): both algorithms are stable and near-identical.**
- FedAvg-noBN: val F1 0.976–0.982 at every logged round (no saturated
  tie), val ROC 0.999 throughout; test ROC 0.913 (r5) → 0.858–0.873
  (r40–50); selected checkpoint (r40, best val F1 0.9824): 0.858 / PR
  0.108.
- FedProx-noBN: val F1 0.971–0.982, val ROC 0.995–0.999; test ROC
  0.949 (r5) → 0.861–0.871 (r45–50); selected (r50): 0.871 / PR 0.128.
Neither arm diverges; the FedAvg–FedProx gap in the shipped numbers
(0.954 vs 0.575 ROC) does not survive BN removal (gap ≤ 0.04 ROC at any
round, ≤ 0.01 at r50).

F6. **Mild, non-catastrophic drift is real.** Both no-BN arms decay in
test PR-AUC over rounds (FedProx-noBN 0.286 → 0.128; FedAvg-noBN
0.199 → 0.108), and the BN architecture under correct normalisation
(arm C/D) reaches higher PR than the no-BN architecture. Client drift
and the optimisation benefits of BN are both genuine; neither produces
catastrophic collapse on its own.

## 2. Locked interpretation (what the paper may and may not claim)

I1. "FedAvg diverges" is retired as a statement about learning. The
correct statement: under an implementation that never transports BN
running statistics, the *evaluated* FedAvg global model drifts out of
the region where init-buffer normalisation is valid (0.950 → 0.575 →
0.098 test ROC across rounds), while the underlying weights retain
~0.95 ROC of discriminative content (F3).

I2. The proximal term's documented "stabilisation" (including the 5/5-seed
multiseed Wilcoxon result) is re-attributed: it stabilises compatibility
with the stale-buffer server evaluation (weights stay near
initialisation, where (0,1) buffers are valid), and it retains a small
genuine early-round optimisation benefit (F5). It is NOT evidence that
plain FedAvg cannot handle this heterogeneity — without BN both
algorithms train stably to near-identical test quality (F5).

I3. The saturated validation-F1 monitor (2p/(1+p) tie) was not merely
uninformative — it was adversarial: it preferred a collapse-round
checkpoint (F2). The checkpoint-independence claim for FedProx is
retired; the honest statement is that both the monitor and the
checkpoint rule are unsafe under a saturated metric and a stale-buffer
evaluation.

I4. The evaluation-protocol failure mode is itself a finding of the
paper: three independent mechanisms (untransported BN statistics, a
saturated thresholded metric, strict-`>` checkpoint selection on that
metric) composed to manufacture a large spurious algorithm gap. This is
a contribution about FL evaluation practice, reported as an audit
finding, not hidden.

I5. Scope discipline: these findings concern the shipped MLP pipeline on
the cleaned substrate (in-partition experiments). They do not retract
the leakage-free fold numbers (which stand as reported), the
raw-substrate encoder-conditional results (LSTM arms, owner-GPU), the
event-level evaluation, or the calibration/threshold-transfer failures —
those are evaluation-side results independent of the BN mechanism and
remain valid as published. The multiseed/sweep/ablation in-partition
studies are retained as protocol-stability evidence with corrected
attribution (I2), not as generalisation estimates.

I6. Precedence: Fu et al. (Research Square preprint, posted 2023-07-10,
"Federated Transfer Learning for Soalr Flare Forecasting" [sic]) precedes
this work. No "first federated solar-flare prediction" claim may appear.
The defensible claim: first federated evaluation on SWAN-SF, first with a
provenance audit against the raw benchmark, and first leakage-free
re-evaluation of federated flare prediction under the benchmark's
intended protocol.

## 3. Numbers the paper must carry (v3.9)

- Diagnostic table (F3) — four BN treatments × rounds 5/20/50, test ROC/PR.
- FedProx artefact arms (F4).
- No-BN control table (F5): both algorithms, selected-checkpoint test
  ROC/PR + per-round trajectory summary.
- Buffer-state facts (F1, F2): num_batches_tracked = 0; round-20 restore.
- Corrected Brier range 0.127–0.544; α=0.5 realised rates 0.3–99.8%
  (seed 42, re-measured); realised partition 28.6–80.5% / 2.3–35.6%;
  communication honesty (70.5 MB vs 47.3 MB cleaned centralisation /
  1.24 GB raw centralisation).
- zhao2018 corrected (arXiv:1806.00582, Zhao/Li/Lai/Suda/Civin/Chandra);
  Fu et al. cited as preprint.

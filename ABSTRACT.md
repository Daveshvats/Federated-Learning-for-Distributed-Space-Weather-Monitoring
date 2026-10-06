# Abstract — SF-9 (revised on the `improvements` branch)

## Federated Solar-Flare Prediction on SWAN-SF: A Benchmark Audit of Provenance, Leakage, and Evaluation Protocols

---

We audit what happens when federated learning is evaluated on the
public SWAN-SF flare benchmark. Six *simulated* clients hold disjoint
Dirichlet-skewed shards of the cleaned export under a frozen protocol:
immutable splits, validation-only selection and calibration, a runtime
leakage audit, and single-pass test evaluation. A provenance audit
against the raw benchmark (normalisation-invariant instance matching,
98.7–100% coverage, 100% test-side label agreement) shows the cleaned
export's same-partition train/test pairing shares instances: every
flaring test window (6,234) also appears in training, and 85.7–90% of
training positives are TimeGAN-synthetic. On a leakage-free fold
(training partitions 1–4, single-pass test on partition 5)
discrimination generalises for all arms (ROC-AUC 0.906–0.978) and the
federated-versus-centralised gap disappears. The deployability verdict,
however, is set by the corrected persistence baselines: on TSS, XGBoost
(0.848) and logistic regression (0.489) alone clear 24-hour-lagged
persistence (0.418), and on HSS no arm clears it (0.434); no arm
beats same-region label inertia (TSS 0.967) at this 24-hour label
geometry. A BatchNorm-transport
diagnostic then shows the in-partition FedAvg collapse (test ROC-AUC
0.575) to be a composite evaluation artefact — untransported
normalisation statistics, a mechanism documented in the
federated-learning literature since FedBN, compounded by a saturated
validation monitor and the checkpoint rule that monitor feeds — rather
than a federated-learning failure: the same weights score 0.951 with
local statistics, and a controlled no-BatchNorm run leaves both FedAvg
and FedProx stable. On the raw, unbalanced substrate the federated MLP
collapse is BatchNorm-under-federation-conditional, and two controls
establish it: the checkpoint-level BN diagnostic finds that
pooled-statistics recalibration — validated as an instrument on the
centralised model, where it should reconstruct the model's own statistics and
does preserve the ROC-AUC ranking — collapses every federated arm to chance, so a
pooled-statistics transport fix would not rescue federation there,
while evaluation-time batch statistics leave the deficit standing; and
a no-BatchNorm re-training, changing nothing else, restores the
federated MLP to the sequence arms' ROC-AUC ranking (FedAvg 0.963/0.366,
FedProx 0.975/0.345, the latter above the centralised MLP on ROC-AUC;
FedProx-LSTM, which also carries no BatchNorm, reaches ROC-AUC 0.970
and PR-AUC 0.404). Calibration, not detection, is the deployability
boundary: no validation-fit decision layer survives the prevalence
shift. Standard verification metrics (TSS, HSS, Brier skill,
persistence and climatology baselines, with reliability for the
in-partition arms) accompany every stored operating point of every
substrate. This is the first federated evaluation on SWAN-SF and the
first instance-level provenance audit of the cleaned export; prior
federated flare forecasting (Fu et al., 2023 preprint) and prior
random-split leakage warnings (the benchmark's own creators, 2019)
both exist. Every number regenerates from committed result artefacts,
and the underlying dataset cache is frozen under a separate 20-file
SHA-256 manifest verified by a committed script.

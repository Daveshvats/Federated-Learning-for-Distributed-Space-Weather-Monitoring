# Abstract — SF-9 (revised on `improvements`, v3.9)

## Federated Solar-Flare Prediction on SWAN-SF: A Provenance, Leakage, and Evaluation-Protocol Audit

---

Grid-resilience operators need flare-probability components trained on
magnetogram histories, and data-locality constraints can preclude
pooling the underlying observations. We audit what actually happens
when federated learning meets the public SWAN-SF benchmark. Six
*simulated* regional clients hold disjoint Dirichlet-skewed shards of
the cleaned benchmark under a frozen protocol: immutable splits,
validation-only selection and calibration, a runtime leakage audit, and
single-pass test evaluation. A provenance audit against the raw
benchmark (normalisation-invariant instance matching, 98.7–100%
coverage, 100% label agreement) shows the cleaned export's
same-partition train/test pairing shares instances: every flaring test
window (6,234) also appears in training, and 85.7–90% of training
positives are TimeGAN-synthetic. On a leakage-free fold (training
partitions 1–4, single-pass test on partition 5) discrimination
generalises for all arms (ROC-AUC 0.906–0.978) and the
federated-versus-centralised gap disappears. A BatchNorm-transport
diagnostic then shows the in-partition FedAvg collapse (test ROC-AUC
0.575) to be an evaluation artefact: the same weights score 0.951 under
correct normalisation, FedAvg's weights never lose discriminative
content (0.953 at round 50), and a controlled no-BatchNorm run leaves
both FedAvg and FedProx stable and near-identical — the published
algorithm gap is substantially mediated by untransported BN statistics,
a saturated validation monitor, and the checkpoint rule that monitor
feeds. On the raw, unbalanced substrate the outcome is
encoder-conditional: federated MLP arms degrade while LSTM arms recover
(FedProx-LSTM 0.970/0.404). Calibration, not detection, is the
deployability boundary. Prior federated flare forecasting exists (Fu et
al., 2023 preprint); this is the first federated evaluation on SWAN-SF
and the first with a provenance audit. An independent byte-verified
re-execution reproduces every headline metric.

**Keywords**: federated learning; solar flare prediction; SWAN-SF; data
provenance; evaluation audit; BatchNorm; non-IID data; class imbalance;
data locality

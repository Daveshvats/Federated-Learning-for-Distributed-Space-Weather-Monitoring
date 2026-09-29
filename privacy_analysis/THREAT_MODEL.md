# SF-9 — Privacy & Security Threat Model

> Status: **documented, mitigation scaffolded, NOT yet implemented in the
> training pipeline.** Until secure aggregation and DP are wired into the
> FL loops, the accurate terminology is **"data-locality-preserving"**, not
> "privacy-preserving" (see `audit/CLAIMS.md`).
>
> **v3.1 note (review-2, finding #21):** this table now has a condensed,
> citable counterpart in the manuscript — `paper/sections/sec_method.tex`
> (Table `tab:threat`, "Threat model and security posture"). The two are
> consistent: the benchmark study runs *undefended*; poisoning/backdoor
> defence is not addressed; secure aggregation is implemented as a
> round-trip-verified module but not wired into training.

## 1. System description

```
 6 simulated clients (one machine)          aggregation server
 ┌────────────┐  w_k (plain weights)   ┌──────────────┐
 │ client k   │ ─────────────────────> │  FedAvg/     │
 │ local data │ <───────────────────── │  FedProx     │
 └────────────┘  global weights        └──────────────┘
```

What is actually transmitted today: **raw float32 parameter tensors**
(full model weights, no masking, no noise). This is the honest starting
point for the threat model.

## 2. Threat actors

| Actor | Capability | Risk today | Mitigation |
|---|---|---|---|
| Honest-but-curious server | sees every client update in clear | **HIGH** — gradient inversion can reconstruct training samples (Zhu et al., 2019 "Deep Leakage from Gradients") | secure aggregation (`secure_aggregation.py` scaffold, Bonawitz 2017 masking) |
| Malicious client | sends crafted updates | model poisoning / backdoor (Bagdasaryan 2020) | norm-bounding (partially present via weight clipping), Byzantine-robust aggregation (future) |
| Network attacker | observes/modifies traffic | update tampering, replay | TLS + update authentication (future) |
| Membership-inference adversary | black-box model + auxiliary data | membership leakage of flare-active regions (Shokri 2017) | differential privacy (future), per-client evaluation |
| Compromised client | full local data + model | exfiltration of its own shard; no cross-client exposure | organisational controls (out of scope) |

## 3. Attack surface — experiments to run (Stage 14 checklist)

| Attack | Method | Metric | Status |
|---|---|---|---|
| Gradient inversion | DLG / iDLG on a captured update | reconstruction MSE vs. true samples | not run (needs torch; protocol documented) |
| Membership inference | shadow-model attack on FL global model | membership AUC vs 0.5 baseline | not run |
| Model poisoning | label-flip / weight-scale malicious client | global AUC degradation | not run |
| Backdoor | targeted flare misclassification | attack success rate | not run |
| Replay | resubmit old updates | drift/rollback effect | not run (dropout schedule exists in `communication_cost.simulate_dropout`) |

## 4. Mitigations

### 4.1 Secure aggregation — scaffolded (`secure_aggregation.py`)

Bonawitz-style pairwise + self masking:
- pairwise masks cancel exactly at the server (round-trip test proves
  the server recovers the SUM only, max error < 1e-9);
- self masks + Shamir shares handle client dropout;
- **honest limitations**: PRNG mock instead of real DH key agreement; no
  authenticated channels; not constant-time; integration path = Flower
  SecAgg / OpenFL.

### 4.2 Differential privacy — designed, not implemented

Target: `(ε, δ)`-DP client-level DP via per-update Gaussian noise +
clip. Expected trade-off curve to measure: ε ∈ {1, 3, 8, ∞} vs
PR-AUC drop. The `communication_cost.py` module already measures the
update-size cost baseline required for this analysis.

## 5. Claims this supports after implementation

| Claim | Requires |
|---|---|
| "server cannot observe individual updates" | secure aggregation wired in + test |
| "training is (ε,δ)-differentially private" | DP-SGD noise + formal accounting |
| "robust to m malicious clients" | Byzantine-robust aggregation + poisoning experiments |

Until then, the manuscript must state: *"Only model updates leave the
client; no raw observations are transmitted. Formal privacy guarantees
(secure aggregation, differential privacy) are designed but not yet
implemented, and gradient-inversion attacks on shared updates remain a
documented risk."*

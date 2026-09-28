# SF-9 — Limitations

> Companion to `audit/CLAIMS.md`. These are the boundaries a reviewer
> will find in 20 minutes; state them before they do.

## Simulation vs. reality

1. **Six simulated clients on one machine.** The federation is
   simulated over a public benchmark (SWAN-SF). No real inter-agency
   network, no NASA/ESA/JAXA/ISRO/KASI/BoM operational data, no
   restricted-data governance constraints were exercised. The
   data-sovereignty motivation is institutional plausibility, not a
   demonstrated constraint — NASA heliophysics data are substantially
   public.
2. **Non-IID structure is simulated label skew** (Dirichlet), not
   physical observatory heterogeneity. Real cross-agency heterogeneity
   is dominated by instrument differences, cadence, coverage duty
   cycle, and processing pipelines. A region-ID-based geographic
   partition (`partition_data_geographic`) exists but the cleaned
   SWAN-SF export lacks region IDs, so it falls back with an explicit
   warning.
3. **No secure aggregation / DP in the training path.** Raw updates are
   transmitted; gradient-inversion risk documented in
   `privacy_analysis/THREAT_MODEL.md`. Claims use
   "data-locality-preserving" terminology.

## Dataset and labelling

4. **The cleaned SWAN-SF export is pre-processed by others** (RUS +
   Tomek links + TimeGAN augmentation, LSBZM normalization, FPCKNN
   imputation). We inherit its labels and any artefacts: training
   prevalence (~49%) is an artefact of RUS balancing, not nature;
   test prevalence (~1.9%) reflects the 24h M/X window definition.
5. **Synthetic-minority provenance**: the training partitions include
   TimeGAN-augmented positives. SMOTE-style interpolation plausibility
   is analysed in `experiments/analyze_smote_validity.py`, but
   TimeGAN-sample validity was not audited by us.
6. **Label horizon** is fixed by the export (24h M/X). No
   horizon-sensitivity analysis was run.

## Evaluation protocol

7. **All pre-improvements numbers came from a flawed protocol**
   (test-set thresholding, stale FL accuracy, non-disjoint shards,
   DA-FL in both arms, single seed). They must not be cited as-is; the
   `improvements` branch re-runs everything under the corrected
   protocol.
8. **Single-seed history**: multi-seed statistics exist as tooling
   (`experiments/run_multiseed.py`) but require compute the owner must
   run.
9. **Calibration under 25x prior shift** is addressed analytically
   (prior-shift/Platt/isotonic/temperature) but validation prevalence
   is still a proxy for operational prevalence.

## Modelling

10. **Architecture confound in the old comparison** (federated MLP vs
    centralized XGBoost): the centralized MLP baseline on this branch
    fixes the federation-cost comparison.
11. **LSTM mode** was not run in the final main-branch configuration
    ("No LSTM run", commit c978c51); LSTM claims are unverified.
12. **SCAFFOLD** is implemented with documented NaN fixes but disabled;
    no validated results exist.

## Scope

13. **No CME/GIC chain** (see `docs/GIC_BOUNDARY.md`): flare
    probability is the first link of a seven-link chain.
14. **SCADA integration is architectural**, not demonstrated.
15. **Communication/deployment costs are measured** for the research
    prototype (bytes/round) but no real WAN deployment was tested.

## What IS solid

- Disjoint, audited partitioning (0 overlap, 100% coverage, runtime
  leakage audit)
- Immutable train/val/test contract, threshold-on-validation
- Full metric set (PR-AUC, Brier, ECE, recall@FPR) appropriate to a
  1.9%-positive operational setting
- Reproducible single-source config + machine-readable results
- Honest claim table with evidence mapping

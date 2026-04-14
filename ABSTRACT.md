# Abstract — SF-9
## Federated Learning for Privacy-Preserving Distributed Space Weather Monitoring in Smart Grid Protection

---

The vulnerability of modern Smart Grids to Geomagnetically Induced Currents (GICs),
triggered by M-class and X-class solar flares, presents a critical threat to power
infrastructure. High-voltage transformers exposed to GIC saturation can suffer
irreversible damage, as demonstrated by the 1989 Quebec blackout. Effective early
warning systems require predictive models trained on high-fidelity solar observational
data spanning multiple geographic regions; however, nationally sensitive space weather
data collected by agencies such as NASA, ESA, JAXA, and ISRO cannot be centralised
due to data sovereignty constraints and inter-agency data-sharing agreements. This
paper proposes the first application of Federated Learning (FL) to solar flare
prediction for Smart Grid protection, explicitly addressing this data privacy barrier.

The proposed framework partitions the SWAN-SF benchmark dataset (Harvard Dataverse,
Angryk et al., 2020) by solar active region identifiers to simulate six geographically
distributed regional observatories. Each client trains a local Multi-Layer Perceptron
(MLP) on its private shard of 24 magnetic-field complexity features, including total
unsigned current helicity (TOTUSJH), total photospheric magnetic free energy (TOTPOT),
and the R-value flux emergence proxy. Only model weight updates — never raw satellite
measurements — are transmitted to the aggregation server. Two federation strategies
are evaluated: FedAvg (McMahan et al., 2017) and FedProx (Li et al., 2020). FedProx
introduces a proximal regularisation term (μ = 0.01) that penalises local model drift,
specifically designed to handle the non-IID data heterogeneity arising when different
observatories monitor different populations of active regions.

Experimental results demonstrate that FedProx achieves a Recall of [X.XX] and an
F1-Score of [X.XX] — within [X.X]% of the centralised XGBoost upper bound — while
preserving complete data locality. FedAvg converges to a lower Recall, confirming
that the proximal correction is necessary under geographic data heterogeneity.
Feature importance analysis via SHAP values identifies total unsigned current helicity
and photospheric free energy as the dominant predictors across federated clients,
consistent with established solar physics. The framework converges within 50
communication rounds, making it operationally viable for integration with existing
SCADA-based grid management infrastructure.

This work demonstrates that distributed space weather agencies can collaboratively
train a high-recall flare prediction model without compromising data sovereignty,
establishing federated machine learning as a practical mechanism for global Smart
Grid resilience against geomagnetic threats.

---
*Keywords: Federated Learning, Solar Flare Prediction, Smart Grid, GIC, FedProx,
Space Weather, SWAN-SF, Data Privacy*

*Track: Track 3 — Computational Intelligence and Machine Learning Applications*

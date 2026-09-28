# SF-9 — Scientific Boundary: Flares, CMEs, and GICs

## The honest physical chain

Space-weather-driven grid risk is a multi-stage causal chain:

```
Magnetogram (24 SHARP-like parameters)
   │
   ▼  [THIS PAPER — federated flare probability]
M/X-class flare probability (24h horizon, SWAN-SF labels)
   │
   ▼  ✗ NOT MODELLED
CME occurrence & association (many flares are "stealth" CME-wise;
   many CMEs are flareless; NOAA distinguishes them explicitly)
   │
   ▼  ✗ NOT MODELLED
CME propagation (1–3 days, direction, speed, ENLIL/CME models)
   │
   ▼  ✗ NOT MODELLED
Solar-wind / IMF Bz coupling at L1 or magnetopause
   │
   ▼  ✗ NOT MODELLED
Geomagnetic storm (Dst/SYM-H, auroral electrojet)
   │
   ▼  ✗ NOT MODELLED
dB/dt at ground observatories
   │
   ▼  ✗ NOT MODELLED
Geoelectric field (Earth conductivity models, e.g. Bolduc 2002)
   │
   ▼  ✗ NOT MODELLED
GIC in transformer neutrals (Boteler 2019 network models)
   │
   ▼  ✗ NOT MODELLED
Transformer thermal stress / grid protection action
```

## Why this matters for the manuscript

GIC risk is driven by **geomagnetic disturbance** (dB/dt, storm
intensity), not by flare occurrence per se. A flare-prediction model is
an **upstream enabler**: it reduces warning-time uncertainty at the
*earliest* link of the chain. The 1989 Quebec event chain ran
flare → CME → storm → GIC; our model addresses only the first step.

## Frozen boundary statement (use verbatim)

> The present system is an upstream flare-probability component. It
> does not model CME association, interplanetary propagation,
> geomagnetic response, geoelectric-field induction, or GIC magnitude.
> Grid-protection integration requires the downstream modules listed
> in the roadmap; this work's contribution is demonstrating that such
> multi-agency warning pipelines can be trained without centralising
> raw observations.

## Downstream roadmap (future modules, each a paper-scale effort)

| Module | Inputs | State of the art | Effort |
|---|---|---|---|
| M1 federated flare probability | SWAN-SF SHARP features | **this work** | done (research prototype) |
| M2 CME association | flare + CDAW/SOHO LASCO catalogues | logistic/ML CME-association models exist | medium |
| M3 CME propagation | CME kinematics | ENLIL, drag-based models | large (physics) |
| M4 geospace response | L1 solar wind | empirical + MHD (e.g., O'Brien-McPherron-type coupling) | medium |
| M5 geoelectric field | dB/dt + MT-derived ground conductivity | Bolduc 2002-type models, AGIC maps | medium |
| M6 GIC + grid stress | geoelectric field + network topology | Boteler 2019 network models | engineering |
| M7 SCADA integration | alert stream | operator latency/workload studies | deployment research |

**Do not** draw figures implying this repo solves M2–M7. The SCADA
integration described in the manuscript is an *architectural sketch*
(claim #5 in `audit/CLAIMS.md`).

## Related terminology corrections

- "GIC-triggering flares" → flares are neither necessary nor sufficient
  for GIC-driving storms; say "flare-CME-progenitor screening".
- The R-value/TOTUSJH SHARP parameters are flare-correlated
  (Schrijver 2007; Bobra 2015); they are *not* GIC predictors.

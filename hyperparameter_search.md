# Hyperparameter Search for Baseline Methods

This document reports the method-specific hyperparameters tuned for each baseline
in our experiments, including their meaning, search range, and the
best-performing configuration reported in the paper.

## Search Protocol

- **Shared settings.** All baselines use the same backbone, optimizer, learning
  rate, training schedule, and batch size as PERO. Selection-based baselines
  (MC-CVaR, OHTM) additionally share the same candidate pool size `B_hat = 64`.
- **Selection criterion.** For each baseline, we perform a grid search over its
  method-specific hyperparameter(s) and select the configuration with the best
  validation-set **accuracy**.
- **Search budget.** <!-- TODO: e.g., "each candidate value is run with 1 seed" or
     "averaged over N seeds" — fill in what was actually done -->
  **[TODO: number of seeds / runs used per candidate value during search]**.
- **Scope.** Each hyperparameter value is searched once and shared across all
  benchmark datasets (i.e., not re-tuned per dataset).
- Random Selection and ERM have no method-specific hyperparameters and are
  therefore not included below.

## MC-CVaR

- **Symbol:** `alpha` (risk level)
- **Meaning:** Defines the tail fraction `(1 - alpha)` of highest-loss samples
  used to estimate the CVaR objective (Eq. mc-cvar in the paper). Larger
  `alpha` focuses optimization on a smaller, higher-risk tail.
- **Search range:** `{0.5, 0.25, 0.125}`, matching the selection ratio
  `r = B / B_hat` searched for PERO's own subset selection (Section 5.4 /
  Fig. 6b: `r in {1/2, 1/4, 1/8}`), so that the two methods are compared
  under the same tail-focus granularity.
- **Best-performing value:** `alpha = 0.5`

## Focal Loss

- **Symbols:** `gamma` (focusing parameter), `alpha` (class-balance weight)
- **Meaning:** `gamma` controls how strongly the loss down-weights well-classified
  (easy) examples; `alpha` balances the contribution of positive/negative or
  minority/majority classes.
- **Search range:**
  - `gamma`: fixed at `2.0` (not searched; adopted directly from the
    original Focal Loss paper's validated best-performing value)
  - `alpha`: `{0.25, 0.5, 1.0}`
- **Best-performing value:** `gamma = 2.0`, `alpha = 1.0`

## TDRO

- **Symbols / config keys:** `rho` (`tdro_rho`), `lambda` (`tdro_lambda`)
- **Meaning:**
  - `rho` (constraint: `0 < rho < 1`) controls the targeted tail-risk
    proportion, i.e., the fraction of the loss distribution the LogSumExp
    surrogate emphasizes.
  - `lambda` (constraint: `lambda > 0`) controls the smoothness of the
    softplus term in the surrogate objective.
- **Search range:**
  - `rho`: `{0.05, 0.1, 0.5}`
  - `lambda`: `{0.05, 0.1, 0.5}`
- **Best-performing value:** `rho = 0.05`, `lambda = 0.05`

## GroupDRO (instance-level adaptation)

- **Symbol:** `tau` (temperature)
- **Meaning:** Controls the sharpness of the softmax weighting applied over
  detached per-sample losses; smaller `tau` concentrates weight more sharply
  on the highest-loss samples, approximating the group-level max-loss
  objective (Eq. gdro in the paper) at the instance level.
- **Search range:** `{0.01, 0.1, 0.5}`
- **Best-performing value:** `tau = 0.01`

## OHTM

- **Symbol:** buffer / pool size
- **Meaning:** Size of the candidate pool from which diverse high-loss samples
  are greedily selected via Gram–Schmidt orthogonalization over
  L2-normalized embeddings.
- **Search range:** `{64, 128, 256}`
- **Best-performing value:** `64` (coincides with the shared candidate pool
  size `B_hat` used by PERO and MC-CVaR)

## Notes

- All best-performing values above are shared across the three main
  benchmarks (USTC-TFC, ISCX-VPN-Service, ISCX-VPN-App); hyperparameters
  were not re-tuned per dataset.
- <!-- TODO: fill in search budget (number of seeds/runs per candidate value) -->

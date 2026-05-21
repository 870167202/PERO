# PERO

Source code and supplementary material for *"PERO: Efficient Robust Post-Training of Foundation Models for Encrypted Traffic Classification"*.

## Overview

PERO is a backbone-agnostic training framework for improving the robustness of encrypted traffic classification foundation models. It introduces a lightweight **pre-evaluation module** that estimates per-sample risk, enabling efficient selection of high-risk training samples without evaluating every candidate using the expensive primary classifier. This design improves tail-risk robustness while reducing computational cost in robust post-training.

## Supplementary Material

Full proofs of the theorems in the main paper are provided in [`Supplementary_Material.pdf`](./Supplementary_Material.pdf).

## Core Components

- `pero_core.py` — Core implementation of the pre-evaluation module (`SurrogateRiskEstimator`) and the PERO training loop.

The training loop follows four steps per iteration:

1. **Pre-Evaluate**: The lightweight surrogate risk estimator scores all candidates.
2. **Select**: The top-*B* highest-risk candidates are selected for training.
3. **Update Classifier**: The primary classifier is updated using the selected subset.
4. **Update Surrogate Estimator**: The surrogate risk estimator is supervised using the classifier's actual per-sample losses.

## Requirements

- Python 3.8+
- PyTorch 1.10+

## Acknowledgments

This code is built upon [ET-BERT](https://github.com/linwhitehat/ET-BERT). We thank the authors for releasing their code and pre-trained models.

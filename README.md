# PERO

Source code and supplementary material for *"PERO: Efficient Robust Post-Training of Foundation Models for Encrypted Traffic Classification"*.

## Overview

PERO is a backbone-agnostic training framework for improving the robustness of encrypted traffic classification foundation models. It introduces a lightweight **pre-evaluation module** that estimates per-sample risk, enabling efficient selection of high-risk training samples without evaluating every candidate using the expensive primary classifier. This design improves tail-risk robustness while reducing computational cost in robust post-training.

## Supplementary Material

Full proofs of the theorems in the main paper are provided in [`Supplementary_Material.pdf`](./Supplementary_Material.pdf).

## Running PERO

Before running PERO, please obtain the `models` directory, including the pre-trained model and vocabulary files, from the official [ET-BERT repository](https://github.com/linwhitehat/ET-BERT/tree/main) and place it in the root directory of this project.

To run PERO on the ISCX-VPN App dataset, use:

```bash
python run_classifier_app.py \
  --pretrained_model_path models/pre-trained_model.bin \
  --vocab_path models/encryptd_vocab.txt \
  --train_path datasets/ISCX-VPN_app_dataset/train_dataset.tsv \
  --dev_path datasets/ISCX-VPN_app_dataset/valid_dataset.tsv \
  --test_path datasets/ISCX-VPN_app_dataset/test_dataset.tsv \
  --method "PERO" \
  --epochs_num 20 \
  --batch_size 32 \
  --embedding word_pos_seg \
  --encoder transformer \
  --mask fully_visible \
  --seq_length 128 \
  --learning_rate 1e-6 \
  --rm_lr 5e-6
```

The example above uses the ISCX-VPN App dataset. To run experiments on the Service or USTC-TFC dataset, replace the `app` field in the script name and dataset paths with `service` or `ustc`, respectively, and use the corresponding dataset directory.

## Requirements

- Python 3.8+
- PyTorch 1.10+

## Acknowledgments

This code is built upon [ET-BERT](https://github.com/linwhitehat/ET-BERT). We thank the authors for releasing their code and pre-trained models.

# EEG-to-Image Decoding: Five Architectures with 5-Fold Cross-Validation and OOF Ensembling

A deep learning framework for decoding visual image representations from EEG signals. The project trains five complementary architectures on the Xu2025 Alljoined dataset, using DINOv2 image embeddings as targets. It combines five-fold cross-validation with out-of-fold (OOF) ensemble optimization to improve image retrieval performance and produces a validated submission package for the Codabench evaluation environment.

## Table of Contents

- [Overview](#overview)
- [Notebook](#notebook)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architectures](#model-architectures)
- [Cross-Validation Strategy](#cross-validation-strategy)
- [Training and Optimization](#training-and-optimization)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)

## Overview

The objective is to learn a mapping from EEG signals to visual image embeddings. Rather than generating images directly, the models predict corresponding image representations, which are compared against a gallery of candidate image embeddings for retrieval.

The framework combines five architectures with different approaches to temporal and spatial EEG feature extraction. Each architecture is trained across five folds, producing up to 25 models. Their predictions are subsequently combined using architecture weights optimized on pooled out-of-fold predictions.

## Notebook

| Notebook | Description |
|---|---|
| `track1_5fold_5arch.ipynb` | Complete pipeline for data preparation, five-architecture training, cross-validation, ensemble optimization and Codabench submission packaging. |

## Dataset

The project uses the **Xu2025 Alljoined** EEG dataset.

EEG epochs are associated with visual stimuli, while DINOv2 image embeddings provide the target visual representations.

### EEG Input

- 32 EEG electrodes.
- 200 time samples per epoch.
- Sampling frequency of 200 Hz.
- Input representation: `(channels, time)`.

### Image Targets

- DINOv2 image embeddings.
- 1,536-dimensional output representation, as configured by the extracted dataset metadata.
- Image identifiers mapped to canonical image and concept identifiers.
- Ten images per concept.

## Methodology

The pipeline consists of the following stages:

1. **Environment preparation:** Configures the Python environment, installs dependencies and retrieves the competition toolkit.
2. **Data preparation:** Loads the Alljoined dataset or reuses an extracted cache. EEG epochs, image embeddings and trial metadata are prepared for training.
3. **Split verification:** Loads the shared three-way split manifest and checks its integrity, subject separation and concept separation.
4. **Five-fold generation:** Divides the training and validation subject and concept pools into five groups.
5. **Model training:** Trains five architectures independently across the folds.
6. **OOF prediction generation:** Collects predictions from each model on its corresponding held-out fold.
7. **Ensemble optimization:** Searches for architecture weights using pooled OOF predictions.
8. **Submission packaging:** Builds and validates the final Codabench submission ZIP.

## Model Architectures

| Architecture | Main Approach |
|---|---|
| REVE | Pretrained EEG representation model with a trainable projection head |
| Bidirectional Conformer | Temporal convolution and attention with forward and reversed EEG sequences |
| AMSPBE | Attentive Modulated Scalp-Pattern Bank Encoder |
| ResCNN | Residual temporal convolutional network |
| TCN-LC | Multi-scale dilated TCN with log-covariance pooling |

### REVE

Uses the pretrained REVE EEG representation model as its backbone, followed by a trainable projection head. The implementation includes token-based and attention-based feature aggregation.

### Bidirectional Conformer

Uses a Conformer architecture to capture temporal relationships in EEG signals. It processes both the original and time-reversed EEG sequence, combining their learned representations.

### AMSPBE

The Attentive Modulated Scalp-Pattern Bank Encoder focuses on spatial and temporal EEG representations through temporal and channel attention, scalp-region grouping, depthwise temporal convolution and a spatial filter bank.

### ResCNN

A residual temporal CNN designed to learn hierarchical EEG features through convolutional stems, depthwise-separable residual blocks, dilated temporal convolutions and an MLP projection head.

### TCN-LC

A multi-scale temporal convolutional network combining temporal features with log-covariance representations. Its temporal and covariance-based representations are concatenated before the final projection.

## Cross-Validation Strategy

The framework uses **5-fold cross-validation** with seed 33.

Rather than randomly splitting individual EEG epochs, folds are constructed by grouping subjects and visual concepts.

For each fold:

- A group of subjects is held out.
- A corresponding group of concepts is held out.
- Training uses the remaining subjects and concepts.
- Validation uses the held-out subject-by-concept block.

The local test partition remains untouched during fold training.

| Configuration | Value |
|---|---:|
| Number of folds | 5 |
| Fold seed | 33 |
| Architectures | 5 |
| Maximum models | 25 |
| Training unit | Subject × concept |
| Local test | Held out from training |

## Training and Optimization

All architectures learn to map EEG epochs into the DINOv2 embedding space.

The main training objective uses temperature-scaled similarity between normalized EEG predictions and image embeddings, optimized through a cross-entropy contrastive objective.

The training pipeline includes:

- AdamW optimizer.
- One-cycle learning-rate scheduling.
- Gradient clipping.
- Mixed-precision training where supported.
- Exponential moving average (EMA) for selected architectures.
- Stochastic weight averaging (SWA).
- Early stopping based on validation performance.

Additional training options include mixup, semantic label smoothing, cosine regression and symmetric contrastive loss, depending on the architecture configuration.

## Evaluation Metrics

The framework evaluates image retrieval performance using rank-based metrics.

### Mean Rank Percentile (MRP)

Measures how highly the correct image ranks among candidate gallery images. Higher values indicate better retrieval performance.

### Top-5 Accuracy

Measures the proportion of queries for which the correct image appears among the five highest-ranked candidates.

The evaluation covers:

- OOF retrieval performance.
- Test retrieval performance.
- Concept-level retrieval using 150 concepts.
- Image-level retrieval using 1,500 images.
- New retrieval using the 200 official test images.

## Results

The five-fold models were evaluated individually using their averaged predictions. Their performance was compared against equal-weight and OOF-optimized ensemble predictions.

### Overall Performance

| Model | OOF MRP | OOF Top-5 /150 Concepts | OOF Top-5 /Images | TEST MRP | TEST Top-5 /150 Concepts | TEST Top-5 /1500 Images | NEW Top-5 /200 Official |
|---|---:|---:|---:|---:|---:|---:|---:|
| REVE | 0.7452 | 0.1719 | 0.0160 | 0.6736 | 0.1166 | 0.0196 | 0.1817 |
| Bidirectional Conformer | 0.7245 | 0.1537 | 0.0134 | 0.6623 | 0.1099 | 0.0149 | 0.2067 |
| AMSPBE | 0.7311 | 0.1578 | 0.0138 | 0.6565 | 0.1098 | 0.0158 | 0.2017 |
| ResCNN | 0.7302 | 0.1604 | 0.0147 | 0.6661 | 0.1098 | 0.0158 | 0.1800 |
| TCN-LC | 0.7303 | 0.1599 | 0.0135 | 0.6646 | 0.1106 | 0.0136 | 0.1700 |
| **Equal-weight Ensemble** | **0.7510** | **0.1802** | **0.0163** | **0.6751** | **0.1190** | **0.0176** | **0.1950** |
| **OOF-optimized Ensemble** | **0.7538** | **0.1836** | **0.0170** | **0.6773** | **0.1217** | **0.0191** | **0.1933** |

### Ensemble Weights

The optimized ensemble assigns different contributions to the five architectures:

| Architecture | Ensemble Weight |
|---|---:|
| REVE | 0.34 |
| Bidirectional Conformer | 0.09 |
| AMSPBE | 0.17 |
| ResCNN | 0.26 |
| TCN-LC | 0.15 |

*Weights are reported as displayed by the notebook.*

### Key Observations

- REVE achieved the strongest individual TEST MRP of **0.6736**.
- The equal-weight ensemble improved TEST MRP to **0.6751**.
- OOF weight optimization further improved TEST MRP to **0.6773**.
- The optimized ensemble achieved an OOF MRP of **0.7538** and concept-level OOF Top-5 accuracy of **0.1836**.
- The optimized ensemble also improved TEST Top-5 accuracy across both the 150-concept and 1,500-image galleries compared with equal weighting.
- The equal-weight ensemble performed better on the NEW official 200-image Top-5 metric (**0.1950 versus 0.1933**), showing that the optimized weights do not improve every evaluation measure.

Overall, the OOF-optimized ensemble provides the strongest TEST MRP and concept-level retrieval performance among the evaluated configurations.



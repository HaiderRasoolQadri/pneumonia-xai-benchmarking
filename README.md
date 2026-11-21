# Pneumonia XAI Benchmarking

Empirical benchmarking of CAM explanations for pneumonia detection: checkpoint-based reproduction, faithfulness, stability, adversarial robustness, external generalization, and lung attention.

## Overview

This repository contains a research project on **explainable AI (XAI) for pneumonia detection from chest X-rays**. The study follows an empirical research approach and includes:

- **Reproduction:** Evaluation of the published pretrained checkpoints of DenseNet121, MobileNetV2, and VGG16 across the original five test folds.
- **Extension:** Comparison of Grad-CAM, Grad-CAM++ (custom approximation), Eigen-CAM, and Score-CAM (Top-32).
- **Explanation quality and robustness:** Insertion/deletion faithfulness, Gaussian noise, cropping, FGSM adversarial attacks, external validation, and Lung Attention Ratio (LAR).

The goal is not only to compare predictive performance but also to investigate how faithful, stable, and anatomically focused the generated explanations are. The study evaluates existing trained models rather than retraining them or proposing a new CNN architecture.

## Research Objectives

1. Verify the baseline classification performance of the three selected CNN architectures using the authors' pretrained checkpoints.
2. Compare four CAM-based explanation methods using quantitative and qualitative evaluation.
3. Evaluate explanation faithfulness and stability under controlled input perturbations.
4. Investigate model and explanation robustness under FGSM adversarial attacks.
5. Assess generalization on an external chest X-ray dataset.
6. Measure attribution within estimated lung regions using segmentation masks.

## Research Questions

- **RQ1:** How do the four CAM variants compare in explanation faithfulness, and are their rankings consistent across datasets?
- **RQ2:** How robust are CAM explanations under Gaussian noise and cropping, and is higher stability associated with weaker faithfulness?
- **RQ3:** To what extent do CAM heatmaps focus on lung regions, and what failure patterns emerge from diffuse or anatomically misplaced explanations?

FGSM experiments extend RQ2 to adversarial perturbations.

## Main Contributions

- Checkpoint-based evaluation across three CNN architectures and five folds (15 checkpoint evaluations).
- Quantitative comparison of four CAM methods using insertion/deletion AUC relative to random patch ordering.
- Explanation stability analysis using SSIM and Spearman rank correlation under Gaussian noise, cropping, and FGSM.
- External classification evaluation on 500 balanced chest X-rays.
- Lung segmentation-based LAR and area-normalized attention enrichment.
- Paired Wilcoxon statistical comparisons, bootstrap confidence intervals, and Holm correction.
- CAM validity audits, Grad-CAM++ reference comparison, and Score-CAM channel-count sensitivity analysis.

## Repository Structure

```text
pneumonia-xai-benchmarking/
├── docs/
│   └── manifest.json
├── experiments/
│   ├── cam_validity_audit/
│   ├── cam_visual_audit/
│   ├── crop_stability/
│   ├── external_cam_stability/
│   ├── external_faithfulness/
│   ├── external_validation/
│   ├── faithfulness_diagnostics/
│   ├── faithfulness_patch_pilot/
│   ├── faithfulness_pilot/
│   ├── fgsm_attack/
│   ├── fgsm_cam_stability/
│   ├── gradcam_pilot/
│   ├── gradcampp_reference_validation/
│   ├── lung_attention_ratio/
│   ├── lung_mask_pilot/
│   ├── multi_image_faithfulness/
│   ├── scorecam_validation/
│   ├── stability_pilot/
│   └── statistical_analysis/
├── notebooks/
│   └── pneumonia-base-paper-reproduction.ipynb
├── reproduction-results/
│   ├── DenseNet121/
│   ├── MobileNetV2/
│   ├── VGG16/
│   └── consolidated metrics and tables
├── .gitignore
├── README.md
└── requirements.txt
```

## Experimental Methodology

| Component | Experimental setting |
|---|---|
| CNN architectures | DenseNet121, MobileNetV2, VGG16 |
| Internal evaluation | Five original test folds per model |
| Extension checkpoints | Fold 1 |
| Image preprocessing | RGB, 224 × 224 pixels, normalized to [0, 1] |
| Classification threshold | 0.5 |
| External classification | 500 images: 250 Normal, 250 Pneumonia |
| External CAM evaluation | 20 images: 10 Normal, 10 Pneumonia |
| Internal faithfulness pilot | 6 images per architecture |
| Internal noise stability | 10 images per architecture |
| FGSM classification | 80 internal images per architecture |
| FGSM CAM stability | 10 internal images per architecture |
| Gaussian noise | σ = 0.01, 0.05 |
| Crop ratios | 80% and 90% of image side length |
| FGSM epsilon | 0, 0.01, 0.03, 0.05 |
| Statistical analysis | Paired Wilcoxon tests with Holm correction |

### Explanation Methods

| Method | Implementation |
|---|---|
| Grad-CAM | Gradient-weighted activation mapping |
| Grad-CAM++ | Custom first-gradient approximation, compared against tf-keras-vis 0.8.7 |
| Eigen-CAM | Uncentered singular-vector projection with orientation correction; class-agnostic |
| Score-CAM Top-32 | Restricted variant using the 32 highest-variance feature channels |

Faithfulness is evaluated using signed true-class logit AUC improvements over random patch ordering. Stability and anatomical attention use the fixed clean predicted class for class-specific CAMs. Eigen-CAM remains class-agnostic. Early pilot results are retained for transparency; corrected implementations and random-controlled patch metrics are the basis of the main findings.

## Results

### 1. Internal checkpoint evaluation

Five-fold mean ± standard deviation. These are evaluation results from the released checkpoints, not independently trained models.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Average Precision | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| DenseNet121 | **0.9544 ± 0.0078** | 0.9781 ± 0.0088 | 0.9590 ± 0.0113 | 0.9684 ± 0.0055 | 0.9887 ± 0.0033 | 0.9956 ± 0.0014 | 0.8872 ± 0.0189 |
| MobileNetV2 | 0.9406 ± 0.0119 | 0.9398 ± 0.0195 | 0.9820 ± 0.0092 | 0.9603 ± 0.0074 | 0.9867 ± 0.0027 | 0.9949 ± 0.0012 | 0.8472 ± 0.0305 |
| VGG16 | 0.9392 ± 0.0055 | 0.9718 ± 0.0034 | 0.9441 ± 0.0062 | 0.9577 ± 0.0039 | 0.9816 ± 0.0016 | 0.9928 ± 0.0008 | 0.8507 ± 0.0129 |

### 2. External validation

Fold-1 checkpoints evaluated on a balanced sample of 500 images from the base paper's released external dataset.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Average Precision | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| DenseNet121 | **0.772** | 0.7036 | 0.940 | 0.8048 | 0.9134 | 0.9230 | 0.5776 |
| MobileNetV2 | 0.638 | 0.5827 | 0.972 | 0.7286 | 0.7597 | 0.7058 | 0.3709 |
| VGG16 | 0.578 | 0.5452 | 0.940 | 0.6902 | 0.7392 | 0.7161 | 0.2262 |

DenseNet121 achieved the highest external accuracy. All models retained high pneumonia recall, but external specificity declined substantially, indicating frequent false-positive predictions.

### 3. Explanation faithfulness

Grad-CAM obtained the highest observed mean deletion and insertion improvement over random ordering for each architecture on the 20-image external sample.

| Model | Grad-CAM deletion improvement | Grad-CAM insertion improvement |
|---|---:|---:|
| DenseNet121 | 1.1615 | 1.1120 |
| MobileNetV2 | 0.3614 | 1.1859 |
| VGG16 | 0.8857 | 0.7654 |

Higher values are better. These are signed-logit AUC differences, not probability AUC values. Eleven of 18 exploratory paired comparisons were significant after Holm correction: six for DenseNet121, five for VGG16, and none for MobileNetV2.

### 4. Explanation stability

Mean external SSIM under Gaussian noise (σ = 0.05):

| CAM method | DenseNet121 | MobileNetV2 | VGG16 |
|---|---:|---:|---:|
| Grad-CAM | 0.6242 | 0.6383 | 0.7401 |
| Grad-CAM++ (approx.) | 0.9490 | 0.8595 | 0.9298 |
| Eigen-CAM | 0.9470 | 0.8781 | 0.9980 |
| Score-CAM Top-32 | 0.9159 | 0.8312 | 0.9593 |

Grad-CAM was less spatially stable than the alternatives in this setting, despite its stronger observed faithfulness. Nearly uniform VGG16 Eigen-CAM maps illustrate why high SSIM alone is insufficient for judging explanation quality.

### 5. FGSM adversarial robustness

| Model | Clean accuracy | Adversarial accuracy (ε = 0.01) | Attack success rate |
|---|---:|---:|---:|
| DenseNet121 | 96.25% | 2.50% | 97.40% |
| MobileNetV2 | 92.50% | 1.25% | 98.65% |
| VGG16 | 91.25% | 2.50% | 97.26% |

Attack success rate is measured among initially correctly classified images. The results demonstrate substantial vulnerability under the tested white-box, untargeted FGSM setting. Some CAM maps remained spatially similar even when predictions flipped.

### 6. Lung attention

Mean lung-attention enrichment on 20 external images:

| CAM method | DenseNet121 | MobileNetV2 | VGG16 |
|---|---:|---:|---:|
| Grad-CAM | 0.9585 | 1.1924 | 1.0286 |
| Grad-CAM++ (approx.) | 1.0682 | 1.3195 | 1.0428 |
| Eigen-CAM | 1.1546 | 1.4391 | 1.0044 |
| Score-CAM Top-32 | 1.1622 | **1.5092** | 1.0209 |

An enrichment of 1 indicates attribution proportional to the estimated lung-mask area. MobileNetV2 Score-CAM Top-32 achieved the highest observed enrichment. The main descriptive trends remained similar after excluding one flagged mask. These results measure anatomical attention, not verified pneumonia lesion localization.

## Main Findings

- High internal classification accuracy does not guarantee external generalization.
- Grad-CAM achieved the strongest observed mean faithfulness on the external sample, while alternative CAM variants generally showed greater spatial stability.
- High explanation stability can be misleading when a heatmap is nearly uniform.
- Small white-box adversarial perturbations substantially reduced classification accuracy in all three CNNs.
- Anatomical attention, faithfulness, and stability capture different properties; no method was established as universally superior.

## Notebook and Reproducibility

The complete implementation is available in [`notebooks/`](notebooks/). All reproduction and extension experiments were performed in a single Kaggle Jupyter notebook. The [`reproduction-results/`](reproduction-results/) and [`experiments/`](experiments/) directories contain saved outputs, including pilot diagnostics, quantitative summaries, and visualizations. The baseline provenance and environment information are recorded in [`docs/manifest.json`](docs/manifest.json).

### Environment

- Python 3.13
- TensorFlow 2.21.0
- pandas 2.3.3
- scikit-learn 1.6.1
- NumPy, SciPy, scikit-image, Matplotlib
- tf-keras-vis 0.8.7
- PyTorch, MONAI, SimpleITK and pydicom (lung segmentation)
- Kaggle GPU environment

See [`requirements.txt`](requirements.txt) for package requirements. Exact compatibility outside the original Kaggle environment has not been independently tested.

### Getting Started

1. Clone this repository:

   ```bash
   git clone https://github.com/HaiderRasoolQadri/pneumonia-xai-benchmarking.git
   cd pneumonia-xai-benchmarking
   ```

2. Open or import the notebook from `notebooks/` into Kaggle and enable a GPU accelerator if available.
3. Attach the required datasets (see below). Install missing dependencies using the notebook setup instructions or `pip install -r requirements.txt` in a compatible environment.
4. Check input paths and run the relevant experiment sections in order. The notebook is the original iterative research workflow; some cells rely on earlier setup or saved artifacts. A fresh one-click Run All has not been verified. Skip dataset creation/versioning and backup cells unless explicitly needed.
5. To inspect the existing results, browse the CSV files and figures directly; retraining or rerunning the experiments is not necessary.

### Datasets and Pretrained Weights

Large datasets and model checkpoints are maintained separately from GitHub:

| Resource | Contents |
|---|---|
| [Pneumonia Reproduction Backup](https://www.kaggle.com/datasets/haiderrasoolqadri/pneumonia-reproduction-backup) | Original data, published CNN checkpoints, fold splits, baseline results |
| [Pneumonia XAI Experiments](https://www.kaggle.com/datasets/haiderrasoolqadri/pneumonia-xai-experiments) | Extension results and lung segmentation checkpoint |
| [Original research artifacts (Zenodo)](https://zenodo.org/records/21620538) | Published research materials |

The Kaggle backups are private and require authorized access. The external evaluation uses a subset of the base paper's released external dataset, not a newly collected cohort. For access to original research assets, consult the authors' distribution and licensing terms.

## Limitations and Future Work

- The original CNNs were not retrained; the study verifies published checkpoints.
- Most XAI extension experiments use Fold 1 and relatively small image samples; broader cross-fold and cross-dataset conclusions require larger evaluations.
- Grad-CAM++ and Score-CAM are approximate or restricted implementations. Reference and channel-sensitivity checks do not establish equivalence to canonical methods.
- Lung masks are automatically generated, not expert-verified lesion annotations. External sample demographics were not independently reverified.
- The failure-case audit is partial; further systematic artifact analysis, larger reference-method experiments, and additional adversarial settings are needed.

This is an exploratory research benchmark, not a clinically validated diagnostic system.


## Citation and Acknowledgments

This research is an independent empirical extension of the following base paper:

> F. Gómez-Vela, A. López-Fernandez, F. Divina, and M. García-Torres, “Performance-interpretability trade-offs and generalization in deep learning for pneumonia detection: A benchmarking study,” *Applied Intelligence*, vol. 56, article 414, 2026. [DOI: 10.1007/s10489-026-07398-5](https://doi.org/10.1007/s10489-026-07398-5).

Credit for the original CNN training, published checkpoints, test splits, datasets and baseline methodology belongs to the original authors. This repository contributes checkpoint-based evaluation, additional CAM comparisons, robustness and faithfulness experiments, lung-attention analysis, and statistical evaluation.

Lung segmentation uses the pretrained model from the [NIAID lung CXR segmentation repository](https://github.com/niaid/lung_cxr_segmentation). Grad-CAM++ implementation checks use [tf-keras-vis](https://github.com/keisen/tf-keras-vis).

If using this work, please cite the original paper and the relevant CAM methods, and acknowledge this repository for its empirical extension.

**Original authors' official GitHub repository:** https://github.com/SynergIA-Lab/pneumoniacnn

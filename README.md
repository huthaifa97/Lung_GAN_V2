Adaptive LungGAN: Hinge-Loss Conditional GAN with a Gap-Only Adaptive Controller for Lung CT Synthesis
Official code for the paper "Adaptive Hinge-Loss GAN with Gap-Aware Dynamic Controller for Medical Lung Image Synthesis" (Al-Khazaleh and Wan Zainon, School of Computer Sciences, Universiti Sains Malaysia).
---
1. Description
This repository contains the two training notebooks used to produce every result in the paper:
Notebook	Role in the paper	Summary
`Lung_GAN_Improved_dynamic_Hinge_v15A_3Seeds.ipynb`	Static LungGAN (v15A), the controlled baseline	Fixed schedule: `N_DIS = 2` discriminator steps per generator step, mode-seeking loss applied on every generator step (λ = 0.06).
`Lung_GAN_Improved_dynamic_Hinge_v15E_3Seeds.ipynb`	Adaptive LungGAN (v15E), the proposed method	Same networks and loss, plus a Gap-Only Adaptive Controller that adjusts the discriminator update frequency (`N_DIS` ∈ {1, 2}) and the probability of applying the stochastic mode-seeking loss.
Both notebooks train a class-conditional GAN on binary lung CT images (COVID-19 vs. Normal) for 200 epochs with three random seeds (42, 123, 777), evaluating with FID, Precision, Recall and LPIPS diversity, and save checkpoints, histories, sample grids, and a summary CSV.
Key idea. Instead of a fixed WGAN-GP gradient penalty (λ = 10), the model uses a static hinge loss (margin 1.0) with spectral normalisation. A lightweight controller monitors the smoothed discrimination gap `E[D(real)] − E[D(fake)]` and spends discriminator compute only when the adversarial balance needs correcting.
Headline results (3 seeds, best-FID checkpoint):
Metric	v15A (static)	v15E (adaptive)
Best FID ↓	36.90 ± 4.84	32.01 ± 0.47
Precision ↑	0.6325 ± 0.0534	0.7175 ± 0.0345
Recall ↑	0.2770 ± 0.0059	0.2877 ± 0.0273
LPIPS ↑	0.3366 ± 0.0067	0.3328 ± 0.0023
Training time (min, mean per seed)	259.1 ± 0.7	176.2 ± 2.1
Controller statistics for v15E: average `N_DIS` = 1.058 ± 0.043, discriminator-step savings of 47.1% ± 2.2%, and mode-seeking savings of 55.1% ± 0.1% (relative to a fixed `N_DIS = 2` schedule and an always-on mode-seeking loss).
---
2. Dataset Information
Dataset: Lung Segmentation Data V2 (COVID-19 vs. Normal lung CT), approximately 14,464 training images.
Classes: 2 (`COVID-19`, `Normal`).
Expected layout: the notebooks read a zip file and look for a `Train` folder in the `torchvision.datasets.ImageFolder` format:
```
Lung Segmentation Data V2.zip
└── .../Train/
    ├── COVID-19/   *.png / *.jpg
    └── Normal/     *.png / *.jpg
```
Source: `<ADD DATASET URL / DOI HERE>`. The dataset is not redistributed in this repository; download it from the original source and respect its licence.
Preprocessing (applied in both notebooks):
Resize to 128 × 128.
CLAHE on the L channel in LAB colour space (`clipLimit = 2.0`, `tileGridSize = 8×8`).
Random horizontal flip (p = 0.5).
Normalisation to [−1, 1].
Batch size 64 with `drop_last=True` gives 226 iterations per epoch.
---
3. Code Information
Each notebook is self-contained (one install cell plus one training cell). Main components, identical across both notebooks unless noted:
Component	Description
`CLAHETransform`	CLAHE contrast enhancement transform.
`ConditionalBatchNorm2d`, `UpBlock`, `ConditionalGenerator`	Generator: `[z (128-d); class embedding (128-d)]` → linear → 512×4×4 → four UpBlocks (nearest upsample → 3×3 conv → conditional BN → ReLU; channels 512→256→128→64) → 3×3 conv + Tanh.
`ProjectionDiscriminator`	Spectrally normalised 4×4 strided-conv discriminator with LeakyReLU(0.2), global average pooling, and a class-projection term: `D(x,y) = fc(h) + ⟨embed(y), h⟩`.
`EMA`	Exponential-moving-average generator (decay 0.999), used only for evaluation and sampling.
`d_hinge_loss`, `g_hinge_loss`	Static hinge loss, margin = 1.0.
`mode_seeking_loss`	Mode-seeking regulariser `1 / (ℓz + ε)` with `ℓz = mean‖G(z₁)−G(z₂)‖₁ / mean‖z₁−z₂‖₁`.
`GapOnlyAdaptiveController`	v15E only. Hysteresis state machine on the EMA-smoothed gap (EMA coefficient 0.98) plus a diversity/volatility monitor that sets the mode-seeking probability.
`compute_metrics`	FID (torchmetrics, 2048-d Inception), k-NN manifold Precision/Recall (k = 3), LPIPS (AlexNet) diversity over 300 pairs; computed on 2,000 samples with the EMA generator. Also run per class.
`save_checkpoint`	Saves G, D, EMA-G, both optimisers, metrics (and controller state in v15E).
Controller logic (v15E)
`GAP_LOW = 0.10`: if smoothed gap < 0.10 → `N_DIS = 2` (train D harder) and reset the calm counter.
`GAP_RELAX = 0.12` with `CALM_PATIENCE = 40`: while in `N_DIS = 2`, after 40 consecutive calm updates (gap > 0.12) → `N_DIS = 1`.
First 3 epochs are a warm-up (`N_DIS = 2`, mode-seeking always on).
Mode-seeking is applied with fixed λ = 0.15. Its probability is `effective_ms / 0.15`, where `effective_ms` moves between 0.06 and 0.15 depending on the diversity score; it is also forced on every 20th step.
Key hyperparameters
Hyperparameter	Value
Image size / batch size	128 × 128 / 64
Latent dim / label-embedding dim	128 / 128
Generator base channels / floor	512 / 64
Discriminator base channels	64
Learning rate (G and D)	2 × 10⁻⁴
Adam betas	(0.0, 0.9)
EMA decay	0.999
Hinge margin	1.0
Epochs	200
Seeds	42, 123, 777
Metrics frequency	epoch 1 and every 25 epochs
v15A: `N_DIS` / `MS_LAMBDA`	2 / 0.06
v15E: mode-seeking λ / `N_DIS` range	0.15 / {1, 2}
---
4. Requirements
The notebooks were developed and run on Google Colab with an NVIDIA L4 GPU and Google Drive mounted.
Python 3.10+
PyTorch and torchvision (CUDA build recommended)
`torchmetrics`, `torch-fidelity` (FID computation)
`lpips`
`numpy`, `pandas`, `opencv-python`, `matplotlib`, `Pillow`, `tqdm`
Install (already included in the first cell of each notebook):
```bash
pip install -q torchmetrics torch-fidelity lpips
pip install -q torch torchvision numpy pandas opencv-python matplotlib pillow tqdm
```
Approximate runtime on one L4 GPU, per seed: ~176 min (v15E) and ~259 min (v15A).
---
5. Usage Instructions
Option A: Google Colab (as used in the paper)
Upload the dataset zip to your Google Drive as `MyDrive/Lung Segmentation Data V2.zip`.
Open a notebook in Colab (`File → Open notebook → GitHub`, or upload the `.ipynb`), and select a GPU runtime (`Runtime → Change runtime type`).
Run the first cell to install dependencies.
Run the second cell. It will:
mount Google Drive,
extract the zip to `/content/lung_data` and locate the `Train` folder,
train all three seeds sequentially for 200 epochs,
print a mean ± std summary and write a CSV.
Run v15A and v15E to reproduce the baseline-vs-proposed comparison (Table 6 in the paper).
Option B: Local machine / other server
The notebooks use Colab-specific code (`google.colab.drive`, `/content/...` paths). To run elsewhere, edit these variables in the config/dataset section:
```python
ZIP_PATH     = "/path/to/Lung Segmentation Data V2.zip"   # or skip extraction and point data_root at your Train folder
EXTRACT_ROOT = "/path/to/lung_data"
CKPT_ROOT    = "/path/to/output_dir"
```
and remove the `from google.colab import drive` / `drive.mount(...)` lines.
Outputs
Written under `CKPT_ROOT/seed_<seed>/`:
File	Content
`lung_gan_best_fid.pth`	Checkpoint at the best FID (used for Tables 2 and 4)
`lung_gan_best_recall.pth`	Checkpoint at the best Recall (used for Table 3)
`lung_gan_epoch_<N>.pth`, `lung_gan_final.pth`	Periodic (every 25 epochs) and final checkpoints
`history_seed_<seed>.pt`	Metric, loss, and (v15E) controller histories, best-checkpoint info, D-step and MS savings
`samples_epoch_<N>.png`	EMA generator sample grids (every 5 epochs)
Summary CSV across seeds: `CKPT_ROOT/v15A_multiseed_results.csv` or `CKPT_ROOT/v15E_multiseed_results.csv`.
Generating images from a trained checkpoint
```python
import torch
ckpt = torch.load("seed_42/lung_gan_best_fid.pth", map_location="cuda")
# Re-create ConditionalGenerator with the same arguments as in the notebook, then:
G_ema = ConditionalGenerator(128, 128, 2, 512, 128).cuda()
G_ema.load_state_dict(ckpt["G_ema"])
G_ema.eval()

z = torch.randn(16, 128, device="cuda")
labels = torch.tensor([0] * 8 + [1] * 8, device="cuda")   # class order: ckpt["classes"]
with torch.no_grad():
    imgs = (G_ema(z, labels) + 1) / 2                      # [0, 1]
```
---
6. Methodology
Data processing: resize → CLAHE (LAB) → random horizontal flip → normalise to [−1, 1].
Model: class-conditional generator with conditional batch norm; spectrally normalised projection discriminator; EMA copy of the generator for evaluation.
Objective: static hinge loss (margin 1.0) in place of WGAN-GP's gradient penalty. Spectral normalisation provides the Lipschitz control.
Stochastic mode-seeking loss (λ = 0.15 in v15E; λ = 0.06 every step in v15A) to counter mode collapse on the imbalanced classes.
Gap-Only Adaptive Controller (v15E): gap `E[D(real)] − E[D(fake)]` → EMA smoothing → hysteresis switching of `N_DIS` between 2 and 1; diversity/volatility monitor modulates mode-seeking probability.
Evaluation: FID, k-NN Precision/Recall, and LPIPS diversity every 25 epochs on 2,000 samples from the EMA generator; per-class metrics; best-FID and best-Recall checkpoints tracked separately.
Reproducibility: each seed (42, 123, 777) resets model weights, data loaders, and controller state; seeds are set for `random`, NumPy, and PyTorch/CUDA. Note that `cudnn.benchmark = True` and `cudnn.deterministic = False` are used for speed, so results may differ slightly across hardware and library versions.
---
7. Citation
If you use this code, please cite:
```bibtex
@article{alkhazaleh2026adaptivelunggan,
  title   = {Adaptive Hinge-Loss GAN with Gap-Aware Dynamic Controller for Medical Lung Image Synthesis},
  author  = {Al-Khazaleh, Huthaifa and Wan Zainon, Wan Mohd Nazmee},
  journal = {PeerJ Computer Science},
  year    = {2026},
  note    = {Code: <ADD REPOSITORY URL / ZENODO DOI>}
}
```
Please also cite the original source of the Lung Segmentation Data V2 dataset: `<ADD DATASET CITATION>`.
---
8. License & Contribution Guidelines
License: `<CHOOSE LICENSE, e.g. MIT>`. See the `LICENSE` file. The dataset is subject to its own licence and is not covered by this repository's licence.
Contributions: issues and pull requests are welcome. For bug reports, please include the notebook name, seed, runtime (GPU type, PyTorch version), and the error message or log excerpt.
9. Contact
Huthaifa Al-Khazaleh and Wan Mohd Nazmee Wan Zainon, School of Computer Sciences, Universiti Sains Malaysia, 11800 USM, Penang, Malaysia. Corresponding author: Wan Mohd Nazmee Wan Zainon (nazmee@usm.my).
10. Acknowledgments
Supported in part by the Research Leadership Facilitation Grant 2026, School of Computer Sciences, Universiti Sains Malaysia.

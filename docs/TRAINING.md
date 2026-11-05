# Full MobileNetV2 training

## Data

The original `D:\Haroon FYP\Dataset.zip`, copied ZIP and raw canonical images are preserved. The authoritative
data is `dataset/processed/`, with train/val/test directories, 12 class labels and the final grouped manifest.
`processed_initial/` is a superseded audit copy and must not be used for training.

`scripts/prepare_dataset.py` validates the source, handles exact/near duplicates and numeric filename parent
families, and prepares 224px RGB copies. It reuses already validated final outputs when the archive is unchanged.
The report explains residual longitudinal-plant grouping limitations.

## Configuration

`training/config.json` requests CUDA, batch size 64, five classifier-head epochs and up to twenty fine-tuning
epochs. Head training freezes the backbone and its batch-normalization statistics. Fine-tuning unfreezes feature
blocks 14 onward at a lower learning rate, while earlier blocks stay frozen.

Training uses inverse-frequency class weights, random crops, flips, rotations and color jitter. Validation/test
use deterministic RGB letterboxing and ImageNet normalization. AdamW, mixed precision, ReduceLROnPlateau and
validation-loss early stopping control optimization. Every executed epoch uses the complete training split.

The test split is excluded from optimization, scheduling, stopping and calibration. After best-checkpoint selection,
validation logits fit a temperature and review threshold, then the untouched test split is evaluated.

## Colab workflow

1. Open `training/complete_training.ipynb` in VS Code and select your existing Colab GPU kernel. Authenticate
   personally if prompted. A local `.venv` kernel is not the GPU runtime.
2. `scripts/upload_colab_inputs.py` prepares source and transfer parts. The project controller and official
   Colab extension handle uploads. If prompted, select the same T4 server as the notebook.
3. Run the complete notebook. It verifies SHA256 values, prepares the exact final split, requires CUDA, and
   runs training, evaluation and export. It does not mount Google Drive.
4. For agent-driven execution, `scripts/start_colab_job.py` queues preparation and a single ordinary training
   subprocess through the selected notebook; a running worker is checked before another is started.
5. `scripts/watch_colab_job.py` monitors the worker's actual status files and retrieves the final archive through
   the supported filesystem provider. Notebook cell completion alone does not establish completed training.

Large transfers use parts with per-part and whole-archive hashes. The executed run reused uploaded image parts
with a compressed final manifest to apply the filename-parent grouping; every reused image was checked against
the final local processed hash. Future final archives already contain those assignments.

## Outputs and recovery

Every epoch atomically saves `last.pt`; validation improvements update `best.pt`. `last.pt` embeds best weights,
optimizer, scheduler, scaler, epoch/phase, patience and RNG state. It checks classes, critical settings and dataset
fingerprint on resume. An interrupted partial epoch is repeated.

Reconnect to the same runtime first after a disconnect. If it is gone, upload source, the prepared dataset and
a locally saved `last.pt` using `scripts/upload_colab_inputs.py --resume models/mobilenet_v2/last.pt`.
To use an intermediate snapshot, copy it into a separate project recovery folder with the filename `last.pt`.
The notebook restores uploaded `/content/last.pt` into its output directory automatically. Keep the same split
and critical settings. Preserve current local outputs separately before importing a different run.
Different library versions, GPUs or worker counts can change exact numerical reproducibility.

`scripts/import_artifacts.py` restores models and evaluation locally, with collision protection. Preserve previous
runs in separate local directories before importing another run. The exact executed source bundle is preserved
as `training/trained_run_source.zip`; runtime versions and configuration accompany the evaluation.
The GPU job never silently falls back to CPU training.

## Method references

- [MobileNetV2 — Sandler et al.](https://arxiv.org/abs/1801.04381)
- [Temperature scaling — Guo et al.](https://proceedings.mlr.press/v70/guo17a.html)
- [Grad-CAM — Selvaraju et al.](https://arxiv.org/abs/1610.02391)
- [Colab extension guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)

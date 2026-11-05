# Measured verification evidence

- `dataset/`: integrity, duplicates, label/quality review candidates, final split and visualizations.
- `mobilenet_v2/`: genuine 826-image test metrics, per-class report, predictions, confusion matrix, training
  curves, calibration, Grad-CAM, runtime provenance and training log.
- `local_model_verification.json`: complete held-out CPU verification of the downloaded model.
- `setup_tests.json` and `.txt`: latest full test-suite result, including explicit skips if present.
- `browser_e2e.json`: actual Microsoft Edge workflows; `app_*.png` are browser captures.
- `colab_monitor.log`: actual epoch progress and successful artifact retrieval.
- `setup_verification.json`: environment, CPU diagnostics, SQLite and dependency verification.

Synthetic infrastructure tests check mechanics and do not measure botanical accuracy.
See [completion report](../FINAL_PROJECT_STATUS.md).

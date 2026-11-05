# Genuine locally saved model

`mobilenet_v2/model.pt` is the completed, ImageNet-initialized MobileNetV2 export for the actual 12 classes.
It contains trained weights, class order, preprocessing, validation-only calibration and split fingerprint.
The local application loads this file directly on CPU. See [model card](../MODEL_CARD.md).

`best.pt` is the validation-selected checkpoint; `last.pt` contains recovery state and the best weights.
`checkpoints/intermediate.pt` is a real checkpoint retrieved while GPU training was running.
`weed_artifacts.zip` preserves the final archive downloaded from temporary Colab storage. Configuration,
classes, history, split manifest and actual CUDA runtime requirements accompany the model.

No Google Drive was used. Preserve previous run directories before importing another archive; collisions
are refused. Resumption requires the same class mapping, split and critical training settings.

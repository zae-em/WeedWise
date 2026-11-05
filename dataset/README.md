# Validated local dataset

The original `D:\Haroon FYP\Dataset.zip` and project copy `Dataset.zip` are preserved. All 11,078 archive
image entries passed CRC validation. The archive repeats the same 5,539-image collection twice; redundant
copies were verified by SHA256 before exclusion. Canonical originals remain in `raw/`.

`processed/` contains 5,536 valid 224 x 224 RGB images after excluding three exact duplicates. Its manifest
records source paths, classes, hashes, duplicate groups and the authoritative split: 3,883 training, 827
validation and 826 test images. All 12 labels are retained (nine weeds, three crops). Detected near duplicates
and shared filename source-frame families stay in one split. Original longitudinal plant IDs are unavailable,
so independence between all related observations cannot be established.

`processed_initial/` is an audit copy superseded before training. `weed_training_224.zip` is the final training
archive. `colab_parts/` preserves transferred parts and the final split delta; every reused image was verified.
See [dataset analysis](../evaluation/dataset/DATASET_REPORT.md), [preparation](../scripts/prepare_dataset.py)
and [limitations](../docs/LIMITATIONS.md). Advisory quality flags are not verified label errors.

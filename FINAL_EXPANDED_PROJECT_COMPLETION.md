# Expanded project completion record

**V2 is trained, verified locally and active in Streamlit as of 5 November 2026.**

| Deliverable | Verified result |
| --- | --- |
| Source dataset | All 11,646,405,955 bytes of the pinned CropAndWeed export verified. FiftyOne imported 8,034 source images with matching masks. The annotation audit checked 111,953 bounding boxes across 929 recording sessions. |
| Combined data | 48 classes, comprising 40 weeds and 8 crops. There are 21,967 training, 5,822 validation and 5,322 test images after crop extraction, filtering and splitting. The original splits were preserved. |
| Training | Started from the existing V1 MobileNetV2 weights and completed 41 epochs on a Colab Tesla T4 in 119 minutes. Validation selected epoch 34. Future runs default to 20–25 epochs. |
| Combined test | 69.67% top-1, 93.91% top-5, 0.619 macro F1 and 86.19% crop-versus-weed accuracy. Local inference reproduced the Colab results across all 5,322 images. |
| Original test | V2 reached 95.28% top-1 and 99.64% top-5 on the same 826 images. V1 scored 93.10% and 99.76%, respectively. |
| New classes | 69.42% top-1 and 91.23% top-5 across 3,077 held-out images. |
| Local artifacts | `models/v2/model.pt`, `best.pt`, `last.pt`, eight immutable recovery snapshots and the SHA-256-verified export. The V1 rollback copy and pre-activation database backup are in `models/v1/`. |
| Application and tests | `models/active.json` selects V2. The release check built the matching species records. All 62 Python tests and 24 browser workflows passed with V2 active. |

Training used prepared classification crops from the full verified download together with the V1 training images. Validation and test images were held out from weight updates. Accuracy on independent Pakistani field images and unknown species remains unmeasured.

- [Test metrics](evaluation/v2/test_metrics.json)
- [Local inference verification](evaluation/v2/local_verification.json)
- [Artifact transfer receipt](evaluation/v2/import_integrity.json)
- [Python test log](evaluation/v2/local_unittest.log)
- [Browser test results](evaluation/ui/browser_e2e.json)

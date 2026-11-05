# Coverage, uncertainty and field-use limits

- **Coverage:** 12 actual classes: nine weed and three crop labels. The proposal's intended 150–250 weed species
  plus five crops are not available in this downloaded dataset.
- **Domain:** cropped seedlings in a restricted acquisition setting. Adult plants, overlapping species, varied
  fields and Pakistani cropping systems are not independently validated.
- **Grouping:** exact copies, conservative near-duplicates and shared filename crop families stay in one split.
  Longitudinal plant and photographer IDs are unavailable; related growth-stage observations can remain.
  Internal accuracy must not be presented as independent field accuracy.
- **Image quality:** valid images remain even when advisory checks flag blur, limited green tissue or poor
  resolution. `evaluation/dataset/image_quality_review.csv` lists 1,044 heuristic flags, not proven label errors.
  Rulers, labels and repeated backgrounds can encourage shortcut learning.
- **Unknown plants:** no independent labeled unknown-species set is available. Validation-only calibration and
  confidence rejection provide a review gate. High-confidence mistakes on unsupported species remain possible.
  No unknown-species accuracy, sensitivity or specificity is invented.
- **Explanation:** Grad-CAM shows class-score influence, not botanical correctness or causal explanation.
- **Knowledge:** sources describe their stated regions and production systems; Pakistani suitability is not
  assumed. Tier B records leave unsupported management details unavailable. Expert review needs are listed in
  `knowledge_base/BOTANICAL_REVIEW.json`.
- **Chemical control:** source-verified herbicide groups and resistance notes only. No brands, doses, mixtures,
  local product approvals or species-specific chemical efficacy are inferred.
- **Chat:** extractive local retrieval supports auditability but limits coverage and conversational flexibility.
  Unverified drafts are excluded from answers.
- **Crop/weed role:** follows dataset categories and does not authorize removal or treatment.
- **Hosting:** local server only, with no public multi-user authentication. Camera capture depends on the device,
  browser and user permission; a headless test cannot establish physical-camera operation.
- **Colab:** allocation and runtime lifetime remain under Google's control. Resume depends on checkpoints
  actually preserved on the laptop.

These limits permit a functional research demonstration but define what has not been validated for field use.

# Using Fieldwise

Start the server from the project folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

Open http://127.0.0.1:8501. The trained model is already local; Colab is not required for identification.

The **Dashboard** shows actual supported class counts, evaluation accuracy, saved/feedback counts, recent
activity and system status. Its primary actions and six shortcuts open the corresponding pages. Navigation
is retained in the page URL; use the sidebar to move between nine sections. On mobile, expand the sidebar
with the arrow at the upper left. Predictions and conversations persist while navigating in a session.

## Identify a seedling

Choose **Upload a photo**, **Use camera**, or **Try a test image**. Photograph one plant clearly in natural
light. JPEG, PNG and WebP uploads up to 12 MB are supported. Camera access depends on the browser, device
and permission. Remove/replace the selected image as needed. Optionally select your field crop; this records
context without changing model rankings. Rice, cotton and sugarcane selections do not add trained model classes.
Press **Identify Plant** to run the genuine model on this laptop.

The result shows the best suggestion, scientific name, five ranked suggestions, confidence and an aggregate
crop/weed score over supported classes. Quality checks and low confidence can flag review. Unsupported
plants can still receive high confidence. **Generate explanation** overlays real Grad-CAM score influence;
it does not establish botanical correctness. Follow the result actions to the profile, management page or chat;
the actual suggested species is transferred as context.

**Save identification** stores the result in SQLite. Retaining the photograph is optional and off by default.
Feedback also saves the identification if needed, using the same retention choice. Unretained photos remain
only in the current session; retained JPEG copies live in `app/uploads/`. Saving twice does not duplicate a
record. **Remove image** clears the input and prediction without deleting any previously saved observation.

## Inspect sources and ask questions

**Species explorer** searches supported common/scientific names and filters role, family and plant type. Tabs show profiles,
management and references. **Weed management** presents source-supported cultural, mechanical, integrated
and group/resistance information. Unsupported sections say so. Sources have regional context.

In **AI chat assistant**, name a supported species or choose plant context. Identification and management
actions pass the species automatically. Suggested questions execute a real query; **Clear chat** clears the
conversation. Species comparisons retrieve cited evidence for each named supported plant. Examples:

- `How can I manage common chickweed without herbicides?`
- `What family is Zea mays in?`
- `What is integrated weed management?`

The local extractive assistant returns existing cited knowledge-base claims. Unsupported species/topics and
pesticide brands, doses and mixtures are declined. No external LLM account or API is needed.

## Review and understand performance

**Prediction history** supports text/ID search, species filtering, UTC date bounds, flagged-only filtering,
details and filtered CSV export. Views show up to the latest 2,000 matching records. Deleting one record or
clearing all history opens a confirmation dialog; type DELETE to enable deletion. Cancel preserves records.
Associated feedback and review notes are deleted together. Shared photographs remain while another record
references them, and deletion never targets source dataset images.

**Feedback & review** separates pending, reviewed, low-confidence and flagged observations. Save a review
decision or feedback with a corrected supported class and comments. Review status is a user's decision,
not an independently verified botanical label. Sourced correction
proposals remain drafts awaiting expert verification and do not alter chatbot evidence.

**Model performance** displays genuine held-out metrics, per-class results, an interactive count/normalized
confusion matrix, interactive learning curves, confidence distribution, calibration, latency, Grad-CAM and
dataset counts. Hover Plotly charts for details. Internal test accuracy does not
measure Pakistani field or unknown-species performance. See [limitations](LIMITATIONS.md).

Back up the model, knowledge base, saved images and database together; see [deployment](DEPLOYMENT.md).

# Fieldwise application

Run from the repository root: `.\.venv\Scripts\python.exe -m streamlit run app/main.py`.
Open http://127.0.0.1:8501. The genuine trained model is reused unchanged on CPU.

`main.py` owns routing, sidebar, layout and error handling. `ui.py` provides cached services, navigation,
source panels and shared record components. `views/` implements Dashboard, Identify plant, Species explorer,
Weed management, AI chat assistant, Prediction history, Feedback & review, Model performance and About.
`style.css` supplies the responsive green/neutral design system using native Streamlit controls and system fonts.

Data logic remains in `weed_assistant/`. Session predictions are separate from explicit SQLite saving.
Retaining a photograph is optional; history deletion confirms its scope and preserves shared photos until
unreferenced. No trained model, dataset or original evaluation artifacts are rewritten by the UI.

See [usage](../docs/USAGE.md) and the [V2 completion report](../FINAL_EXPANDED_PROJECT_COMPLETION.md).

import streamlit as st
from app.ui import *


def render():
    heading("ABOUT THE PROJECT","Built for careful, local plant review.","AI Weed Identification & Management Assistant brings identification, botanical evidence and human review into one workspace.")
    a,b=st.columns([1.5,1])
    with a,st.container(border=True):
        st.subheader("From observation to informed action")
        st.write("Seedlings can look alike, and an incorrect identification can lead to ineffective management or crop damage. This project helps students, researchers and agricultural advisers compare model suggestions with sourced botanical information.")
        st.markdown("**Our objectives**\n\n- Identify supported seedlings from photographs.\n- Make confidence and model limitations visible.\n- Link management guidance to its sources.\n- Preserve observations and human review locally.")
    with b,st.container(border=True):
        st.subheader("The implemented system")
        st.markdown("**MobileNetV2** · ImageNet transfer learning\n\n**Streamlit + Plotly** · interactive interface\n\n**PyTorch** · local CPU inference\n\n**SQLite** · history and review\n\n**TF-IDF retrieval** · grounded local chat")
    data,method,scope=st.tabs(["Dataset & coverage","Training & evaluation","Scope & safety"])
    with data:
        if ACTIVE_VERSION=="v1":
            st.write("The provided archive repeated a 5,539-image seedling collection. After verifying redundant copies and excluding three exact duplicates, 5,536 valid processed images remained. All originals were preserved. The structure matches the Aarhus Plant Seedlings collection.")
        else:
            st.write("The original seedling collection is combined with annotated individual plant crops from CropAndWeed. Growth stages are merged, ambiguous labels are excluded, and recording-session groups stay within one split. The original held-out split is preserved.")
            st.caption("CropAndWeed comes from Austrian agricultural fields and has a custom non-commercial research licence. Pakistani field performance is unverified.")
        references(["dataset"])
        st.dataframe(pd.DataFrame([{"Class":r["class_label"],"Scientific name":r["scientific_name"],"Role":r["role"],"Family":r["family"]} for r in knowledge().records]),hide_index=True,width="stretch")
        records=knowledge().records
        st.info(f"Actual coverage: {len(records)} species, including {sum(r['role']=='weed' for r in records)} weed and {sum(r['role']=='crop' for r in records)} crop classes.")
    with method:
        metrics=artifact(EVALUATION_PATH/"test_metrics.json") or {}
        history=artifact(MODEL_PATH.parent/"history.json") or []
        st.write(f"Model {ACTIVE_VERSION} completed {len(history)} epochs with validation-based selection. Its measured evaluation uses {metrics.get('test_samples','unavailable')} held-out images.")
        st.write("Temperature and the confidence review threshold were fitted on validation data only. The saved evaluation report contains measured accuracy, class results and limitations.")
        nav_button("Inspect measured results","Model performance",key="about_performance")
    with scope:
        st.warning("Research and decision support only. Confirm identity and field treatment with a qualified local expert.")
        st.write("Independent Pakistani field performance and unknown-species detection are unverified. High confidence is not proof of identity. The crop/weed label reflects the dataset, not whether a plant should be removed from your field.")
        st.write("Knowledge records cite their sources and regional contexts. Pakistani suitability, product registration and botanical expert verification are not inferred. No pesticide brands, doses or mixtures are provided. Physical camera support depends on the device and browser permission.")
    st.subheader("Your storage, your choice")
    st.write("Inference runs on this laptop. Predictions are saved only when you choose; retaining a photo is optional. SQLite stores history, feedback and review. Chat stays in the browser session. Colab was used for temporary GPU training; no Google Drive or paid LLM is required.")
    st.code(f"Model: {MODEL_PATH.relative_to(ROOT).as_posix()}\nDatabase: app/data/weed_assistant.sqlite3\nRetained photos: app/uploads/\nEvaluation: {EVALUATION_PATH.relative_to(ROOT).as_posix()}/",language=None)

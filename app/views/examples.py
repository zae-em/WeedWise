"""Run real inference on labelled test photos without saving an observation."""

import streamlit as st

from app.samples import heldout_samples
from app.ui import badge, database, knowledge
from weed_assistant.service import IdentificationService


def open_example(label):
    st.session_state.pop("examples_role", None)
    st.session_state.update(examples_species=label, examples_photo=0, species_tab="Examples")


def _reset_photo():
    st.session_state["examples_photo"] = 0


def _show_result(result, expected, records, open_profile):
    predictions = result["predictions"]
    best = predictions[0]
    predicted = records[best["class_label"]]
    rank = next((rank for rank, item in enumerate(predictions, 1) if item["class_label"] == expected), None)
    expected_name = records[expected]["common_name"]

    st.markdown(badge("Needs review" if result["needs_review"] else "Suggested match",
                      "amber" if result["needs_review"] else "green") + badge(predicted["role"].title(), "neutral"),
                unsafe_allow_html=True)
    st.subheader(predicted["common_name"])
    st.markdown(f'*{predicted["scientific_name"]}*')
    confidence, timing = st.columns(2)
    confidence.metric("Top confidence", f"{best['confidence']:.1%}")
    timing.metric("Inference", f"{result['inference_ms']:.0f} ms")

    if rank == 1:
        st.success(f"Top prediction matches the dataset label: {expected_name}.", icon=":material/check_circle:")
    elif rank:
        st.info(f"The dataset label, {expected_name}, appears at rank {rank}.", icon=":material/info:")
    else:
        st.warning(f"The dataset label, {expected_name}, is outside the five closest matches.")

    st.markdown("**Five closest matches**")
    for rank, item in enumerate(predictions, 1):
        name = records[item["class_label"]]["common_name"]
        st.progress(item["confidence"], text=f"{rank}. {name} · {item['confidence']:.1%}")
    st.caption(f"Crop score {result['crop_probability']:.1%} · Weed score {result['weed_probability']:.1%}")
    st.button("Open predicted species profile", icon=":material/eco:", key="demo_profile",
              on_click=open_profile, args=(best["class_label"],), width="stretch")


def render(classifier, records, open_profile):
    st.subheader("Try the model on a labelled photo")
    st.write("Choose a held-out photo and compare the model’s five suggestions with the dataset label.")
    samples = heldout_samples(classifier)
    if not samples:
        st.info("Local test images are unavailable for this model.")
        return

    by_label = {record["class_label"]: record for record in records}
    left, right = st.columns([1, 1.08], gap="large")
    with left, st.container(border=True, key="demo_photo"):
        st.markdown("**1. Pick a test photo**")
        role = st.segmented_control("Example plant role", ["All", "Weeds", "Crops"], key="examples_role",
                                    default="All", required=True, on_change=_reset_photo, width="stretch")
        labels = [label for label, record in by_label.items()
                  if label in samples and role in ("All", record["role"].title() + "s")]
        if not labels:
            st.info("No test photos are available for this plant role.")
            return
        if st.session_state.get("examples_species") not in labels:
            st.session_state["examples_species"] = labels[0]
        label = st.selectbox("Example species", labels, format_func=lambda label: by_label[label]["common_name"],
                             key="examples_species", on_change=_reset_photo)
        photos = samples[label]
        if st.session_state.get("examples_photo") not in range(len(photos)):
            _reset_photo()
        number = st.selectbox("Test photo", range(len(photos)), format_func=lambda i: f"Photo {i + 1} of {len(photos)}",
                              key="examples_photo")
        path, origin = photos[number]
        data = path.read_bytes()
        result_key = (classifier.model_hash, label, path)
        if st.button("Run example", type="primary", icon=":material/play_arrow:", width="stretch"):
            with st.spinner("Running the trained model on this laptop…"):
                identification = IdentificationService(classifier, database(), knowledge()).identify(data, save=False)
                st.session_state["demo_result"] = (result_key, identification["result"])
        source = "CropAndWeed" if origin == "cropandweed" else "Original seedling dataset"
        st.image(data, caption=f"Dataset label: {by_label[label]['common_name']} · {source}", width="stretch")
        st.caption("The demo runs locally. Example predictions stay out of your observation history.")

    with right, st.container(border=True, key="demo_result"):
        st.markdown("**2. Compare the prediction**")
        saved = st.session_state.get("demo_result")
        if saved and saved[0] == result_key:
            _show_result(saved[1], label, by_label, open_profile)
        else:
            st.markdown('''<div class="demo-empty"><span aria-hidden="true">◎</span>
                        <h3>What does the model see?</h3><p>Run the selected photo to see the ranked
                        matches, confidence scores and how they compare with the dataset label.</p></div>''',
                        unsafe_allow_html=True)

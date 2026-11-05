import hashlib
import html
import pandas as pd
import plotly.express as px
import streamlit as st
from app.ui import *
from app.samples import heldout_samples
from weed_assistant.quality import decode_image,assess_quality
from weed_assistant.service import IdentificationService


def reset_input():
    st.session_state["input_generation"]=st.session_state.get("input_generation",0)+1
    for key in ["input_data","input_hash","input_image","input_quality","identification","explanation"]:st.session_state.pop(key,None)


def input_changed(key):
    if st.session_state.get(key) is None:reset_input()


def render_feedback(saved,prefix="prediction"):
    st.caption("Submitting feedback also saves this identification if needed, using your photograph-retention choice.")
    with st.form(prefix+"_feedback"):
        verdict=st.radio("Does the suggestion look right?",["Looks correct","Incorrect","Unsure"],horizontal=True)
        corrected=st.selectbox("Suggested correction",["No correction"]+list(knowledge().by_class)+["Other / unsupported"])
        note=st.text_area("Your observations",max_chars=4000,placeholder="Describe what you noticed about the leaves or growing conditions.")
        if st.form_submit_button("Submit feedback",type="primary"):
            service=IdentificationService(model(),database(),knowledge())
            service.save(saved,retain_image=st.session_state.get("retain_photo",False))
            database().add_feedback(saved["id"],verdict,None if corrected=="No correction" else corrected,note)
            st.success("Feedback saved locally. The model has not been retrained.")


def render():
    heading("AI PLANT IDENTIFICATION","A closer look at what’s growing.","Start with one clear seedling photo. Compare the suggestions before deciding what to do next.")
    classifier=model();left,right=st.columns([1,1.17],gap="large")
    with left:
        with st.container(border=True):
            st.subheader("1. Add your photograph")
            source=st.segmented_control("Image source",["Upload a photo","Use camera","Try a test image"],default="Upload a photo",key="image_source") or "Upload a photo"
            incoming=None;generation=st.session_state.get("input_generation",0)
            if source=="Upload a photo":
                upload_key="photo_"+str(generation)
                upload=st.file_uploader("Choose a plant photograph",type=["jpg","jpeg","png","webp"],key=upload_key,on_change=input_changed,args=(upload_key,),help="Maximum 12 MB. One plant, clear leaves, natural light.")
                if upload:incoming=upload.getvalue()
            elif source=="Use camera":
                st.caption("Your browser will ask for camera permission. You can use a file if a camera is unavailable.")
                camera_key="camera_"+str(generation)
                capture=st.camera_input("Take a photograph",key=camera_key,on_change=input_changed,args=(camera_key,))
                if capture:incoming=capture.getvalue()
            else:
                samples=heldout_samples(classifier)
                if samples:
                    label=st.selectbox("Example class",list(samples),format_func=lambda label:knowledge().get(label)["common_name"])
                    if st.button("Use this held-out image",icon=":material/image:"):
                        incoming=samples[label][0][0].read_bytes()
                    st.caption("A labelled photo from this model’s held-out test split.")
                else:st.info("Local test images are unavailable for this model.")
            if incoming is not None:
                digest=hashlib.sha256(incoming).hexdigest()
                if digest!=st.session_state.get("input_hash"):
                    for key in ["identification","explanation","input_image","input_data"]:st.session_state.pop(key,None)
                    try:
                        image=decode_image(incoming)
                        st.session_state.update(input_data=incoming,input_hash=digest,input_image=image,input_quality=assess_quality(image))
                    except ValueError as error:
                        st.error(str(error))
                        st.button("Choose another image",on_click=reset_input,icon=":material/restart_alt:")
            if "input_image" in st.session_state:
                image=st.session_state["input_image"];quality=st.session_state["input_quality"]
                st.image(image,width="stretch")
                st.caption(f"{image.width:,} × {image.height:,} pixels · Processed locally")
                st.button("Remove image",icon=":material/close:",on_click=reset_input,width="stretch")
                for warning in quality["warnings"]:st.warning(warning,icon=":material/photo_camera:")
            else:
                st.markdown('<div class="image-empty"><span class="material-symbol">↟</span><h3>A clear photo makes a difference</h3><p>Photograph one plant in natural light, with its leaves in focus.</p></div>',unsafe_allow_html=True)
            st.divider();st.subheader("2. Add field context")
            crop=st.selectbox("Crop in your field (optional)",["Not Sure","Wheat","Rice","Cotton","Maize","Sugarcane"],key="crop_context",help="Context only. No crop-specific reranking method is validated for this model.")
            st.caption("Crop context is recorded with your observation. It does not change the model’s ranking.")
            if st.button("Identify Plant",type="primary",icon=":material/center_focus_strong:",width="stretch",disabled="input_data" not in st.session_state):
                with st.spinner("Examining your photograph on this laptop…"):
                    saved=IdentificationService(classifier,database(),knowledge()).identify(st.session_state["input_data"],save=False,crop_context=crop)
                    st.session_state["identification"]=saved;st.session_state.pop("explanation",None)
    with right:
        saved=st.session_state.get("identification")
        if not saved:
            with st.container(key="result_empty"):
                st.markdown('<div class="eyebrow">YOUR IDENTIFICATION</div><h2>A suggestion.<br>The evidence.<br>Your next step.</h2><p>Your five closest matches, calibrated confidence and botanical information will appear here.</p>',unsafe_allow_html=True)
                a,b=st.columns(2);a.metric("Supported classes",str(len(classifier.classes)));b.metric("Analysis","On-device")
            with st.container(border=True):
                st.markdown("**A careful approach to identification**")
                st.write("1. Photograph a single seedling.\n2. Compare the top candidates and leaf characteristics.\n3. Save your observation or ask for a review.")
                st.caption(f"This model supports {len(model().classes)} dataset classes. An unknown plant can still receive high confidence.")
        else:
            result=saved["result"];best=result["predictions"][0];record=knowledge().get(best["class_label"])
            with st.container(border=True,key="prediction_card"):
                st.markdown(badge("Needs review" if result["needs_review"] else "Suggested match","amber" if result["needs_review"] else "green")+" "+badge(record["role"].title(),"neutral"),unsafe_allow_html=True)
                st.header(record["common_name"]);st.markdown("*"+record["scientific_name"]+"*")
                a,b,c=st.columns(3);a.metric("Top confidence",f"{best['confidence']:.1%}");b.metric("Plant family",record["family"]);c.metric("Inference",f"{result['inference_ms']:.0f} ms")
                if result["needs_review"]:st.warning("This suggestion needs review. The species may be unsupported or the photograph may lack detail. Try a clearer photo and compare leaf characteristics.")
                st.caption(f"Review threshold: {classifier.threshold:.1%}, fitted on known validation images. It is not a validated unknown-species detector.")
                for claim in record["claims"]:
                    if claim["category"]=="description":st.write(claim["text"]);break
                st.caption("Crop context: "+result.get("crop_context","Not Sure")+" · No reranking applied")
                st.checkbox("Retain photograph with saved record",key="retain_photo",value=False,help="Optional. Otherwise only prediction details are stored; the photo stays in this session.")
                already_saved=bool(saved.get("id") and database().get_prediction(saved["id"]))
                if already_saved:st.caption("Record ID: "+saved["id"])
                if st.button("Identification saved" if already_saved else "Save identification",type="primary",icon=":material/bookmark:",disabled=already_saved,width="stretch"):
                    IdentificationService(classifier,database(),knowledge()).save(saved,retain_image=st.session_state.get("retain_photo",False))
                    st.success("Identification saved locally.");st.rerun()
            st.subheader("Your five closest matches")
            table=pd.DataFrame([{"Species":knowledge().get(p["class_label"])["common_name"],"Confidence":p["confidence"]} for p in result["predictions"]])
            fig=px.bar(table,x="Confidence",y="Species",orientation="h",text=table.Confidence.map(lambda x:f"{x:.1%}"),color_discrete_sequence=[GREEN])
            fig.update_layout(height=270,yaxis=dict(autorange="reversed"),showlegend=False);fig.update_xaxes(range=[0,1],tickformat=".0%")
            chart(fig,"prediction_chart")
            with st.expander("Compare all ranked candidates",expanded=False):
                for rank,prediction in enumerate(result["predictions"],1):
                    candidate=knowledge().get(prediction["class_label"])
                    st.markdown(f"**{rank:02d} · {candidate['common_name']}** · {prediction['confidence']:.1%}")
                    st.progress(float(prediction["confidence"]))
                    nav_button("View "+candidate["common_name"],"Species explorer",candidate["class_label"],key="candidate_"+str(rank))
            st.caption(f"Crop score {result['crop_probability']:.1%} · Weed score {result['weed_probability']:.1%}. Aggregated over supported classes only.")
            a,b=st.columns(2)
            with a:nav_button("Species details","Species explorer",record["class_label"],key="result_details")
            with b:nav_button("Ask about this plant","AI chat assistant",record["class_label"],key="result_chat")
            if record["role"]=="weed":nav_button("Open weed management","Weed management",record["class_label"],key="result_management")
            with st.expander("See where the model focused (Grad-CAM)"):
                if st.button("Generate explanation",icon=":material/blur_on:"):
                    with st.spinner("Computing the class activation map…"):st.session_state["explanation"]=classifier.explain(saved["image"])
                if "explanation" in st.session_state:
                    overlay,label=st.session_state["explanation"];st.image(overlay,caption="Score influence for "+label,width="stretch")
                    st.caption("A heatmap explains score sensitivity; it does not verify plant identity.")
            with st.expander("Help review this identification"):render_feedback(saved)

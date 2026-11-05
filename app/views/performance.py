import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from app.ui import *


@st.cache_data(show_spinner=False)
def test_probabilities(path,version):
    with np.load(path) as saved:return saved["probabilities"],saved["labels"]


def render():
    heading("MODEL PERFORMANCE","Measured performance. Visible limits.",f"Explore the genuine evaluation artifacts for model {ACTIVE_VERSION}.")
    metrics=artifact(EVALUATION_PATH/"test_metrics.json");history=artifact(MODEL_PATH.parent/"history.json")
    if not metrics:st.warning("Saved evaluation artifacts are unavailable. No substitute metrics are shown.");return
    a,b,c,d=st.columns(4)
    a.metric("Held-out top-1",f"{metrics['top1']:.2%}");b.metric("Held-out top-5",f"{metrics['top5']:.2%}");c.metric("Macro F1",f"{metrics['macro_f1']:.4f}");d.metric("Test images",metrics["test_samples"])
    a,b,c,d=st.columns(4)
    provenance=artifact(EVALUATION_PATH/"provenance.json") or artifact(EVALUATION_PATH/"run_provenance.json") or {}
    a.metric("Architecture","MobileNetV2");b.metric("Training epochs",len(history) if history else "Unavailable");c.metric("Model classes",len(model().classes));d.metric("Training device",provenance.get("gpu","Tesla T4" if ACTIVE_VERSION=="v1" else "Unavailable"))
    overview,classes,learning,confidence=st.tabs(["Overview","Class performance","Learning curves","Confidence & latency"])
    with overview:
        a,b,c=st.columns(3);a.metric("Macro precision",f"{metrics['precision_macro']:.4f}");b.metric("Macro recall",f"{metrics['recall_macro']:.4f}");c.metric("Weighted F1",f"{metrics['weighted_f1']:.4f}")
        report=artifact(EVALUATION_PATH/"classification_report.json")
        worst=min(model().classes,key=lambda label:report[label]["f1-score"])
        st.info(f"{worst} is the weakest class (F1 {report[worst]['f1-score']:.4f}). Inspect the confusion matrix before interpreting the overall accuracy.")
        st.subheader("Data used in this run")
        summary=artifact(DATASET_SUMMARY_PATH)
        if summary:
            frame=pd.DataFrame([{"Class":label,**counts} for label,counts in summary["per_class"].items()])
            chart(px.bar(frame,x="Class",y=["train","val","test"],barmode="stack",labels={"value":"Images","variable":"Split"}),"data_distribution")
            st.caption("Saved split counts for this model. Detected duplicate and source-frame groups stay within a split.")
        for group,values in metrics.get("groups",{}).items():
            st.write(f"{group.replace('_',' ').title()}: {values['samples']} images; top-1 {values['top1']:.2%}" if values.get('top1') is not None else f"{group}: no test samples")
        if "original_heldout_top1_delta_from_v1" in metrics:
            st.caption(f"Original held-out top-1 change from v1: {metrics['original_heldout_top1_delta_from_v1']:+.2%} points.")
        st.download_button("Download measured evaluation report",(EVALUATION_PATH/"EVALUATION_REPORT.md").read_text(encoding="utf-8"),"EVALUATION_REPORT.md","text/markdown",icon=":material/download:")
    with classes:
        report=artifact(EVALUATION_PATH/"classification_report.json");labels=model().classes
        frame=pd.DataFrame([{"Species":label,"Precision":report[label]["precision"],"Recall / per-class accuracy":report[label]["recall"],"F1":report[label]["f1-score"],"Test images":int(report[label]["support"])} for label in labels])
        metric=st.selectbox("Compare class metric",["F1","Precision","Recall / per-class accuracy"])
        fig=px.bar(frame.sort_values(metric),x=metric,y="Species",orientation="h",color_discrete_sequence=[GREEN]);fig.update_layout(height=480);fig.update_xaxes(range=[0,1])
        chart(fig,"class_metrics")
        st.dataframe(frame,hide_index=True,width="stretch")
        st.caption("Per-class accuracy here is recall: correctly identified images of a true class divided by its held-out images.")
        st.subheader("Where the model confuses classes")
        normalize_matrix=st.toggle("Normalize confusion matrix by true class")
        matrix=np.loadtxt(EVALUATION_PATH/"confusion_matrix.csv",delimiter=",")
        if normalize_matrix:matrix=matrix/np.maximum(matrix.sum(1,keepdims=True),1)
        fig=px.imshow(matrix,x=labels,y=labels,text_auto=".0%" if normalize_matrix else ".0f",color_continuous_scale=["#f3f6ec","#9db994","#285e43"],labels={"x":"Predicted class","y":"True class","color":"Fraction" if normalize_matrix else "Images"},aspect="auto")
        fig.update_layout(height=610);fig.update_xaxes(tickangle=-45)
        chart(fig,"confusion_matrix")
    with learning:
        if history:
            rows=[{"Epoch":row["epoch"]+1,"Phase":row["stage"],"Split":split.capitalize(),"Loss":row[split]["loss"],"Accuracy":row[split]["top1"]} for row in history for split in ["train","val"]]
            frame=pd.DataFrame(rows)
            a,b=st.columns(2)
            with a:
                st.subheader("Training & validation loss");chart(px.line(frame,x="Epoch",y="Loss",color="Split",markers=True),"loss_curves")
            with b:
                st.subheader("Training & validation accuracy")
                fig=px.line(frame,x="Epoch",y="Accuracy",color="Split",markers=True);fig.update_yaxes(tickformat=".0%",range=[0,1]);chart(fig,"accuracy_curves")
            st.caption("The saved history records head training and fine-tuning. Validation loss selects the best checkpoint and controls early stopping.")
        else:st.info("No saved learning history is available.")
    with confidence:
        calibration=artifact(EVALUATION_PATH/"calibration.json");local=artifact(ROOT/("evaluation/v2/local_verification.json" if ACTIVE_VERSION=="v2" else "evaluation/local_model_verification.json"))
        a,b,c=st.columns(3);a.metric("ECE before scaling",f"{metrics['ece_before']:.5f}");b.metric("ECE after scaling",f"{metrics['ece_after']:.5f}");c.metric("Review threshold",f"{calibration['unknown_threshold']:.2%}")
        st.caption("Temperature was fitted on validation data only. Compare the measured ECE before and after scaling; no test-data recalibration was performed.")
        path=EVALUATION_PATH/"test_predictions.npz"
        if path.exists():
            probabilities,truth=test_probabilities(str(path),path.stat().st_mtime_ns)
            frame=pd.DataFrame({"Confidence":probabilities.max(1),"Outcome":np.where(probabilities.argmax(1)==truth,"Correct","Incorrect")})
            fig=px.histogram(frame,x="Confidence",color="Outcome",nbins=25,barmode="stack");fig.add_vline(x=calibration["unknown_threshold"],line_dash="dash",line_color="#b57832");chart(fig,"test_confidence")
        a,b=st.columns(2);a.metric("GPU median model inference",f"{metrics['model_inference_median_ms']:.2f} ms");b.metric("Local CPU median model inference",f"{local['median_cpu_model_inference_ms']:.2f} ms" if local else "Unavailable")
        st.caption("Warm-model timings exclude upload and preprocessing. Known-class acceptance at this threshold: "+f"{metrics['known_class_acceptance_rate']:.2%}"+". This is not unknown-species detection accuracy.")
    with st.expander("Grad-CAM: inspect actual held-out examples"):
        examples=sorted(EVALUATION_PATH.glob("gradcam_*.png"))
        for path in examples:st.image(str(path),width="stretch")
        st.caption("Examples include real mistakes. Heatmaps show score influence, not botanical proof.")
    with st.container(key="safety_panel"):
        st.markdown("**What these results do not establish**")
        st.write(f"The model supports {len(model().classes)} trained plant classes. Independent Pakistani field-image accuracy and unknown-species recognition are unverified. Related plant observations may remain because original plant IDs were unavailable. Adult plants, mixed plants and unfamiliar backgrounds can reduce reliability.")

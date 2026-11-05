import pandas as pd
import streamlit as st
from app.ui import *


def render():
    heading("FEEDBACK & REVIEW","Turn observations into better records.","Review uncertain suggestions and capture corrections without changing the trained model.")
    stats=database().stats();a,b,c,d=st.columns(4)
    a.metric("Pending reviews",stats["pending"]);b.metric("Reviewed records",stats["reviewed"]);c.metric("Flagged predictions",stats["flagged"]);d.metric("Feedback entries",stats["feedback"])
    predictions,feedback,records=st.tabs(["Review predictions","Feedback log","Species records"])
    with predictions:
        a,b=st.columns(2)
        queue=a.selectbox("Review queue",["All observations","Pending reviews","Reviewed predictions","Low-confidence predictions","Flagged predictions"],key="review_queue")
        species=b.selectbox("Review species",["All species"]+list(knowledge().by_class),key="review_filter")
        rows=database().history(limit=2000,species=None if species=="All species" else species,flagged_only=queue=="Flagged predictions",low_confidence=model().threshold if queue=="Low-confidence predictions" else None)
        if queue=="Pending reviews":rows=[r for r in rows if r["review_status"] in {"pending","unreviewed","needs expert"}]
        if queue=="Reviewed predictions":rows=[r for r in rows if r["review_status"] in {"confirmed","rejected"}]
        st.caption(f"{len(rows)} records in this view · Latest 2,000 matching observations")
        if not rows:st.info("No observations in this review queue.")
        else:
            row=record_picker(rows,"Review an observation",key="review_selected")
            with st.container(border=True):record_details(row)
            with st.form("review_decision_"+row["id"]):
                st.subheader("Reviewer decision")
                statuses=["pending","confirmed","needs expert","rejected"]
                status=st.selectbox("Review status",statuses,index=statuses.index(row["review_status"]) if row["review_status"] in statuses else 0)
                note=st.text_area("Reviewer note",value=row.get("reviewer_note") or "",max_chars=4000)
                if st.form_submit_button("Save review",type="primary"):
                    database().review(row["id"],status,note);st.session_state["notice"]="Review saved locally.";st.rerun()
            with st.expander("Submit prediction feedback"):
                with st.form("review_feedback_"+row["id"]):
                    verdict=st.radio("Prediction feedback",["Looks correct","Incorrect","Unsure"],horizontal=True)
                    corrected=st.selectbox("Corrected species",["No correction"]+list(knowledge().by_class)+["Other / unsupported"])
                    comment=st.text_area("Feedback comments",max_chars=4000)
                    if st.form_submit_button("Save feedback",type="primary"):
                        database().add_feedback(row["id"],verdict,None if corrected=="No correction" else corrected,comment)
                        st.session_state["notice"]="Feedback saved locally. No retraining was performed.";st.rerun()
    with feedback:
        entries=database().feedback()
        if entries:
            frame=pd.DataFrame(entries);st.dataframe(frame,hide_index=True,width="stretch")
            chart(px.bar(frame.verdict.value_counts().rename_axis("Verdict").reset_index(name="Count"),x="Verdict",y="Count",color_discrete_sequence=[GREEN]),"feedback_counts")
        else:st.info("Submitted feedback will appear here.")
    with records:
        st.write("Curated records are available for all supported model classes. Proposals remain separate from source-checked evidence.")
        st.dataframe(pd.DataFrame([{"Species":r["common_name"],"Scientific name":r["scientific_name"],"Family":r["family"],"Tier":r["tier"],"Review":r["expert_review_status"]} for r in knowledge().records]),hide_index=True,width="stretch")
        with st.form("species_correction"):
            label=st.selectbox("Species to review",list(knowledge().by_class))
            note=st.text_area("Proposed correction",max_chars=4000)
            source=st.text_input("Supporting HTTPS source",placeholder="https://…")
            if st.form_submit_button("Submit review note"):
                try:database().add_draft(label,note,source);st.success("Saved as awaiting expert verification.")
                except ValueError as error:st.error(str(error))
        drafts=database().drafts()
        if drafts:st.dataframe(pd.DataFrame(drafts),hide_index=True,width="stretch")
    st.caption("User feedback and review decisions are not independently verified training labels. No automatic model updates occur.")

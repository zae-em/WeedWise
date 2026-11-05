import json
import streamlit as st
from app.ui import *


@st.dialog("Delete saved observations",icon=":material/delete:")
def confirm_delete(identifiers):
    st.warning(f"Delete {len(identifiers)} selected record(s), their feedback and review notes? This cannot be undone.")
    st.caption("Unreferenced retained photographs in the app's uploads folder will also be removed. Model, dataset and species records are preserved.")
    suffix=str(hash(tuple(identifiers)))
    confirmation=st.text_input("Type DELETE to confirm",key="confirm_"+suffix)
    a,b=st.columns(2)
    if a.button("Delete permanently",type="primary",disabled=confirmation!="DELETE",key="delete_"+suffix):
        result=database().delete_predictions(identifiers,uploads=UPLOADS_PATH)
        st.session_state["notice"]=f"Deleted {result['deleted']} observation(s) and {result['images_removed']} unreferenced photograph(s)."
        st.session_state["delete_warnings"]=result["cleanup_warnings"]
        if st.session_state.get("identification",{}).get("id") in identifiers:st.session_state["identification"]["id"]=None
        st.rerun()
    if b.button("Cancel",key="cancel_"+suffix):st.rerun()


def render():
    heading("PREDICTION HISTORY","Your observations, in one place.","Find a saved identification, revisit its evidence or export records for further review.")
    stats=database().stats()
    a,b,c=st.columns(3);a.metric("Saved observations",stats["predictions"]);b.metric("Flagged for review",stats["flagged"]);c.metric("Feedback records",stats["feedback"])
    for warning in st.session_state.pop("delete_warnings",[]):st.warning("A retained photo could not be removed: "+warning)
    with st.container(border=True):
        a,b=st.columns([1.6,1])
        search=a.text_input("Search history",placeholder="Species name or record ID",icon=":material/search:")
        species=b.selectbox("Filter by species",["All species"]+list(knowledge().by_class))
        a,b,c=st.columns(3)
        start=a.date_input("From date (UTC)",value=None);end=b.date_input("To date (UTC)",value=None)
        flagged=c.checkbox("Flagged observations only")
    if start and end and start>end:st.error("The start date must be before or equal to the end date.");return
    rows=database().history(limit=2000,search=search,species=None if species=="All species" else species,date_from=start,date_to=end,flagged_only=flagged)
    if not rows:
        st.info("No saved observations match. Try another filter or save your first identification.")
        nav_button("Identify a plant","Identify plant",key="history_empty_identify")
    else:
        st.caption(f"Showing {len(rows):,} matching records (up to the latest 2,000). Dates are UTC.")
        frame=history_frame(rows)
        st.dataframe(frame.drop(columns=["ID"]),hide_index=True,width="stretch",column_config={"Confidence":st.column_config.ProgressColumn(format="percent",min_value=0,max_value=1)})
        st.download_button("Export filtered history CSV",frame.to_csv(index=False),"fieldwise_history.csv","text/csv",icon=":material/download:")
        st.divider();st.subheader("Observation details")
        row=record_picker(rows,key="history_selected")
        with st.container(border=True):record_details(row)
        a,b=st.columns(2)
        with a:nav_button("Review this observation","Feedback & review",row["top_class"],key="history_review",record=row["id"])
        with b:
            if st.button("Delete this record",icon=":material/delete:",width="stretch"):confirm_delete([row["id"]])
    if stats["predictions"]:
        with st.expander("History storage and privacy"):
            st.write("Photographs are retained only when chosen while saving. Deleting a record also removes its feedback and review. Shared photographs remain while another record references them.")
            if st.button("Clear all prediction history",icon=":material/delete_sweep:"):confirm_delete(database().all_prediction_ids())

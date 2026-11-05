import streamlit as st
from app.ui import *


def render():
    heading("WEED MANAGEMENT CENTER","Practical steps, with their sources.","Understand the plant, explore the options and check what applies to your crop and location.")
    kb=knowledge();weeds=[r["class_label"] for r in kb.records if r["role"]=="weed"]
    context=st.session_state.get("context_species")
    selected=st.selectbox("Weed species",weeds,index=weeds.index(context) if context in weeds else 0,key="management_species")
    record=kb.get(selected)
    with st.container(border=True):
        st.markdown(badge(record["family"],"neutral")+" "+badge("Tier "+record["tier"]),unsafe_allow_html=True)
        st.subheader(record["common_name"]);st.markdown("*"+record["scientific_name"]+"*")
        claims(record,{"description","competition"},"No source-verified description of crop competition is available for this record.")
        st.caption("Crops affected / Pakistani suitability: "+record["pakistan_relevance"])
    cultural,mechanical,chemical,integrated=st.tabs(["Cultural control","Mechanical control","Chemical groups","Integrated management"])
    for tab,title,categories in [(cultural,"Work with the growing system",{"cultural"}),(mechanical,"Physical removal and prevention",{"mechanical"}),(chemical,"Herbicide groups and resistance",{"chemical","resistance"}),(integrated,"Bring complementary methods together",{"integrated"})]:
        with tab,st.container(border=True):
            st.subheader(title);claims(record,categories)
            if tab==chemical:
                st.caption("No product brands, doses, mixing instructions or local approvals are inferred.")
    with st.expander("General management principles and group terminology"):
        for claim in kb.general:
            if claim["category"]!="classification":
                st.write(claim["text"]);references(claim["source_ids"])
    with st.container(key="safety_panel"):
        st.markdown(":material/health_and_safety: **Before acting in a field**")
        st.write("Confirm the plant identity with a qualified expert. Check local registration and the approved label for your crop and location. Resistance can limit chemical control; a group name does not establish efficacy or crop safety.")
        st.caption("Source context: HRAC mode-of-action classification; the regional management sources linked above.")
        references(["hrac"])
    a,b=st.columns(2)
    with a:nav_button("Open botanical profile","Species explorer",selected,key="management_profile")
    with b:nav_button("Ask a management question","AI chat assistant",selected,key="management_chat")

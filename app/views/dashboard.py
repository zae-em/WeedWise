import html
import pandas as pd
import streamlit as st
from app.ui import *


def render():
    kb=knowledge();classifier=model();stats=database().stats();metrics=artifact(EVALUATION_PATH/"test_metrics.json")
    heading("YOUR FIELD WORKSPACE","Better decisions start with identification.","AI Weed Identification & Management Assistant")
    with st.container(key="dashboard_hero"):
        left,right=st.columns([1.65,1])
        with left:
            st.markdown('<div class="hero-overline">LOOK CLOSER. GROW WITH CONFIDENCE.</div><h2>From a seedling photo<br>to an informed next step.</h2><p>Identify supported plants, compare the evidence and explore source-backed management. Your observations stay on this laptop.</p>',unsafe_allow_html=True)
            a,b,c=st.columns(3)
            with a:nav_button("Identify a Plant","Identify plant",key="home_identify",primary=True)
            with b:nav_button("Explore Species","Species explorer",key="home_explore")
            with c:nav_button("Try Examples","Species explorer",key="home_examples",icon="play_circle",tab="Examples")
        with right:
            st.markdown('''<div class="growth-art" aria-hidden="true"><svg viewBox="0 0 280 220"><circle cx="140" cy="102" r="84" fill="none" stroke="#769e79" stroke-opacity=".3"/><circle cx="140" cy="102" r="62" fill="none" stroke="#769e79" stroke-opacity=".2"/><path d="M140 183V88M140 140C85 140 74 105 81 77C120 78 141 104 140 140Z" fill="#7fa679" stroke="#b6d3a5" stroke-width="2"/><path d="M140 111C186 112 211 79 201 47C165 50 138 77 140 111Z" fill="#b1c59a" stroke="#d4ddbb" stroke-width="2"/><path d="M87 192H193M110 201H170" stroke="#96b48b" stroke-width="2"/><circle cx="216" cy="150" r="5" fill="#d8bb79"/></svg><span>OBSERVE · IDENTIFY · REVIEW</span></div>''',unsafe_allow_html=True)
    st.subheader("What the model can identify")
    counts=[("Supported species",len(classifier.classes),"All","eco"),
            ("Weed species",sum(r=="weed" for r in classifier.class_roles.values()),"Weeds","grass"),
            ("Crop species",sum(r=="crop" for r in classifier.class_roles.values()),"Crops","agriculture")]
    for column,(label,value,role,icon) in zip(st.columns(3),counts):
        with column,st.container(border=True,key="coverage_"+role.lower()):
            st.metric(label,str(value))
            nav_button("Browse "+("all species" if role=="All" else role.lower()),"Species explorer",key="browse_"+role,role=role,icon=icon)
    st.subheader("A snapshot of your workspace")
    for column,(label,value) in zip(st.columns(3),[("Held-out top-1",f"{metrics['top1']:.2%}" if metrics else "Unavailable"),("Saved identifications",str(stats["predictions"])),("Feedback records",str(stats["feedback"]))]):column.metric(label,value)
    st.caption(f"Model {classifier.model_version}: accuracy measured on {metrics['test_samples'] if metrics else 'unavailable'} internal held-out images; independent field accuracy is unverified.")
    st.subheader("What would you like to do?")
    actions=[("Identify Plant","Identify plant","Upload a seedling and compare five matches."),("Explore Weed Species","Species explorer","Browse botanical profiles and sources."),("Weed Management","Weed management","Explore cultural, mechanical and integrated guidance."),("AI Chat Assistant","AI chat assistant","Ask questions and inspect supporting citations."),("Prediction History","Prediction history","Find saved observations and export records."),("Model Performance","Model performance","Inspect measured results and limitations.")]
    for start in [0,3]:
        for column,(label,page,text) in zip(st.columns(3),actions[start:start+3]):
            with column,st.container(border=True):
                st.markdown("**"+label+"**");st.caption(text);nav_button("Open "+label,page,key="quick_"+PAGES[page],role="Weeds" if label=="Explore Weed Species" else None)
    activity,status=st.columns([1.65,1])
    with activity:
        st.subheader("Recent activity")
        rows=database().history(limit=5)
        if rows:
            frame=history_frame(rows)
            st.dataframe(frame[["Date (UTC)","Species","Confidence","Review"]],hide_index=True,width="stretch",column_config={"Confidence":st.column_config.ProgressColumn(format="percent",min_value=0,max_value=1)})
            nav_button("View all observations","Prediction history",key="recent_history")
        else:st.info("Your saved identifications will appear here. Start with a photo, then choose Save identification.")
    with status:
        st.subheader("System status")
        with st.container(border=True):
            for title,detail in [("Trained model loaded","MobileNetV2 · CPU inference"),("Knowledge base available",f"{len(kb.records)} profiles · {len(kb.sources)} sources"),("Chat assistant ready","Local, source-grounded retrieval"),("Database connected","SQLite · local storage")]:
                st.markdown(":material/check_circle: **"+title+"**");st.caption(detail)
            assistant()

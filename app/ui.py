"""Shared UI, cached services and navigation. Model weights and training code are read-only."""
import html
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from weed_assistant.config import ROOT,MODEL_PATH,KNOWLEDGE_PATH,UPLOADS_PATH,EVALUATION_PATH,ACTIVE_VERSION,DATASET_SUMMARY_PATH
from weed_assistant.database import Database
from weed_assistant.knowledge import KnowledgeBase,normalize
from weed_assistant.chatbot import GroundedChatbot
from weed_assistant.inference import PlantClassifier

PAGES={"Dashboard":"dashboard","Identify plant":"identify","Species explorer":"species","Weed management":"management",
       "AI chat assistant":"chat","Prediction history":"history","Feedback & review":"review","Model performance":"performance","About":"about"}
ICONS={"Dashboard":"dashboard","Identify plant":"center_focus_strong","Species explorer":"eco","Weed management":"agriculture",
       "AI chat assistant":"forum","Prediction history":"history","Feedback & review":"fact_check","Model performance":"monitoring","About":"info"}
GREEN="#285e43";GOLD="#caa75e"


@st.cache_data(show_spinner=False)
def read_json(path,version):return json.loads(Path(path).read_text(encoding="utf-8"))


def artifact(path):
    path=Path(path)
    return read_json(str(path),path.stat().st_mtime_ns) if path.is_file() else None


@st.cache_resource(show_spinner=False)
def _knowledge(version):return KnowledgeBase()


def knowledge():return _knowledge(KNOWLEDGE_PATH.stat().st_mtime_ns)


@st.cache_resource(show_spinner=False)
def _database(version):
    db=Database();db.sync_species(knowledge().records);return db


def database():return _database(KNOWLEDGE_PATH.stat().st_mtime_ns)


@st.cache_resource(show_spinner=False)
def _assistant(version):return GroundedChatbot(knowledge())


def assistant():return _assistant(KNOWLEDGE_PATH.stat().st_mtime_ns)


@st.cache_resource(show_spinner="Loading the trained plant model…")
def _model(version):
    classifier=PlantClassifier()
    database().register_model(classifier.model_hash,classifier.model_version,classifier.classes,classifier.class_roles)
    return classifier


def model():return _model(MODEL_PATH.stat().st_mtime_ns)


def go(page,species=None,role=None,record=None,tab=None):
    st.session_state["navigation"]=page
    st.query_params["page"]=PAGES[page]
    if species:
        st.session_state["context_species"]=species
        for key in ["explorer_species","management_species","chat_species"]:st.session_state.pop(key,None)
        if page=="Species explorer":
            st.session_state.update(explorer_role="All",explorer_family="All families",explorer_type="All types",explorer_search="")
    if role:st.session_state["explorer_role"]=role
    if page=="Species explorer":
        st.session_state.update(species_tab=tab or ("Plant profile" if species else "Species library"), explorer_page=0)
    if record:
        st.session_state.update(review_selected=record,review_queue="All observations",review_filter="All species")


def nav_button(label,page,species=None,key=None,primary=False,icon=None,role=None,record=None,tab=None):
    st.button(label,key=key,type="primary" if primary else "secondary",icon=":material/"+(icon or ICONS[page])+":",
              on_click=go,args=(page,species,role,record,tab),width="stretch")


def notice():
    if "notice" in st.session_state:st.success(st.session_state.pop("notice"))


def heading(kicker,title,description):
    st.markdown('<div class="eyebrow">'+html.escape(kicker)+'</div>',unsafe_allow_html=True)
    st.title(title);st.markdown('<p class="page-intro">'+html.escape(description)+'</p>',unsafe_allow_html=True)


def badge(text,kind="green"):
    return '<span class="tag tag-'+kind+'">'+html.escape(text)+'</span>'


def references(ids):
    for source in knowledge().references(ids):
        st.markdown(f"[{source['title']}]({source['url']})")
        st.caption(source["context"]+" · checked "+source["accessed"])


def claims(record,categories,empty="No source-verified guidance for this section is available in this record."):
    rows=[c for c in record["claims"] if c["category"] in categories]
    if not rows:st.info(empty)
    for index,claim in enumerate(rows):
        st.write(claim["text"])
        with st.expander("Sources for this guidance",expanded=False):references(claim["source_ids"])


def species_detail(record):
    st.markdown(badge(record["role"].title())+" "+badge("Tier "+record["tier"],"neutral")+" "+badge(record["plant_type"].title(),"neutral"),unsafe_allow_html=True)
    st.subheader(record["common_name"]);st.markdown("*"+record["scientific_name"]+"*")
    a,b,c=st.columns(3)
    a.metric("Botanical family",record["family"]);b.metric("Life cycle",record["life_cycle"]);c.metric("Record detail","Detailed" if record["tier"]=="A" else "Essential")
    profile,management,sources=st.tabs(["Botanical profile","Management","References"])
    with profile:
        claims(record,{"description","identification","lookalikes","competition"})
        st.caption("Crop associations / Pakistani relevance: "+record["pakistan_relevance"])
    with management:
        if record["role"]=="crop":st.info("This is a crop class. The model's crop/weed category is contextual and does not justify removal or treatment.")
        else:
            for title,categories in [("Cultural control",{"cultural"}),("Mechanical control",{"mechanical"}),("Integrated management",{"integrated"}),("Herbicide groups & resistance",{"chemical","resistance"})]:
                st.markdown("**"+title+"**");claims(record,categories)
    with sources:
        references(list(dict.fromkeys(i for claim in record["claims"] for i in claim["source_ids"])))
        st.caption(record["expert_review_status"])


def chart(fig,key=None):
    fig.update_layout(font=dict(family="Segoe UI, Arial",color="#304337"),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10,r=15,t=30,b=15),colorway=[GREEN,GOLD,"#6d8f79","#b56c51"])
    fig.update_xaxes(showgrid=False);fig.update_yaxes(gridcolor="#e7ede5")
    st.plotly_chart(fig,width="stretch",key=key,config={"displaylogo":False,"responsive":True})


def history_frame(rows):
    data=[]
    for row in rows:
        result=json.loads(row["result_json"])
        data.append({"ID":row["id"],"Date (UTC)":row["created_at"][:19].replace("T"," "),"Species":row["top_class"],
                     "Role":result.get("crop_weed_label","unknown"),"Confidence":row["confidence"],
                     "Feedback":row.get("feedback_status") or "Not submitted","Review":row.get("review_status") or "unreviewed",
                     "Corrected species":row.get("corrected_class") or "","Crop context":result.get("crop_context") or "Not Sure"})
    return pd.DataFrame(data)


def retained_photo(row):
    value=row.get("image_path")
    if value:
        path=Path(value).resolve()
        if path.is_relative_to(UPLOADS_PATH.resolve()) and path.is_file():return path
    return None


def record_picker(rows,label="Select a record",key="selected_record"):
    names={row["id"]:row["top_class"]+" · "+row["created_at"][:16].replace("T"," ")+" · "+row["id"][:8] for row in rows}
    if st.session_state.get(key) not in names:st.session_state.pop(key,None)
    chosen=st.selectbox(label,list(names),format_func=names.get,key=key)
    return next(row for row in rows if row["id"]==chosen)


def record_details(row):
    left,right=st.columns([1,1.6])
    with left:
        photo=retained_photo(row)
        if photo:st.image(str(photo),width="stretch")
        else:st.info("No photograph retained for this record.",icon=":material/no_photography:")
    with right:
        result=json.loads(row["result_json"]);quality=json.loads(row["quality_json"])
        st.subheader(knowledge().get(row["top_class"])["common_name"])
        st.caption(row["created_at"]+" · "+row["id"][:8])
        a,b=st.columns(2);a.metric("Confidence",f"{row['confidence']:.1%}");b.metric("Review",row["review_status"])
        st.dataframe(pd.DataFrame(result["predictions"]),hide_index=True,width="stretch")
        for warning in quality.get("warnings",[]):st.caption("Quality note: "+warning)
        st.caption("Crop context: "+(result.get("crop_context") or "Not Sure")+". No crop-based reranking applied.")
        nav_button("Open species profile","Species explorer",row["top_class"],key="record_profile_"+row["id"])

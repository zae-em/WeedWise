"""Fieldwise Streamlit application. Reuses the completed model and existing services."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import sqlite3
import streamlit as st
from app.ui import PAGES,ICONS,knowledge,database,model,go,notice,ACTIVE_VERSION
from weed_assistant.inference import ModelUnavailable
from app.views import dashboard,identify,species,management,chat,history,review,performance,about

st.set_page_config(page_title='Fieldwise | AI Weed Assistant',page_icon=':material/eco:',layout='wide',initial_sidebar_state='auto')
st.markdown((ROOT/'app/style.css').read_text(encoding='utf-8-sig'),unsafe_allow_html=True)

if 'navigation' not in st.session_state:
    slug=st.query_params.get('page','dashboard')
    st.session_state['navigation']=next((name for name,value in PAGES.items() if value==slug),'Dashboard')
if st.session_state['navigation'] not in PAGES:st.session_state['navigation']='Dashboard'

def sync_navigation():st.query_params['page']=PAGES[st.session_state['navigation']]

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-icon">&#8599;</span> fieldwise<span class="brand-dot">.</span></div><div class="brand-sub">IDENTIFY. UNDERSTAND. MANAGE.</div>',unsafe_allow_html=True)
    st.caption('WORKSPACE')
    st.radio('Explore',list(PAGES),key='navigation',label_visibility='collapsed',on_change=sync_navigation)
    st.divider()
    st.markdown('<div class="sidebar-note"><strong>Your field notes stay local.</strong><p>Real model predictions. Source-backed guidance. Human review.</p></div>',unsafe_allow_html=True)
    st.caption('RESEARCH RELEASE '+ACTIVE_VERSION.upper())
    records=knowledge().records
    st.caption(f"{len(records)} supported classes / {sum(r['role']=='weed' for r in records)} weeds / {sum(r['role']=='crop' for r in records)} crops")

st.markdown('<div class="workspace-bar"><span>AI WEED ASSISTANT <span class="workspace-divider">/</span> '+st.session_state['navigation'].upper()+'</span><span class="local-status"><i></i> LOCAL WORKSPACE</span></div>',unsafe_allow_html=True)
notice()
try:
    kb=knowledge();db=database();classifier=model()
    if set(classifier.classes)!=set(kb.by_class):raise ValueError('Model and knowledge-base class mappings differ. Restore the matching artifacts.')
    views={'Dashboard':dashboard,'Identify plant':identify,'Species explorer':species,'Weed management':management,'AI chat assistant':chat,'Prediction history':history,'Feedback & review':review,'Model performance':performance,'About':about}
    views[st.session_state['navigation']].render()
except (ValueError,ModelUnavailable,OSError,KeyError,sqlite3.Error) as error:
    st.error('This action could not be completed: '+str(error),icon=':material/error:')
    st.caption('Check the local model, image, knowledge records or database, then try again. Existing records and model files are preserved.')
st.markdown(f'<div class="app-footer"><strong>fieldwise.</strong><span>Observe carefully. Check the evidence. Keep learning.</span><span>{len(knowledge().records)} model classes &middot; Local research application</span></div>',unsafe_allow_html=True)

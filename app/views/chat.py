import streamlit as st
from app.ui import *


def clear_chat():
    st.session_state["messages"]=[];st.session_state.pop("queued_prompt",None)


def queue_prompt(value):st.session_state["queued_prompt"]=value


def render():
    heading("AI CHAT ASSISTANT","Ask a question. See the evidence.","A local assistant grounded in the curated species records. Every supported answer includes its sources.")
    left,right=st.columns([3,1])
    options=["Detect from question"]+list(knowledge().by_class);context=st.session_state.get("context_species")
    with left:
        selected=st.selectbox("Plant context",options,index=options.index(context) if context in options else 0,key="chat_species")
    with right:st.button("Clear chat",icon=":material/delete_sweep:",on_click=clear_chat,width="stretch")
    if selected!="Detect from question":
        st.caption("Active context: "+knowledge().get(selected)["common_name"]+". You can ask “How can I manage this plant?”")
    st.session_state.setdefault("messages",[])
    if not st.session_state["messages"]:
        with st.container(key="chat_welcome"):
            st.markdown(":material/forum: **Start with something you’ve noticed.**")
            st.write("Ask about leaf characteristics, a supported species or integrated weed management. I will say when a verified answer is unavailable.")
        suggestions=[("Identification","What are the identifying characteristics of "+(selected if selected!="Detect from question" else "common chickweed")+"?"),
                     ("Management","How can I manage "+(selected if selected!="Detect from question" else "common chickweed")+" without herbicides?"),
                     ("Compare species","Compare the identifying features of blackgrass and loose silky-bent.")]
        for column,(label,prompt) in zip(st.columns(3),suggestions):
            column.button(label,key="suggest_"+label,on_click=queue_prompt,args=(prompt,),width="stretch")
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"],avatar=":material/person:" if message["role"]=="user" else ":material/eco:"):
            st.markdown(message["content"])
            for reference in message.get("references",[]):st.markdown(f"[{reference['id']}: {reference['title']}]({reference['url']})")
    typed=st.chat_input("Ask about a plant or weed-management topic",max_chars=2000)
    prompt=typed or st.session_state.pop("queued_prompt",None)
    if prompt:
        st.session_state["messages"].append({"role":"user","content":prompt})
        with st.chat_message("user",avatar=":material/person:"):st.write(prompt)
        with st.chat_message("assistant",avatar=":material/eco:"):
            with st.spinner("Checking the curated sources…"):
                answer=assistant().answer(prompt,None if selected=="Detect from question" else selected)
            st.markdown(answer["answer"])
            for reference in answer["references"]:st.markdown(f"[{reference['id']}: {reference['title']}]({reference['url']})")
        st.session_state["messages"].append({"role":"assistant","content":answer["answer"],"references":answer["references"]})
    st.caption("Conversation stays in this browser session. No external LLM or paid API is used. Unverified feedback never becomes evidence.")

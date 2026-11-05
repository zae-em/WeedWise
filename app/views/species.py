import html

import streamlit as st

from app.ui import badge, heading, knowledge, model, nav_button, normalize, species_detail
from app.views import examples


PAGE_SIZE = 9


def filtered_species(records,search="",role="All",family="All families",plant_type="All types"):
    query=normalize(search)
    return [record for record in records
            if (role=="All" or record["role"]==("weed" if role=="Weeds" else "crop"))
            and (family=="All families" or record["family"]==family)
            and (plant_type=="All types" or record["plant_type"]==plant_type)
            and (not query or query in normalize(" ".join([record["common_name"],record["class_label"],record["scientific_name"],*record.get("aliases",[])])))]


def _set_page(index=0):
    st.session_state["explorer_page"] = index


def _clear_filters():
    st.session_state.update(explorer_search="", explorer_role="All", explorer_family="All families",
                            explorer_type="All types", explorer_page=0)


def open_profile(label):
    st.session_state.update(context_species=label, explorer_species=label, species_tab="Plant profile")


def _library(records):
    with st.container(border=True, key="library_filters"):
        search = st.text_input("Search common or scientific name", placeholder="Try chickweed, Zea mays or black-grass",
                               icon=":material/search:", key="explorer_search", on_change=_set_page)
        a, b, c = st.columns(3)
        role = a.selectbox("Plant role", ["All", "Weeds", "Crops"], key="explorer_role", on_change=_set_page)
        family = b.selectbox("Botanical family", ["All families"] + sorted({r["family"] for r in records}),
                             key="explorer_family", on_change=_set_page)
        plant_type = c.selectbox("Plant type", ["All types"] + sorted({r["plant_type"] for r in records}),
                                 key="explorer_type", on_change=_set_page)
    matches = filtered_species(records, search, role, family, plant_type)
    summary, reset = st.columns([2, 1], vertical_alignment="center")
    summary.caption(f"{len(matches)} of {len(records)} supported profiles")
    with reset:
        if search or role != "All" or family != "All families" or plant_type != "All types":
            st.button("Clear filters", icon=":material/filter_alt_off:", on_click=_clear_filters, width="stretch")
    if not matches:
        st.info("No supported species matches these filters. Try a different name or clear the filters.")
        return

    page_count = (len(matches) + PAGE_SIZE - 1) // PAGE_SIZE
    page = min(st.session_state.get("explorer_page", 0), page_count - 1)
    visible = matches[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
    for start in range(0, len(visible), 3):
        for column, record in zip(st.columns(3), visible[start:start + 3]):
            role = record["role"]
            with column, st.container(border=True, key=f"species_tile_{role}_{record['id']}"):
                mark = f'<div class="species-mark species-mark-{role}" aria-hidden="true">{html.escape(record["common_name"][:1])}</div>'
                st.markdown(mark + badge(role.title(), "amber" if role == "weed" else "green") + badge(record["family"], "neutral"),
                            unsafe_allow_html=True)
                st.subheader(record["common_name"])
                st.markdown(f'*{record["scientific_name"]}*')
                st.caption(f'{record["plant_type"].capitalize()} · {record["life_cycle"]}')
                st.button("View profile", icon=":material/arrow_forward:", key="species_open_" + record["id"],
                          on_click=open_profile, args=(record["class_label"],), width="stretch")
    if page_count > 1:
        previous, status, following = st.columns(3, vertical_alignment="center")
        previous.button("Previous", key="species_prev", disabled=page == 0, on_click=_set_page,
                        args=(page - 1,), icon=":material/chevron_left:", width="stretch")
        status.caption(f"Page {page + 1} of {page_count} · Showing {page * PAGE_SIZE + 1}–{page * PAGE_SIZE + len(visible)}")
        following.button("Next", key="species_next", disabled=page + 1 == page_count, on_click=_set_page,
                         args=(page + 1,), icon=":material/chevron_right:", width="stretch")


def _profile(records):
    labels = {record["class_label"]: record for record in records}
    if st.session_state.get("explorer_species") not in labels:
        context = st.session_state.get("context_species")
        st.session_state["explorer_species"] = context if context in labels else next(iter(labels))
    selected = st.selectbox("Species details", list(labels), format_func=lambda label: labels[label]["common_name"],
                            key="explorer_species")
    record = labels[selected]
    with st.container(border=True, key="plant_profile"):
        species_detail(record)
    left, middle, right = st.columns(3)
    left.button("Try an example photo", icon=":material/photo_camera:", on_click=examples.open_example,
                args=(selected,), width="stretch")
    with middle:
        nav_button("Ask about this species", "AI chat assistant", selected, key="library_chat")
    if record["role"] == "weed":
        with right:
            nav_button("Explore management", "Weed management", selected, key="library_management")


def render():
    classifier = model()
    records = sorted(knowledge().records, key=lambda record: record["common_name"].casefold())
    heading("SPECIES EXPLORER", "Know the plant before the next step.",
            f"Explore {len(records)} plants supported by {classifier.model_version.upper()}, read their botanical profiles and try the model on labelled photos.")
    library, profile, demo = st.tabs(["Species library", "Plant profile", "Examples"],
                                     key="species_tab", on_change="rerun")
    with library:
        _library(records)
    with profile:
        _profile(records)
    if demo.open:
        with demo:
            examples.render(classifier, records, open_profile)

import hashlib
import json
from typing import Mapping, List

import streamlit as st
from langchain_core.documents import Document
import plotly.graph_objects as go

from src.search_engine.search_engine import SearchEngine
import matplotlib.pyplot as plt
import numpy as np
from functools import lru_cache

st.set_page_config(page_title="CoffeeBeanDream", layout="wide")

def compute_search_signature(query: str, roast: str, origins: list[str], avoid: str) -> str:
    data = {
        "query": query,
        "roast": roast,
        "origins": sorted(origins),
        "avoid_terms": avoid
    }
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()

@st.cache_resource
def get_engine():
    print("SearchEngine initialized!")
    return SearchEngine()

engine = get_engine()

def run_search(query: str, roast_sel: str, origins_sel: list[str], avoid_terms_raw: str):
    """Call the back‑end search, then (optionally) post‑filter by origins."""
    filters = {}
    if roast_sel and roast_sel != "All":
        filters["roast"] = roast_sel

    negative_terms = [t.strip().lower() for t in avoid_terms_raw.split(",") if t.strip()]

    docs = engine.search(query, filters, negative_terms=negative_terms)

    if origins_sel and "All" not in origins_sel:
        wanted = set(origins_sel)
        docs = [
            d
            for d in docs
            if wanted & {d.metadata.get(k) for k in ("origin_1", "origin_2")}
        ]

    return docs

def run_slider_search(roast_sel: str, origins_sel: list[str], avoid_terms_raw: str) -> list[Document]:
    """Call the back-end slider-based flavor preference search."""
    filters = {}
    if roast_sel and roast_sel != "All":
        filters["roast"] = roast_sel

    negative_terms = [t.strip().lower() for t in avoid_terms_raw.split(",") if t.strip()]

    # Build preferences dict from sliders
    preferences = {
        "sweet_bitter": st.session_state["sweet_vs_bitter"] / 5.0,
        "acid_smooth": st.session_state["acid_vs_smooth"] / 5.0,
        "fruit_nut": st.session_state["fruity_vs_earthy"] / 5.0,
        "citrus_chocolate": st.session_state["citrus_vs_chocolate"] / 5.0,
        "floral_wood": st.session_state["floral_vs_spicy"] / 5.0,
    }

    docs = engine.search_by_flavor_preferences(
        preferences=preferences,
        filters=filters,
        negative_terms=negative_terms
    )

    if origins_sel and "All" not in origins_sel:
        wanted = set(origins_sel)
        docs = [
            d for d in docs
            if wanted & {d.metadata.get(k) for k in ("origin_1", "origin_2")}
        ]

    return docs

@st.cache_data(show_spinner=False)
def _flavor_radar(flavor_vec: Mapping[str, float]) -> go.Figure:
    """
    Turn a 10-dimension flavor vector into a Plotly radar figure.

    Expected keys (lower-case): acid, bitter, chocolate, citrus, floral,
    fruit, nut, smooth, sweet, wood.  Values should already be scaled 0-1.
    """
    # Fixed axis order so all charts line up visually
    axes = [
        "bitter",  # axis 1
        "acid",  # axis 2
        "nut",  # axis 3
        "chocolate",  # axis 4
        "wood",  # axis 5
        "sweet",  # axis 1 opposite
        "smooth",  # axis 2 opposite
        "fruit",  # axis 3 opposite
        "citrus",  # axis 4 opposite
        "floral",  # axis 5 opposite
    ]

    r = [flavor_vec.get(a, 0.0) for a in axes]  # default 0 if missing
    r.append(r[0])          # close loop
    theta = axes + [axes[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=r,
                theta=theta,
                fill="toself",
                hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
            )
        ],
        layout=go.Layout(
            margin=dict(l=10, r=10, t=10, b=10),
            polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
            showlegend=False,
        ),
    )
    return fig


def plot_3axis_radar(dimensions, values):
    values = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=1)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_ylim(0, 10)
    return fig


def main():
    st.title("☕ CoffeeBeanDream")

    # session‑state defaults
    defaults = {
        "search_mode": "Description",
        "search_query": "",
        "roast_filter": "All",
        "origin_filter_multi": ["All"],
        "avoid_terms": "",
        # polar slider defaults
        "sweet_vs_bitter": 0,
        "acid_vs_smooth": 0,
        "fruity_vs_earthy": 0,
        "citrus_vs_chocolate": 0,
        "floral_vs_spicy": 0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    # dynamic filter vocab
    chunks = getattr(engine.index, "chunks", []) or []
    all_roasts = sorted({d.metadata.get("roast", "Unknown") for d in chunks})
    all_origins = sorted({d.metadata.get("origin_1", "Unknown") for d in chunks})
    roast_options = ["All"] + all_roasts
    origin_options = ["All"] + all_origins

    # search‑mode toggle
    st.session_state.search_mode = st.radio(
        "Search Mode", ["Description", "Flavor Profile"], horizontal=True
    )

    # query capture
    if st.session_state.search_mode == "Description":
        st.session_state.search_query = st.text_input(
            "Describe your ideal coffee…",
            placeholder="e.g. fruity, chocolatey, bright acidity",
            value=st.session_state.search_query,
        )
        full_query = st.session_state.search_query.strip()
    else:
        # 5‑pole sliders laid out **with pole labels**
        polar_defs = [
            ("sweet_vs_bitter", "Sweet", "Bitter"),
            ("acid_vs_smooth", "Acidic", "Smooth"),
            ("fruity_vs_earthy", "Fruity", "Nutty/Earthy"),
            ("citrus_vs_chocolate", "Citrusy", "Chocolatey"),
            ("floral_vs_spicy", "Floral", "Spicy/Woody"),
        ]

        cols = st.columns(len(polar_defs), gap="large")
        for col, (key, left, right) in zip(cols, polar_defs):
            with col:
                # two‑column header for poles
                h_left, h_right = st.columns([1, 1])
                with h_left:
                    st.markdown(
                        f"<div style='text-align:left;font-weight:600'>{left}</div>",
                        unsafe_allow_html=True,
                    )
                with h_right:
                    st.markdown(
                        f"<div style='text-align:right;font-weight:600'>{right}</div>",
                        unsafe_allow_html=True,
                    )

                # unique‑key slider; value saved automatically in session_state[key]
                st.slider(
                    label=" ",
                    min_value=-5,
                    max_value=5,
                    value=st.session_state[key],
                    step=1,
                    key=key,
                    label_visibility="collapsed",
                )

        # translate slider deltas to verbal query tokens
        def pole_phrase(val, l, r):
            if val == 0:
                return None
            degree = ["slightly", "moderately", "very"][min(abs(val) // 2, 2)]
            return f"{degree} {(l if val < 0 else r).lower()}"

        phrases = [pole_phrase(st.session_state[k], l, r) for k, l, r in polar_defs]
        full_query = ", ".join([p for p in phrases if p]) or "balanced profile"

    # filters area
    with st.expander("Filters", expanded=False):
        f1, f2 = st.columns([1, 1], gap="medium")
        if st.button("Reset Filters"):
            st.session_state.roast_filter = "All"
            st.session_state.origin_filter_multi = ["All"]
            st.session_state.avoid_terms = ""

        with f1:
            st.selectbox("Roast", roast_options, key="roast_filter")
        with f2:
            st.multiselect(
                "Origin(s)",
                origin_options,
                default=st.session_state.origin_filter_multi,
                key="origin_filter_multi",
            )
        st.text_input(
            "Flavors to avoid (comma-separated)",
            key="avoid_terms",
            placeholder="e.g. smoky, liquorice",
            help="Any coffee whose description contains one of these words will be excluded.",
        )

    if st.button("Search Beans"):
        if not full_query:
            st.warning("Please enter a query or adjust sliders first.")
        else:
            current_signature = compute_search_signature(
                full_query,
                st.session_state.roast_filter,
                st.session_state.origin_filter_multi,
                st.session_state.avoid_terms
            )

            previous_signature = st.session_state.get("last_signature")

            # only clear state if query or filters changed
            if previous_signature != current_signature:
                for key in list(st.session_state.keys()):
                    if key.startswith("flavor_vector_") or key.startswith("explanation_"):
                        del st.session_state[key]

            with st.spinner("Searching…"):
                docs = run_search(
                    full_query,
                    st.session_state.roast_filter,
                    st.session_state.origin_filter_multi,
                    st.session_state.avoid_terms,
                )

                if not docs:
                    st.info("No coffees found matching those criteria.")
                    st.session_state.pop("last_docs", None)
                    st.session_state.pop("last_query", None)
                else:
                    st.session_state.last_docs = docs
                    st.session_state.last_query = full_query
                    st.session_state.last_signature = current_signature

    if "last_docs" in st.session_state:
        st.subheader("Recommended Coffees")
        for i, d in enumerate(st.session_state.last_docs, 1):
            m = d.metadata
            name = m.get("name", "Unknown")
            origin_list = {m.get(k, "") for k in ("origin_1", "origin_2")}
            origins = ", ".join(o for o in origin_list if o) or "Unknown"
            link = f" • [link]({m['link']})" if m.get("link") else ""

            # card container keeps the layout tight
            with st.container():
                # two columns: text on the left, radar on the right
                left, right = st.columns([2, 3], gap="medium")

            with left:
                st.markdown(f"### {i}. {name}{link}")
                st.markdown(f"**Origins:** {origins}")
                st.markdown(f"**Roast:** {m.get('roast', 'Unknown')}")
                st.markdown(f"**Roaster:** {m.get('roaster', 'Unknown')}")

                if d.page_content:
                    st.markdown(f"**Flavor description:** *{d.page_content}*")

                with st.expander("Why this match?"):
                    explain_key = f"explanation_{i}"
                    if st.button("Explain Match", key=f"explain_btn_{i}"):
                        with st.spinner("Generating explanation..."):
                            st.session_state[explain_key] = engine.explain_result(
                                st.session_state.last_query, d)

                    if explain_key in st.session_state:
                        st.markdown(f"**Explanation:** {st.session_state[explain_key]}")

            with right:
                flavor_key = f"flavor_vector_{i}"
                flavor_btn_key = f"flavor_btn_{i}"

                # Step 1: Handle analysis button click
                if st.button("Analyze Flavor Profile", key=flavor_btn_key):
                    with st.spinner("Analyzing flavor profile..."):
                        analyzed_doc = engine.flavor_radar.analyze_document(d)
                        flavor_vec = analyzed_doc.metadata.get("flavor_vector")
                        if flavor_vec:
                            st.session_state[flavor_key] = flavor_vec

                # Step 2: Render if available after rerun
                flavor_vec = st.session_state.get(flavor_key) or m.get("flavor_vector")
                if flavor_vec:
                    st.plotly_chart(_flavor_radar(flavor_vec), use_container_width=True)

            st.markdown("---")


if __name__ == "__main__":
    main()
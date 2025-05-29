import streamlit as st
from src.search_engine.search_engine import SearchEngine
import matplotlib.pyplot as plt
import numpy as np

if "engine" not in st.session_state:
    st.session_state.engine = SearchEngine()
engine = st.session_state.engine

def run_search(query: str, roast_sel: str, origins_sel: list[str], avoid_terms_raw: str):
    """Call the back‑end search, then (optionally) post‑filter by origins."""
    filters = {}
    if roast_sel and roast_sel != "All":
        filters["roast"] = roast_sel

    docs = engine.search(query, filters)

    if origins_sel and "All" not in origins_sel:
        wanted = set(origins_sel)
        docs = [
            d
            for d in docs
            if wanted & {d.metadata.get(k) for k in ("origin_1", "origin_2")}
        ]
    terms = [t.strip().lower() for t in avoid_terms_raw.split(",") if t.strip()]
    if terms:
        docs = [
            d for d in docs
            if not any(term in (d.page_content or "").lower() for term in terms)
        ]

    return docs


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
    st.set_page_config(page_title="CoffeeBeanDream", layout="wide")
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
            ("acid_vs_smooth", "Bright (Acidic)", "Smooth"),
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

    # search action
    if st.button("Search Beans"):
        if not full_query:
            st.warning("Please enter a query or adjust sliders first.")
        else:
            with st.spinner("Searching…"):
                docs = run_search(
                    full_query,
                    st.session_state.roast_filter,
                    st.session_state.origin_filter_multi,
                    st.session_state.avoid_terms, 
                )
                if not docs:
                    st.info("No coffees found matching those criteria.")
                else:
                    st.subheader("Recommended Coffees")
                    for i, d in enumerate(docs, 1):
                        m = d.metadata
                        name = m.get("name", "Unknown")
                        origin_list = {m.get(k, "") for k in ("origin_1", "origin_2")}
                        origins = ", ".join(o for o in origin_list if o) or "Unknown"
                        link = f" • [link]({m['link']})" if m.get("link") else ""

                        st.markdown(f"### {i}. {name}{link}")
                        st.markdown(f"**Origins:** {origins}")
                        st.markdown(f"**Roast:** {m.get('roast','Unknown')}")
                        st.markdown(f"**Roaster:** {m.get('roaster','Unknown')}")

                        if d.page_content:
                            st.markdown(f"**Flavor description:** *{d.page_content}*")

                        # with st.expander("Why this match?"):
                        #     st.write(engine.explain_result(full_query, d))

                        st.markdown("---")



if __name__ == "__main__":
    main()

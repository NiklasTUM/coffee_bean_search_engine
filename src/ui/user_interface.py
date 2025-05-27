import streamlit as st
from src.search_engine.search_engine import SearchEngine
import matplotlib.pyplot as plt
import numpy as np


def _apply_meta_filters(docs, roast_sel: str, origin_sel: str):
    """
    Return only docs that match the roast and origin filters.
    'All' in either field means no filtering on that field.
    """
    filtered = []
    for d in docs:
        roast_ok = roast_sel == "All" or d.metadata.get("roast") == roast_sel
        origin_ok = origin_sel == "All" or d.metadata.get("origin") == origin_sel
        if roast_ok and origin_ok:
            filtered.append(d)
    return filtered


def plot_3axis_radar(dimensions, values, title=None):
    # close the loop
    values = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=1)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_ylim(0, 10)

    return fig


# Initialize RAGChain in session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = SearchEngine()


# Helper to post-filter retrieved docs
def _apply_meta_filters(docs, roast_sel, origin_sel):
    filtered = []
    for d in docs:
        roast_ok = (roast_sel == "All") or (d.metadata.get("roast") == roast_sel)
        origin_ok = (origin_sel == "All") or (d.metadata.get("origin") == origin_sel)
        if roast_ok and origin_ok:
            filtered.append(d)
    return filtered


def main():
    st.set_page_config(page_title="CoffeeBeanDream", layout="centered")
    st.title("☕ CoffeeBeanDream")

    # ————————————————
    # 1) Ensure session_state defaults
    # ————————————————
    if "search_mode" not in st.session_state: st.session_state.search_mode = "Description"
    if "search_query" not in st.session_state: st.session_state.search_query = ""
    if "sweetness" not in st.session_state: st.session_state.sweetness = 5
    if "bitterness" not in st.session_state: st.session_state.bitterness = 5
    if "acidity" not in st.session_state: st.session_state.acidity = 5
    if "roast_filter" not in st.session_state: st.session_state.roast_filter = "All"
    if "origin_filter" not in st.session_state: st.session_state.origin_filter = "All"

    # Grab the chain (already in session_state)
    chain = st.session_state.rag_chain

    # ————————————————
    # 2) Build dynamic filter options
    # ————————————————
    # We use the indexed chunks' metadata to get all roast/origin values:
    all_roasts = sorted({d.metadata.get("roast", "Unknown") for d in chain.index.chunks})
    all_origins = sorted({d.metadata.get("origin", "Unknown") for d in chain.index.chunks})
    roast_options = ["All"] + all_roasts
    origin_options = ["All"] + all_origins

    # ————————————————
    # 3) Mode toggle
    # ————————————————
    st.session_state.search_mode = st.radio(
        "Choose Search Mode",
        ["Description", "Flavor Profile"],
        index=["Description", "Flavor Profile"].index(st.session_state.search_mode),
        horizontal=True
    )

    # ————————————————
    # 4) Input area
    # ————————————————
    if st.session_state.search_mode == "Description":
        st.session_state.search_query = st.text_input(
            "Describe your ideal coffee",
            value=st.session_state.search_query,
            placeholder="e.g. fruity, chocolatey, bright acidity"
        )
        full_query = st.session_state.search_query.strip()

    else:
        c1, c2, c3 = st.columns(3, gap="small")
        # with c1:
        #     st.session_state.sweetness = st.slider(
        #         "Sweet", 0, 10, st.session_state.sweetness
        #     )
        # with c2:
        #     st.session_state.bitterness = st.slider(
        #         "Bitter", 0, 10, st.session_state.bitterness
        #     )
        # with c3:
        #     st.session_state.acidity = st.slider(
        #         "Acid", 0, 10, st.session_state.acidity
        #     )
        #

        with c1:
            st.session_state.sweet_vs_bitter = st.slider(
                "Sweet ←→ Bitter", -5, 5, st.session_state.get("sweet_vs_bitter", 0)
            )

        with c2:
            st.session_state.acid_vs_smooth = st.slider(
                "Bright/Acidic ←→ Smooth", -5, 5, st.session_state.get("acid_vs_smooth", 0)
            )

        with c3:
            st.session_state.fruity_vs_earthy = st.slider(
                "Fruity ←→ Earthy", -5, 5, st.session_state.get("fruity_vs_earthy", 0)
            )

        def _verbal(v, dim):
            return f"{'low' if v <= 3 else 'medium' if v <= 6 else 'high'} {dim}"

        full_query = ", ".join([
            _verbal(st.session_state.sweetness, "sweetness"),
            _verbal(st.session_state.bitterness, "bitterness"),
            _verbal(st.session_state.acidity, "acidity"),
        ])

    # # ─── 5) Collapsible Filters ─────────────────────────────────────────────────
    with st.expander("Filters", expanded=False):

        # ─── Row 2: Controls ───────────────────────────────────────────────────────
        ctl1, ctl2, ctl3 = st.columns([1, 1, 0.5], gap="small")
        with ctl3:
            st.markdown("<div style='padding-top:1.75rem'></div>", unsafe_allow_html=True)
            if st.button("Reset Filters"):
                # clear the filters (and any inputs you want)
                st.session_state.roast_filter = "All"
                st.session_state.origin_filter = "All"
                st.session_state.pop("search_query", None)
                for k in ("sweetness", "bitterness", "acidity"):
                    st.session_state.pop(k, None)
                # no need for experimental_rerun()

        with ctl1:
            st.selectbox(
                "Roast",
                options=roast_options,
                index=roast_options.index(st.session_state.get("roast_filter", "All")),
                key="roast_filter",
                help="Filter by roast level",
            )
        with ctl2:
            st.selectbox(
                "Origin",
                options=origin_options,
                index=origin_options.index(st.session_state.get("origin_filter", "All")),
                key="origin_filter",
                help="Filter by country of origin",
            )

    # ————————————————
    # 6) Search button
    # ————————————————
    search_clicked = st.button("Search Beans")

    # ————————————————
    # 7) Execute search
    # ————————————————
    if search_clicked:
        if not full_query:
            st.warning("Please enter a description or adjust the sliders first.")
        else:
            with st.spinner("Searching coffee beans…"):
                # Retrieve and then post-filter by roast/origin
                chain = st.session_state.rag_chain
                roast_filter = st.session_state.get("roast_filter", "All")
                origin_filter = st.session_state.get("origin_filter", "All")

                # TODO put this into filters dict
                filters = {
                    "roast": roast_filter,
                    #"origin_1": origin_filter
                }

                docs = chain.search(full_query, {})

                # st.subheader("Recommended Coffees")
                # if not docs:
                #     st.info("No beans match those filters.")
                # for i, d in enumerate(docs, 1):
                #     m = d.metadata
                #     st.markdown(f"### {i}. {m.get('name','?')}")
                #     st.markdown(
                #         f"**Origin:** {m.get('origin','?')}  \n"
                #         f"**Roast:** {m.get('roast','?')}  \n"
                #         f"**Roaster:** {m.get('roaster','?')}"
                #     )
                #     with st.expander("Why this match?"):
                #         st.markdown(chain.explain_match(d, full_query))
                #     st.markdown("---")
                st.subheader("Recommended Coffees")
                for i, d in enumerate(docs, 1):
                    m = d.metadata
                    name = m.get("name", "Unknown")
                    origin = m.get("origin_1", "?")
                    roast = m.get("roast", "?")

                    # 1) Text info
                    st.markdown(f"### {i}. {name}")
                    st.markdown(f"**Origin:** {origin}  \n**Roast:** {roast}")

                    # 2) Radar chart
                    #   — make sure these keys exist in your metadata as numbers 0–10
                    dims = ["Sweetness", "Bitterness", "Acidity"]
                    vals = [
                        float(m.get("sweetness", 5)),
                        float(m.get("bitterness", 5)),
                        float(m.get("acidity", 5)),
                    ]
                    # fig = plot_3axis_radar(dims, vals, title=f"{m.get('name','')} Profile")
                    # st.pyplot(fig)

                    # 3) Explanation
                    with st.expander("Why this match?"):
                        explanation = chain.explain_result(full_query, d)
                        translated_explanation = chain.translator.translate_text(explanation, chain.user_language)["translated_text"]
                        st.markdown(translated_explanation)

                    st.markdown("---")

    # ————————————————
    # 8) Update Index button
    # ————————————————
    if st.button("Update Index"):
        with st.spinner("Re-indexing…"):
            chain.update_index()
            st.success("Index updated successfully!")


# Call main once
main()

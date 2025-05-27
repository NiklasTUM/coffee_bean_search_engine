import streamlit as st
from src.rag_chain.search_engine_chain import SearchEngineChain
import matplotlib.pyplot as plt
import numpy as np

def _apply_meta_filters(docs, roast_sel: str, origin_sel: str):
    """
    Return only docs that match the roast and origin filters.
    'All' in either field means no filtering on that field.
    """
    filtered = []
    for d in docs:
        roast_ok  = roast_sel == "All"   or d.metadata.get("roast")   == roast_sel
        origin_ok = origin_sel == "All"  or d.metadata.get("origin")  == origin_sel
        if roast_ok and origin_ok:
            filtered.append(d)
    return filtered


def plot_3axis_radar(dimensions, values, title=None):
    # close the loop
    values = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3,3), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=1)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions)
    ax.set_yticks([2,4,6,8,10])
    ax.set_ylim(0,10)

    return fig

# Initialize RAGChain in session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = SearchEngineChain()


# def main():
#     st.set_page_config(page_title="AromaAtlas", layout="centered")
#     st.title("☕ AromaAtlas")

#     st.sidebar.title("Search Settings")
#     search_mode = st.sidebar.radio(
#         "Select Search Mode:",
#         ["Search by Description", "Search by Flavor Profile"],
#         index=0
#     )

#     # Input for Mode 1: Text-based search
#     if search_mode == "Search by Description":
#         search_query = st.text_input(
#             "Describe your ideal coffee:",
#             placeholder="e.g., fruity, chocolatey, low-acidity"
#         )
#         full_query = search_query.strip()

#     # Input for Mode 2: Slider-based profile search
#     elif search_mode == "Search by Flavor Profile":
#         st.sidebar.header("Flavor Profile Sliders")
#         sweetness = st.sidebar.slider("Sweetness", 0, 10, 5)
#         bitterness = st.sidebar.slider("Bitterness", 0, 10, 5)
#         acidity = st.sidebar.slider("Acidity", 0, 10, 5)
#         body = st.sidebar.slider("Body (light to full)", 0, 10, 5)

#         def scale_to_text(value, dim):
#             if value <= 3: return f"low {dim}"
#             elif value <= 6: return f"medium {dim}"
#             else: return f"high {dim}"

#         flavor_words = ", ".join([
#             scale_to_text(sweetness, "sweetness"),
#             scale_to_text(bitterness, "bitterness"),
#             scale_to_text(acidity,   "acidity"),
#             scale_to_text(body,      "body")
#         ])
#         full_query = flavor_words

#     # Trigger search
#     if st.button("🔍 Search Beans"):
#         if full_query:
#             with st.spinner("Searching coffee beans..."):
#                 rag_chain = st.session_state.rag_chain
#                 results = rag_chain.retriever.retrieve(full_query)

#                 st.subheader("☕ Recommended Coffees")
#                 for i, doc in enumerate(results, 1):
#                     meta = doc.metadata
#                     name = meta.get("name", "Unknown")
#                     origin = meta.get("origin", "Unknown")
#                     roast = meta.get("roast", "Unknown")
#                     roaster = meta.get("roaster", "Unknown")

#                     with st.container():
#                         st.markdown(f"### {i}. {name}")
#                         st.markdown(f"**Origin**: {origin}  \n**Roast**: {roast}  \n**Roaster**: {roaster}")
#                         with st.expander("Why this match?"):
#                             explanation = rag_chain.explain_match(doc, full_query)
#                             st.markdown(explanation)
#                         st.markdown("---")
#         else:
#             st.warning("Please provide input based on the selected search mode.")

#     # Optional: update index
#     if st.button("Update Index"):
#         with st.spinner("Re-indexing coffee beans..."):
#             st.session_state.rag_chain.update_index()
#             st.success("Index updated successfully!")
# main()


# def main() -> None:
#     st.set_page_config(page_title="CoffeeBeanDream", layout="centered")
#     st.title("☕ CoffeeBeanDream")

#     # ── 1  Search-mode toggle ────────────────────────────────────────────────
#     search_mode = st.radio(
#         "Choose Search Mode",
#         ["Description", "Flavor Profile"],
#         horizontal=True,
#     )

#     input_col, button_col = st.columns([5, 1], gap="small")

#     with input_col:
#         if search_mode == "Description":
#             # label is inside the widget → no separate heading, no anchor icon
#             search_query = st.text_input(
#                 "Describe your ideal coffee",
#                 placeholder="e.g. fruity, chocolatey, bright acidity",
#             )
#             full_query = search_query.strip()

#         else:  # Flavor Profile sliders
#             c1, c2, c3 = st.columns(3, gap="small")
#             with c1:
#                 sweet  = st.slider("Sweet",  0, 10, 5)
#             with c2:
#                 bitter = st.slider("Bitter", 0, 10, 5)
#             with c3:
#                 acid   = st.slider("Acid",   0, 10, 5)

#             def taste(value, dim):
#                 return f"{'low' if value <= 3 else 'medium' if value <= 6 else 'high'} {dim}"

#             full_query = ", ".join([
#                 taste(sweet,  "sweetness"),
#                 taste(bitter, "bitterness"),
#                 taste(acid,   "acidity"),
#             ])

#     with button_col:
#         st.markdown("<div style='height:1.8em'></div>", unsafe_allow_html=True)  # vertical align
#         search_clicked = st.button("Search")

#     # ── 3  Update-index button (own row) ─────────────────────────────────────
#     if st.button("Update Index"):
#         with st.spinner("Re-indexing…"):
#             st.session_state.rag_chain.update_index()
#             st.success("Index updated!")

#     filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])

#     with filter_col1:
#         roast_filter = st.selectbox(
#             "Roast",
#             ["All", "Light", "Medium", "Medium-Dark", "Dark"]
#         )

#     with filter_col2:
#         # Populate origins dynamically if you like; here’s a static example
#         origin_filter = st.selectbox(
#             "Origin",
#             ["All", "Ethiopia", "Honduras", "Colombia", "Kenya", "Peru", "Indonesia"]
#         )

#     with filter_col3:
#         st.markdown("### &nbsp;")
#         if st.button("Reset Filters"):
#             roast_filter = "All"
#             origin_filter = "All"
#             # also clear text input / sliders:
#             st.session_state.pop("Describe your ideal coffee", None)
#             for key in ("Sweet", "Bitter", "Acid"):
#                 if key in st.session_state:  # slider values
#                     st.session_state[key] = 5


#     # ── 4  Run the search ───────────────────────────────────────────────────
#     if search_clicked:
#         if full_query:
#             with st.spinner("Searching coffee beans…"):
#                 chain   = st.session_state.rag_chain
#                 results = chain.retriever.retrieve(full_query)
#                 results = _apply_meta_filters(results, roast_filter, origin_filter)
                
#                 st.subheader("☕ Recommended Coffees")
#                 for i, doc in enumerate(results, 1):
#                     m = doc.metadata
#                     st.markdown(f"### {i}. {m.get('name', 'Unknown')}")
#                     st.markdown(
#                         f"**Origin:** {m.get('origin','?')}  \n"
#                         f"**Roast:** {m.get('roast','?')}  \n"
#                         f"**Roaster:** {m.get('roaster','?')}"
#                     )
#                     with st.expander("Why this match?"):
#                         st.markdown(chain.explain_match(doc, full_query))
#                     st.markdown("---")
#         else:
#             st.warning("Please enter a description or adjust the sliders first.")

# main()


# Helper to post-filter retrieved docs
def _apply_meta_filters(docs, roast_sel, origin_sel):
    filtered = []
    for d in docs:
        roast_ok  = (roast_sel == "All") or (d.metadata.get("roast")  == roast_sel)
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
    if "search_mode"   not in st.session_state: st.session_state.search_mode   = "Description"
    if "search_query"  not in st.session_state: st.session_state.search_query  = ""
    if "sweetness"     not in st.session_state: st.session_state.sweetness     = 5
    if "bitterness"    not in st.session_state: st.session_state.bitterness    = 5
    if "acidity"       not in st.session_state: st.session_state.acidity       = 5
    if "roast_filter"  not in st.session_state: st.session_state.roast_filter  = "All"
    if "origin_filter" not in st.session_state: st.session_state.origin_filter = "All"

    # Grab the chain (already in session_state)
    chain = st.session_state.rag_chain

    # ————————————————
    # 2) Build dynamic filter options
    # ————————————————
    # We use the indexed chunks' metadata to get all roast/origin values:
    all_roasts  = sorted({d.metadata.get("roast","Unknown")  for d in chain.index.chunks})
    all_origins = sorted({d.metadata.get("origin","Unknown") for d in chain.index.chunks})
    roast_options  = ["All"] + all_roasts
    origin_options = ["All"] + all_origins

    # ————————————————
    # 3) Mode toggle
    # ————————————————
    st.session_state.search_mode = st.radio(
        "Choose Search Mode",
        ["Description", "Flavor Profile"],
        index=["Description","Flavor Profile"].index(st.session_state.search_mode),
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
        with c1:
            st.session_state.sweetness = st.slider(
                "Sweet", 0, 10, st.session_state.sweetness
            )
        with c2:
            st.session_state.bitterness = st.slider(
                "Bitter", 0, 10, st.session_state.bitterness
            )
        with c3:
            st.session_state.acidity = st.slider(
                "Acid", 0, 10, st.session_state.acidity
            )

        def _verbal(v, dim):
            return f"{'low' if v <= 3 else 'medium' if v <= 6 else 'high'} {dim}"

        full_query = ", ".join([
            _verbal(st.session_state.sweetness,  "sweetness"),
            _verbal(st.session_state.bitterness, "bitterness"),
            _verbal(st.session_state.acidity,   "acidity"),
        ])

    # # ─── 5) Collapsible Filters ─────────────────────────────────────────────────
    with st.expander("Filters", expanded=False):

        # ─── Row 2: Controls ───────────────────────────────────────────────────────
        ctl1, ctl2, ctl3 = st.columns([1, 1, 0.5], gap="small")
        with ctl3:
            st.markdown("<div style='padding-top:1.75rem'></div>", unsafe_allow_html=True)
            if st.button("Reset Filters"):
                # clear the filters (and any inputs you want)
                st.session_state.roast_filter  = "All"
                st.session_state.origin_filter = "All"
                st.session_state.pop("search_query", None)
                for k in ("sweetness","bitterness","acidity"):
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
                chain   = st.session_state.rag_chain
                roast_filter  = st.session_state.get("roast_filter", "All")
                origin_filter = st.session_state.get("origin_filter", "All")
                docs = chain.retriever.retrieve(full_query)
                docs = _apply_meta_filters(
                    docs,
                    roast_filter,
                    origin_filter
                )

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
                    name   = m.get("name","Unknown")
                    origin = m.get("origin","?")
                    roast  = m.get("roast","?")

                    # 1) Text info
                    st.markdown(f"### {i}. {name}")
                    st.markdown(f"**Origin:** {origin}  \n**Roast:** {roast}")

                    # 2) Radar chart
                    #   — make sure these keys exist in your metadata as numbers 0–10
                    dims   = ["Sweetness","Bitterness","Acidity"]
                    vals   = [
                        float(m.get("sweetness",5)),
                        float(m.get("bitterness",5)),
                        float(m.get("acidity",5)),
                    ]
                    # fig = plot_3axis_radar(dims, vals, title=f"{m.get('name','')} Profile")
                    # st.pyplot(fig)

                    # 3) Explanation
                    with st.expander("Why this match?"):
                        st.markdown(chain.explain_match(d, full_query))

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

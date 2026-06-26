import streamlit as st
import json
import os
import time
import re
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from urllib.parse import quote

# Import our core components
from config import DATA_DIR, CORPUS_DIR, EVAL_DATASET_PATH
from rag_pipeline import RAGPipeline
from evaluator import RAGEvaluator
from retriever import HybridRetriever

# 1. Page Configuration
st.set_page_config(
    page_title="RAG Evaluation & Playground",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Theme Toggle State
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# Define colors based on theme
BG = "#09090b" if IS_DARK else "#ffffff"
BG_SUBTLE = "#0c0c0f" if IS_DARK else "#f9fafb"
CARD = "#0c0c0f" if IS_DARK else "#ffffff"
CARD_HOVER = "#131316" if IS_DARK else "#f4f4f5"
BORDER = "#1e1e24" if IS_DARK else "#e4e4e7"
BORDER_SUBTLE = "#16161a" if IS_DARK else "#f0f0f2"
TEXT = "#fafafa" if IS_DARK else "#09090b"
TEXT_MUTED = "#71717a"
ACCENT = "#2563eb"
GREEN = "#22c55e" if IS_DARK else "#16a34a"
GREEN_MUTED = "rgba(34,197,94,0.12)" if IS_DARK else "rgba(22,163,74,0.08)"
RED = "#ef4444" if IS_DARK else "#dc2626"
RED_MUTED = "rgba(239,68,68,0.12)" if IS_DARK else "rgba(220,38,38,0.08)"
AMBER = "#f59e0b" if IS_DARK else "#d97706"
AMBER_MUTED = "rgba(245,158,11,0.12)" if IS_DARK else "rgba(217,119,6,0.08)"
SHADOW = "none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"

# Inject Custom CSS Design System
st.markdown(f"""
<style>
    /* Global Variables */
    :root {{
        --bg: {BG};
        --bg-subtle: {BG_SUBTLE};
        --card: {CARD};
        --card-hover: {CARD_HOVER};
        --border: {BORDER};
        --border-subtle: {BORDER_SUBTLE};
        --text: {TEXT};
        --text-muted: {TEXT_MUTED};
        --accent: {ACCENT};
        --shadow: {SHADOW};
        --radius: 10px;
    }}
    
    /* Hide Streamlit Chrome */
    header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
    div[data-testid="stSidebarCollapsedControl"] {{
        display: none !important;
    }}
    
    /* Layout Overrides */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
    }}
    .block-container {{
        padding: 1.5rem 2.5rem 3rem !important;
        max-width: 1360px !important;
    }}
    
    /* Custom Components */
    .brand {{
        margin-bottom: 1.5rem;
    }}
    .brand-name {{
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, var(--accent) 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .brand-subtitle {{
        font-size: 0.8rem;
        color: var(--text-muted);
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.1rem 1.25rem;
        box-shadow: var(--shadow);
    }}
    .metric-label {{
        font-size: 0.75rem;
        color: var(--text-muted);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .metric-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.03em;
        margin-top: 0.2rem;
    }}
    .metric-delta {{
        font-size: 0.72rem;
        font-weight: 500;
        margin-top: 0.3rem;
        padding: 1px 6px;
        border-radius: 5px;
        display: inline-flex;
        align-items: center;
        gap: 3px;
    }}
    .delta-up {{ color: {GREEN}; background: {GREEN_MUTED}; }}
    .delta-down {{ color: {RED}; background: {RED_MUTED}; }}
    .delta-warn {{ color: {AMBER}; background: {AMBER_MUTED}; }}
    
    /* Tabs Customization */
    button[data-baseweb="tab"] {{
        background: transparent !important;
        color: var(--text-muted) !important;
        font-size: 0.835rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 1rem !important;
        border: 1px solid transparent !important;
        border-radius: 7px !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--text) !important;
        background: var(--card) !important;
        border-color: var(--border) !important;
    }}
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
        display: none !important;
    }}
    [data-baseweb="tab-list"] {{
        gap: 4px !important;
        background: var(--bg-subtle) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 3px;
    }}
    
    /* Chart and Containers */
    .chart-wrap {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.2rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.25rem;
    }}
    .chart-title {{ font-size: 0.85rem; font-weight: 600; color: var(--text); }}
    .chart-subtitle {{ font-size: 0.72rem; color: var(--text-muted); margin-bottom: 0.8rem; }}
    
    /* Cards for retrieved chunks */
    .chunk-card {{
        background: var(--bg-subtle);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.9rem;
        margin-bottom: 0.8rem;
    }}
    .chunk-header {{
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-bottom: 0.4rem;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.2rem;
    }}
    .chunk-source {{ font-weight: 600; color: var(--accent); }}
    .chunk-text {{
        font-size: 0.8rem;
        line-height: 1.4;
        color: var(--text);
    }}
    
    /* Custom Data Tables */
    .data-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }}
    .data-table th {{
        text-align: left;
        padding: 0.6rem 0.8rem;
        color: var(--text-muted);
        font-weight: 500;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border-bottom: 1px solid var(--border);
    }}
    .data-table td {{
        padding: 0.65rem 0.8rem;
        color: var(--text);
        border-bottom: 1px solid var(--border-subtle);
    }}
    .data-table tr:last-child td {{
        border-bottom: none;
    }}
    
    /* Citations list */
    .citation-tag {{
        display: inline-block;
        padding: 2px 8px;
        background: rgba(37, 99, 235, 0.1);
        border: 1px solid rgba(37, 99, 235, 0.2);
        color: var(--accent);
        border-radius: 5px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        text-decoration: none;
    }}
    .citation-tag:hover {{
        background: rgba(37, 99, 235, 0.2);
    }}
</style>
""", unsafe_allow_html=True)

# 3. Header with Brand & Theme Toggle
head_left, head_right = st.columns([6, 1])
with head_left:
    st.markdown("""
    <div class="brand">
        <div class="brand-name">◆ RAG Evaluation & Playground Platform</div>
        <div class="brand-subtitle">Production-grade evaluations, semantic search diagnostics, and parameter tuning</div>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    theme_label = "☀️ Light" if IS_DARK else "🌙 Dark"
    st.button(theme_label, on_click=toggle_theme, use_container_width=True)

# Plotly styling based on theme
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#fafafa" if IS_DARK else "#09090b", size=11),
    margin=dict(l=40, r=20, t=30, b=30),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.06)" if IS_DARK else "rgba(0,0,0,0.06)",
        zerolinecolor="rgba(255,255,255,0.06)" if IS_DARK else "rgba(0,0,0,0.06)",
        tickfont=dict(size=10, color="#71717a"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.06)" if IS_DARK else "rgba(0,0,0,0.06)",
        zerolinecolor="rgba(255,255,255,0.06)" if IS_DARK else "rgba(0,0,0,0.06)",
        tickfont=dict(size=10, color="#71717a"),
    ),
)

# 4. Helper UI Functions
def render_metric_card(label, value, delta=None, delta_type="up"):
    cls = f"delta-{delta_type}"
    arrow = "↑" if delta_type == "up" else ("↓" if delta_type == "down" else "→")
    delta_html = f'<div class="metric-delta {cls}">{arrow} {delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# Navigation via Tabs
tab_chat, tab_eval, tab_corpus = st.tabs([
    "💬 RAG Chat Playground", 
    "📈 Evaluation Report Card", 
    "📁 Corpus & Database Management"
])

# ==========================================
# TAB 1: RAG CHAT PLAYGROUND
# ==========================================
with tab_chat:
    st.info("⚡ **Demo Notice**: This application is hosted on Google Cloud Run free-tier. If the first query takes 10-15 seconds, the container is cold-starting. Subsequent operations respond in real-time.")
    
    col_ctrl, col_main = st.columns([1, 3])
    
    with col_ctrl:
        st.subheader("Configuration")
        mode = st.radio("Pipeline Configuration", ["improved", "baseline"], help="Baseline: 500 char chunks, no rewrite, no re-rank. Improved: 1000 char chunks, query rewriting, and LLM re-ranking.")
        
        st.subheader("Sample Queries")
        sample_queries = [
            "-- Select a pre-defined sample --",
            "Who is known as the father of information theory?",
            "How did Alan Turing's theoretical machine impact the concept of computability, and how does the halting problem define its limits?",
            "How do Docker and Kubernetes collaborate in a microservices architecture?",
            "What is the difference between TCP and UDP?",
            "What is the recipe for baking a chocolate chip cookie?",
            "How do you implement caching in a software application?"
        ]
        selected_sample = st.selectbox("Quick Select", sample_queries)
        
    with col_main:
        # Build playground logic
        st.subheader("Query Sandbox")
        
        # Initialize session state for user query
        if "user_query" not in st.session_state:
            st.session_state.user_query = ""
            
        # Quick sample click buttons
        st.markdown("**Quick Demo Queries (Click to try instantly):**")
        btn_cols = st.columns(3)
        if btn_cols[0].button("🟢 Claude Shannon (Direct QA)", use_container_width=True):
            st.session_state.user_query = "Who is known as the father of information theory?"
        if btn_cols[1].button("🟡 Microservices (Multi-hop QA)", use_container_width=True):
            st.session_state.user_query = "How do Docker and Kubernetes collaborate in a microservices architecture?"
        if btn_cols[2].button("🔴 Cookie Recipe (Out of Domain)", use_container_width=True):
            st.session_state.user_query = "What is the recipe for baking a chocolate chip cookie?"
            
        # Update session state if selected from selectbox
        if selected_sample != "-- Select a pre-defined sample --":
            st.session_state.user_query = selected_sample
            
        user_query = st.text_input(
            "Ask a question about the corpus:", 
            value=st.session_state.user_query, 
            placeholder="e.g. What is the P vs NP problem?",
            max_chars=1000
        )
        
        if st.button("Query RAG System", type="primary") and user_query:
            with st.spinner("Retrieving, re-ranking, and generating answer..."):
                try:
                    pipeline = RAGPipeline(mode=mode)
                    result = pipeline.query(user_query)
                    
                    # 1. Answer Card
                    st.markdown("### Answer")
                    st.info(result["answer"])
                    
                    # 2. Metrics row
                    st.markdown("### Transaction Performance")
                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                        render_metric_card("Total Latency", f"{result['metrics']['total_latency']:.3f}s", "API execution", "warn")
                    with mc2:
                        total_toks = result["metrics"]["input_tokens"] + result["metrics"]["output_tokens"]
                        render_metric_card("Token Consumption", f"{total_toks} tokens", f"in: {result['metrics']['input_tokens']} | out: {result['metrics']['output_tokens']}", "up")
                    with mc3:
                        render_metric_card("Estimated Cost", f"${result['metrics']['total_cost']:.6f}" if "total_cost" in result["metrics"] else f"${result['metrics']['cost']:.6f}", "Gemini API pricing", "up")
                    
                    # 3. Query rewrite (if improved)
                    if mode == "improved" and result.get("rewritten_query") and result["rewritten_query"] != user_query:
                        st.markdown("#### Optimized Search Query (Query Rewriter)")
                        st.code(result["rewritten_query"])
                        
                    # 4. Citations
                    st.markdown("### Source Citations")
                    if result["citations"]:
                        citations_html = ""
                        for src in result["citations"]:
                            citations_html += f'<a class="citation-tag" href="#">📄 {src}</a>'
                        st.markdown(citations_html, unsafe_allow_html=True)
                    else:
                        st.write("No source documents cited in this answer.")
                        
                    # 5. Retrieved Chunks details
                    st.markdown("### Retrieved Chunks (Hybrid Search & Re-ranked)")
                    for idx, chunk in enumerate(result["retrieved_chunks"]):
                        with st.expander(f"Chunk [{idx}] — Source: {chunk['source']} | RRF Rank: {chunk.get('rank', idx+1)}"):
                            score_label = f"Score: {chunk['score']:.4f}" if "score" in chunk else f"RRF Score: {chunk.get('rrf_score', 0):.4f}"
                            st.markdown(f"""
                            <div class="chunk-card">
                                <div class="chunk-header">
                                    <span class="chunk-source">{chunk['source']}</span>
                                    <span>{score_label}</span>
                                </div>
                                <div class="chunk-text">{chunk['text']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                except Exception as e:
                    st.error(f"Error querying RAG system: {e}")

# ==========================================
# TAB 2: EVALUATION REPORT CARD
# ==========================================
with tab_eval:
    st.subheader("Interactive Evaluation Report Card")
    
    report_path = DATA_DIR / "evaluation_report.json"
    if not report_path.exists():
        st.warning("No evaluation report found. Please run the evaluation runner first, or go to the Management tab to trigger index building and evaluation.")
    else:
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
            
        base_sum = report_data["baseline_summary"]
        imp_sum = report_data["improved_summary"]
        st.write("BASE SUMMARY")
        st.json(base_sum)

        st.write("IMPROVED SUMMARY")
        st.json(imp_sum)

        st.stop()
        
        # Overall comparison row
        st.markdown("### Overall System Comparison")
        oc1, oc2, oc3, oc4, oc5 = st.columns(5)
        
        with oc1:
            imp_precision = imp_sum.get("overall", {}).get("precision")
            base_precision = base_sum.get("overall", {}).get("precision")

            if imp_precision is not None and base_precision is not None:
                p_diff = imp_precision - base_precision
                delta_str = f"{p_diff*100:+.1f}%"
                dtype = "up" if p_diff >= 0 else "down"

                render_metric_card(
                    "Precision @ 5",
                    f"{imp_precision*100:.1f}%",
                    f"Baseline: {base_precision*100:.1f}% ({delta_str})",
                    dtype,
            )
            else:
                st.error("Precision metric not found in evaluation results.")
                st.write("Improved Summary")
                st.json(imp_sum)
                st.write("Baseline Summary")
                st.json(base_sum)
            
        with oc2:
            imp_recall = imp_sum.get("overall", {}).get("recall")
            base_recall = base_sum.get("overall", {}).get("recall")

            if imp_recall is not None and base_recall is not None:
                r_diff = imp_recall - base_recall
                delta_str = f"{r_diff*100:+.1f}%"
                dtype = "up" if r_diff >= 0 else "down"

                render_metric_card(
                    "Recall @ 5",
                    f"{imp_recall*100:.1f}%",
                    f"Baseline: {base_recall*100:.1f}% ({delta_str})",
                    dtype,
            )
            else:
                st.error("Recall metric not found.")


        with oc3:
            imp_faith = imp_sum.get("overall", {}).get("faithfulness")
            base_faith = base_sum.get("overall", {}).get("faithfulness")

            if imp_faith is not None and base_faith is not None:
               f_diff = imp_faith - base_faith
               delta_str = f"{f_diff*100:+.1f}%"
               dtype = "up" if f_diff >= 0 else "down"

               render_metric_card(
                   "Faithfulness (Judge)",
                   f"{imp_faith*100:.1f}%",
                   f"Baseline: {base_faith*100:.1f}% ({delta_str})",
                   dtype,
            )
            else:
                st.error("Faithfulness metric not found.")


        with oc4:
            imp_rel = imp_sum.get("overall", {}).get("relevance")
            base_rel = base_sum.get("overall", {}).get("relevance")

            if imp_rel is not None and base_rel is not None:
               rel_diff = imp_rel - base_rel
               delta_str = f"{rel_diff:+.2f}"
               dtype = "up" if rel_diff >= 0 else "down"

               render_metric_card(
                   "Relevance (1-5)",
                   f"{imp_rel:.2f}/5",
                   f"Baseline: {base_rel:.2f}/5 ({delta_str})",
                   dtype,
            )
            else:
                st.error("Relevance metric not found.")


        with oc5:
            imp_latency = imp_sum.get("overall", {}).get("latency")
            base_latency = base_sum.get("overall", {}).get("latency")

            if imp_latency is not None and base_latency is not None:
               l_diff = imp_latency - base_latency
               delta_str = f"{l_diff:+.3f}s"
               dtype = "down" if l_diff > 0 else "up"

               render_metric_card(
                   "Avg Latency",
                   f"{imp_latency:.3f}s",
                   f"Baseline: {base_latency:.3f}s ({delta_str})",
                   dtype,
            )
            else:
                st.error("Latency metric not found.")
                st.markdown("### Metrics Segmentation by Question Category")
        
        ccategories = list(base_sum.keys())
        categories_display = categories
        # 1. Retrieval Performance Plot
        st.markdown('<div class="chart-wrap"><div class="chart-title">Retrieval Precision and Recall Comparison</div><div class="chart-subtitle">Comparing baseline retriever vs re-ranked improved retriever across query types</div>', unsafe_allow_html=True)
        
        fig_ret = go.Figure()
        
        # Precision
        fig_ret.add_trace(go.Bar(
            name='Baseline Precision',
            x=categories_display,
            y=[
                base_sum.get(cat, {}).get("precision", 0)
                for cat in categories
            ]
            marker_color='#ef4444'
        ))
        fig_ret.add_trace(go.Bar(
            name='Improved Precision',
            x=categories_display,
            y=[
                imp_sum.get(cat, {}).get("precision", 0)
                for cat in categories
            ]
            marker_color='#22c55e'
        ))
        
        # Recall
        fig_ret.add_trace(go.Bar(
            name='Baseline Recall',
            x=categories_display,
            y=[
                base_sum.get(cat, {}).get("recall", 0)
                for cat in categories
            ]
            marker_color='#f87171'
        ))
        fig_ret.add_trace(go.Bar(
            name='Improved Recall',
            x=categories_display,
            y=[
                imp_sum.get(cat, {}).get("recall", 0)
                for cat in categories
            ]
            marker_color='#4ade80'
        ))
        
        fig_ret.update_layout(barmode='group', height=400, **PLOT_LAYOUT)
        st.plotly_chart(fig_ret, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 2. Generation Performance Plot
        st.markdown('<div class="chart-wrap"><div class="chart-title">Generation Groundedness & Relevance Score</div><div class="chart-subtitle">Faithfulness rate and Relevance ratings by category</div>', unsafe_allow_html=True)
        
        fig_gen = go.Figure()
        fig_gen.add_trace(go.Bar(
            name='Baseline Faithfulness',
            x=categories_display,
            y=[
                base_sum.get(cat, {}).get("faithfulness", 0)
                for cat in categories
            ]
            marker_color='#ef4444'
        ))
        fig_gen.add_trace(go.Bar(
            name='Improved Faithfulness',
            x=categories_display,
            y=[
                imp_sum.get(cat, {}).get("faithfulness", 0)
                for cat in categories
        ]
            marker_color='#22c55e'
        ))
        fig_gen.add_trace(go.Bar(
            name='Baseline Relevance (Normalised)',
            x=categories_display,
            # Normalize 1-5 score to 0-1 for plotting consistency
           y=[
               (base_sum.get(cat, {}).get("relevance", 1)-1)/4
               for cat in categories
            ]
            marker_color='#3b82f6'
        ))
        fig_gen.add_trace(go.Bar(
            name='Improved Relevance (Normalised)',
            x=categories_display,
            y=[
                (imp_sum.get(cat, {}).get("relevance", 1)-1)/4
                for cat in categories
            ]
            marker_color='#60a5fa'
        ))
        
        fig_gen.update_layout(barmode='group', height=400, **PLOT_LAYOUT)
        st.plotly_chart(fig_gen, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Judge Validation Details
        st.markdown("### LLM-as-a-Judge Validation against Human Benchmarks")
        st.write(f"The LLM-as-a-judge (evaluator) agreement rate with manual labels is **{report_data['judge_validation']['accuracy']*100:.1f}%**.")
        
        val_rows = ""
        for item in report_data["judge_validation"]["results"]:
            agr_badge = '<span class="badge badge-green">Agree</span>' if item["agreement"] else '<span class="badge badge-red">Disagree</span>'
            val_rows += f"""
            <tr>
                <td>{item['type']}</td>
                <td>{item['query']}</td>
                <td>{item['human_label']}</td>
                <td>{item['judge_label']} (Score: {item['judge_score']:.2f})</td>
                <td>{agr_badge}</td>
            </tr>
            """
            
        st.markdown(f"""
        <table class="data-table">
            <thead>
                <tr>
                    <th>Type</th>
                    <th>Query</th>
                    <th>Human Label</th>
                    <th>LLM Judge Evaluation</th>
                    <th>Agreement</th>
                </tr>
            </thead>
            <tbody>
                {val_rows}
            </tbody>
        </table>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 3: CORPUS & DATABASE MANAGEMENT
# ==========================================
with tab_corpus:
    st.subheader("Document Corpus Registry")
    
    # 1. Corpus stats
    files = list(CORPUS_DIR.glob("*.txt"))
    st.write(f"Current document count in corpus folder: **{len(files)}** files.")
    
    col_up, col_actions = st.columns(2)
    
    with col_up:
        st.markdown("### Upload New Document")
        st.warning("⚠️ **Notice**: Documents uploaded here are saved ephemerally. They will be cleared when the container scales down or restarts.")
        up_file = st.file_uploader("Upload a text document (.txt):", type=["txt"])
        
        # Enforce size limits and basic text validation
        is_file_valid = True
        if up_file is not None:
            file_bytes = up_file.getvalue()
            file_size_kb = len(file_bytes) / 1024.0
            if file_size_kb > 200.0:
                st.error(f"❌ File size ({file_size_kb:.1f}KB) exceeds the 200KB limit for public playground uploads.")
                is_file_valid = False
            else:
                try:
                    # Verify it decodes cleanly to plain text
                    file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    st.error("❌ The uploaded file is not a valid UTF-8 text file.")
                    is_file_valid = False

        doc_title = st.text_input("Enter Document Title:", placeholder="e.g. Backpropagation Algorithm", max_chars=100)
        
        if st.button("Add to Corpus") and up_file and doc_title and is_file_valid:
            try:
                # Sanitize filename
                filename = re.sub(r'[\\/*?:"<>|]', "", doc_title).replace(" ", "_") + ".txt"
                filepath = CORPUS_DIR / filename
                
                content = up_file.read().decode("utf-8")
                # Format file content
                formatted_content = f"Title: {doc_title}\nSource: User Upload\n\n{content}"
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(formatted_content)
                    
                st.success(f"Successfully added document to corpus: '{doc_title}' ({len(content)} chars)")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to add document: {e}")
                
    with col_actions:
        st.markdown("### Index & Evaluation Rebuilder")
        st.write("Clicking the button below will trigger a complete rebuild of the Qdrant and BM25 baseline and improved indices from the current corpus files. Once built, it will execute the evaluation suite (20 queries) and rewrite the report card.")
        
        eval_sample_size = st.slider("Evaluation Sample Size", min_value=4, max_value=100, value=20, step=4, help="Select number of questions to evaluate (should be a multiple of 4)")
        
        if st.button("Rebuild Indexes & Re-Run Evaluation", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 1. Build Baseline Index
                status_text.text("Building Baseline Index (Chunk size: 500)...")
                progress_bar.progress(10)
                baseline_retriever = HybridRetriever(mode="baseline")
                baseline_retriever.build_index()
                
                # 2. Build Improved Index
                status_text.text("Building Improved Index (Chunk size: 1000)...")
                progress_bar.progress(40)
                improved_retriever = HybridRetriever(mode="improved")
                improved_retriever.build_index()
                
                # 3. Running Evaluator
                status_text.text("Executing evaluation suite queries...")
                progress_bar.progress(70)
                
                # Run the run_evaluation main programmatically
                import subprocess
                cmd = f"uv run python run_evaluation.py --skip-indexing --sample-size {eval_sample_size}"
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if res.returncode == 0:
                    status_text.text("Rebuilding complete!")
                    progress_bar.progress(100)
                    st.success("Successfully rebuilt database indexes and re-evaluated RAG metrics!")
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.error(f"Evaluation script failed: {res.stderr}")
                    
            except Exception as e:
                st.error(f"Rebuild failed: {e}")
                
    # 2. List corpus files
    st.markdown("### Current Documents List")
    
    file_rows = ""
    for idx, f in enumerate(files[:50]): # Display top 50
        size_kb = f.stat().st_size / 1024
        file_rows += f"""
        <tr>
            <td>{idx+1}</td>
            <td>{f.name}</td>
            <td>{size_kb:.2f} KB</td>
            <td><a href="https://en.wikipedia.org/wiki/{quote(f.name.replace('.txt','').replace('_',' '))}" target="_blank">View Wiki</a></td>
        </tr>
        """
        
    st.markdown(f"""
    <div style="max-height: 400px; overflow-y: auto;">
        <table class="data-table">
            <thead>
                <tr>
                    <th>Index</th>
                    <th>Filename</th>
                    <th>Size (KB)</th>
                    <th>Wikipedia Source</th>
                </tr>
            </thead>
            <tbody>
                {file_rows}
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

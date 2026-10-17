"""TwinMind visual theme: dark hero header, mint page background, gradient accents.

app.py calls inject_theme() (CSS) and render_hero() (header). Numbers in the
hero chips are read from the saved result files, never typed in.
"""

import base64
import json
from functools import lru_cache
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"


@lru_cache(maxsize=None)
def _data_uri(name):
    try:
        raw = (ASSETS / name).read_bytes()
    except OSError:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def _read_json(*parts):
    try:
        with open(HERE.joinpath(*parts), "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
:root { color-scheme: light !important; }
html, body, .stApp, h1, h2, h3, h4, p, li, label, input, textarea,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { font-family: 'Poppins', 'Segoe UI', Arial, sans-serif; }
.stApp { background: linear-gradient(180deg, #e8f2f4 0%, #dde9ec 100%) fixed !important; color: #1e293b !important; }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: transparent !important; }
[data-testid="stHeader"] { background: rgba(255, 255, 255, 0.75) !important; }
.block-container { max-width: 1600px; padding-top: 1.5rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #cfdde2 !important; }
[data-testid="stSidebar"]::before { content: ""; display: block; height: 6px; background: linear-gradient(90deg, #a855f7, #06b6d4); }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span { color: #334155 !important; }
h1, h2, h3, h4 { color: #1e0f55 !important; letter-spacing: -0.01em; font-weight: 700; }
p, li, label { color: #334155; }
[data-testid="stCaptionContainer"] { color: #475569 !important; }
[data-testid="stMetric"] { background: #ffffff !important; border: 1px solid #dbe5ea !important; border-radius: 24px; padding: 20px 24px; box-shadow: 0 8px 24px rgba(30, 15, 85, 0.08); }
[data-testid="stMetricLabel"] { color: #64748b !important; }
[data-testid="stMetricValue"] { color: #5b1ae8 !important; font-weight: 800; }
div[data-testid="stAlert"] { border-radius: 16px; }
.stButton > button, [data-testid="stDownloadButton"] > button { background: #ffffff !important; color: #5b1ae8 !important; border: 1px solid #c9bdf5 !important; border-radius: 999px; font-weight: 600; transition: all 0.2s ease; }
.stButton > button p, [data-testid="stDownloadButton"] > button p { color: inherit !important; }
.stButton > button[kind="primary"] { background: linear-gradient(90deg, #8b5cf6, #06b6d4) !important; color: #ffffff !important; border: none !important; }
.stButton > button:hover, [data-testid="stDownloadButton"] > button:hover { border-color: #5b1ae8 !important; box-shadow: 0 4px 14px rgba(91, 26, 232, 0.18); }
.stButton > button:focus-visible, [data-testid="stDownloadButton"] > button:focus-visible,
[data-baseweb="select"] > div:focus-within, [data-baseweb="input"] > div:focus-within { outline: 2px solid #5b1ae8; outline-offset: 2px; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] { background: #ffffff !important; border-color: #cbd5e1 !important; border-radius: 12px; color: #1e293b !important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid #c3d3d9; }
[data-testid="stTabs"] button[data-baseweb="tab"] { color: #475569 !important; padding: 10px 9px; font-size: 0.9rem; }
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] { color: #5b1ae8 !important; border-bottom: 3px solid #5b1ae8 !important; background: #ffffff !important; border-radius: 12px 12px 0 0; font-weight: 600; }
[data-testid="stTabs"] button[data-baseweb="tab"]:hover { color: #5b1ae8 !important; }
[data-testid="stExpander"] { background: #ffffff !important; border: 1px solid #dbe5ea !important; border-radius: 16px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { background: #ffffff !important; color: #334155 !important; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"], [data-testid="stTable"] { background: #ffffff !important; border: 1px solid #dbe5ea !important; border-radius: 20px; overflow: hidden; }
[data-testid="stTable"] table, [data-testid="stTable"] th, [data-testid="stTable"] td { background: #ffffff !important; color: #334155 !important; border-color: #e2e8f0 !important; }
hr { border-color: #c3d3d9 !important; opacity: 1; }
.tm-hero { position: relative; overflow: hidden; border-radius: 32px; padding: 56px 56px 40px; margin: 0.4rem 0 1.2rem; color: #ffffff; background: linear-gradient(90deg, #0d1117 30%, rgba(13, 17, 23, 0.25) 100%), __HERO__ #0d1117; box-shadow: 0 18px 48px rgba(13, 27, 36, 0.25); }
.tm-eyebrow { color: #22d3ee !important; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 10px; }
.tm-title { color: #ffffff !important; font-size: clamp(2.4rem, 5vw, 4.4rem); font-weight: 800; line-height: 1.05; margin: 0 0 12px; }
.tm-grad { background: linear-gradient(90deg, #a855f7, #06b6d4); -webkit-background-clip: text; background-clip: text; color: transparent !important; -webkit-text-fill-color: transparent; }
.tm-subtitle { color: #e2e8f0 !important; font-size: 1.05rem; line-height: 1.6; max-width: 640px; margin: 0 0 22px; }
.tm-line { height: 3px; width: 240px; border-radius: 2px; background: linear-gradient(90deg, #a855f7, #06b6d4); margin-bottom: 22px; }
.tm-chips { display: flex; flex-wrap: wrap; gap: 14px; }
.tm-chip { background: rgba(17, 27, 46, 0.85); border: 1px solid #22304a; border-radius: 16px; padding: 12px 20px; }
.tm-chip b { display: block; color: #22d3ee; font-size: 1.6rem; font-weight: 800; line-height: 1.1; }
.tm-chip span { color: #cbd5e1; font-size: 0.85rem; }
.tm-disclaimer { color: #475569 !important; background: #ffffff !important; border: 1px solid #cfdde2 !important; border-left: 5px solid #5b1ae8 !important; border-radius: 14px; padding: 12px 16px; margin: 0.5rem 0 1.5rem; font-size: 0.88rem; line-height: 1.55; }
.st-key-section_nav { background: #ffffff; border: 1px solid #dbe5ea; border-radius: 20px; padding: 10px 14px; box-shadow: 0 6px 18px rgba(30, 15, 85, 0.06); margin-bottom: 0.6rem; }
.st-key-section_nav [role="radiogroup"] { gap: 6px; flex-wrap: wrap; }
.st-key-section_nav label[data-baseweb="radio"] { background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 999px; padding: 6px 12px; margin: 0; cursor: pointer; }
.st-key-section_nav label[data-baseweb="radio"] > div:first-child { display: none; }
.st-key-section_nav label[data-baseweb="radio"] p { font-size: 0.85rem; font-weight: 500; }
.st-key-section_nav label[data-baseweb="radio"]:hover { border-color: #5b1ae8; }
.st-key-section_nav label[data-baseweb="radio"]:has(input:checked) { background: linear-gradient(90deg, #8b5cf6, #06b6d4); border-color: transparent; }
.st-key-section_nav label[data-baseweb="radio"]:has(input:checked) * { color: #ffffff !important; }
@media (max-width: 768px) { .block-container { padding: 1rem; } .tm-hero { padding: 28px 22px; } }
"""


def inject_theme():
    """Apply the TwinMind CSS and chart colors."""
    hero = _data_uri("twinmind_hero.jpg")
    css = _CSS.replace(
        "__HERO__", f"url({hero}) right center / cover no-repeat," if hero else ""
    )
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    try:  # chart colors follow the deck palette
        import plotly.io as pio

        pio.templates["twinmind"].layout.colorway = [
            "#5b1ae8", "#06b6d4", "#a855f7", "#3b82f6", "#f59e0b", "#64748b",
        ]
    except Exception:  # noqa: BLE001
        pass


def render_hero():
    """Header with headline numbers read from the saved result files."""
    chips = []
    timing = _read_json("reports", "amd_gpu", "temporal_inference_timing.json")
    if timing and timing.get("best_cpu_over_gpu"):
        chips.append((f"{timing['best_cpu_over_gpu']:.1f}x", "GPU vs tuned CPU (temporal model)"))
    meta = _read_json("reports", "temporal_gpu", "experiment_metadata.json")
    if meta and meta.get("accuracy") is not None:
        chips.append((f"{meta['accuracy'] * 100:.1f}%", "holdout accuracy (temporal model)"))
    if meta and meta.get("classes"):
        chips.append((str(len(meta["classes"])), "motor conditions classified"))
    chip_html = "".join(f'<div class="tm-chip"><b>{n}</b><span>{label}</span></div>' for n, label in chips)

    html = (
        '<section class="tm-hero">'
        '<div class="tm-eyebrow">Industrial Intelligence / Digital Twin</div>'
        '<div class="tm-title">TwinMind <span class="tm-grad">AI</span></div>'
        '<p class="tm-subtitle">Motor condition monitoring, sensor exploration, fault '
        "classification and historical-data analysis, with AMD GPU-accelerated screening and a PyTorch temporal model.</p>"
        '<div class="tm-line"></div>'
        f'<div class="tm-chips">{chip_html}</div>'
        "</section>"
        '<div class="tm-disclaimer"><strong>Prototype notice:</strong> This application '
        "analyzes historical sensor recordings and estimates condition classes. It is not "
        "connected to a live motor and does not independently confirm physical faults or "
        "provide safety-critical predictive maintenance.</div>"
    )
    st.markdown(html, unsafe_allow_html=True)
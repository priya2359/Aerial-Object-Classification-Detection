# filename: streamlit_app/app.py
# purpose:  Main Streamlit entry point — sidebar health status + 3 prediction/analytics tabs
# version:  1.0

# stdlib
import os

# third-party
import requests
import streamlit as st

# internal
from components import analytics, batch_predictor, predictor

# ── Config ────────────────────────────────────────────────────────────────────
# In Docker: FASTAPI_URL=http://fastapi:8000 (service name, never localhost)
# Local dev:  FASTAPI_URL=http://localhost:8000
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000").rstrip("/")

st.set_page_config(
    page_title="Aerial Object Detection",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛸 Aerial Detection")
    st.caption("Bird vs Drone · Production System")

    st.divider()

    # Live health check
    st.subheader("API Status")
    try:
        health = requests.get(f"{FASTAPI_URL}/health", timeout=5).json()
        api_ok = health.get("status") == "ok"
    except Exception:
        health = {}
        api_ok = False

    if api_ok:
        st.success("FastAPI connected")
    else:
        st.error("FastAPI unreachable")
        st.caption(f"Expected: `{FASTAPI_URL}`")

    if health:
        redis_ok = health.get("redis") == "connected"
        pg_ok    = health.get("postgres") == "connected"
        drift_ok = health.get("drift") == "enabled"

        col1, col2 = st.columns(2)
        col1.markdown("Redis")
        col2.markdown("🟢" if redis_ok else "🔴")
        col1.markdown("Postgres")
        col2.markdown("🟢" if pg_ok else "🔴")
        col1.markdown("Drift")
        col2.markdown("🟢" if drift_ok else "🔴")

        st.divider()
        st.caption(f"Model: **{health.get('model_type', 'n/a')}**")
        st.caption(f"Version: `{health.get('model_version', 'n/a')}`")
        st.caption(f"Threshold: `{health.get('threshold', 'n/a')}`")
        st.caption(f"Uptime: {health.get('uptime_s', 0):.0f}s")

    st.divider()

    # Recent session predictions (last 5)
    st.subheader("Recent Predictions")
    history = st.session_state.get("history", [])
    if not history:
        st.caption("No predictions yet.")
    else:
        for item in history[:5]:
            icon = "🚨" if item["is_alert"] else ("🤖" if item["label"] == "drone" else "🐦")
            label_str = item["label"].upper()
            conf_str  = f"{item['confidence']:.1%}"
            st.markdown(f"{icon} `{label_str}` · {conf_str}")
            st.caption(item["filename"])

    st.divider()
    st.caption("Aerial Detection System  ·  v1.0")
    st.caption(f"API: `{FASTAPI_URL}`")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single Predict", "📦 Batch Predict", "📊 Analytics"])

with tab1:
    predictor.render(FASTAPI_URL)

with tab2:
    batch_predictor.render(FASTAPI_URL)

with tab3:
    analytics.render(FASTAPI_URL)

# filename: streamlit_app/components/analytics.py
# purpose:  Tab 3 — model metrics, system health, drift report, session history, hot-reload
# version:  1.0

# third-party
import plotly.graph_objects as go
import requests
import streamlit as st


def _gauge(value: float, title: str, max_val: float = 1.0) -> go.Figure:
    """Plotly gauge — green < 0.15, yellow < 0.30, red ≥ 0.30 (matches alert_rules.yml)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        number={"suffix": "", "valueformat": ".4f"},
        gauge={
            "axis": {"range": [0, max_val]},
            "bar":  {"color": "#2c3e50"},
            "steps": [
                {"range": [0, 0.15],   "color": "#2ecc71"},
                {"range": [0.15, 0.3], "color": "#f39c12"},
                {"range": [0.3, max_val], "color": "#e74c3c"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.75,
                "value": 0.3,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def _get(url: str, timeout: int = 10) -> dict | None:
    """GET request with silent error handling — returns None on failure."""
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _post(url: str, timeout: int = 30) -> dict | None:
    try:
        r = requests.post(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def render(fastapi_url: str) -> None:
    """Render the Analytics tab — model info, health, drift, session history, admin."""
    st.header("System Analytics")

    # ── Section 1: Model Performance ──────────────────────────────────────────
    st.subheader("Model Performance")
    info = _get(f"{fastapi_url}/model/info")

    if info is None or info.get("metrics") == "unavailable":
        st.warning("Model metrics not available (metrics JSON not found on server).")
    elif isinstance(info, dict) and "metrics" in info and isinstance(info["metrics"], dict):
        m = info["metrics"]
        mt = info.get("model_name", info.get("model_type", "unknown"))
        st.caption(f"Model: **{mt}**  ·  Optimal threshold (Youden's J): `{info.get('optimal_threshold', {}).get('threshold', 'n/a')}`")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{m.get('accuracy',  0):.4f}")
        c2.metric("F1 Score",  f"{m.get('f1',        0):.4f}")
        c3.metric("Precision", f"{m.get('precision', 0):.4f}")
        c4.metric("Recall",    f"{m.get('recall',    0):.4f}")

        c5, c6 = st.columns(2)
        c5.metric("AUC-ROC",       f"{m.get('auc_roc', 0):.4f}")
        c6.metric("Inference (ms)", f"{m.get('inference_time_ms', 'n/a')}")

        # Training info if available
        if "training" in info:
            tr = info["training"]
            with st.expander("Training details"):
                st.json(tr)
    else:
        st.json(info)

    st.divider()

    # ── Section 2: System Health ───────────────────────────────────────────────
    st.subheader("System Health")
    health = _get(f"{fastapi_url}/health")

    if health is None:
        st.error("FastAPI is unreachable.")
    else:
        def _dot(status: str) -> str:
            ok_values = {"ok", "connected", "enabled"}
            return "🟢" if str(status).lower() in ok_values else "🔴"

        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("API",      _dot(health.get("status", "")) + " " + health.get("status", "unknown"))
        h2.metric("Redis",    _dot(health.get("redis", ""))  + " " + health.get("redis", "unknown"))
        h3.metric("Postgres", _dot(health.get("postgres", "")) + " " + health.get("postgres", "unknown"))
        h4.metric("Drift",    _dot(health.get("drift", ""))  + " " + health.get("drift", "unknown"))
        h5.metric("Uptime",   f"{health.get('uptime_s', 0):.0f}s")

        st.caption(
            f"Model: **{health.get('model_type', 'n/a')}**  ·  "
            f"Version: `{health.get('model_version', 'n/a')}`  ·  "
            f"Embedding dim: {health.get('embedding_dim', 'n/a')}  ·  "
            f"Threshold: {health.get('threshold', 'n/a')}"
        )

    st.divider()

    # ── Section 3: Data Drift Report ───────────────────────────────────────────
    st.subheader("Data Drift Monitor")
    st.caption("Compares recent prediction embeddings against the EDA reference distribution (200 EfficientNetB0 embeddings).")

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_drift = st.button("Refresh Drift Report", type="secondary")

    if run_drift or "drift_report" in st.session_state:
        if run_drift:
            with st.spinner("Running Evidently DataDriftReport (may take a few seconds)..."):
                drift = _get(f"{fastapi_url}/drift/report", timeout=60)
            st.session_state["drift_report"] = drift
        else:
            drift = st.session_state.get("drift_report")

        if drift is None:
            st.error("Drift report request failed — check API logs.")
        elif drift.get("status") == "disabled":
            st.warning(f"Drift detection disabled: {drift.get('reason', 'see API logs')}")
        elif drift.get("status") == "insufficient_data":
            st.info(
                f"Not enough predictions yet — {drift.get('n_current', 0)} accumulated, "
                f"need {drift.get('required', 50)}. Make more predictions to enable drift detection."
            )
        elif drift.get("status") == "error":
            st.error(f"Drift computation error: {drift.get('error')}")
        else:
            d1, d2 = st.columns([1, 2])
            with d1:
                st.plotly_chart(
                    _gauge(drift.get("drift_score", 0), "Mean KS Drift Score"),
                    use_container_width=True,
                )
            with d2:
                status_colour = "🔴" if drift.get("drift_status") == "warning" else "🟢"
                st.metric("Drift Status", f"{status_colour} {drift.get('drift_status', 'unknown').upper()}")
                st.metric("Drifted PCA Components",
                          f"{drift.get('n_drifted_columns', 0)} / {drift.get('n_total_columns', 50)}")
                st.metric("Share Drifted", f"{drift.get('share_drifted', 0):.1%}")
                st.caption(
                    f"Current buffer: {drift.get('n_current', 0)} predictions  ·  "
                    f"Reference: {drift.get('n_reference', 200)} embeddings"
                )
                if drift.get("generated_at"):
                    st.caption(f"Generated: {drift['generated_at']}")
    else:
        st.info("Click **Refresh Drift Report** to run the Evidently analysis.")

    st.divider()

    # ── Section 4: Session Prediction History ─────────────────────────────────
    st.subheader("Session Prediction History")
    history = st.session_state.get("history", [])

    if not history:
        st.info("No predictions made in this session. Use Tab 1 or Tab 2 to classify images.")
    else:
        import pandas as pd
        hist_df = pd.DataFrame(history)
        hist_df["confidence"] = hist_df["confidence"].map("{:.1%}".format)
        hist_df["is_alert"] = hist_df["is_alert"].map(lambda x: "🚨 YES" if x else "no")
        hist_df["cached"]   = hist_df["cached"].map(lambda x: "⚡" if x else "")
        st.dataframe(
            hist_df[["filename", "label", "confidence", "is_alert", "cached", "inf_ms"]].rename(
                columns={"inf_ms": "ms"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ── Section 5: Admin — Model Reload ───────────────────────────────────────
    st.subheader("Admin")

    with st.expander("Hot-Reload Production Model"):
        st.warning(
            "This reloads `production_model.txt` and the corresponding `.h5` from disk. "
            "Use after copying a new model to the `models/` directory. "
            "In-flight requests are not interrupted."
        )
        confirm = st.checkbox("I understand — reload the model now")
        if confirm and st.button("Reload Model", type="primary"):
            with st.spinner("Reloading..."):
                result = _post(f"{fastapi_url}/model/reload")
            if result is None:
                st.error("Reload failed — check API logs.")
            else:
                st.success(
                    f"Reloaded: **{result.get('model_type')}**  ·  "
                    f"threshold={result.get('threshold')}  ·  "
                    f"time={result.get('reload_time_s', '?')}s"
                )
                # Clear drift cache — new model means different embedding space
                if "drift_report" in st.session_state:
                    del st.session_state["drift_report"]

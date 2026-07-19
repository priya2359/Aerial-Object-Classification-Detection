# filename: streamlit_app/components/batch_predictor.py
# purpose:  Tab 2 — batch upload (≤20 images), POST /batch_predict, table + chart + CSV
# version:  1.0

# stdlib
import io
import time

# third-party
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

BATCH_MAX = 20


def _build_chart(df: pd.DataFrame) -> go.Figure:
    """Plotly horizontal bar chart showing bird vs drone counts and alert breakdown."""
    counts = df["label"].value_counts()
    bird_count  = int(counts.get("bird", 0))
    drone_count = int(counts.get("drone", 0))
    alert_count = int(df["is_alert"].sum())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[bird_count], y=["Prediction"], orientation="h",
        name="Bird", marker_color="#2ecc71", text=[bird_count],
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        x=[drone_count - alert_count], y=["Prediction"], orientation="h",
        name="Drone (no alert)", marker_color="#e67e22", text=[drone_count - alert_count],
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        x=[alert_count], y=["Prediction"], orientation="h",
        name="Drone ALERT", marker_color="#e74c3c", text=[alert_count],
        textposition="auto",
    ))
    fig.update_layout(
        barmode="stack",
        height=200,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        title="Prediction Distribution",
    )
    return fig


def render(fastapi_url: str) -> None:
    """Render the Batch Predict tab. Calls POST /batch_predict with one multipart request."""
    st.header("Batch Prediction")
    st.caption(f"Upload up to {BATCH_MAX} images and classify them in one API call.")

    uploaded_files = st.file_uploader(
        "Choose images (hold Ctrl/Cmd to select multiple)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if not uploaded_files:
        st.info("Upload images above and click **Run Batch** to classify.")
        return

    n = len(uploaded_files)
    if n > BATCH_MAX:
        st.error(f"Too many files: {n} uploaded, maximum is {BATCH_MAX}. Remove {n - BATCH_MAX} images.")
        return

    st.caption(f"{n} image{'s' if n > 1 else ''} selected.")

    if not st.button("Run Batch Predict", type="primary"):
        return

    # ── Call POST /batch_predict ───────────────────────────────────────────────
    with st.spinner(f"Classifying {n} images..."):
        try:
            t0 = time.time()
            # Field name "files" must match `files: list[UploadFile]` in api/main.py
            multipart = [
                ("files", (f.name, f.getvalue(), f.type or "image/jpeg"))
                for f in uploaded_files
            ]
            response = requests.post(
                f"{fastapi_url}/batch_predict",
                files=multipart,
                timeout=120,
            )
            elapsed_ms = round((time.time() - t0) * 1000, 1)
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach FastAPI. Is the API running?")
            return
        except requests.exceptions.Timeout:
            st.error("Batch request timed out after 120s.")
            return

    if response.status_code != 200:
        st.error(f"API error {response.status_code}: {response.text}")
        return

    data        = response.json()
    predictions = data["predictions"]
    total_ms    = data.get("total_time_ms", elapsed_ms)
    n_cached    = data.get("n_cached", 0)
    n_total     = data.get("n_total", n)

    # ── Summary row ───────────────────────────────────────────────────────────
    n_drone  = sum(1 for p in predictions if p["predicted_label"] == "drone")
    n_bird   = n_total - n_drone
    n_alerts = sum(1 for p in predictions if p.get("is_alert"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", n_total)
    c2.metric("Bird", n_bird)
    c3.metric("Drone", n_drone)
    c4.metric("Alerts 🚨", n_alerts)
    c5.metric("Time", f"{total_ms:.0f} ms")

    if n_alerts:
        st.error(f"🚨 {n_alerts} high-confidence drone detection{'s' if n_alerts > 1 else ''} in this batch.")

    # ── Build results DataFrame ────────────────────────────────────────────────
    rows = []
    for i, (pred, f) in enumerate(zip(predictions, uploaded_files)):
        rows.append({
            "filename":   f.name,
            "label":      pred["predicted_label"],
            "confidence": pred["confidence"],
            "is_alert":   pred.get("is_alert", False),
            "cached":     pred.get("cached", False),
            "hash":       pred.get("image_hash", ""),
        })
    df = pd.DataFrame(rows)

    # ── Chart ─────────────────────────────────────────────────────────────────
    st.plotly_chart(_build_chart(df), use_container_width=True)

    # ── Results table ──────────────────────────────────────────────────────────
    st.subheader("Per-Image Results")

    def _style_row(row):
        if row["is_alert"]:
            return ["background-color: #fde8e8"] * len(row)
        if row["label"] == "drone":
            return ["background-color: #fef3e2"] * len(row)
        return [""] * len(row)

    display_df = df[["filename", "label", "confidence", "is_alert", "cached"]].copy()
    display_df["confidence"] = display_df["confidence"].map("{:.1%}".format)
    st.dataframe(
        display_df.style.apply(_style_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # ── Cache stats + CSV download ─────────────────────────────────────────────
    cache_pct = round(n_cached / n_total * 100) if n_total else 0
    st.caption(f"Cache hits: {n_cached}/{n_total} ({cache_pct}%)  ·  Model: {predictions[0].get('model_version', 'v1')}")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results CSV",
        data=csv_bytes,
        file_name="batch_predictions.csv",
        mime="text/csv",
    )

    # ── Append to session history ──────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state["history"] = []
    for row in rows:
        st.session_state["history"].insert(0, {
            "filename":   row["filename"],
            "label":      row["label"],
            "confidence": row["confidence"],
            "is_alert":   row["is_alert"],
            "cached":     row["cached"],
            "inf_ms":     round(total_ms / n_total, 1),
        })
    st.session_state["history"] = st.session_state["history"][:20]

# filename: streamlit_app/components/predictor.py
# purpose:  Tab 1 — single image upload, POST /predict, render result with alert banner
# version:  1.0

# stdlib
import time

# third-party
import requests
import streamlit as st


def _confidence_colour(confidence: float, label: str) -> str:
    if label == "drone" and confidence >= 0.9:
        return "red"
    if label == "drone":
        return "orange"
    return "green"


def render(fastapi_url: str) -> None:
    """Render the Single Predict tab. Calls POST /predict and displays the result."""
    st.header("Single Image Prediction")
    st.caption("Upload one aerial image — the model classifies it as Bird or Drone.")

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="single_uploader",
    )

    if uploaded is None:
        st.info("Upload an image above to get a prediction.")
        return

    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.image(uploaded, caption=uploaded.name, use_container_width=True)

    with col_result:
        with st.spinner("Running inference..."):
            try:
                t0 = time.time()
                response = requests.post(
                    f"{fastapi_url}/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=30,
                )
                elapsed_ms = round((time.time() - t0) * 1000, 1)
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach FastAPI. Is the API running?")
                return
            except requests.exceptions.Timeout:
                st.error("Request timed out after 30s.")
                return

        if response.status_code != 200:
            st.error(f"API error {response.status_code}: {response.text}")
            return

        data = response.json()
        label      = data["predicted_label"]
        confidence = data["confidence"]
        is_alert   = data.get("is_alert", False)
        cached     = data.get("cached", False)
        inf_ms     = data.get("inference_time_ms", elapsed_ms)
        drift_score  = data.get("drift_score", 0.0)
        drift_status = data.get("drift_status", "unknown")

        # ── Security alert banner ──────────────────────────────────────────────
        if is_alert:
            st.error("🚨 DRONE ALERT — High-confidence drone detected (confidence ≥ 0.90)")
        elif label == "drone":
            st.warning("⚠️ Drone detected (below 0.90 alert threshold)")
        else:
            st.success("✅ Bird detected — no security concern")

        # ── Prediction metrics ─────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", label.upper())
        m2.metric("Confidence", f"{confidence:.1%}")
        m3.metric("Inference", f"{inf_ms:.0f} ms")

        # Confidence bar — colour-coded by label
        colour = _confidence_colour(confidence, label)
        st.markdown(f"**Confidence** ({colour})")
        st.progress(confidence)

        # ── Secondary info ─────────────────────────────────────────────────────
        badge_cached = "⚡ Redis cache hit" if cached else "🔄 Model inference"
        st.caption(f"{badge_cached} · Drift score: {drift_score:.4f} ({drift_status})")
        st.caption(f"Hash: `{data.get('image_hash', 'n/a')}`  ·  Model: {data.get('model_type', 'n/a')} {data.get('model_version', '')}")

    # ── Save to session history ────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state["history"] = []

    st.session_state["history"].insert(0, {
        "filename":   uploaded.name,
        "label":      label,
        "confidence": confidence,
        "is_alert":   is_alert,
        "cached":     cached,
        "inf_ms":     inf_ms,
    })
    # Keep only the last 20 in session state
    st.session_state["history"] = st.session_state["history"][:20]

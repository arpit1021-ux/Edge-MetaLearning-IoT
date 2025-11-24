# dashboard/app.py
import streamlit as st
import numpy as np
import requests
import json
import time
import os
import matplotlib.pyplot as plt

# ------------------- CONFIG -------------------
API_PREDICT = "http://127.0.0.1:8000/predict"
API_FEEDBACK = "http://127.0.0.1:8000/feedback"
LOCAL_PPT_PATH = "/mnt/data/SDL_ppt.pptx"

ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

PAGE_NAMES = [
    "Prediction", "Manual Test", "Live Simulation", "Model Info", "Kanban & Export"
]

st.set_page_config(page_title="Edge IoT Meta-Learning Dashboard", layout="wide")

# --------------- Custom CSS for sidebar buttons ---------------
st.markdown("""
<style>
.sidebar-button {
    padding: 12px 20px;
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 16px;
    font-weight: 600;
    width: 100%;
    text-align: left;
    background-color: #2b2b2b;
    color: white;
    border: 1px solid #444;
}
.sidebar-button:hover {
    background-color: #4b4b4b;
    cursor: pointer;
}
.active-button {
    background-color: #8a2be2 !important;
    color: white !important;
    border: 1px solid #b57cff;
}
</style>
""", unsafe_allow_html=True)

# --------------- Sidebar Navigation ----------------
st.sidebar.title("📚 Navigation")
clicked_page = st.session_state.get("page", "Prediction")

for name in PAGE_NAMES:
    if st.sidebar.button(
        name,
        key=name,
        use_container_width=True,
        help=f"Open {name}",
    ):
        st.session_state["page"] = name
        clicked_page = name

# Highlight active button
st.sidebar.markdown(
    f"<script>"
    f"var btn = window.parent.document.querySelector('button[kind=\"secondary\"]:contains({clicked_page})');"
    f"</script>",
    unsafe_allow_html=True,
)

page = clicked_page

# ----------- Backend Links -------------
st.sidebar.markdown("---")
st.sidebar.write("### Backend")
st.sidebar.write(f"🔵 Predict API: `{API_PREDICT}`")
st.sidebar.write(f"🟢 Feedback API: `{API_FEEDBACK}`")

# ---------------- Helper Functions ----------------
def send_predict(window):
    try:
        r = requests.post(API_PREDICT, json={"window": window}, timeout=5)
        return r
    except:
        return None

def default_window():
    row = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    return [row] * 128

def plot_signals(window):
    w = np.array(window)
    t = np.arange(w.shape[0])
    fig, axs = plt.subplots(3, 2, figsize=(10, 6), sharex=True)
    names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

    for i, ax in enumerate(axs.ravel()):
        ax.plot(t, w[:, i])
        ax.set_title(names[i])
    plt.tight_layout()
    st.pyplot(fig)

def map_prediction(pred):
    bins = [-np.inf, 0.2, 0.6, 1.0, 1.4, 1.8, np.inf]
    idx = np.digitize(pred, bins)
    idx = min(max(idx, 1), 6)
    return ACTIVITY_LABELS[idx]

# ------------------- PAGES ----------------------

# ====================== Prediction =========================
if page == "Prediction":
    st.title("📡 Real-Time Prediction")
    st.write("Upload a **128×6 JSON window** or paste JSON manually.")

    # Show prediction mapping box
    st.info("""
    ### 📘 Prediction Value Interpretation  
    Model output → Activity label  
    - < 0.2 → WALKING  
    - 0.2–0.6 → WALKING_UPSTAIRS  
    - 0.6–1.0 → WALKING_DOWNSTAIRS  
    - 1.0–1.4 → SITTING  
    - 1.4–1.8 → STANDING  
    - > 1.8 → LAYING  
    """)

    col1, col2 = st.columns([2, 1])

    # Upload / Paste Input
    with col1:
        uploaded = st.file_uploader("Upload JSON", type=["json"])
        pasted = st.text_area("Or paste JSON manually", height=120)

    with col2:
        if st.button("Use Default Sample Window", type="primary", use_container_width=True):
            st.session_state["window"] = default_window()

    window = None

    if uploaded:
        try:
            window = json.load(uploaded)["window"]
            st.success("JSON file loaded successfully.")
        except:
            st.error("Invalid JSON file.")

    elif pasted.strip():
        try:
            window = json.loads(pasted)["window"]
            st.success("Pasted JSON loaded successfully.")
        except:
            st.error("Invalid JSON format.")

    elif "window" in st.session_state:
        window = st.session_state["window"]

    if window is None:
        st.warning("Please upload or paste a JSON window.")
        st.stop()

    # Preview
    st.subheader("Preview (first 5 rows)")
    st.write(window[:5])

    # Plots
    with st.expander("📈 Show 6 Sensor Plots"):
        plot_signals(window)

    st.markdown("---")

    if st.button("🚀 Send to Model", type="primary", use_container_width=True):
        t0 = time.time()
        resp = send_predict(window)
        t1 = time.time()

        if resp is None:
            st.error("❌ Server unreachable.")
        elif resp.status_code != 200:
            st.error(f"Server Error {resp.status_code}: {resp.text}")
        else:
            val = resp.json().get("prediction", None)
            label = map_prediction(val)

            st.success(f"Prediction received in {int((t1-t0)*1000)} ms")

            colA, colB = st.columns(2)
            colA.metric("Raw Prediction Value", f"{val:.4f}")
            colB.metric("Predicted Activity", label)

# ====================== Manual Test =========================
elif page == "Manual Test":
    st.title("🧪 Manual Test Window")
    num_rows = st.slider("Rows", 10, 128, 128)
    v = st.number_input("Base value", value=0.1)

    template = [v, v+0.1, v+0.2, v+0.3, v+0.4, v+0.5]
    window = [template] * num_rows

    if st.button("Predict", type="primary"):
        resp = send_predict(window)
        if resp:
            st.metric("Prediction", resp.json().get("prediction", 0.0))
        else:
            st.error("Prediction failed.")

# ====================== Live Simulation =========================
elif page == "Live Simulation":
    st.title("🔴 Live Simulation")

    start = st.button("Start")
    stop = st.button("Stop")

    if "sim" not in st.session_state:
        st.session_state.sim = False

    if start: st.session_state.sim = True
    if stop: st.session_state.sim = False

    placeholder = st.empty()

    while st.session_state.sim:
        base = np.sin(np.linspace(0, 6.28, 128)) * 0.2
        window = [[
            float(base[i] + np.random.randn()*0.02),
            float(base[i] + 0.01 + np.random.randn()*0.02),
            float(base[i] - 0.01 + np.random.randn()*0.02),
            float(np.cos(base[i]) * 0.1 + np.random.randn()*0.01),
            float(np.cos(base[i]) * 0.05 + np.random.randn()*0.01),
            float(np.sin(base[i]*0.5) * 0.03 + np.random.randn()*0.01)
        ] for i in range(128)]

        resp = send_predict(window)

        if resp:
            pred = resp.json().get("prediction", 0.0)
            label = map_prediction(pred)
            placeholder.metric("Live Prediction", label)
        else:
            placeholder.error("Server unreachable")

        time.sleep(1)

# ====================== Model Info =========================
elif page == "Model Info":
    st.title("📘 Model Information")
    st.write("""
    - **Type**: Meta-Learning Prototype Network  
    - **Base Model**: CNN  
    - **Input**: 128×6 Sensor Window  
    - **Dataset**: UCI HAR  
    - **Epochs**: 20  
    """)

# ====================== Kanban =========================
elif page == "Kanban & Export":
    st.title("🗂 Project Status")
    st.write("- ✔ Backend Functional")
    st.write("- ✔ Dashboard Complete")
    st.write("- ✔ Real Sensor Data Tested")
    st.write("- ⏳ Export to TFLite")

    st.markdown("---")

    if os.path.exists(LOCAL_PPT_PATH):
        with open(LOCAL_PPT_PATH, "rb") as f:
            st.download_button("Download PPT", f, file_name="SDL_ppt.pptx")

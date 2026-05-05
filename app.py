import streamlit as st
import numpy as np
from PIL import Image
import os

# Load tflite interpreter without tensorflow
try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MaskGuard AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #f0ede8;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,210,130,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(0,180,255,0.07) 0%, transparent 55%),
        #0a0a0f !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 750px !important; margin: auto !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px dashed rgba(255,255,255,0.18) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(0,210,130,0.5) !important;
    background: rgba(0,210,130,0.04) !important;
}
[data-testid="stImage"] img {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
.stProgress > div > div {
    background: linear-gradient(90deg, #00d282, #00b4ff) !important;
    border-radius: 99px !important;
}
.stProgress > div {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 99px !important;
    height: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    interpreter = Interpreter(model_path="face_mask_detector.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter    = load_model()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    model_loaded   = True
except Exception as e:
    model_loaded = False

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(image):
    img_resized = image.resize((128, 128))
    img_array   = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch   = np.expand_dims(img_array, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    interpreter.invoke()
    prob = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
    return prob

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:3rem 0 1.5rem;">
    <div style="display:inline-flex;align-items:center;gap:6px;
        background:rgba(0,210,130,0.1);border:1px solid rgba(0,210,130,0.3);
        border-radius:99px;padding:6px 16px;font-size:0.72rem;font-weight:500;
        letter-spacing:0.12em;text-transform:uppercase;color:#00d282;margin-bottom:1.4rem;">
        ● &nbsp;AI Powered Detection
    </div>
    <h1 style="font-family:'Syne',sans-serif;font-size:3.2rem;font-weight:800;
        letter-spacing:-0.03em;line-height:1.05;margin:0.5rem 0 0.8rem;
        background:linear-gradient(135deg,#ffffff 30%,#00d282 70%,#00b4ff 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        MaskGuard AI
    </h1>
    <p style="font-size:1rem;color:rgba(240,237,232,0.5);font-weight:300;
        line-height:1.7;max-width:460px;margin:0 auto;">
        Upload a face photo and our CNN model instantly detects whether a mask is being worn — with a confidence score.
    </p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Could not load model. Make sure `face_mask_detector.tflite` is in the same folder as `app.py`.")
    st.stop()

st.markdown("<hr style='border:none;height:1px;background:rgba(255,255,255,0.07);margin:0.5rem 0 1.5rem;'>", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂  Drop your image here or click to browse  ·  JPG / PNG / JFIF",
    type=["jpg", "jpeg", "png", "jfif"],
    label_visibility="visible"
)

# ── Prediction ────────────────────────────────────────────────────────────────
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, use_container_width=True, caption="Uploaded Image")

    with st.spinner("🔍  Analyzing image..."):
        prob       = predict(image)
        pred_class = 1 if prob > 0.5 else 0
        confidence = prob if pred_class == 1 else 1 - prob
        conf_pct   = round(confidence * 100, 1)

    is_mask = pred_class == 0

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    if is_mask:
        bg, border, color = "rgba(0,210,130,0.10)", "rgba(0,210,130,0.35)", "#00d282"
        emoji, label      = "✅", "Mask Detected"
        subtext           = "This person is wearing a face mask."
    else:
        bg, border, color = "rgba(255,75,75,0.10)", "rgba(255,75,75,0.35)", "#ff4b4b"
        emoji, label      = "❌", "No Mask Detected"
        subtext           = "This person is NOT wearing a face mask."

    st.markdown(f"""
    <div style="background:{bg};border:1.5px solid {border};border-radius:18px;
        padding:1.8rem 2rem;display:flex;align-items:center;gap:1.2rem;margin-bottom:1.5rem;">
        <div style="font-size:2.8rem;flex-shrink:0;">{emoji}</div>
        <div>
            <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
                letter-spacing:-0.02em;color:{color};">{label}</div>
            <div style="font-size:0.87rem;color:rgba(240,237,232,0.5);margin-top:3px;">{subtext}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
        font-size:0.78rem;color:rgba(240,237,232,0.45);text-transform:uppercase;
        letter-spacing:0.08em;font-weight:500;margin-bottom:0.5rem;">
        <span>Confidence Score</span>
        <span style="color:{color};font-weight:700;font-size:1rem;">{conf_pct}%</span>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(conf_pct))

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    def stat_box(col, val, lbl):
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                border-radius:14px;padding:1rem;text-align:center;">
                <div style="font-family:'Syne',sans-serif;font-size:1.25rem;
                    font-weight:700;color:#f0ede8;">{val}</div>
                <div style="font-size:0.7rem;color:rgba(240,237,232,0.4);
                    text-transform:uppercase;letter-spacing:0.08em;margin-top:4px;">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    stat_box(c1, f"{conf_pct}%", "Confidence")
    stat_box(c2, "CNN", "Model Type")
    stat_box(c3, "128px", "Input Size")

else:
    st.markdown("""
    <div style="margin-top:1rem;">
        <div style="font-family:'Syne',sans-serif;font-size:0.8rem;font-weight:700;
            color:rgba(240,237,232,0.4);text-transform:uppercase;
            letter-spacing:0.12em;margin-bottom:1rem;">How it works</div>
    </div>
    """, unsafe_allow_html=True)

    for num, title, desc in [
        ("1", "Upload an image", "any JPG, PNG or JFIF photo containing a face"),
        ("2", "CNN processes it", "the model resizes and analyzes facial features"),
        ("3", "Get instant results", "mask / no-mask prediction with confidence score"),
    ]:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:0.9rem;
            background:rgba(255,255,255,0.03);border-radius:12px;
            padding:0.9rem 1.1rem;margin-bottom:0.6rem;">
            <div style="min-width:26px;height:26px;border-radius:8px;
                background:rgba(0,210,130,0.15);border:1px solid rgba(0,210,130,0.3);
                color:#00d282;font-family:'Syne',sans-serif;font-size:0.78rem;font-weight:700;
                display:flex;align-items:center;justify-content:center;">{num}</div>
            <div style="font-size:0.88rem;color:rgba(240,237,232,0.6);line-height:1.5;">
                <strong style="color:#f0ede8;font-weight:500;">{title}</strong> — {desc}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:2.5rem 0 0;
    font-size:0.75rem;color:rgba(240,237,232,0.2);letter-spacing:0.05em;">
    MaskGuard AI · Built with TensorFlow Lite & Streamlit
</div>
""", unsafe_allow_html=True)

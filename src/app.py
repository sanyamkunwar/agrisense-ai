import sys
import os

# Ensure project root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from cv_module.infer import predict
from rag_module.generate import generate_answer


# ---- Load environment ----
load_dotenv()
APP_NAME = os.getenv("APP_NAME", "Intelligent Agricultural Assistant")

# ---- Page Config ----
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🌱",
    layout="wide"
)

# ---- Sidebar ----
with st.sidebar:
    st.title("🌿 AgriSense AI")
    st.caption("Crop Disease Detection + RAG Advice")
    st.info("Upload a leaf → detect disease → get treatment advice.")

# ---- Chat History ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Header ----
st.title("🌾 Intelligent Agricultural Assistant")
st.write("Upload a crop leaf image to diagnose diseases and get expert farming guidance.")
st.write("---")


# ============================================================
#     IMAGE UPLOAD + COMPUTER VISION DISEASE DETECTION
# ============================================================

uploaded = st.file_uploader("📸 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(img, caption="Uploaded Leaf", width=350)

    with col2:
        st.subheader("🔍 Disease Detection (Computer Vision)")

        result = predict(uploaded, top_k=3)
        st.success(f"Detected: **{result['label']}** ({result['confidence']*100:.2f}%)")

        # Save disease name
        st.session_state.last_detected_disease = result["label"]


        st.write("### Top-3 Predictions")
        for cls, prob in result["topk"]:
            st.write(f"- **{cls.replace('_', ' ')}** → {prob*100:.2f}%")

        st.write("---")

        # Automatic query for treatment advice
        query = f"What is {result['label']} and how do I treat it?"

        st.subheader("💬 Treatment Advice (RAG + LLM)")
        with st.spinner("Generating treatment advice..."):
            answer, sources = generate_answer(query)

        st.write(answer)

        st.write("### Sources Used")
        for i, s in enumerate(sources, 1):
            st.write(f"{i}. {s}")

st.write("---")

# ============================================================
#                     CHATBOT SECTION
# ============================================================

st.subheader("💬 Ask Any Farming Question")

# user_q = st.text_input("Your question:")

with st.form(key="chat_form", clear_on_submit=True):
    user_q = st.text_input("Ask your question:")
    send = st.form_submit_button("Send")

if send:
    if user_q.strip():

        st.session_state.chat_history.append(("User", user_q))

        # Disease-aware query injection
        disease = st.session_state.get("last_detected_disease", None)

        if disease:
            final_query = (
                f"The detected crop disease is: {disease}. "
                f"Answer the following question strictly with respect to this disease only: {user_q}"
            )
        else:
            final_query = user_q

        with st.spinner("Thinking..."):
            answer, sources = generate_answer(final_query)

        st.session_state.chat_history.append(("Assistant", answer))



# Render chat history — simple clean text
for role, msg in st.session_state.chat_history:
    st.write(f"**{role}:** {msg}")



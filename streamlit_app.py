import streamlit as st
from src.inference import (
    predict_lstm,
    predict_distilbert,
    predict_bert
)

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Hate Speech Detection",
    page_icon="🛑",
    layout="centered"
)

# ===============================
# STYLE (BLUE SOFT + KONTRAS JELAS)
# ===============================
st.markdown("""
<style>
.stApp {
    background-color: #e0f2fe;
}

/* Global text */
h1, h2, h3, h4, h5, h6, p, span, label {
    color: #0f172a !important;
}

/* Button */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    height: 3em;
    font-size: 16px;
}

/* Result Card */
.result-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #2563eb;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}

/* Result Label */
.result-label {
    font-size: 24px;
    font-weight: bold;
    color: #000000; /* HITAM */
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.title("🛑 Hate Speech Detection")
st.write(
    "Aplikasi klasifikasi ujaran kebencian pada teks media sosial "
    "menggunakan **LSTM**, **DistilBERT**, dan **BERT**."
)

st.divider()

# ===============================
# INPUT
# ===============================
text_input = st.text_area(
    "✍️ Masukkan teks:",
    placeholder="Contoh: i hate you",
    height=120
)

model_choice = st.selectbox(
    "🤖 Pilih Model:",
    ["LSTM", "DistilBERT", "BERT"]
)

# ===============================
# PREDICT
# ===============================
if st.button("🔍 Prediksi"):
    if text_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong.")
    else:
        with st.spinner("Sedang memproses prediksi..."):
            if model_choice == "LSTM":
                result = predict_lstm(text_input)
            elif model_choice == "DistilBERT":
                result = predict_distilbert(text_input)
            else:
                result = predict_bert(text_input)

        st.divider()

        label = result["label"]
        confidence = result["confidence"]

        # ===============================
        # RESULT CARD
        # ===============================
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        # Label utama (HITAM)
        st.markdown(
            f'<div class="result-label">Hasil Prediksi: {label}</div>',
            unsafe_allow_html=True
        )

        st.markdown("<br><b>Confidence (%)</b>", unsafe_allow_html=True)

        # ===============================
        # CONFIDENCE BAR (AMAN 0–1)
        # ===============================
        for lbl, score in confidence.items():
            score = float(score)
            score = max(0.0, min(score, 1.0))  # ⛑️ pengaman

            percent = round(score * 100, 2)
            st.write(f"**{lbl}** : {percent}%")
            st.progress(score)

        st.markdown('</div>', unsafe_allow_html=True)

        st.caption("Confidence menunjukkan probabilitas prediksi model.")

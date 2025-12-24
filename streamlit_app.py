import streamlit as st
from src.inference import (
    predict_lstm,
    predict_distilbert,
    predict_bert
)

# ===============================
# FUNSI UTILITY
# ===============================
def safe_progress(value):
    """
    Normalisasi nilai confidence agar aman dipakai di st.progress()
    - Nilai >1 dianggap persen → dibagi 100
    - NaN / inf → diubah menjadi 0
    - Clamp antara 0.0 - 1.0
    """
    try:
        value = float(value)

        # jika sudah dalam persen (misal 75 atau 132)
        if value > 1:
            value = value / 100

        # NaN atau inf → 0
        if value != value or value == float("inf"):
            return 0.0

        # clamp
        return max(0.0, min(value, 1.0))
    except:
        return 0.0

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Hate Speech Detection",
    page_icon="🛑",
    layout="centered"
)

# ===============================
# STYLE
# ===============================
st.markdown("""
<style>
.stApp { background-color: #e0f2fe; }
h1, h2, h3, h4, h5, h6, p, span, label { color: #0f172a !important; }
.stButton > button { background-color: #2563eb; color: white; border-radius: 8px; height: 3em; font-size: 16px; }
.result-card { background-color: #ffffff; padding: 20px; border-radius: 12px; border-left: 6px solid #2563eb; box-shadow: 0 4px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
.result-label { font-size: 24px; font-weight: bold; color: #000000; }
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
# INPUT TEKS DAN PILIH MODEL
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
# BUTTON PREDIKSI
# ===============================
if st.button("🔍 Prediksi"):
    if text_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong.")
    else:
        with st.spinner("Sedang memproses prediksi..."):
            # ===============================
            # PILIH MODEL DAN PREDIKSI
            # ===============================
            if model_choice == "LSTM":
                result = predict_lstm(text_input)
            elif model_choice == "DistilBERT":
                result = predict_distilbert(text_input)
            else:
                result = predict_bert(text_input)

        st.divider()

        label = result.get("label", "Unknown")
        confidence = result.get("confidence", {})

        # ===============================
        # RESULT CARD
        # ===============================
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        # Label utama
        st.markdown(
            f'<div class="result-label">Hasil Prediksi: {label}</div>',
            unsafe_allow_html=True
        )

        st.markdown("<br><b>Confidence (%)</b>", unsafe_allow_html=True)

        # ===============================
        # CONFIDENCE BAR (AMAN 0–1)
        # ===============================
        if isinstance(confidence, dict) and confidence:
            for lbl, score in confidence.items():
                safe_value = safe_progress(score)
                percent = round(safe_value * 100, 2)

                st.write(f"**{lbl}** : {percent}%")
                st.progress(safe_value)
        else:
            st.write("Tidak ada confidence yang tersedia.")
            st.progress(0.0)

        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Confidence menunjukkan probabilitas prediksi model.")

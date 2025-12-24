# 🛑 Hate Speech Detection — Text Classification on Social Media

Aplikasi klasifikasi ujaran kebencian pada teks media sosial menggunakan **Deep Learning** dan **Transformer-Based Models**.  
Proyek ini dikembangkan sebagai bagian dari **Ujian Akhir Praktikum (UAP) Pembelajaran Mesin**.

---

## 🌟 Overview
**Hate Speech Detection** adalah sistem klasifikasi teks untuk mendeteksi ujaran kebencian (*hate speech*) dan bahasa ofensif di media sosial. Sistem ini memanfaatkan beberapa pendekatan modern dalam *Deep Learning*:

✔ **LSTM** (Long Short-Term Memory) sebagai baseline model  
✔ **BERT** (Bidirectional Encoder Representations from Transformers) untuk pemahaman konteks dua arah  
✔ **DistilBERT**, versi ringan BERT, lebih cepat dan efisien  

Sistem juga dilengkapi dengan **user interface berbasis Streamlit**, sehingga pengguna dapat langsung memasukkan teks dan melihat prediksi secara interaktif.

---

## 🔎 Dataset
📂 Dataset yang digunakan: **Hate Speech and Offensive Language Dataset**  
Sumber: [Kaggle](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)

### Karakteristik Dataset:
- Jumlah data: >20.000 teks  
- Bahasa: Inggris  
- Jenis data: Teks media sosial  
- Label:
  - **Hate Speech**  
  - **Offensive Language**  
  - **Neither (Normal)**

Setiap entri berisi kolom teks dan label kategorikal, siap digunakan untuk pelatihan dan pengujian model.

---

## 🔎 Exploratory Data Analysis (EDA)
Analisis awal dilakukan untuk memahami distribusi data dan karakteristik teks:

| Analisis | Insight |
|----------|--------|
| Distribusi kelas | Tidak sepenuhnya seimbang; label "Neither" lebih dominan |
| Panjang teks | Bervariasi, sebagian besar berupa teks pendek |

📌 **Kesimpulan**: Perlu **tokenisasi dan padding** agar model dapat memproses input teks secara konsisten.

---

## 🧠 Preprocessing Data
Tahapan persiapan data sebelum pelatihan model:

1. **Lowercase** — Mengubah seluruh teks menjadi huruf kecil  
2. **Cleaning** — Menghapus tanda baca atau karakter khusus yang tidak relevan  
3. **Tokenisasi**  
   - LSTM → Tokenizer Keras  
   - BERT / DistilBERT → Tokenizer spesifik model transformer  
4. **Padding** — Menyamakan panjang sequence untuk LSTM  
5. **Encode Label** — Mengubah label kategorikal menjadi representasi numerik  

---

## 📊 Models Implemented

| Model | Tipe | Deskripsi |
|-------|------|-----------|
| **LSTM** | Neural Network Base | Model berbasis LSTM untuk menangkap urutan kata dalam teks |
| **BERT** | Pretrained Transformer | Memahami konteks teks dua arah secara kontekstual |
| **DistilBERT** | Pretrained Transformer (Light) | Versi ringan BERT, lebih cepat, performa kompetitif |

---

## 📈 Evaluation Summary

| Model | Accuracy | Analisis |
|-------|----------|----------|
| **LSTM** | 0.55 | Akurasi terendah karena embedding sederhana dan tokenisasi statis, kesulitan membedakan konteks ofensif |
| **BERT** | 0.66 | Pretrained, memahami konteks dua arah, performa terbatas karena fine-tuning singkat dan CPU-only |
| **DistilBERT** | 0.68 | Lebih ringan dan stabil, generalisasi lebih baik, cocok untuk deployment Streamlit |

📌 **Kesimpulan**: Transformer-based models unggul dibanding LSTM, dan DistilBERT memberikan kombinasi efisiensi dan akurasi terbaik.

---

## 🛠 How It Works

### Input
Pengguna memasukkan teks bebas melalui **form input Streamlit**.

### Output
Setelah klik **Predict**, sistem menampilkan:

- Prediksi label: *Hate Speech*, *Offensive Language*, atau *Neither*  
- Confidence score (%)  
- Progress bar yang menunjukkan probabilitas tiap kelas  

---

## 🖼 Feature Screenshot & Live Demo
### Input Form
![Input Form](screenshots/input_form.png)

### Result View
![Result View](screenshots/result_view.png)

💻 **Live Demo:** [Streamlit App]([https://share.streamlit.io/amandarachma/hate-speech/main/streamlit_app.py](https://requirementstxt-8appf5qte4lxyw5efbymerm.streamlit.app/))





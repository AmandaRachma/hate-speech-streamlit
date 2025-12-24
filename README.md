# 🛑 Hate Speech Detection on Social Media Text

Aplikasi klasifikasi ujaran kebencian pada teks media sosial menggunakan **Deep Learning** dan **Transformer-Based Models**.  
Proyek ini dikembangkan sebagai bagian dari **Ujian Akhir Praktikum (UAP) Pembelajaran Mesin**.

---

## 📌 Latar Belakang

Media sosial merupakan sarana komunikasi digital yang sangat populer, namun sering disalahgunakan untuk menyebarkan ujaran kebencian (*hate speech*) dan bahasa ofensif. Konten semacam ini dapat berdampak negatif terhadap individu maupun kelompok tertentu serta berpotensi memicu konflik sosial.

Oleh karena itu, diperlukan sistem otomatis yang mampu mendeteksi dan mengklasifikasikan ujaran kebencian pada teks media sosial secara akurat. Pemanfaatan teknik *machine learning* dan *deep learning* diharapkan dapat membantu proses moderasi konten secara lebih efisien.

---

## 📊 Dataset

Dataset yang digunakan dalam penelitian ini adalah:

**Hate Speech and Offensive Language Dataset**  
Sumber: Kaggle  
Link: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

### Karakteristik Dataset:
- Jumlah data: lebih dari 20.000 teks
- Bahasa: Bahasa Inggris
- Jenis data: Teks media sosial
- Kelas label:
  - **Hate Speech**
  - **Offensive Language**
  - **Neither (Normal)**

Dataset ini banyak digunakan dalam penelitian karena merepresentasikan kasus ujaran kebencian yang nyata di media sosial.

---

## 🧠 Metode dan Model Klasifikasi

Penelitian ini menggunakan tiga pendekatan model klasifikasi teks, yaitu:

### 1. LSTM (Long Short-Term Memory)
- Model *non-pretrained*
- Menggunakan tokenisasi dan padding manual
- Digunakan sebagai **baseline model**

### 2. DistilBERT
- Model transformer hasil distilasi dari BERT
- Lebih ringan dan cepat
- Cocok untuk deployment aplikasi

### 3. BERT (Bidirectional Encoder Representations from Transformers)
- Model transformer kontekstual dua arah
- Memiliki performa terbaik dalam memahami konteks teks
- Namun membutuhkan sumber daya komputasi lebih besar

---

## 🏗️ Arsitektur Sistem


# RINGKASAN PENELITIAN / RESEARCH SUMMARY
## Proposal Tugas Akhir - Mujahid Ansori Majid (1197050093)

---

## 📋 JUDUL PENELITIAN
**"Perbandingan Wav2Vec2 dan Data2Vec terhadap Fitur Laten Audio"**

*(Comparison of Wav2Vec2 and Data2Vec on Latent Audio Features)*

---

## 🎯 LATAR BELAKANG (BACKGROUND)

### 1. Perkembangan Teknologi
- Deep Learning dalam pemrosesan audio didominasi oleh model **Self-Supervised Learning (SSL)**
- **Wav2Vec2** menjadi arsitektur pionir yang sangat berpengaruh
- Model-model SSL telah diterapkan dalam konteks tilawah Al-Qur'an
- Efektivitas tinggi dalam transkripsi otomatis (ASR) dengan Word Error Rate (WER) dan Character Error Rate (CER) rendah

### 2. Masalah yang Ditemukan
**ASR ≠ Kualitas Representasi Fonetic**
- Akurasi transkripsi (WER/CER) **tidak selalu setara** dengan kualitas representasi fonetik
- Dibutuhkan untuk mendeteksi **detail tajwid yang kompleks**
- ASR menghasilkan teks yang rentan terhadap error tajwid

### 3. Solusi: Representasi Laten (Latent Embeddings)
**Wav2Vec2 dan Data2Vec:**
- Dikembangkan dengan paradigma SSL
- Mempelajari representasi audio mendalam dari data mentah
- **Tanpa memerlukan label transkripsi**
- Representasi laten adalah kunci untuk tugas-tugas **non-transkripsi**

### 4. Aplikasi dalam Konteks Hafalan Al-Qur'an
**Yang dibutuhkan:**
- **Audio Similarity Search / Audio Retrieval**
- Mencocokkan potongan bacaan dengan ayat paling mirip di database
- Berdasarkan **kemiripan vector embeddings** (bukan teks ASR)
- Lebih akurat karena tidak rentan error tajwid

### 5. Perbedaan Filosofi Arsitektur
| Aspek | Wav2Vec2 | Data2Vec |
|-------|----------|----------|
| **Metode** | Contrastive Learning | Self-Distillation |
| **Target** | Diskrit (quantized) | Kontinu (teacher-student) |
| **Philosophy** | Bedakan true vs distractor | Prediksi representasi guru |
| **Sukses di** | Speaker identification (Arab) | Speaker identification (Arab) |

### 6. Research Gap (Celah Penelitian)
**Pertanyaan Fundamental:**
> "Sejauh mana perbedaan filosofi ini memengaruhi kualitas fitur laten untuk tugas yang sangat bergantung pada fonetik, seperti Audio Retrieval?"

**Masalah yang Ada:**
- Perbandingan yang ada di literatur masih didominasi metrik **transkripsi (WER/CER)**
- Belum ada pengujian empiris menggunakan metrik **similarity search (MAP)**
- Khususnya untuk **Retrieval Ayat Al-Qur'an**

---

## ❓ RUMUSAN MASALAH (PROBLEM STATEMENT)

**Permasalahan Utama:**
Bagaimana membandingkan efektivitas fitur laten audio dari Wav2Vec2 dan Data2Vec dalam mendukung tugas retrieval ayat Al-Qur'an menggunakan metrik similarity search?

**Pertanyaan Penelitian:**
1. Bagaimana kualitas representasi laten Wav2Vec2 dibandingkan Data2Vec untuk audio Al-Qur'an?
2. Metrik similarity search (MAP, Precision@k, Recall@k) seperti apa yang optimal untuk mengevaluasi retrieval ayat?
3. Arsitektur SSL mana (contrastive vs self-distillation) yang lebih baik untuk fonetik Arab?

---

## 🎯 TUJUAN PENELITIAN (RESEARCH OBJECTIVES)

### Tujuan Utama:
Melakukan perbandingan komprehensif antara Wav2Vec2 dan Data2Vec untuk mengukur efektivitas fitur laten audio dalam mendukung tugas **retrieval ayat Al-Qur'an**.

### Tujuan Spesifik:
1. Mengekstrak dan membandingkan representasi laten dari kedua model
2. Mengimplementasikan sistem similarity search untuk retrieval ayat
3. Mengevaluasi kinerja menggunakan metrik retrieval (MAP, Precision, Recall)
4. Memberikan rekomendasi arsitektur optimal untuk fonetik Al-Qur'an

---

## 🔬 METODE PENELITIAN (RESEARCH METHODOLOGY)

### Tahapan Penelitian:

#### **1. Pengumpulan Data**
- Dataset audio Al-Qur'an (murottal)
- Beberapa qari dengan tajwid yang berbeda
- Variasi bacaan untuk menguji robustness

#### **2. Tahap Pre-processing Data**
- Normalisasi audio (sample rate, amplitude)
- Segmentasi ayat/potongan audio
- Data augmentation (opsional)

#### **3. Tahap Ekstraksi Representasi Laten (Vector Embedding)**
- Load pre-trained Wav2Vec2 (facebook/wav2vec2-base-960h)
- Load pre-trained Data2Vec (facebook/data2vec-audio-base)
- Ekstrak embeddings dari layer-layer tertentu
- Pooling: mean, cls, atau last

#### **4. Tahap Implementasi Similarity Search**
- Cosine similarity antara query dan database
- k-Nearest Neighbor (k-NN) search
- Indexing untuk efisiensi (FAISS/Annoy)

#### **5. Tahap Evaluasi Kinerja (Perbandingan)**
**Metrik yang Digunakan:**
- **MAP (Mean Average Precision)**: Overall retrieval quality
- **Precision@k**: Akurasi top-k hasil
- **Recall@k**: Coverage hasil relevan
- **MRR (Mean Reciprocal Rank)**: Posisi hasil pertama yang relevan

**Aspek yang Dievaluasi:**
- Clustering quality (Silhouette score)
- Discriminative power (classification accuracy)
- Similarity structure (correlation matrices)
- Information retrieval performance

#### **6. Tahap Analisis Hasil**
- Analisis statistik (paired t-test)
- Visualisasi embedding (t-SNE, UMAP)
- Interpretasi hasil dalam konteks fonetik Arab
- Rekomendasi arsitektur optimal

---

## 📊 KERANGKA PEMIKIRAN (CONCEPTUAL FRAMEWORK)

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT: Audio Al-Qur'an                       │
│              (Berbagai qari, variasi tajwid)                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              PRE-PROCESSING                                     │
│  - Normalisasi (16kHz)                                          │
│  - Segmentasi ayat                                              │
│  - Augmentasi (opsional)                                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────┴─────────────────────────────────────┐
│              EKSTRAKSI FITUR LATEN                              │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │    WAV2VEC2         │    │     DATA2VEC        │            │
│  │  (Contrastive)      │    │ (Self-Distillation) │            │
│  │                     │    │                     │            │
│  │ • CNN Feature       │    │ • CNN Feature       │            │
│  │   Extractor         │    │   Extractor         │            │
│  │ • Quantization      │    │ • Transformer       │            │
│  │ • Transformer (12)  │    │   (12 layers)       │            │
│  │ • Contrastive Loss  │    │ • Regression Loss   │            │
│  └──────────┬──────────┘    └──────────┬──────────┘            │
│             │                          │                       │
│             ▼                          ▼                       │
│    Embedding: (768-d)         Embedding: (768-d)               │
│             │                          │                       │
└─────────────┼──────────────────────────┼───────────────────────┘
              │                          │
              ▼                          ▼
┌───────────────────────────┐  ┌───────────────────────────┐
│   SIMILARITY SEARCH       │  │   SIMILARITY SEARCH       │
│   (Cosine Similarity)     │  │   (Cosine Similarity)     │
│                           │  │                           │
│   Query → Top-k Matches   │  │   Query → Top-k Matches   │
└───────────┬───────────────┘  └───────────┬───────────────┘
            │                              │
            └──────────────┬───────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│              EVALUASI & PERBANDINGAN                      │
│                                                           │
│   Metrik:                                                 │
│   • MAP (Mean Average Precision)                         │
│   • Precision@k, Recall@k                                │
│   • MRR (Mean Reciprocal Rank)                           │
│   • Silhouette Score (clustering)                        │
│   • Classification Accuracy                              │
│                                                           │
│   Analisis:                                               │
│   • Statistical significance (t-test)                    │
│   • Visualisasi (t-SNE, UMAP)                            │
│   • Interpretasi fonetik                                 │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│              OUTPUT: REKOMENDASI                         │
│                                                           │
│   • Model terbaik untuk retrieval ayat Qur'an           │
│   • Analisis kekuatan/kelemahan masing-masing           │
│   • Strategi optimal untuk fonetik Arab                 │
└──────────────────────────────────────────────────────────┘
```

---

## 💡 MANFAAT PENELITIAN (RESEARCH BENEFITS)

### 1. Manfaat Teoritis
- Memperkaya literatur perbandingan model SSL untuk audio Arab
- Mengisi gap penelitian similarity search vs ASR metrics
- Kontribusi metodologi evaluasi retrieval ayat Al-Qur'an

### 2. Manfaat Praktis
- Rekomendasi arsitektur optimal untuk aplikasi hafalan Qur'an
- Landasan teknis pengembangan alat bantu hafalan berbasis AI
- Peningkatan akurasi sistem pencocokan tilawah

### 3. Manfaat Sosial
- Mendukung pembelajaran Al-Qur'an melalui teknologi
- Membantu pendeteksi kesalahan tajwid secara otomatis
- Aplikasi edukasi yang lebih efektif

---

## 🔍 PERBEDAAN PENDEKATAN: Wav2Vec2 vs Data2Vec

### Wav2Vec2 (Contrastive Learning)
**Filosofi:** Belajar dengan membedakan yang benar dari yang salah

**Proses:**
1. Feature Extractor (CNN) → Latent representations
2. Quantization → Discrete codebook vectors
3. Context Network (Transformer) → Contextualized representations
4. Training: Prediksi quantized target dari posisi yang di-mask
5. Objective: Contrastive loss (bedakan true vs negatives)

**Karakteristik:**
- Menggunakan discrete targets (vektor quantized)
- Diversity penalty untuk penggunaan codebook yang merata
- Masked prediction ~50% time steps

### Data2Vec (Self-Distillation)
**Filosofi:** Belajar dengan meniru representasi guru yang lebih baik

**Proses:**
1. Feature Extractor (CNN/Transformer)
2. Student Network → Latent representations
3. Teacher Network (EMA dari student) → Target representations
4. Training: Prediksi target teacher (kontinu, bukan diskrit)
5. Objective: Regression loss (MSE terhadap teacher)

**Karakteristik:**
- Tidak memerlukan quantization
- Teacher-student framework
- Unified architecture across modalities (vision, NLP, speech)
- Continuous target representations

### Perbandingan untuk Retrieval
| Aspek | Wav2Vec2 | Data2Vec |
|-------|----------|----------|
| **Target Type** | Diskrit | Kontinu |
| **Training Signal** | Classification | Regression |
| **Quantization** | Diperlukan | Tidak |
| **Architecture** | Speech-specific | Multi-modal unified |
| **Potential for Retrieval** | Good | Potentially better (smooth representations) |

---

## 📈 HASIL YANG DIHARAPKAN (EXPECTED OUTCOMES)

### 1. Hasil Kuantitatif
- Nilai MAP untuk kedua model pada dataset Qur'an
- Perbandingan Precision@k dan Recall@k
- Analisis statistical significance (p-values)
- Perbandingan clustering quality (Silhouette score)

### 2. Hasil Kualitatif
- Visualisasi embedding space (t-SNE/UMAP)
- Analisis layer mana yang optimal untuk retrieval
- Interpretasi fonetik dari representasi laten
- Identifikasi kekuatan/kelemahan masing-masing model

### 3. Rekomendasi
- Model terbaik untuk retrieval ayat Qur'an
- Strategi fine-tuning yang optimal
- Arah pengembangan aplikasi hafalan berbasis AI

---

## 📝 DAFTAR PUSTAKA UTAMA (KEY REFERENCES)

1. Shaklawoon et al. - Penerapan Wav2Vec2 untuk tilawah Al-Qur'an
2. Baevski et al. - wav2vec 2.0: A Framework for Self-Supervised Learning
3. Baevski et al. - Data2Vec: A General Framework for Self-supervised Learning
4. Research pada domain Arab untuk speaker identification
5. Studi tentang audio similarity search dan cosine similarity

---

## ✅ CHECKLIST PERSIAPAN

### Data & Resources
- [ ] Dataset audio Al-Qur'an (minimal 10 qari, 30 juz)
- [ ] Pre-trained models: Wav2Vec2-base-960h
- [ ] Pre-trained models: Data2Vec-audio-base
- [ ] Storage yang cukup (~50GB untuk models dan data)

### Tools & Environment
- [ ] Python 3.8-3.11
- [ ] PyTorch + Transformers + Fairseq
- [ ] GPU (optional tapi direkomendasikan)
- [ ] Jupyter notebook untuk eksplorasi

### Metodologi
- [ ] Framework evaluasi (code yang sudah dibuat)
- [ ] Metrik: MAP, Precision@k, Recall@k, MRR
- [ ] Statistical testing (t-test, Wilcoxon)
- [ ] Visualisasi (t-SNE, UMAP, similarity matrices)

---

## 🎓 KESIMPULAN RINGKAS

**Penelitian ini bertujuan untuk:**
1. Membandingkan dua arsitektur SSL (Wav2Vec2 vs Data2Vec) untuk fitur laten audio
2. Fokus pada aplikasi **retrieval ayat Al-Qur'an** (bukan ASR)
3. Menggunakan metrik **similarity search (MAP)** alih-alih WER/CER
4. Memberikan rekomendasi arsitektur optimal untuk fonetik Arab

**Kontribusi:**
- Mengisi gap literatur perbandingan SSL untuk audio Arab
- Metodologi evaluasi retrieval ayat menggunakan vector embeddings
- Landasan teknis untuk pengembangan aplikasi hafalan Qur'an

**Tools yang Dibuat:**
✅ Framework perbandingan lengkap (Python)
✅ Ekstraktor embeddings Wav2Vec2
✅ Ekstraktor embeddings Data2Vec
✅ Pipeline evaluasi multi-metrik
✅ Visualisasi komprehensif

---

*Dibuat untuk membantu memahami kembali fokus penelitian dan persiapan presentasi/uji.*

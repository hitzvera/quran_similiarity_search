# Rencana Penyesuaian State of the Art (Bab 2) — Skripsi Wav2Vec2 vs Data2Vec Retrieval Al-Quran

## TL;DR

> **Ringkasan**: Menyelaraskan bagian "The State of The Art" pada `Skripsi.md` dengan argumen baru di Bab 1 (propagasi error ASR→retrieval), menambah 8 entri paper ke tabel SotA (4 sudah ada di pustaka: WavRAG, ArTST, Li, TranUSR + 4 baru: Hossain, Abdelfattah, SUPERB, Meghanani&Hain), memperbaiki 4 inkonsistensi, dan menulis ulang paragraf sintesis agar research gap tegas.
>
> **Deliverables**:
> - Tabel 1 SotA direorganisasi per tema, ~18 baris, tanpa orphan row
> - 8 entri naratif poin baru untuk paper yang masuk tabel
> - Paragraf sintesis ditulis ulang (alur ASR→retrieval + gap eksplisit)
> - 4 entri Daftar Pustaka baru ([18]-[21], format IEEE)
> - 4 bug diperbaiki (nama peneliti baris 1, kolom Tujuan baris 6, "AV-data2vec" tanpa sitasi, sitasi [17]→[11])
>
> **Estimated Effort**: Short (4-8 jam)
> **Parallel Execution**: NO — sekuensial (edit satu file yang sama, risiko konflik)
> **Critical Path**: Fix bug → tambah entri tabel → tambah narasi → tulis ulang sintesis → tambah pustaka → QA

---

## Context

### Original Request
User sedang mengerjakan skripsi Bab 2 (State of the Art). Bab 1 telah diubah, sehingga SotA perlu diselaraskan. User minta: sarankan paper baru, hapus/rapikan yang sudah ada.

### File Target
- `C:\skripsi\quran_similiarity_search\laporan\Skripsi\Skripsi.md`
- Format: Markdown/Pandoc (hasil konversi Word). Tabel = grid table `+---+`. Format ini BENAR, pertahankan.
- Bagian SotA: baris ~235-701 (narasi poin a-j + Tabel 1 + paragraf sintesis).
- Daftar Pustaka: baris ~1010-1089 ([1]-[17]).

### Perubahan Bab 1 yang jadi acuan
Bab 1 (baris 115-123) kini menegaskan: *"sistem retrieval berbasis latent embedding... mengungguli pendekatan dua tahap dari ASR ke retrieval, karena kesalahan transkripsi pada tahap ASR akan berpropagasi..."* — merujuk [4] Gemini Embedding 2 & [5] WavRAG. Argumen inilah yang harus tercermin di SotA.

### Keputusan User (dikonfirmasi)
1. Masukkan semua paper relevan dari pustaka ke tabel: [5] WavRAG, [6] ArTST, [7] Li, [8] TranUSR.
2. Tambah 4 paper baru ke tabel: Hossain 2026 [18], Abdelfattah 2025 [19], Yang SUPERB 2024 [20], Meghanani&Hain 2024 [21].
3. [4] Gemini Embedding 2 & [9] Quranic Conversations = konteks narasi saja (tidak masuk tabel).
4. Struktur tabel: PER TEMA (5 tema).
5. Cakupan: State of the Art SAJA. Studi Literatur 7.1-7.3 TIDAK disentuh.

### Konsultasi Metis (rambu)
- Kunci nomor sitasi [1]-[17]; append baru [18]+ (hindari cascade break).
- Setiap baris tabel WAJIB punya ≥1 kalimat naratif (no orphan row).
- Akhiri SotA dengan paragraf research gap → kontribusi eksplisit.
- Verifikasi metadata paper via DOI (sudah dilakukan librarian).
- Konsistensi istilah: pilih "Wav2Vec2" & "Data2Vec".

### Konsultasi Oracle (verifikasi, VERDICT: GO 5/5)
Mengonfirmasi 4 bug di dokumen aktual + menemukan bug ke-4: baris 652 sitasi [17] salah untuk speaker recognition (seharusnya [11]).

---

## Work Objectives

### Core Objective
Menyelaraskan bagian State of the Art dengan argumen Bab 1 (embedding langsung > ASR-then-retrieve), memperkaya referensi yang relevan dengan tugas retrieval/Quran/komparasi SSL, dan memperbaiki inkonsistensi teknis — sehingga research gap skripsi tampil tegas dan konsisten.

### Concrete Deliverables
- Tabel 1 SotA: ~18 baris berkelompok per tema, semua entri terverifikasi.
- 8 entri naratif baru (poin k-r atau disisipkan sesuai gaya) untuk paper yang masuk tabel.
- Paragraf sintesis SotA yang ditulis ulang.
- 4 entri Daftar Pustaka baru [18]-[21].
- 4 perbaikan bug.

### Must Have
- Nomor sitasi [1]-[17] TIDAK bergeser.
- Setiap baris tabel disebut minimal 1x di narasi.
- Paragraf penutup SotA memuat pernyataan celah penelitian → kontribusi.
- Semua metadata paper akurat (sesuai temuan librarian).

### Must NOT Have (Guardrails)
- JANGAN menyentuh Bab 1, Studi Literatur 7.1-7.3, Metode Penelitian, Jadwal.
- JANGAN menambah paper ke tabel selain 8 yang disepakati.
- JANGAN memasukkan [4] & [9] ke tabel (narasi saja).
- JANGAN merusak struktur grid table `+---+` (kolom harus tetap sejajar).
- JANGAN mengubah nomor sitasi existing.
- JANGAN menghalusinasi detail paper — gunakan metadata terverifikasi di plan ini.

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO (dokumen skripsi markdown, bukan kode).
- **Automated tests**: None.
- **QA**: Verifikasi manual + grep-based citation consistency check.

### QA Policy
- Cek grep tiap nomor sitasi [1]-[21] → tidak ada yang loncat/hilang.
- Cek tiap nama peneliti di kolom tabel muncul di narasi (no orphan).
- Cek paragraf akhir SotA mengandung frasa celah penelitian + kontribusi.
- Cek grid table tetap valid (jumlah kolom & separator konsisten).

---

## Execution Strategy

> Semua tugas menyentuh SATU file (`Skripsi.md`), sehingga dieksekusi SEKUENSIAL untuk menghindari konflik edit. Urutan penting: perbaiki bug dulu (perubahan kecil terisolasi), lalu tambah konten besar, terakhir QA.

```
Sekuensial (satu file, hindari konflik):
Task 1: Fix nama peneliti tabel baris 1
Task 2: Fix kolom Tujuan tabel baris 6 (DQ-Data2vec)
Task 3: Fix sitasi [17]→[11] baris 652
Task 4: Reorganisasi + tambah 8 baris tabel per tema
Task 5: Tambah 8 entri naratif poin baru
Task 6: Tulis ulang paragraf sintesis (gap eksplisit)
Task 7: Tambah 4 entri Daftar Pustaka [18]-[21]
Task 8: QA — audit sitasi, orphan row, konsistensi istilah
```

**Agent Dispatch**: Semua task cocok untuk `writing` atau `unspecified-high` (editing dokumen akademik bahasa Indonesia, butuh presisi bahasa + ketelitian sitasi).

---

## Metadata Paper Terverifikasi (rujukan untuk penulisan)

**Existing (sudah di pustaka, masuk tabel):**
- [5] WavRAG — Yifu Chen, Shengpeng Ji, Haoxiao Wang, dkk. ACL 2025. WavRetriever di atas Qwen2-Audio + contrastive. Bypass ASR total → 10× lebih cepat, hilangkan propagasi error transkripsi.
- [6] ArTST — Hawau O. Toyin, Amirbek Djanibekov, Ajinkya Kulkarni, Hanan Aldarmaki. ArabicNLP 2023. SpeechT5 di-pretrain from scratch untuk Arab; monolingual Arab > multibahasa (Whisper, MMS).
- [7] Li et al. — Shuyue Stella Li, dkk. ICNLSP 2023. Probing 5 SSL English via metrik PSR/DGCCA; English SSL → fitur fonetik kurang informatif lintas bahasa; wav2vec2 contrastive > masked prediction cross-lingual.
- [8] TranUSR — Hongfei Xue, Qijie Shao, dkk. INTERSPEECH 2023. UniData2vec (target kontinu ala Data2vec) + P2W Transcoder; target kontinu turunkan PER 5.3% relatif vs diskrit (UniSpeech).

**Baru (append [18]-[21]):**
- [18] Hossain et al. 2026, arXiv:2606.19747 — "A Comparative Study of Pretrained Transformer Models for Quranic ASR". Bandingkan Wav2Vec2, HuBERT, XLS-R di 870+ jam Quran; XLSR-53 terbaik (WER 0.08). Paling dekat dgn tesis.
- [19] Abdelfattah et al. 2025, arXiv:2509.00094 — "Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners Using Deep Learning". wav2vec2-BERT utk segmentasi waqf Quran; dataset 850+ jam; PER 0.16%.
- [20] Yang et al. 2024, IEEE/ACM TASLP, arXiv:2404.09385 — "A Large-Scale Evaluation of Speech Foundation Models" (SUPERB extended). Data2vec Large = konten/fonetik terbaik & speaker-disentangled; wav2vec2/HuBERT simpan lebih banyak info speaker.
- [21] Meghanani & Hain 2024, EACL 2024 — "Improving Acoustic Word Embeddings through Correspondence Training of Self-supervised Speech Representations". AWE dari SSL (HuBERT/wav2vec2/WavLM) utk word discrimination lintas bahasa tanpa fine-tune encoder.

---

## TODOs

- [ ] 1. **Fix nama peneliti Tabel 1 baris 1** — `quick`
  Di grid tabel baris 1 (sekitar baris 412-414), kolom "Judul Jurnal dan Peneliti", ganti nama peneliti yang salah menjadi peneliti sitasi [1] yang benar. Pertahankan lebar sel grid table (padding spasi agar `|` tetap sejajar).
  - CARI (blok sel):
    ```
    |        | R Prajana,       |                       | mempelajari             |
    |        | Prof.Kavitha S N |                       | representasi langsung   |
    |        | (2021)           |                       | dari sinyal audio       |
    ```
  - GANTI:
    ```
    |        | Alexei Baevski,  |                       | mempelajari             |
    |        | dkk (2020)       |                       | representasi langsung   |
    |        |                  |                       | dari sinyal audio       |
    ```
  - Rasional: sitasi [1] = A. Baevski, H. Zhou, A. Mohamed, M. Auli (2020), bukan "R Prajana, Kavitha (2021)".

- [ ] 2. **Fix kolom Tujuan Tabel 1 baris 6 (DQ-Data2vec)** — `writing`
  Baris 6 (sekitar 523-540) kolom "Tujuan" salah tempel (isinya membahas Wav2Vec2 embedding & fonologi = copy dari baris 5). Tulis ulang kolom Tujuan agar sesuai isi DQ-Data2vec [13]. Pertahankan lebar sel grid (~23 char per baris, wrap manual).
  - Isi Tujuan yang BENAR (parafrasekan agar pas ke grid): *"Bertujuan memisahkan (decoupling) informasi bahasa dan fonem dalam SSL melalui dua K-means quantizer, menghasilkan representasi multibahasa yang lebih terpisah dan bersih. Pada CommonVoice, menurunkan phoneme error rate (PER) relatif 9,51% dan word error rate (WER) 11,58% dibanding Data2vec dan UniData2vec."*
  - Catatan: peneliti tabel baris 6 tertulis "Qijie Shao (2025)" — konsisten dgn [13]; boleh dilengkapi "dkk".

- [ ] 3. **Fix sitasi [17]→[11] pada paragraf sintesis (baris ~652)** — `quick`
  Pada kalimat *"penelitian mengenai fine-tuning Wav2Vec2 untuk pengenalan pembicara [17]"*, sitasi [17] salah ([17]=Librispeech). Ganti ke [11] (Vaessen & van Leeuwen, fine-tuning wav2vec2 speaker recognition).
  - CATATAN: bila task 6 (tulis ulang paragraf sintesis) mencakup kalimat ini, lakukan fix di task 6 sekalian dan tandai task 3 selesai.

- [ ] 4. **Reorganisasi + tambah 8 baris Tabel 1 (per tema)** — `writing`
  Susun ulang Tabel 1 menjadi berkelompok per tema, tambahkan 8 baris baru. Total ~18 baris. Pertahankan format grid `+---+` dengan 4 kolom (No | Judul & Peneliti | Metode | Tujuan). Boleh menambah baris "header tema" sebagai sel spanning ATAU cukup mengurutkan baris sesuai tema (pilih yang paling rapi untuk render Pandoc — urutan per tema lebih aman daripada spanning cell).

  **Urutan final baris tabel (per tema):**

  *Tema A — Fondasi Arsitektur Model SSL*
  1. wav2vec 2.0 [1]
  2. data2vec [12]

  *Tema B — Perbandingan & Analisis Representasi SSL*
  3. SUPERB / Yang et al. (2024) [20] — BARU
  4. Searching for Structure / probing [2]
  5. DQ-Data2vec [13]
  6. TranUSR [8] — BARU ke tabel

  *Tema C — SSL untuk Domain Arab/Quran*
  7. Hossain et al. (2026) [18] — BARU
  8. Abdelfattah et al. (2025) [19] — BARU
  9. ArTST [6] — BARU ke tabel
  10. Arabic SER BAVED [10]

  *Tema D — Retrieval / Similarity Search berbasis Embedding*
  11. WavRAG [5] — BARU ke tabel
  12. Query-by-Example / Öberg [3]
  13. Meghanani & Hain (2024) [21] — BARU
  14. Siamese Network spoofing [14]

  *Tema E — Generalisasi, Adaptasi Domain & Lintas-Bahasa*
  15. Li et al. cross-lingual [7] — BARU ke tabel
  16. XLSR / Conneau [16]
  17. CPT-Boosted wav2vec2 [15]
  18. Fine-tuning speaker recognition [11]

  **Isi ringkas 8 baris BARU (Metode | Tujuan) — parafrase ke grid:**
  - **[20] SUPERB (Yang et al. 2024)** | Metode: Evaluasi skala besar 36 model SSL (wav2vec2, HuBERT, Data2vec, WavLM) pada 15 tugas SUPERB (PR, SID, ASR, QbE, dll) via probing. | Tujuan: Membandingkan kapabilitas representasi antar-model SSL; menemukan Data2vec unggul pada konten/fonetik dan paling *speaker-disentangled*, sedangkan wav2vec2 & HuBERT menyimpan lebih banyak info pembicara — memberi dasar hipotesis pemilihan model untuk retrieval berbasis konten.
  - **[8] TranUSR** | Metode: UniData2vec (mengganti target diskrit UniSpeech dengan target kontinu-kontekstual ala Data2vec) + Phoneme-to-Word Transcoder. | Tujuan: Meningkatkan pembelajaran representasi fonetik lintas bahasa; target kontinu menurunkan phoneme error rate (PER) 5,3% relatif dibanding target diskrit, membuktikan keunggulan paradigma Data2vec untuk fitur fonetik.
  - **[18] Hossain et al. (2026)** | Metode: Fine-tuning & perbandingan sistematis Wav2Vec2, HuBERT, XLS-R pada 870+ jam audio Quran; ablasi feature extractor, format label, komposisi dataset. | Tujuan: Mengevaluasi model SSL mana yang menghasilkan representasi terbaik untuk ASR Quran; Wav2Vec2-XLSR-53 terbaik (WER 0,08). Menegaskan celah: komparasi SSL untuk domain Quran ada pada ASR, belum pada retrieval.
  - **[19] Abdelfattah et al. (2025)** | Metode: Pipeline anotasi Quran 98% otomatis memakai wav2vec2-BERT untuk segmentasi waqf; model CTC multi-level; rilis dataset 850+ jam. | Tujuan: Membuktikan representasi SSL (wav2vec2) dapat diadaptasi efektif untuk ucapan Arab/Quran yang kaya tajwid (PER 0,16%), sekaligus menyediakan sumber data domain Quran.
  - **[6] ArTST** | Metode: SpeechT5 (Transformer unified text-speech) di-*pretrain from scratch* pada MSA, lalu fine-tune ASR/TTS/dialek. | Tujuan: Menunjukkan model SSL Arab-monolingual mengungguli model multibahasa besar (Whisper, MMS) pada tugas Arab — mendukung argumen bahwa pra-pelatihan spesifik-bahasa menghasilkan representasi fonetik superior.
  - **[5] WavRAG** | Metode: WavRetriever (di atas Qwen2-Audio) dengan contrastive learning menyatukan audio & teks dalam satu ruang embedding; retrieval end-to-end via cosine similarity. | Tujuan: Menyediakan retrieval audio yang sepenuhnya melewati ASR; dibanding pipeline ASR-Teks-RAG, mencapai performa setara dengan percepatan 10× dan menghilangkan propagasi error transkripsi — bukti empiris inti argumen skripsi.
  - **[21] Meghanani & Hain (2024)** | Metode: Correspondence Auto-Encoder (CAE) di atas representasi beku HuBERT/Wav2vec2/WavLM untuk menghasilkan Acoustic Word Embeddings; evaluasi diskriminasi kata di 5 bahasa. | Tujuan: Membuktikan embedding SSL dapat dipakai untuk retrieval/diskriminasi kata lintas bahasa tanpa fine-tuning encoder — memvalidasi pendekatan metodologis skripsi (embedding beku + kemiripan vektor).
  - **[7] Li et al.** | Metode: Probing 5 model SSL terlatih-Inggris sebagai feature extractor beku untuk ASR lintas bahasa; metrik Phonetic-Syntax Ratio (PSR) via DGCCA. | Tujuan: Mengukur secara kuantitatif bahwa encoder SSL terlatih-Inggris menghasilkan fitur fonetik kurang informatif ketika diterapkan lintas bahasa; korelasi positif PSR dengan performa lintas bahasa — menjustifikasi kebutuhan model teradaptasi-Arab.

- [ ] 5. **Tambah 8 entri naratif poin untuk paper baru** — `writing`
  Bagian narasi SotA (poin a-j, baris ~243-395) hanya memuat 10 poin. Tambahkan 8 poin baru (lanjut huruf k-r, atau sisipkan sesuai tema mengikuti gaya paragraf existing) untuk: [20] SUPERB, [8] TranUSR, [18] Hossain, [19] Abdelfattah, [6] ArTST, [5] WavRAG, [21] Meghanani&Hain, [7] Li et al. Gaya: 1 paragraf per paper, sebut peneliti+tahun di awal, deskripsi metode + relevansi ke tugas retrieval/Quran/komparasi, akhiri dgn nomor sitasi. Gunakan metadata terverifikasi di atas.
  - GUARDRAIL: pastikan tiap nama peneliti di tabel (task 4) muncul di sini (no orphan). [4] & [9] TIDAK dibuat poin naratif tersendiri — masuk di paragraf sintesis (task 6).

- [ ] 6. **Tulis ulang paragraf sintesis SotA** — `writing`
  Paragraf sintesis (baris ~619-694) ditulis ulang agar: (a) mengalir dengan argumen ASR→retrieval, (b) menghapus penyebutan "AV-data2vec maupun AV2vec" yang tidak punya sitasi (baris ~646-649), (c) memperbaiki sitasi [17]→[11] (T3), (d) menyebut [4] Gemini Embedding 2 sebagai konteks native embedding retrieval, (e) menyebut [9] Quranic Conversations sebagai baseline pencarian Quran berbasis TEKS yang di-supersede pendekatan audio, (f) diakhiri paragraf research gap eksplisit → kontribusi skripsi.
  - Struktur alur yang disarankan: fondasi SSL [1][12] → ranah komparasi/analisis representasi [20][2][13][8] → domain Arab/Quran [18][19][6][10] → pergeseran ke retrieval berbasis embedding & bukti bypass ASR [5][3][21][14][4] → keterbatasan lintas-bahasa & adaptasi [7][16][15][11] → baseline Quran masih berbasis teks [9] → CELAH: komparasi langsung Wav2Vec2 vs Data2Vec untuk kualitas embedding pada similarity search/audio retrieval ayat Quran belum pernah dilakukan → kontribusi skripsi.
  - GUARDRAIL: jangan mengubah kalimat di luar paragraf sintesis; jangan menyentuh Gambar 1 Kerangka Pemikiran.

- [ ] 7. **Tambah 4 entri Daftar Pustaka [18]-[21]** — `writing`
  Di akhir DAFTAR PUSTAKA (setelah [17], baris ~1085-1089), tambahkan 4 entri format IEEE (samakan gaya dengan entri existing):
  - `[18] N. M. Hossain, R. Islam, and U. Obaidellah, "A Comparative Study of Pretrained Transformer Models for Quranic ASR: Speech Representations, Label Formats, and Dataset Composition," 2026, *arXiv*: arXiv:2606.19747.`
  - `[19] A. Abdelfattah, M. I. Khalil, and H. Abbas, "Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners Using Deep Learning," Aug. 2025, *arXiv*: arXiv:2509.00094.`
  - `[20] S. Yang et al., "A Large-Scale Evaluation of Speech Foundation Models," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2024. doi: 10.48550/arXiv.2404.09385.`
  - `[21] A. Meghanani and T. Hain, "Improving Acoustic Word Embeddings through Correspondence Training of Self-supervised Speech Representations," in *Proceedings of the 18th Conference of the European Chapter of the ACL (EACL 2024)*, Mar. 2024, pp. 1959–1967. doi: 10.18653/v1/2024.eacl-long.118.`
  - CATATAN: verifikasi ulang detail volume/halaman [20] jika ingin presisi maksimal (librarian menyebut IEEE/ACM TASLP 2024 + arXiv:2404.09385).

## Final Verification Wave

- [ ] F1. **Audit Sitasi & Orphan (QA)** — `unspecified-high`
  grep semua `\[1\]`–`\[21\]` di `Skripsi.md`. Pastikan: (a) [1]-[17] tidak bergeser makna, (b) [18]-[21] muncul di tabel/narasi DAN di Daftar Pustaka, (c) setiap nama peneliti di kolom tabel muncul ≥1x di narasi SotA. Laporkan pelanggaran dengan nomor baris.

- [ ] F2. **Validitas Grid Table & Konsistensi Istilah** — `unspecified-high`
  Cek Tabel 1 grid `+---+` masih valid (jumlah kolom & baris separator konsisten, render Pandoc tidak rusak). Cek konsistensi istilah "Wav2Vec2"/"Data2Vec" di seluruh bagian SotA. Cek paragraf akhir SotA memuat frasa celah penelitian → kontribusi.

---

## Commit Strategy
Tidak ada instruksi commit dari user. Jangan commit kecuali diminta.

## Success Criteria
- [ ] 4 bug diperbaiki.
- [ ] Tabel SotA ~18 baris, per tema, no orphan.
- [ ] 8 entri naratif baru sinkron dengan tabel.
- [ ] Paragraf sintesis ditulis ulang, gap eksplisit, [4] & [9] disebut sebagai konteks.
- [ ] Daftar Pustaka [18]-[21] ditambahkan.
- [ ] Nomor sitasi [1]-[17] tidak bergeser.

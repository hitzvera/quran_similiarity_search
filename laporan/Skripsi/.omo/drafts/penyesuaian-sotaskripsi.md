# Draft: Penyesuaian State of the Art (Bab 2) dengan Bab 1 — Skripsi Wav2Vec2 vs Data2Vec Retrieval Al-Quran

## Konteks Dokumen
- File target: `laporan/Skripsi/Skripsi.md`
- Topik skripsi: Perbandingan Wav2Vec2 dan Data2Vec terhadap fitur laten audio untuk retrieval tilawah Al-Quran.
- Fokus metrik: Similarity search (Cosine Similarity, Top-K Accuracy, MAP) — BUKAN transkripsi ASR.
- Bab 2 = "The State of The Art" (narasi poin a-j + Tabel 1 SotA + paragraf sintesis) dan "Studi Literatur".

## Requirements (dikonfirmasi user)
1. **Perubahan Bab 1 utama**: argumen propagasi error ASR→retrieval baru ditambahkan (ref [4] Gemini Embedding 2, [5] WavRAG). Ini yang jadi acuan penyelarasan SotA.
2. **Paper baru dari daftar pustaka**: masukkan SEMUA yang relevan ke tabel SotA — yaitu [4] Gemini Embedding 2, [5] WavRAG, [6] ArTST, [7] Li et al., [8] TranUSR, [9] Quranic Conversations.
3. **Saran paper baru dari luar daftar pustaka**: YA — cari & sarankan (prioritas: Quran/Arabic recitation audio, perbandingan SSL, audio retrieval SSL embedding).
4. **Perbaikan inkonsistensi tabel SotA**: perbaiki SEMUA.

## Inkonsistensi yang ditemukan (untuk diperbaiki)
- **Tabel SotA baris 1**: peneliti tertulis "R Prajana, Prof.Kavitha S N (2021)" — SALAH. Sitasi [1] = A. Baevski, H. Zhou, A. Mohamed, M. Auli (2020).
- **Tabel SotA baris 6 (DQ-Data2vec)**: kolom "Tujuan" salah tempel — isinya membahas Wav2Vec 2.0 embedding & fonologi, bukan DQ-Data2vec decoupling quantization.
- **Paragraf sintesis setelah tabel**: menyebut "AV-data2vec maupun AV2vec" TANPA sitasi di daftar pustaka. Penomoran ref campur ([16]/[17] dipakai untuk hal berbeda; [17] adalah Librispeech tapi dirujuk sebagai "fine-tuning speaker recognition").
- Narasi poin a-j: entri baru ([4],[5],[6],[7],[8],[9]) belum ada di daftar poin naratif maupun tabel.
- Nomor sub-bab Studi Literatur ganjil (7.1, 7.2, 7.3 padahal bab-nya bernomor lain) — perlu dicek konsistensi penomoran.

## Struktur SotA saat ini (10 entri tabel)
1. wav2vec 2.0 [1] — nama peneliti salah
2. data2vec [12]
3. Arabic SER wav2vec2/HuBERT (BAVED) [10]
4. Query-by-Example audio search wav2vec2 (Oberg) [3]
5. Searching for Structure (probing) [2]
6. DQ-Data2vec [13] — kolom tujuan salah
7. Siamese Network spoofing [14]
8. Fine-tuning wav2vec2 speaker recognition [11]
9. CPT-Boosted wav2vec2 [15]
10. Unsupervised Cross-Lingual (XLSR) [16]

## Rencana penambahan entri SotA (kandidat, menunggu verifikasi librarian)
- [4] Gemini Embedding 2 — dukung argumen native embedding retrieval > two-stage ASR.
- [5] WavRAG — audio-integrated retrieval menghindari propagasi error transkripsi.
- [6] ArTST — Arab-monolingual SSL > multibahasa.
- [7] Li et al. — encoder SSL terlatih-Inggris kurang informatif lintas bahasa.
- [8] TranUSR — target kontinu ala Data2Vec menurunkan cross-lingual PER 5.3%.
- [9] Quranic Conversations — embedding + cosine similarity untuk pencocokan konten Quran (domain grounding).
- + paper baru hasil pencarian librarian (menunggu).

## Research Findings

### Verifikasi 6 paper existing (dari daftar pustaka) — SIAP MASUK TABEL
- **[4] Gemini Embedding 2** — Shanbhogue et al., **2026** (BUKAN 2026 di judul tapi arXiv:2605.27295 = Mei 2026). CATATAN: tahun harus 2026. Native multimodal embedding (audio+text 1 ruang), contrastive learning. Dukung argumen audio→text retrieval tanpa ASR (implisit, bukan komparasi eksplisit ASR-pipeline).
- **[5] WavRAG** — Yifu Chen et al., **2025, ACL 2025** (bukan sekadar arXiv). WavRetriever di atas Qwen2-Audio + contrastive. Bypass ASR total → 10x lebih cepat + hilangkan propagasi error. BUKTI EMPIRIS KUAT untuk argumen inti Bab 1.
- **[6] ArTST** — Toyin et al., 2023, **ArabicNLP 2023**. SpeechT5 di-pretrain from scratch untuk Arab. Monolingual Arab > multibahasa (Whisper, MMS).
- **[7] Li et al.** — Shuyue Stella Li et al., 2023, **ICNLSP 2023** (bukan sekadar "2023"). Probing 5 SSL English via metrik PSR/DGCCA. English SSL → fitur fonetik kurang informatif lintas bahasa. wav2vec2 contrastive > masked prediction untuk cross-lingual.
- **[8] TranUSR** — Hongfei Xue, Qijie Shao et al., 2023, **INTERSPEECH 2023** (6 penulis). UniData2vec (target kontinu Data2vec) + P2W Transcoder. Target kontinu turunkan PER 5.3% relatif vs UniSpeech discrete. Langsung membenturkan target kontinu (Data2vec) vs diskrit (Wav2Vec2).
- **[9] Quranic Conversations** — Shohoud et al., 2023, arXiv. Word2Vec(CBOW) di 30+ tafsir + cosine similarity. TEXT-ONLY (bukan audio). Evaluasi preliminer (1 query, cosine 0.97). Frame sebagai baseline pencarian Quran berbasis TEKS yang di-supersede pendekatan audio.

### CATATAN KOREKSI SITASI (untuk daftar pustaka)
- [4] tahun = **2026** (bukan yang tertulis, cek konsistensi).
- [7] venue = ICNLSP 2023.
- [8] = 6 penulis (Xue, Shao, Chen, Guo, Xie, Liu).
- [9] hasil 0.97 = 1 data point, frame hati-hati.

### Paper BARU disarankan (7 kandidat, terverifikasi, tidak duplikat)
PRIORITAS TINGGI (domain Quran + komparasi SSL retrieval):
- **Hossain et al. (2026)** arXiv:2606.19747 — "Comparative Study of Pretrained Transformer Models for Quranic ASR". Bandingkan Wav2Vec2, HuBERT, XLS-R di 870+ jam Quran. XLSR-53 terbaik (WER 0.08). PALING DEKAT dgn tesis → perkuat gap: komparasi utk ASR ada, utk RETRIEVAL belum. [HIGH]
- **Abdelfattah et al. (2025)** arXiv:2509.00094 — wav2vec2-BERT utk Quran, dataset 850+ jam open-source, PER 0.16%. [HIGH]
- **Salman et al. (2026)** arXiv:2601.17880 — "Quran-MD" dataset multimodal verse-level 32 reciter, dibuat utk multimodal embedding & semantic retrieval. [HIGH — bisa utk dataset section]
- **Al-Rfooh et al. (2025)** IJEECS DOI:10.11591/ijeecs.v40.i3.pp1486-1499 — "QV Finder" retrieval ayat via Whisper + string matching. Baseline ASR-then-retrieve. [MEDIUM]
KOMPARASI SSL (framework teoretis):
- **Yang et al. (2024)** IEEE/ACM TASLP arXiv:2404.09385 — SUPERB besar. Data2vec = konten paling murni & speaker-disentangled; wav2vec2/HuBERT simpan info speaker. HIPOTESIS FALSIFIABEL utk retrieval. [HIGH]
- **Huo & Dunbar (2025)** Interspeech DOI:10.21437/Interspeech.2025-514 — kenapa HuBERT != wav2vec2 (iterasi, bukan objektif). [MEDIUM]
RETRIEVAL SSL:
- **Meghanani & Hain (2024)** EACL DOI:10.18653/v1/2024.eacl-long.118 — Acoustic Word Embeddings dari SSL utk retrieval lintas bahasa tanpa fine-tune encoder. [HIGH]

## Open Questions — RESOLVED
- Paper baru masuk tabel: Hossain 2026, Yang 2024 SUPERB, Abdelfattah 2025, Meghanani&Hain 2024 (4 paper).
- Struktur tabel: PER TEMA.
- Cakupan: State of the Art SAJA. Studi Literatur (7.1-7.3) TIDAK disentuh.
- Format file: Markdown/Pandoc (grid table `+---+`), hasil konversi Word. Tabel grid = format BENAR.
- Nomor sitasi [1]-[17] DIKUNCI. Paper baru di-append [18]+.

## KOMPOSISI FINAL TABEL SotA (per tema) — 18 baris
Nomor sitasi baru: [18]=Hossain2026, [19]=Abdelfattah2025, [20]=Yang2024 SUPERB, [21]=Meghanani&Hain2024

**Tema A — Fondasi Arsitektur Model SSL**
1. wav2vec 2.0 [1] (FIX nama: Baevski, Zhou, Mohamed, Auli 2020)
2. data2vec [12]

**Tema B — Perbandingan & Analisis Representasi SSL**
3. SUPERB / Yang et al. 2024 [20] (BARU) — Data2vec konten-murni vs wav2vec2 simpan speaker
4. Searching for Structure / probing [2]
5. DQ-Data2vec [13] (FIX kolom Tujuan yang salah tempel)
6. TranUSR [8] (BARU ke tabel) — target kontinu Data2vec turun PER 5.3%

**Tema C — SSL untuk Domain Arab/Quran**
7. Hossain et al. 2026 [18] (BARU) — komparasi SSL Quran ASR, XLSR terbaik
8. Abdelfattah et al. 2025 [19] (BARU) — wav2vec2-BERT Quran, dataset 850 jam
9. ArTST [6] (BARU ke tabel) — Arab monolingual > multibahasa
10. Arabic SER BAVED [10]

**Tema D — Retrieval / Similarity Search berbasis Embedding**
11. WavRAG [5] (BARU ke tabel) — bypass ASR, 10x cepat, no error propagation
12. Query-by-Example / Oberg [3]
13. Meghanani & Hain 2024 [21] (BARU) — AWE dari SSL utk retrieval lintas bahasa
14. Siamese Network spoofing [14]

**Tema E — Generalisasi, Adaptasi Domain & Lintas-Bahasa**
15. Li et al. cross-lingual [7] (BARU ke tabel) — English SSL degradasi lintas bahasa
16. XLSR [16]
17. CPT-Boosted wav2vec2 [15]
18. Fine-tuning speaker recognition [11]

**KONTEKS NARASI SAJA (tidak masuk tabel):**
- [4] Gemini Embedding 2 — dukung native embedding retrieval (paragraf pembuka).
- [9] Quranic Conversations — baseline pencarian Quran berbasis TEKS (paragraf gap).

## DAFTAR PERUBAHAN (checklist eksekusi)
1. [ ] FIX tabel baris 1: nama peneliti → Baevski dkk 2020.
2. [ ] FIX tabel baris 6: kolom Tujuan DQ-Data2vec (tulis ulang benar).
3. [ ] Tambah 8 baris tabel baru ([20],[8],[18],[19],[6],[5],[21],[7]) — total jadi 18, reorganisasi per tema.
4. [ ] Tambah entri naratif poin (a-j → tambah) untuk paper baru yang masuk tabel.
5. [ ] Tulis ulang paragraf sintesis: alur argumen ASR->retrieval, hapus "AV-data2vec/AV2vec" tanpa sitasi, rapikan penomoran (FIX [17]->[11] di baris 652), tegaskan research gap + sebut [4] & [9] sebagai konteks.
6. [ ] Tambah entri Daftar Pustaka [18]-[21].
7. [ ] Update Daftar Tabel bila caption/isi berubah (Tabel 1 tetap).
8. [ ] Konsistensi istilah: pilih "Wav2Vec2" & "Data2Vec".

## QA (dari Metis)
- Setiap baris tabel punya >=1 kalimat naratif (no orphan).
- Paragraf akhir SotA eksplisit "celah penelitian" -> kontribusi.
- Nomor sitasi [1]-[17] tak bergeser; baru contiguous [18]-[21].

## Scope Boundaries
- INCLUDE: Bab 2 (State of the Art naratif + Tabel 1 + paragraf sintesis), Daftar Pustaka (paper baru [18]-[21]), Daftar Tabel/penomoran terkait.
- EXCLUDE: Bab 1 (acuan saja), Studi Literatur 7.1-7.3, Metode Penelitian, Jadwal Penelitian.

## BUG TAMBAHAN (temuan Oracle)
- **Baris 652 Skripsi.md**: "*fine-tuning* Wav2Vec2 untuk pengenalan pembicara [17]" — sitasi SALAH. [17]=Librispeech. Speaker recognition = [11]. FIX: ganti [17]->[11].

# Draft Jawaban Munaqosah: Q38–Q56 + Q56A

> **Catatan**: Q38–Q49 sudah ada di `Panduan-Pertanyaan-Sidang.md`. Dokumen ini menambahkan Q50–Q56 dan Q56A untuk melengkapi topik yang sering muncul dalam sidang. Jawaban ditulis dalam gaya bicara akademis Indonesia, singkat, dan didasarkan pada naskah skripsi. Angka dan klaim tidak boleh diganti dengan tebakan.

---

## Q38–Q49 (Sudah Ada di Panduan)

*Jawaban untuk Q38–Q49 sudah tersedia di file `Panduan-Pertanyaan-Sidang.md` bagian 9–11. Baca file tersebut untuk jawaban lengkap.*

---

## Q50. Bisa sebutkan judul persis skripsi ini?

**Jawaban Singkat:**
"Perbandingan Wav2Vec2 dan Data2Vec terhadap Fitur Laten Audio untuk Tilawah Al-Quran."

**Jawaban Lengkap:**
Judul lengkap skripsi ini adalah **Perbandingan Wav2Vec2 dan Data2Vec terhadap Fitur Laten Audio untuk Tilawah Al-Quran**. Penelitian ini membandingkan dua model audio *self-supervised* dalam tugas *retrieval* ayat Al-Quran langsung dari audio, tanpa melalui tahap transkripsi. Parameter model dibekukan, embedding diekstrak dari 13 lapisan Transformer, dan evaluasi dilakukan menggunakan metrik MAP, MRR, dan Top-K pada 13 sel eksperimen.

**Evidence:**
- Halaman judul skripsi (halaman 1)
- Rumusan masalah dan tujuan penelitian (Bab 1.2 dan 1.3)

---

## Q51. Apa motivasi penelitian ini dan di mana celah penelitiannya?

**Jawaban Singkat:**
Motivasi utamanya adalah perbedaan paradigma *pretraining* antara Wav2Vec2 (kontrastif) dan Data2Vec (*self-distillation*) belum diuji pada tugas *retrieval* audio Al-Quran. Celah penelitiannya: studi sebelumnya hanya menguji model kontrastif seperti Wav2Vec2 untuk ASR lintas bahasa, sedangkan Data2Vec dengan target kontekstual kontinu belum dievaluasi pada domain fonetik spesifik seperti bacaan Al-Quran dengan metrik *similarity search*.

**Jawaban Lengkap:**
Penelitian ini berangkat dari fenomena bahwa model SSL seperti Wav2Vec2 dan Data2Vec mampu menghasilkan representasi laten dari audio mentah. Namun, bukti empiris tentang keunggulan representasi ini pada tugas *retrieval* Al-Quran masih terbatas. Studi terdahulu seperti Li et al. (2023) dan Toyin et al. (2023) menunjukkan bahwa model SSL yang dilatih pada bahasa Inggris kurang informatif untuk fonetik Arab, tetapi bukti tersebut hanya terbatas pada arsitektur kontrastif seperti Wav2Vec2. Data2Vec dengan paradigma *self-distillation* dan target representasi kontekstual penuh belum diuji pada skenario serupa.

Lebih jauh, perbandingan langsung Wav2Vec2 dan Data2Vec untuk tugas *audio retrieval* dengan metrik MAP, MRR, dan Top-K—bukan metrik transkripsi seperti WER—masih minim di literatur. Ketiadaan perbandingan ini menyisakan pertanyaan: apakah perbedaan paradigma *pretraining* menghasilkan kualitas representasi fonetik yang berbeda untuk tugas *retrieval* ayat Al-Quran.

**Evidence:**
- Latar Belakang (Bab 1.1), paragraf 2–6
- Tinjauan Pustaka (Bab 2.1), diskusi tentang Li et al. [7] dan Toyin et al. [6]
- Kerangka Pemikiran (Bab 1.6)

---

## Q52. Apa perbedaan penelitian ini dengan studi sebelumnya?

**Jawaban Singkat:**
Perbedaan utamanya ada tiga: (1) tugas yang dievaluasi adalah *retrieval* ayat dari audio, bukan ASR atau klasifikasi pembicara; (2) parameter model dibekukan tanpa *fine-tuning*, sehingga yang diuji adalah kemampuan representasi bawaan; (3) protokol evaluasi 13 sel yang memvariasikan skenario sumber data dan rasio pemilik, dengan penguncian konfigurasi di *dev set* dan uji bootstrap berpasangan untuk MAP.

**Jawaban Lengkap:**
Penelitian ini berbeda dari studi ASR Al-Quran seperti Aswat (Alkanhal et al., 2023) dalam beberapa hal. Pertama, Aswat mengevaluasi kinerja model untuk tugas transkripsi dengan metrik WER, sedangkan penelitian ini mengevaluasi *retrieval* dengan metrik MAP, MRR, dan Top-K. Kedua, Aswat melakukan *fine-tuning* pada data Arab, sedangkan penelitian ini membekukan parameter model untuk menguji representasi bawaan. Ketiga, penelitian ini merancang 13 sel eksperimen yang memvariasikan sumber data dan rasio pemilik, yang tidak ada dalam studi Aswat.

Perbedaan juga terlihat dibandingkan studi *query-by-example* seperti Öberg (2025). Öberg menggunakan Wav2Vec2 untuk pencarian audio umum, sedangkan penelitian ini secara khusus mengevaluasi retrieval ayat Al-Quran dengan label relevansi berdasarkan pasangan (surah, ayat) dan desain eksperimen yang mengontrol kebocoran data melalui pemisahan identitas pembaca.

**Evidence:**
- Tinjauan Pustaka (Bab 2.1), Tabel 2.1 State of the Art
- Latar Belakang (Bab 1.1), paragraf tentang perbedaan ASR dan retrieval
- Metodologi (Bab 3), penjelasan 13 sel dan protokol evaluasi

---

## Q53. Apa rumusan masalah dan tujuan penelitian?

**Jawaban Singkat:**
Rumusan masalah: (1) Bagaimana implementasi Wav2Vec2 dan Data2Vec dalam tugas *retrieval* ayat Al-Quran? (2) Bagaimana kinerja *frozen embedding* kedua model dalam tugas tersebut? Tujuan: (1) Mengimplementasikan kedua model untuk *retrieval* ayat. (2) Mengevaluasi kinerja *frozen embedding* mereka menggunakan metrik MAP, MRR, dan Top-K.

**Jawaban Lengkap:**
Rumusan masalah penelitian ini terbagi menjadi dua pertanyaan. Pertama, bagaimana implementasi model Wav2Vec2 dan Data2Vec dalam tugas *retrieval* ayat Al-Quran? Kedua, bagaimana kinerja *frozen embedding* Wav2Vec2 dan Data2Vec dalam tugas *retrieval* ayat Al-Quran?

Tujuan penelitian menjawab kedua rumusan tersebut: (1) mengimplementasikan model Wav2Vec2 dan Data2Vec dalam tugas *retrieval* ayat Al-Quran, mencakup persiapan audio, ekstraksi representasi dari 13 lapisan, pembentukan vektor embedding, dan pemeringkatan kandidat berdasarkan *cosine similarity*; (2) mengevaluasi kinerja *frozen embedding* kedua model menggunakan metrik MAP, MRR, Top-1, Top-5, dan Top-10 pada 13 sel eksperimen yang memvariasikan skenario sumber data dan rasio pemilik.

**Evidence:**
- Rumusan Masalah (Bab 1.2)
- Tujuan Penelitian (Bab 1.3)
- Kesimpulan (Bab 5.1) yang menjawab kedua rumusan

---

## Q54. Apa manfaat penelitian ini?

**Jawaban Singkat:**
Manfaat penelitian ini memberikan pemahaman ilmiah tentang efektivitas model representasi audio Wav2Vec2 dan Data2Vec dalam domain bacaan Al-Quran, khususnya untuk tugas *retrieval* ayat. Hasil penelitian juga menjadi pedoman bagi peneliti dan pengembang dalam memilih model yang sesuai untuk tugas *retrieval* ayat Al-Quran berdasarkan skenario dan rasio pemilik.

**Jawaban Lengkap:**
Penelitian ini memiliki manfaat teoretis dan praktis. Secara teoretis, penelitian memberikan pemahaman ilmiah tentang seberapa efektif model representasi audio seperti Wav2Vec2 dan Data2Vec dalam domain pembacaan ayat Al-Quran, khususnya dalam tugas *retrieval* ayat tanpa melalui tahap transkripsi. Secara praktis, hasil penelitian menjadi pedoman atau acuan bagi peneliti dan pengembang perangkat lunak dalam memilih model yang cocok untuk tugas *retrieval* ayat Al-Quran. Misalnya, untuk domain Quran-MD murni (skenario B), Wav2Vec2 titik 7 memberikan hasil terkuat; untuk skenario lintas sumber (A) atau lintas mahasiswa + Quran-MD (D), Data2Vec titik 5 atau 6 lebih tinggi MAP-nya.

**Evidence:**
- Manfaat Penelitian (Bab 1.5)
- Deployment Konseptual (Bab 4.6), rekomendasi model per skenario

---

## Q55. Apakah penelitian ini memiliki hipotesis H1 dan H2?

**Jawaban Singkat:**
**Tidak**. Penelitian ini bersifat komparatif-empiris, bukan eksperimen dengan hipotesis formal H1/H2. Tujuan penelitian adalah membandingkan kinerja Wav2Vec2 dan Data2Vec pada berbagai kondisi evaluasi, bukan menguji hipotesis bahwa satu model secara inheren lebih baik dari yang lain.

**Jawaban Lengkap:**
Penelitian ini tidak merumuskan hipotesis formal H1 atau H2. Sifat penelitian adalah komparatif-empiris: membandingkan kinerja *frozen embedding* Wav2Vec2 dan Data2Vec pada tugas *retrieval* ayat Al-Quran dalam berbagai kondisi skenario dan rasio pemilik. Tujuan penelitian adalah memperoleh gambaran empiris mengenai kinerja masing-masing model, bukan membuktikan bahwa satu model secara universal lebih baik.

Hasil penelitian menunjukkan tidak ada pemenang universal. Data2Vec memiliki MAP numerik lebih tinggi pada 9 dari 13 sel, Wav2Vec2 pada 4 sel. Namun, dua sel tidak menunjukkan perbedaan signifikan secara statistik. Pola kemenangan bergantung pada skenario dan rasio pemilik, bukan pada arsitektur secara universal. Klaim yang dapat dipertanggungjawabkan adalah klaim yang terikat pada *evidence* dalam naskah, bukan generalisasi tentang keunggulan satu model.

**Evidence:**
- Rumusan Masalah (Bab 1.2) dan Tujuan Penelitian (Bab 1.3) tidak menyebutkan hipotesis
- Kesimpulan (Bab 5.1): "tidak terdapat satu model yang dapat dinyatakan paling baik untuk seluruh kondisi dan metrik"
- Bab 4.5.2: "Tidak ada pemenang universal"

---

## Q56. Teori apa saja yang digunakan dalam penelitian ini?

**Jawaban Singkat:**
Penelitian ini menggunakan landasan teori: (1) *Self-Supervised Learning* (SSL) untuk representasi ucapan; (2) arsitektur Wav2Vec2 dengan *contrastive learning*; (3) arsitektur Data2Vec dengan *self-distillation* ke target kontekstual; (4) *cosine similarity* sebagai fungsi skor; (5) metrik evaluasi retrieval MAP, MRR, dan Top-K; (6) metode CRISP-DM yang diadaptasi.

**Jawaban Lengkap:**
Landasan teori penelitian ini mencakup beberapa konsep utama. Pertama, *Self-Supervised Learning* (SSL) untuk representasi ucapan, yang memungkinkan model mempelajari representasi dari audio mentah tanpa label transkripsi. Kedua, arsitektur Wav2Vec2 yang menggunakan *contrastive learning* dengan *quantized discrete units* (Baevski et al., 2020). Ketiga, arsitektur Data2Vec yang menggunakan *self-distillation* dengan target representasi kontekstual penuh dari jaringan *teacher* (Baevski et al., 2022). Keempat, *cosine similarity* sebagai fungsi skor untuk menyusun peringkat kandidat, bukan sebagai metrik evaluasi akhir. Kelima, metrik evaluasi retrieval yaitu MAP, MRR, dan Top-K (Manning et al., 2008). Keenam, metode penelitian CRISP-DM yang diadaptasi menjadi enam fase: *business understanding*, *data understanding*, *data preparation*, *modeling*, *evaluation*, dan *deployment*.

**Evidence:**
- Landasan Teori (Bab 2.2): 2.2.1–2.2.11
- Daftar Pustaka: [1] Baevski 2020 (Wav2Vec2), [8] Baevski 2022 (Data2Vec), [21] Manning 2008 (metrik retrieval), [26] Schröer 2021 (CRISP-DM)

---

## Q56A. Apakah data dalam penelitian ini termasuk data primer atau data sekunder?

**Jawaban Singkat:**
Penelitian ini menggunakan **keduanya**. Data primer adalah rekaman mahasiswa yang dikumpulkan langsung dari tugas Tahfidz. Data sekunder adalah dataset Quran-MD yang sudah tersedia dan digunakan sebagai referensi.

**Jawaban Lengkap:**
Penelitian ini menggunakan dua jenis data. **Data primer** adalah rekaman audio mahasiswa yang dikumpulkan langsung dari pengumpulan tugas Tahfidz. Rekaman ini berupa bacaan satu surah penuh per berkas, yang kemudian disegmentasi menjadi klip per ayat menggunakan stempel waktu kata dari WhisperX. Setelah validasi teknis, diperoleh 60 mahasiswa dengan 25.829 klip final.

**Data sekunder** adalah dataset Quran-MD (Salman et al., 2026) yang sudah tersedia dan menyediakan audio tingkat ayat beserta metadata qari, surah, dan ayat. Quran-MD digunakan sebagai data referensi pada skenario A, B, dan bagian dari basis data pada skenario D. Setelah validasi, diperoleh 17.127 referensi Quran-MD yang dapat digunakan.

Kedua sumber data ini tidak selalu ditempatkan pada sisi yang sama dalam eksperimen. Peran mereka bervariasi tergantung skenario: sebagai kueri, basis data referensi, atau keduanya. Desain ini memungkinkan evaluasi kinerja model dalam berbagai kondisi sumber data dan rasio pemilik.

**Evidence:**
- Data Understanding (Bab 3.2 dan 4.2)
- Tabel 3.2: Peran sumber data dalam rancangan eksperimen
- Daftar Pustaka [27]: Quran-MD (Salman et al., 2026)

---

## Catatan Tambahan untuk Munaqosah

### Tentang Bootstrap dan Signifikansi

Jika ditanya lebih lanjut tentang bootstrap:
- Bootstrap dilakukan pada **selisih AP berpasangan per kueri** (AP Data2Vec dikurangi AP Wav2Vec2)
- B = 10.000 replikasi, *seed* 42
- Interval kepercayaan 95% dari persentil 2,5 dan 97,5
- Jika interval tidak mencakup nol, selisih MAP dianggap signifikan
- **Hanya MAP yang diuji signifikansinya**; MRR dan Top-K hanya dibahas deskriptif

### Tentang Hasil vs Pembahasan

Jika ditanya perbedaan hasil dan pembahasan:
- **Hasil** = angka metrik yang diperoleh dari eksperimen (MAP, MRR, Top-K per sel)
- **Pembahasan** = interpretasi pola hasil, kaitan dengan literatur, dan implikasi
- Contoh: Hasil = "Data2Vec MAP 1,67% di sel A, Wav2Vec2 1,50%"
- Pembahasan = "Pola ini menunjukkan Data2Vec lebih tinggi secara numerik di skenario lintas sumber, tetapi kinerja absolut masih rendah sehingga sistem lebih cocok sebagai pembangkit kandidat"

### Tentang Tie dalam Bootstrap

Jika ditanya bagaimana menangani kasus tie (interval mencakup nol):
- Dua sel (C-60:40 dan C-80:20) memiliki interval kepercayaan yang mencakup nol
- Artinya, perbedaan MAP di kedua sel tersebut **tidak signifikan secara statistik**
- Klaim yang aman: "Data2Vec unggul numerik, tetapi perbedaan belum cukup kuat untuk membedakan kedua model secara statistik di sel tersebut"
- Tidak boleh mengklaim "Data2Vec lebih baik" di sel yang tidak signifikan

### Tentang Acknowledgments dan Abstract

Jika ditanya tentang bagian pembuka skripsi:
- **Acknowledgments** (Kata Pengantar): ucapan terima kasih kepada pembimbing, narasumber, keluarga. Tidak perlu dibaca detail saat sidang, cukup sebutkan jika ditanya.
- **Abstract**: ringkasan singkat latar belakang, metode, hasil utama, dan kesimpulan. Jika ditanya, bacakan abstract dan siap jelaskan setiap kalimatnya. Abstract harus konsisten dengan isi Bab 1–5.

### Tentang Motivasi, Celah, dan Perbedaan

Jika ditanya lebih lanjut tentang motivasi:
- **Motivasi**: perbedaan paradigma *pretraining* Wav2Vec2 (kontrastif) vs Data2Vec (*self-distillation*) belum diuji pada *retrieval* Al-Quran
- **Celah (gap)**: studi sebelumnya hanya menguji model kontrastif untuk ASR lintas bahasa; Data2Vec belum dievaluasi pada domain fonetik spesifik seperti Al-Quran dengan metrik *similarity search*
- **Perbedaan**: tugas *retrieval* (bukan ASR), parameter beku (bukan *fine-tuning*), protokol 13 sel (bukan evaluasi tunggal)

---

## Fact Sheet Tambahan (untuk Q50–Q56A)

| Fakta | Nilai |
|---|---|
| Judul skripsi | Perbandingan Wav2Vec2 dan Data2Vec terhadap Fitur Laten Audio untuk Tilawah Al-Quran |
| Rumusan masalah | 2 pertanyaan (implementasi dan kinerja) |
| Tujuan penelitian | 2 tujuan (mengimplementasikan dan mengevaluasi) |
| Hipotesis formal | **Tidak ada** (penelitian komparatif-empiris) |
| Jenis data | Data primer (rekaman mahasiswa) + data sekunder (Quran-MD) |
| Teori utama | SSL, Wav2Vec2, Data2Vec, cosine similarity, metrik retrieval, CRISP-DM |
| Sel tidak signifikan | 2 (C-60:40 dan C-80:20) |
| Bootstrap B | 10.000 replikasi |
| Seed | 42 |

---

## Kalimat Penutup yang Siap Diucapkan

> "Penelitian ini tidak menghasilkan pemenang universal antara Wav2Vec2 dan Data2Vec. Yang dihasilkan adalah **pola komparatif yang bergantung pada skenario dan rasio pemilik**, protokol evaluasi 13 sel yang dapat direproduksi, serta rancangan deployment konseptual yang jujur mengenai syarat yang belum diuji. Klaim yang dapat saya pertanggungjawabkan adalah klaim yang terikat pada evidence dalam naskah—dan di luar itu, saya menyampaikan interpretasi sebagai interpretasi, bukan fakta."

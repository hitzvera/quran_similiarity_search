# Panduan Pertanyaan Sidang Skripsi

> **Tujuan**: panduan latihan lisan untuk menjawab pertanyaan sidang. Jawaban ditulis dalam gaya bicara akademis Indonesia, singkat (1–3 paragraf), dan didasarkan pada skripsi. Angka, istilah, dan klaim tidak boleh diganti dengan tebakan.
>
> **Cara memakai**: baca *Ringkasan Cepat* dulu, hafalkan *Fact Sheet* di akhir, lalu latihan jawab *Pertanyaan Jebakan* dengan suara keras. Setiap pertanyaan diikuti **follow-up** yang sering diajukan penguji.
>
> **Konvensi**: istilah *evidence* = apa yang didukung angka pada naskah; *interpretasi* = penjelasan masuk akal yang tidak diklaim sebagai hubungan kausal. Jika ragu, tarik jawaban ke evidence.

---

## 1. Ringkasan Cepat

**Q1. Bisa dijelaskan inti skripsi ini dalam 60 detik?**
Penelitian membandingkan kualitas representasi laten dua model audio pralatih—Wav2Vec2 dan Data2Vec—pada tugas *retrieval* ayat Al-Qur'an langsung dari audio. Tidak ada pelatihan ulang; parameter model dibekukan, embedding diekstrak dari 13 lapisan Transformer, diringkas dengan *mean pooling* menjadi vektor 768 dimensi, lalu dibandingkan dengan *cosine similarity*. Evaluasi dilakukan pada 13 sel yang terbentuk dari empat skenario (A, B, C, D) dengan metrik MAP, MRR, Top-1/5/10. Pemilihan titik representasi dikunci di *development set*, lalu diuji sekali di *test set*. Hasil menunjukkan tidak ada pemenang universal: Data2Vec unggul secara numerik di 9 dari 13 sel (A, empat C, empat D), sedangkan Wav2Vec2 unggul di empat sel B.

> **Follow-up**: "Lalu apa kontribusi konkretnya?" → Alur ekstraksi, matriks 13 sel, dan protokol evaluasi yang dapat direproduksi, serta temuan empiris bahwa pilihan model harus disesuaikan dengan skenario.

**Q2. Apa yang membedakan penelitian ini dengan studi *speaker verification* atau ASR pada bacaan Al-Qur'an yang sudah ada?**
Penelitian ini tidak melakukan pengenalan pembicara atau transkripsi, melainkan *retrieval* ayat dari audio. Relevansi ditentukan oleh pasangan *(surah, ayat)*, bukan identitas qari atau transkripsi. Fokusnya adalah kualitas representasi bawaan dua model pada kondisi domain dan rasio pemilik yang bervariasi, bukan arsitektur klasifikasi baru.

> **Follow-up**: "Tapi kan sudah ada SUPERB?" → SUPERB adalah benchmark umum untuk representasi ucapan, sedangkan penelitian ini secara khusus mengevaluasi retrieval pada korpus bacaan Al-Qur'an dengan sumber mahasiswa dan Quran-MD serta desain 13 sel.

---

## 2. Latar Belakang dan *Novelty*

**Q3. Mengapa Wav2Vec2 dan Data2Vec yang dipilih?**
Keduanya adalah model audio *self-supervised* yang banyak dijadikan rujukan dan memiliki mekanisme pralatih berbeda. Wav2Vec2 memakai pembelajaran kontrastif, sedangkan Data2Vec memakai *self-distillation* ke target representasi kontekstual. Perbedaan tersebut perlu diuji pada tugas *retrieval* karena keberhasilan pada ASR tidak otomatis berlaku pada pemeringkatan audio.

> **Follow-up**: "Kenapa tidak HuBERT atau wavLM juga?" → Cakupan penelitian dibatasi pada dua model dengan arsitektur inti yang berbeda agar perbandingan tetap terkontrol. Model tambahan bisa menjadi penelitian lanjutan.

**Q4. Apa *novelty* penelitian ini?**
Ada tiga: (1) formulasi tugas *retrieval* ayat dari audio mahasiswa dengan label relevansi *(surah, ayat)*; (2) protokol evaluasi 13 sel yang memisahkan pengaruh domain, pembaca, dan rasio pemilik; (3) penyapuan 13 titik representasi + penguncian konfigurasi di *dev set* + uji bootstrap berpasangan untuk MAP.

> **Follow-up**: "Apakah *novelty*-nya arsitektur?" → Bukan arsitektur baru, melainkan **rancangan eksperimen komparatif** pada korpus Al-Qur'an dengan pembekuan parameter dan evaluasi ketat.

**Q5. Apa rumusan masalahnya?**
"Bagaimana kualitas representasi laten Wav2Vec2 dan Data2Vec pada tugas *retrieval* audio ayat Al-Qur'an dalam berbagai kondisi domain dan rasio pemilik?" Jawaban tidak bisa hanya "Data2Vec lebih baik"—harus disebutkan **pada sel dan metrik mana**.

**Q6. Kenapa *retrieval*, bukan klasifikasi surah?**
Klasifikasi memaksa keputusan tunggal ke satu label, sedangkan pada kenyataannya satu kueri perlu daftar kandidat terurut. *Retrieval* dengan MAP dan Top-K lebih realistis untuk skenario pencarian bacaan ayat, di mana pengguna butuh beberapa ayat yang mungkin cocok.

---

## 3. *Data Preparation*

**Q7. Dari mana data berasal dan berapa jumlahnya?**
Dua sumber: (1) rekaman mahasiswa dari 81 folder awal, yang setelah validasi teknis menghasilkan korpus **60 mahasiswa**; (2) Quran-MD sebagai data referensi tingkat ayat. Setelah segmentasi dan pembersihan, diperoleh **25.829 klip mahasiswa final** dan **17.127 referensi Quran-MD** yang dapat digunakan (3 dari 17.130 baris gagal dikeluarkan lewat *union filtering*).

> **Follow-up**: "81 menjadi 60, kenapa?" → Angka 81 adalah jumlah folder pada pengumpulan awal, sedangkan 60 adalah jumlah mahasiswa yang masuk korpus tervalidasi. Naskah tidak menyediakan rincian alasan per folder, sehingga saya tidak mengklaim bahwa seluruh selisih tersebut disebabkan oleh satu jenis kegagalan tertentu.

**Q8. Apa itu *union filtering* dan mengapa penting?**
Jika satu klip gagal diekstrak di salah satu model, baris yang sama juga dikeluarkan dari model lain. Akibatnya, kedua model dievaluasi pada **himpunan klip yang persis sama**. Tanpa ini, perbedaan metrik bisa terjadi karena perbedaan input, bukan karena representasi model. Pada data Quran-MD, 3 baris gagal dikeluarkan dari kedua model; pada klip mahasiswa, seluruh 25.829 berhasil diproses kedua model.

**Q9. Bagaimana segmentasi klip mahasiswa?**
Rekaman tingkat surah dipotong per ayat memakai stempel waktu kata dari **WhisperX**. Alokasi dilakukan berdasarkan jumlah kata per ayat. Jika penyelarasan kata tidak bisa dipakai, sistem jatuh ke **pembagian waktu proporsional** sebagai *fallback*. Dari 25.829 klip final: 24.872 audit berhasil, 710 *fallback*, 247 tanpa baris audit pasangan.

> **Follow-up**: "Apakah batas segmentasinya pasti benar?" → **Tidak**. Naskah secara eksplisit menyatakan bahwa ini validasi teknis dan *provenance*, bukan pembuktian manual ketepatan fonetik tiap batas. Ini harus diakui sebagai keterbatasan.

**Q10. Bagaimana kebocoran dicegah?**
Untuk sel B, C, D, identitas pemilik (qari atau mahasiswa) dipisah saling lepas antara kueri dan basis data. Audit memeriksa (a) tidak ada pemilik yang muncul di kedua sisi, (b) tidak ada jalur berkas yang sama persis. Seluruh audit lolos. Selain itu, pada setiap sel, kueri hanya dipertahankan jika basis data memiliki minimal satu dokumen dengan pasangan *(surah, ayat)* yang sama.

**Q11. Apa itu *owner_ratio*?**
Fraksi pemilik identitas yang ditempatkan di sisi basis data. Rasio 60/40 berarti 60% pemilik di basis data, 40% di kueri. Rasio yang diuji: 60/40, 70/30, 80/20, 90/10. Pada skenario B dengan 30 qari: 60/40 → 12 qari kueri vs 18 qari basis data; 90/10 → 3 qari kueri vs 27 qari basis data. Pada skenario C/D dengan 60 mahasiswa: 60/40 → 24 vs 36; 90/10 → 6 vs 54.

---

## 4. Wav2Vec2 dan Data2Vec

**Q12. Apakah model-nya dilatih ulang?**
**Tidak**. Parameter model dibekukan. Tidak ada *fine-tuning*, tidak ada kepala baru yang dilatih, tidak ada gradien yang diperbarui. Tujuan eksperimen adalah mengukur **kemampuan representasi pralatih**, bukan adaptasi pada korpus penelitian.

> **Follow-up**: "Lalu adil tidak kalau tidak di-*fine-tune*?" → Adil untuk pertanyaan penelitian ini karena yang dibandingkan adalah representasi bawaan. *Fine-tuning* akan mengaburkan perbedaan representasi pralatih dengan pengaruh adaptasi terhadap data latih korpus.

**Q13. Bagaimana embedding diekstrak?**
Setiap klip masuk sebagai gelombang mono 16 kHz. Keluaran diambil pada **13 titik**: titik 0 (sebelum blok Transformer pertama) dan titik 1–12 (keluaran blok Transformer ke-1 sampai ke-12). Tiap titik menghasilkan matriks temporal $H_l \in \mathbb{R}^{T_l \times 768}$ yang diringkas dengan **mean pooling** menjadi vektor tunggal $\mathbf{e}_l \in \mathbb{R}^{768}$. Satu klip menghasilkan 13 vektor × 768 dimensi.

**Q14. Kenapa *mean pooling*, bukan *max pooling* atau vektor CLS?**
*Mean pooling* dipilih karena (a) menjamin vektor berdimensi tetap sehingga klip dengan durasi berbeda dapat dibandingkan; (b) perbedaan hasil antar-titik murni berasal dari representasi model, bukan dari perbedaan dimensi. Pendekatan ini konsisten dengan praktik penyapuan *layerwise* pada model audio (Pasad et al., 2021).

**Q15. Kenapa harus 13 lapisan?**
Representasi pada lapisan berbeda menangkap informasi akustik dan linguistik yang berbeda. Lapisan awal cenderung lebih akustik; lapisan akhir lebih linguistik/abstrak. Tanpa penyapuan, kita tidak tahu lapisan mana yang terbaik untuk tugas ini. Hasil eksperimen mengonfirmasi: Wav2Vec2 konsisten memilih lapisan 7 (12 dari 13 sel), sedangkan Data2Vec memilih lapisan 5 atau 6—bukan lapisan terakhir.

---

## 5. Rancangan 13 Sel

**Q16. Bisa digambar skenario A, B, C, D?**

| Sel | Sisi kueri | Sisi basis data | Pemisahan | Jumlah sel |
|---|---|---|---|---|
| **A** | Mahasiswa | Quran-MD | Lintas sumber data | **1** |
| **B** | Quran-MD | Quran-MD | 4 rasio qari (60/40, 70/30, 80/20, 90/10) | **4** |
| **C** | Mahasiswa | Mahasiswa | 4 rasio mahasiswa | **4** |
| **D** | Mahasiswa | Mahasiswa + seluruh Quran-MD | 4 rasio mahasiswa | **4** |

**Total = 1 + 4 + 4 + 4 = 13 sel.** Dengan 2 model, ada **26 baris model–sel**.

> **Follow-up**: "Skenario A disebut paling sulit?" → **Hindari kata "paling sulit" secara universal**. Naskah hanya menyebut A sebagai kondisi lintas sumber yang menghasilkan MAP absolut rendah. D juga rendah, tetapi diduga karena basis data gabungan yang lebih besar. Keduanya rendah **dengan alasan yang kemungkinan berbeda**—ini interpretasi, bukan bukti kausal.

**Q17. Kenapa D berbeda dari C?**
D menambahkan seluruh 17.127 klip Quran-MD ke basis data mahasiswa. Kueri di D dan C pada rasio yang sama **identik**, hanya basis data yang berbeda. Ini menguji apakah penambahan domain kedua mengganggu atau membantu retrieval.

**Q18. Berapa ukuran kueri dan basis data tiap sel?**
- A: 25.829 kueri × 17.127 basis data
- B-60/40: 6.849 × 10.278; B-90/10: 1.713 × 15.414
- C-60/40: 9.386 × 16.443; C-90/10: 1.833 × 23.996
- D-60/40: 9.386 × 33.570; D-90/10: 1.833 × 41.123

Basis data D selalu lebih besar daripada C pada rasio yang sama karena penambahan Quran-MD.

**Q19. Mengapa perlu empat rasio?**
Rasio memvariasikan komposisi pemilik pada sisi kueri dan basis data. Rasio 60/40 relatif lebih seimbang; 90/10 menempatkan lebih banyak pemilik pada basis data dan menyisakan lebih sedikit pemilik kueri. Pola perubahan metrik terhadap rasio memberi gambaran deskriptif tentang sensitivitas hasil, tetapi tidak boleh dianggap sebagai hubungan sebab-akibat karena ukuran dan komposisi data juga berubah.

---

## 6. *Cosine* dan *Layer Sweep*

**Q20. Kenapa *cosine similarity*, bukan Euclidean atau learned metric?**
*Cosine* dipilih sebagai **skor penyusun peringkat**, bukan metrik evaluasi akhir. Alasannya: (a) vektor dinormalisasi secara implisit oleh rumus kosinus sehingga besar vektor tidak memengaruhi peringkat; (b) banyak dipakai pada representasi audio (Dehak et al., 2011); (c) penelitian tidak melatih metric baru, sehingga *learned metric* di luar ruang lingkup. Metrik evaluasinya tetap MAP/MRR/Top-K.

> **Follow-up**: "Tidakkah Euclidean memberi hasil berbeda?" → Mungkin, tapi itu bukan pertanyaan penelitian ini. Penelitian ini mengisolasi pengaruh model dan lapisan; pemilihan fungsi skor adalah keputusan metodologis yang dikunci.

**Q21. Bagaimana *layer sweep* dilakukan?**
Untuk **setiap sel**, MAP dihitung pada *dev set* untuk titik 0 sampai 12, terpisah untuk masing-masing model. Titik dengan MAP dev tertinggi dipilih. Setelah titik terpilih, *test set* dibuka sekali untuk evaluasi final. Tidak ada pemilihan ulang berdasarkan hasil test.

**Q22. Lapisan berapa yang terpilih?**
- **Wav2Vec2**: lapisan **7** pada 12 dari 13 sel, lapisan **8** khusus pada B-90:10.
- **Data2Vec**: lapisan **5** pada A, D-60:40, D-70:30; lapisan **6** pada seluruh sel B, seluruh sel C, D-80:20, dan D-90:10.

Keduanya **bukan lapisan terakhir**. Ini konsisten dengan temuan Pasad et al. (2021) bahwa lapisan tengah sering memberi keseimbangan terbaik antara informasi akustik lokal dan konteks yang sudah diolah. Pernyataan ini adalah **interpretasi**, bukan bukti kausal tentang isi tiap lapisan.

**Q23. Mengapa memakai *dev/test* terkunci dan bukan *cross-validation*?**
*Dev sweep* dan *locked test* dipilih agar data pengujian tidak digunakan untuk memilih lapisan. *Cross-validation* sebenarnya juga dapat dirancang secara valid, misalnya dengan prosedur bersarang, tetapi tidak digunakan dalam ruang lingkup penelitian ini. Konsekuensinya, hasil penelitian terikat pada satu matriks pembagian dengan *seed* 42 dan hal itu dinyatakan sebagai keterbatasan.

---

## 7. Metrik dan Bootstrap

**Q24. Bedanya MAP, MRR, Top-K?**
- **MAP**: rata-rata presisi di setiap posisi dokumen relevan, lalu dirata-ratakan antar-kueri. Menilai sebaran dokumen relevan di seluruh peringkat.
- **MRR**: kebalikan peringkat dokumen relevan pertama. Menekankan kehadiran dokumen relevan di puncak.
- **Top-1/5/10**: proporsi kueri yang memiliki minimal satu dokumen relevan dalam K teratas.

Karena setiap kueri memiliki **banyak** dokumen relevan (banyak klip ayat yang sama di basis data), MAP dan MRR menangkap aspek berbeda.

**Q25. Mengapa hanya bootstrap untuk MAP?**
Bootstrap dilakukan pada **selisih AP berpasangan per kueri** (AP Data2Vec dikurangi AP Wav2Vec2), dengan B = 10.000 dan *seed* 42. Interval kepercayaan 95% didapat dari persentil 2,5 dan 97,5. Jika interval tidak mencakup nol, selisih MAP dianggap signifikan. **Penelitian tidak melakukan uji inferensial untuk MRR maupun Top-K**—perbedaan di metrik tersebut hanya boleh dibahas deskriptif. Ini harus ditegaskan jika penguji bertanya signifikansi Top-1.

> **Follow-up**: "Kenapa Top-K tidak diuji signifikansinya?" → Karena tidak dirancang demikian; uji inferensial pada proporsi Top-K memerlukan asumsi tambahan yang tidak diambil dalam penelitian ini.

**Q26. Apa itu *random baseline* dan *lift*?**
Untuk setiap sel, *random baseline* = rata-rata (jumlah dokumen relevan per kueri / ukuran basis data). **Lift** = MAP / *random baseline*, bernilai **kelipatan (kali)**, bukan persentase. Contoh: lift 7,78 di sel A (Wav2Vec2) berarti MAP model 7,78× baseline acak. Lift tertinggi ada di B-60:40 Wav2Vec2 = **78,70 kali**; terendah di A Wav2Vec2 = **7,78 kali**. Lift antar-sel tidak boleh dibanding mentah-mentah karena baseline dan komposisi uji berbeda.

> **Jebakan**: Jangan pernah sebut lift sebagai "7,78%" — itu **salah**. Sebut "7,78 kali" atau "lift 7,78×".

**Q27. Mengapa *seed* 42?**
*Seed* menjamin **reproduksibilitas**. Pembagian pemilik, split dev/test, dan bootstrap semua memakai generator yang di-seed 42. Hasil dapat direproduksi persis oleh peneliti lain.

**Q28. Apakah B = 10.000 cukup?**
Penelitian menggunakan 10.000 replikasi sebagai konfigurasi bootstrap yang ditetapkan. Jumlah tersebut menghasilkan interval empiris yang digunakan dalam laporan, tetapi penelitian tidak melakukan analisis konvergensi untuk membuktikan bahwa jumlah yang lebih besar pasti tidak akan mengubah batas interval. Karena itu, jawaban yang aman adalah menyebut konfigurasi dan prosedurnya, bukan mengklaim optimalitas B = 10.000.

---

## 8. Temuan Utama

**Q29. Model mana yang menang?**
**Tidak ada pemenang universal.** Data2Vec unggul secara numerik pada **9 dari 13 sel** (A, empat C, empat D). Wav2Vec2 unggul pada **4 sel B**. Namun, **C-60:40 dan C-80:20 tidak signifikan secara statistik**. Jadi klaim yang aman adalah: "pola kemenangan bergantung pada skenario dan rasio pemilik."

> **Follow-up**: "Jadi Data2Vec lebih baik dong?" → **Koreksi**: Data2Vec lebih tinggi MAP-nya secara numerik di 9 sel, tetapi 2 sel tidak signifikan, dan di skenario B justru Wav2Vec2 yang menang telak di semua metrik. Generalisasi harus dihindari.

**Q30. Seberapa besar perbedaannya?**
Lihat Tabel 4.5: selisih MAP terbesar ada di B-60:40 (Wav2Vec2 unggul 2,36 poin persentase). Selisih terkecil ada di C-60:40 (+0,07 pp, tidak signifikan) dan D-80:20 (+0,12 pp, signifikan tetapi tipis). Selisih kecil di A (+0,24 pp) signifikan, tetapi kinerja absolutnya rendah (MAP ~1,5%).

**Q31. Apakah lift membuktikan model berguna?**
Lift > 1 di **seluruh 13 sel**, artinya kedua model melampaui tebakan acak. Namun lift tinggi tidak otomatis berarti siap pakai. Top-1 tertinggi hanya 52,67% (B-60:40 Wav2Vec2). Di A, Top-1 hanya 5,37%. Sistem di domain A dan D lebih cocok sebagai **pembangkit kandidat**, bukan penentu ayat tunggal.

**Q32. Apa arti sel D secara praktis?**
D mensimulasikan kondisi di mana basis data mahasiswa digabung dengan referensi profesional. Kueri tetap mahasiswa. Hasil: MAP lebih rendah dari B dan sebanding dengan C, menunjukkan penambahan domain kedua **berasosiasi** dengan retrieval yang lebih menantang. Ini asosiasi, bukan hubungan kausal.

**Q33. Apakah sel B paling mudah?**
B secara numerik memiliki MAP tertinggi karena kedua sisi dari Quran-MD dengan unit ayat yang sama. Namun, **sebut "paling tinggi" lebih aman daripada "paling mudah"**. Penurunan MAP pada B seiring rasio basis data membesar (Wav2Vec2: 13,78% → 9,75%) menunjukkan ukuran basis data dan jumlah kueri yang sedikit ikut berpengaruh.

---

## 9. Pertanyaan Jebakan dan Cara Menjawab Aman

> Bagian ini berisi pertanyaan yang sering menjurus ke klaim berlebihan. Jawaban aman = kembali ke *evidence*, hindari generalisasi, dan bedakan hasil numerik vs signifikan.

**Q34. "Jadi Wav2Vec2 atau Data2Vec yang lebih bagus?"**
Jawaban aman: "Kedua model tidak dapat disimpulkan unggul secara universal. Data2Vec memiliki MAP numerik lebih tinggi pada 9 sel, Wav2Vec2 pada 4 sel. Secara statistik, 11 sel signifikan, dua sel—C-60:40 dan C-80:20—tidak. Rekomendasi bergantung pada sel sasaran: untuk domain Quran-MD murni (skenario B), Wav2Vec2 titik 7 memiliki bukti terkuat; untuk skenario lintas domain A dan lintas mahasiswa D, Data2Vec titik 5 atau 6 lebih tinggi MAP-nya dengan dukungan signifikansi."

**Q35. "Apakah Signifikan = Penting?"**
**Tidak**. Signifikansi statistik tidak sama dengan kepentingan praktis, terutama ketika selisih MAP kecil dan kinerja absolut masih rendah. Contohnya, D-80:20 signifikan dengan selisih sekitar 0,12 poin persentase; dampak praktisnya tetap perlu dinilai berdasarkan kebutuhan aplikasi dan tidak boleh disimpulkan hanya dari label signifikan.

**Q36. "Kenapa tidak pakai uji signifikansi untuk Top-K?"**
Jawaban: "Karena desain penelitian mengunci uji inferensial hanya pada selisih AP berpasangan untuk MAP. Top-K adalah proporsi kueri, bukan rata-rata AP; memerlukan uji berbeda yang tidak diambil. Pembahasan Top-K dilakukan secara deskriptif."

**Q37. "Apakah lift 78,70× di B-60:40 berarti model luar biasa?"**
Jawaban: "Lift besar harus dibaca relatif terhadap baseline acak pada sel tersebut. Nilainya merupakan MAP dibagi rasio rata-rata dokumen relevan terhadap ukuran basis data. Lift tertinggi bukan bukti model terbaik secara universal. MAP absolut B-60:40 Wav2Vec2 adalah 13,78%, sehingga lift dan metrik absolut perlu dibaca bersama."

**Q38. "Kalau Qwen atau model lain lebih baru, kenapa tidak dimasukkan?"**
Jawaban aman: "Evaluasi pada penelitian ini dibatasi pada Wav2Vec2 dan Data2Vec. Model lain, termasuk yang lebih baru, di luar ruang lingkup penelitian ini dan tidak dibandingkan. Penelitian lanjutan dapat memperluas cakupan." Jangan mengarang angka untuk model di luar cakupan.

**Q39. "Apakah sistem ini sudah bisa dipakai di produksi?"**
**Tidak**. Fase *deployment* dalam penelitian ini bersifat **konseptual**. Tidak ada pengujian layanan produksi, latensi, keamanan API, kapasitas serentak, atau pemantauan. Rancangan konseptual terdiri atas proses luring (penyiapan indeks embedding) dan proses daring (skor cosine terhadap kueri), dengan catatan sel A dan D disarankan hanya sebagai pembangkit kandidat.

**Q40. "Apakah lapisan 7 Wav2Vec2 secara inheren lebih baik?"**
**Tidak**. Lapisan 7 hanya terpilih di 12 dari 13 sel pada korpus ini. Naskah menyebut ini **interpretasi** bahwa lapisan tengah memberi keseimbangan akustik-kontekstual. Pernyataan kausal tentang isi lapisan tidak didukung.

**Q41. "Kenapa MAP rendah di A? Artinya model gagal?"**
Jawaban aman: "A adalah kondisi lintas sumber data yang sulit—kueri dari mahasiswa, basis data dari Quran-MD. Perbedaan perangkat, lingkungan, dan segmentasi memengaruhi geometri embedding. MAP rendah bukan berarti model gagal; lift masih 7,78–9,18× di atas baseline acak, dan secara statistik selisih A signifikan. Namun untuk aplikasi, A lebih cocok sebagai pembangkit kandidat."

**Q42. "Apakah hasil ini bisa digeneralisasi ke surah lain di luar Al-Fatihah dan Juz Amma?"**
**Tidak otomatis**. Cakupan data hanya Al-Fatihah dan Juz Amma. Generalisasi ke surah lain memerlukan evaluasi ulang karena karakteristik akustik dan panjang ayat dapat berbeda.

**Q43. "Kenapa tidak pakai *cross-validation* 5-fold?"**
Jawaban: "Penelitian membatasi evaluasi pada matriks 13 sel dengan *dev/test split* terkunci dan *seed* 42. *Cross-validation* dapat dilakukan secara valid jika pemilihan lapisan ditempatkan di dalam setiap fold, tetapi itu memerlukan biaya komputasi dan rancangan eksperimen tambahan. Karena belum dilakukan, generalisasi terhadap split lain saya nyatakan sebagai keterbatasan."

**Q44. "Klaim apa yang TIDAK boleh saya sampaikan di sidang?"**
Daftar hitam klaim:
- ❌ "Data2Vec lebih baik daripada Wav2Vec2" secara umum.
- ❌ "Skenario A paling sulit" tanpa kualifikasi.
- ❌ "Lift 7,78%" — salah satuan, seharusnya **kali**.
- ❌ "Perbedaan Top-K signifikan" — tidak diuji.
- ❌ "MRR berbeda signifikan" — tidak diuji.
- ❌ "Lapisan 7 secara inheren lebih baik."
- ❌ "Sistem sudah siap produksi."
- ❌ Mengaitkan arsitektur model secara kausal dengan performa observasi.

---

## 10. Deployment

**Q45. Bagaimana deployment konseptual sistem ini?**
Dua tahap:
1. **Luring**: audio referensi dinormalisasi (mono 16 kHz), embedding diekstrak dengan model + titik terpilih per sel, lalu disimpan menjadi indeks beserta metadata surah/ayat.
2. **Daring**: kueri dinormalisasi, embedding diekstrak dengan konfigurasi yang sama dengan sel sasaran, skor cosine dihitung terhadap indeks, kandidat Top-K dikembalikan, dan disarankan ada verifikasi tambahan setelah Top-K karena kinerja A/D belum mendukung keputusan otomatis presisi tinggi.

> **Follow-up**: "Berapa latensinya?" → Tidak diuji. Ini di luar ruang lingkup penelitian.

**Q46. Model mana yang direkomendasikan untuk deployment?**
- **Domain Quran-MD murni (skenario B)**: Wav2Vec2 titik 7, bukti terkuat di eksperimen ini (MAP tertinggi di keempat rasio, semua signifikan).
- **Lintas sumber (A) atau lintas mahasiswa + Quran-MD (D)**: Data2Vec titik 5 atau 6, MAP lebih tinggi dengan dukungan signifikansi.
- **Skenario C**: Data2Vec unggul numerik di 4 rasio, tetapi hanya 2 rasio signifikan; perlu pertimbangan tambahan.

Rekomendasi **tidak universal** dan tidak boleh dipindahkan ke korpus lain tanpa evaluasi pengembangan baru.

---

## 11. Keterbatasan dan Pekerjaan Masa Depan

**Q47. Apa keterbatasan utama penelitian ini?**
1. Satu matriks eksperimen dengan satu *seed* (42); generalisasi ke split atau populasi lain belum diuji.
2. Lapisan terpilih dari *dev sweep* pada 13 titik (0–12); kesimpulan terikat pada ruang lapisan ini.
3. Rerata makro antar-sel memperlakukan semua sel setara, padahal `n_test` berbeda; angka agregat bukan estimasi performa per-kueri gabungan.
4. Bootstrap hanya untuk selisih MAP; tidak ada uji inferensial untuk MRR atau Top-K.
5. Hasil retrieval cosine tidak dapat langsung dipakai untuk menyimpulkan performa konfigurasi DTW atau eksperimen lain.
6. Literatur tidak menetapkan pemenang universal untuk korpus Al-Qur'an; temuan komparatif-deskriptif untuk konfigurasi yang diuji, bukan hubungan kausal.

**Q48. Apa saran untuk penelitian lanjutan?**
- Evaluasi manual ketepatan batas ayat pada subset.
- Adaptasi domain (fine-tuning) pada korpus Al-Qur'an.
- Eksplorasi strategi *pooling* temporal selain mean pooling.
- Pengujian model lain (HuBERT, wavLM, atau model yang lebih baru) dengan protokol 13 sel yang sama.
- Pengukuran efisiensi indeks, latensi, skalabilitas, dan pemantauan untuk deployment nyata.
- Ekstensi cakupan surah di luar Al-Fatihah dan Juz Amma.

**Q49. Apakah hasil 25.829 klip ini bisa dipakai untuk tugas lain?**
Bisa sebagai sumber embedding beku untuk eksperimen retrieval lain pada korpus yang sama, dengan catatan: (a) konfigurasi ekstraksi harus identik, (b) evaluasi harus memakai dev/test split yang sama atau split baru dengan seed yang dilaporkan, (c) metrik yang dipilih harus sesuai dengan hipotesis baru.

---

## 12. *Rapid-Review Fact Sheet*

> Hafalkan angka-angka ini. Sering ditanya langsung.

| Fakta | Nilai |
|---|---|
| Folder mahasiswa awal | **81** |
| Mahasiswa tervalidasi | **60** |
| Kandidat segmentasi | 25.945 |
| Berkas nol dikeluarkan | 116 |
| **Klip mahasiswa final** | **25.829** |
| Provenance audit berhasil | 24.872 |
| Provenance fallback | 710 |
| Tanpa baris audit pasangan | 247 |
| Baris Quran-MD awal | 17.130 |
| Baris Quran-MD gagal | 3 |
| **Referensi Quran-MD final** | **17.127** |
| Jumlah skenario | 4 (A, B, C, D) |
| Jumlah sel | **13** (1 + 4 + 4 + 4) |
| Jumlah baris model–sel | **26** |
| Titik representasi | 0–12 (**13 titik**) |
| Dimensi embedding | **768** |
| Agregasi temporal | **Mean pooling** |
| Status parameter model | **Beku (frozen)**, tanpa fine-tuning |
| Skor penyusun peringkat | **Cosine similarity** |
| Seed | **42** (reproduksibilitas) |
| Bootstrap B | **10.000** |
| Metrik dengan uji inferensial | **Hanya MAP** (via selisih AP berpasangan) |
| Metrik tanpa uji inferensial | MRR, Top-1, Top-5, Top-10 (deskriptif) |
| Sel dengan MAP Data2Vec numerik lebih tinggi | **9** (A, C-60/40, C-70/30, C-80/20, C-90/10, D-60/40, D-70/30, D-80/20, D-90/10) |
| Sel dengan MAP Wav2Vec2 numerik lebih tinggi | **4** (semua sel B) |
| Sel signifikan | **11** |
| Sel tidak signifikan | **2** (C-60:40 dan C-80:20) |
| Lift tertinggi | 78,70× (B-60/40 Wav2Vec2) |
| Lift terendah | 7,78× (A Wav2Vec2) |
| Lapisan terpilih Wav2Vec2 | 7 (12/13 sel), 8 (B-90:10) |
| Lapisan terpilih Data2Vec | 5 (A, D-60:40, D-70:30), 6 (sisanya) |
| Deployment | **Konseptual**, bukan produksi |

---

### Kalimat Penutup yang Siap Diucapkan

> "Penelitian ini tidak menghasilkan pemenang universal antara Wav2Vec2 dan Data2Vec. Yang dihasilkan adalah **pola komparatif yang bergantung pada skenario dan rasio pemilik**, protokol evaluasi 13 sel yang dapat direproduksi, serta rancangan deployment konseptual yang jujur mengenai syarat yang belum diuji. Klaim yang dapat saya pertanggungjawabkan adalah klaim yang terikat pada evidence dalam naskah—dan di luar itu, saya menyampaikan interpretasi sebagai interpretasi, bukan fakta."

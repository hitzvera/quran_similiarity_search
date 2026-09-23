# Draft Jawaban Munaqosah: Pertanyaan 57–73

Dokumen ini berisi jawaban lisan yang siap dipakai untuk pertanyaan dosen penguji nomor 57 sampai 73 dari CHECKLIST_MUNAQOSAH.md. Setiap jawaban merujuk fakta yang terverifikasi di BAB IV revisi (BAB-IV-revisi-layer-dan-split.md) dan Skripsi.md. Jawaban dirancang agar terdengar natural saat diucapkan, bukan seperti membaca naskah.

---

## Q57 — Populasi, sampel, dan teknik penarikan sampel

**Pertanyaan:** Siapa populasi penelitian ini? Sebutkan populasinya, ukuran populasi, dan teknik penarikan sampel yang digunakan.

**Jawaban:**

Penelitian ini bukan penelitian sosial yang mengambil sampel dari responden manusia, melainkan eksperimen komputasional. Jadi istilah "populasi" dan "sampel" di sini merujuk pada kumpulan data audio, bukan orang.

Populasi dalam penelitian ini adalah seluruh rekaman bacaan Al-Qur'an surah Al-Fatihah dan surah-surah Juz Amma yang tersedia dari dua sumber. Pertama, rekaman mahasiswa UIN Sunan Gunung Djati Bandung yang mengumpulkan tugas Tahfidz. Kedua, dataset Quran-MD yang bersifat publik.

Ukuran populasi final setelah validasi teknis adalah 25.829 klip mahasiswa dan 17.127 baris referensi Quran-MD. Jadi totalnya sekitar 42.956 klip audio yang siap dipakai dalam eksperimen.

Teknik pengumpulan datanya adalah purposive sampling untuk rekaman mahasiswa, yaitu mahasiswa yang mengambil mata kuliah Tahfidz dan merekam bacaannya. Dari 81 folder awal, 60 mahasiswa lolos validasi teknis. Untuk Quran-MD, datanya sudah tersedia sebagai dataset publik sehingga tidak ada proses sampling tambahan.

Dalam eksperimen, seluruh klip yang lolos validasi digunakan, bukan diambil sampelnya lagi. Jadi ukuran sampel sama dengan ukuran populasi final. Pembagian ke dalam sisi kueri dan basis data dilakukan berdasarkan skenario dan rasio pemilik, bukan dengan teknik sampling statistik konvensional.

---

## Q58 — Teknik pengumpulan data

**Pertanyaan:** Apa teknik pengumpulan data yang dilakukan? Bagaimana caranya?

**Jawaban:**

Ada dua sumber data dengan teknik pengumpulan yang berbeda.

Untuk rekaman mahasiswa, pengumpulan dilakukan melalui tugas mata kuliah Tahfidz. Mahasiswa merekam bacaan Al-Fatihah dan surah Juz Amma, lalu mengumpulkannya dalam bentuk folder. Setiap folder berisi satu atau lebih berkas audio. Setelah terkumpul 81 folder, dilakukan validasi teknis: pemeriksaan format berkas, kelengkapan metadata, dan keterbacaan audio. Hasilnya 60 mahasiswa yang lolos.

Selanjutnya, setiap rekaman mahasiswa yang panjang dipotong menjadi klip per ayat menggunakan WhisperX. WhisperX menghasilkan stempel waktu tingkat kata, lalu batas ayat dihitung melalui alokasi urutan kata berdasarkan jumlah kata per ayat. Jika jalur tersebut tidak tersedia, sistem menggunakan pembagian waktu proporsional sebagai fallback.

Untuk Quran-MD, datanya sudah tersedia dalam bentuk audio per ayat beserta metadata qari, surah, dan ayat. Jadi tidak perlu segmentasi ulang.

Seluruh audio kemudian dinormalisasi menjadi gelombang mono 16 kHz agar konsisten untuk ekstraksi embedding.

---

## Q59 — Skala pengukuran

**Pertanyaan:** Skala apa yang digunakan? Mengapa skala itu yang dipilih?

**Jawaban:**

Dalam konteks eksperimen komputasional ini, skala pengukuran merujuk pada jenis metrik yang dipakai untuk menilai kinerja retrieval.

Metrik utama yang digunakan adalah MAP (Mean Average Precision), MRR (Mean Reciprocal Rank), dan Top-K Accuracy (Top-1, Top-5, Top-10). Seluruh metrik ini berskala rasio, yaitu memiliki nilai antara 0% sampai 100% dengan titik nol yang bermakna absolut. MAP 0% berarti tidak ada dokumen relevan yang berhasil ditemukan, sedangkan MAP 100% berarti seluruh dokumen relevan berada di peringkat teratas.

Cosine similarity yang digunakan sebagai fungsi skor juga berskala interval, yaitu antara minus 1 sampai plus 1. Tapi cosine similarity di sini bukan metrik evaluasi, melainkan hanya fungsi untuk menyusun peringkat kandidat.

Alasan memilih metrik rasio seperti MAP, MRR, dan Top-K adalah karena metrik-metrik ini standar dalam evaluasi information retrieval, sebagaimana dijelaskan oleh Manning dkk. dalam buku Introduction to Information Retrieval (2008). Metrik ini memungkinkan perbandingan kuantitatif yang ketat antar model dan antar skenario.

---

## Q60 — Uji validitas dan reliabilitas

**Pertanyaan:** Apa teknik uji validitas dan reliabilitas yang digunakan?

**Jawaban:**

Karena ini eksperimen komputasional, validitas dan reliabilitas dijaga melalui desain eksperimen, bukan melalui uji statistik seperti pada kuesioner.

Untuk validitas internal, penelitian menerapkan beberapa pengendalian. Pertama, union filtering memastikan kedua model menerima baris audio yang identik. Kedua, dimensi embedding dan metode mean pooling dibuat sama. Ketiga, relevansi ditentukan oleh label surah dan ayat yang sama, bukan oleh penilaian subjektif. Keempat, pemilihan konfigurasi dilakukan pada development set, sedangkan evaluasi akhir hanya pada test set yang terkunci. Kelima, audit kebocoran data memastikan tidak ada identitas pemilik atau jalur berkas yang tumpang tindih antara kueri dan basis data.

Untuk reliabilitas, seluruh proses menggunakan seed deterministik, yaitu seed 42. Ini berarti pembagian data, permutasi pemilik, dan bootstrap dapat direproduksi persis oleh peneliti lain yang menjalankan pipeline yang sama.

Tidak ada triangulasi dalam arti penelitian kualitatif, tetapi fungsi analognya dipenuhi melalui penggunaan empat skenario berbeda dan lima metrik evaluasi. Jika suatu pola muncul konsisten di beberapa skenario dan metrik, kepercayaan terhadap temuan tersebut lebih kuat.

---

## Q61 — Pengolahan dan analisis data

**Pertanyaan:** Bagaimana anda mengolah dan menganalisis data?

**Jawaban:**

Seluruh pengolahan data dilakukan secara komputasional menggunakan pipeline Python yang saya bangun.

Tahapannya sebagai berikut. Pertama, normalisasi audio ke mono 16 kHz. Kedua, segmentasi menggunakan WhisperX untuk menghasilkan klip per ayat. Ketiga, ekstraksi embedding menggunakan Wav2Vec2 dan Data2Vec dalam keadaan frozen, menghasilkan 13 matriks layer per model. Keempat, mean pooling untuk merangkum representasi temporal menjadi vektor per klip. Kelima, perhitungan cosine similarity antara setiap kueri dan seluruh basis data untuk menyusun peringkat. Keenam, evaluasi menggunakan MAP, MRR, dan Top-K. Ketujuh, uji bootstrap berpasangan dengan 10.000 iterasi untuk menguji signifikansi selisih MAP.

Seluruh proses ini saya lakukan sendiri dengan script yang terdokumentasi. Tidak ada pihak lain yang mengolahkan data. Script dan manifes tersedia untuk audit jika diperlukan.

---

## Q62 — Hasil penelitian

**Pertanyaan:** Bagaimana hasil penelitiannya?

**Jawaban:**

Hasil utama adalah perbandingan kinerja Wav2Vec2 dan Data2Vec pada 13 sel evaluasi.

Berdasarkan MAP pada test set, Data2Vec memiliki nilai numerik lebih tinggi pada 9 dari 13 sel, yaitu Skenario A, seluruh Skenario C (4 rasio), dan seluruh Skenario D (4 rasio). Wav2Vec2 lebih tinggi pada 4 sel, yaitu seluruh Skenario B (4 rasio).

Namun, uji bootstrap menunjukkan bahwa 11 dari 13 sel memiliki perbedaan yang signifikan secara statistik. Dua sel yang tidak signifikan adalah C-60:40 dan C-80:20, di mana selisih MAP numeriknya kecil (0,07 dan 0,17 poin persentase) dan interval kepercayaan 95% mencakup nol.

Penting untuk dicatat bahwa arah MRR dan Top-K tidak selalu sejalan dengan MAP. Pada beberapa sel C dan D, Data2Vec lebih tinggi MAP-nya, tetapi Wav2Vec2 lebih tinggi MRR dan Top-K-nya. Ini menunjukkan bahwa MAP, MRR, dan Top-K menangkap aspek pemeringkatan yang berbeda.

Jadi tidak ada pemenang universal. Keunggulan model bergantung pada skenario, rasio pemilik, dan metrik yang menjadi prioritas.

---

## Q63 — Kesimpulan penelitian

**Pertanyaan:** Apa kesimpulan penelitian anda?

**Jawaban:**

Kesimpulan penelitian ini menjawab dua rumusan masalah.

Pertama, implementasi Wav2Vec2 dan Data2Vec sebagai pengekstrak representasi laten dalam keadaan frozen untuk tugas retrieval audio ayat Al-Qur'an berhasil dilakukan. Implementasi mencakup persiapan audio, segmentasi, ekstraksi representasi dari 13 lapisan, pembentukan vektor embedding per ayat, serta pemeringkatan kandidat berdasarkan cosine similarity. Kedua model mampu membentuk peringkat yang lebih baik daripada peringkat acak, dengan lift tertinggi mencapai 78,70 kali pada B-60:40.

Kedua, kinerja kedua model bergantung pada kondisi retrieval. Wav2Vec2 unggul konsisten ketika kueri dan basis data sama-sama dari Quran-MD dengan qari yang dipisahkan (Skenario B). Data2Vec cenderung lebih baik berdasarkan MAP ketika kueri berasal dari rekaman mahasiswa, baik pada kondisi lintas sumber, antar-mahasiswa, maupun basis data gabungan. Namun, pada sebagian kondisi antar-mahasiswa, perbedaannya belum signifikan secara statistik.

Representasi terbaik bagi kedua model berasal dari lapisan tengah, bukan lapisan terakhir. Wav2Vec2 konsisten memilih layer 7 pada 12 dari 13 sel, sedangkan Data2Vec memilih layer 5 atau 6.

Tidak ada satu rasio pemilik yang dapat direkomendasikan sebagai pilihan terbaik secara umum, karena perubahan rasio juga mengubah komposisi kueri, basis data, dan data pengujian secara bersamaan.

---

## Q64 — Alasan pemberian saran

**Pertanyaan:** Mengapa anda memberikan saran-saran tersebut?

**Jawaban:**

Setiap saran dalam kesimpulan didasarkan pada keterbatasan atau temuan spesifik dalam penelitian ini.

Saran pertama tentang fine-tuning dan metric learning didasarkan pada fakta bahwa kinerja absolut pada kondisi lintas sumber dan basis data gabungan masih rendah. Top-1 tertinggi pada Skenario A hanya 5,37%, dan pada Skenario D hanya 24,23%. Frozen embedding saja tidak cukup untuk aplikasi yang menuntut ketepatan tinggi. Fine-tuning pada korpus Al-Qur'an dapat membantu model mempelajari kedekatan antara bacaan dari ayat yang sama.

Saran kedua tentang perluasan korpus didasarkan pada batasan masalah yang hanya mencakup Al-Fatihah dan Juz Amma. Generalisasi ke seluruh Al-Qur'an dan pembaca yang lebih beragam perlu diuji untuk memastikan hasil perbandingan tidak hanya berlaku pada cakupan surah dan karakteristik pembaca dalam penelitian ini.

Saran ketiga tentang metode agregasi lain didasarkan pada fakta bahwa penelitian ini hanya menggunakan mean pooling. Penggabungan representasi dari beberapa lapisan atau metode pooling lain mungkin menghasilkan kinerja yang berbeda.

Saran keempat tentang analisis kesalahan kualitatif didasarkan pada keterbatasan bahwa hasil agregat tidak menyediakan contoh daftar peringkat individual yang cukup untuk menuliskan studi kasus. Analisis kesalahan secara kualitatif dapat membantu mengidentifikasi apakah kegagalan retrieval berkaitan dengan kualitas segmentasi, variasi pembaca, kondisi audio, atau kemiripan fonetik antar ayat.

---

## Q65 — Hipotesis dan pembuktiannya

**Pertanyaan:** Apakah hipotesis terbukti? Ada yang tidak terbukti?

**Jawaban:**

Penelitian ini tidak merumuskan hipotesis statistik dalam bentuk H0 dan H1 yang diuji dengan uji parametrik. Pertanyaan penelitiannya bersifat komparatif-deskriptif: bagaimana kinerja frozen embedding Wav2Vec2 dan Data2Vec dalam tugas retrieval ayat Al-Qur'an, dan model mana yang lebih baik pada kondisi tertentu.

Jika hipotesis diartikan sebagai dugaan awal berdasarkan literatur, maka latar belakang menyebutkan bahwa Aswat (Alkanhal dkk., 2023) menemukan Data2vec unggul untuk ASR bahasa Arab. Pertanyaannya adalah apakah keunggulan itu transfer ke tugas retrieval tanpa fine-tuning.

Hasil eksperimen menunjukkan bahwa jawabannya tidak sederhana. Data2Vec memang lebih tinggi MAP-nya pada 9 dari 13 sel, terutama ketika kueri berasal dari rekaman mahasiswa. Tetapi Wav2Vec2 justru unggul konsisten pada Skenario B di mana kedua sisi berasal dari Quran-MD. Jadi keunggulan Data2Vec pada ASR tidak secara otomatis transfer ke retrieval, dan sebaliknya Wav2Vec2 tidak selalu kalah.

Dua sel C (C-60:40 dan C-80:20) bahkan tidak menunjukkan perbedaan signifikan. Jadi pada kondisi antar-mahasiswa dengan rasio tertentu, kedua model secara statistik tidak dapat dibedakan.

---

## Q66 — Konsistensi daftar pustaka

**Pertanyaan:** Mengapa buku ini ada di daftar pustaka tapi di dalam skripsi tidak ada? Atau sebaliknya?

**Jawaban:**

Saya telah memeriksa konsistensi antara sitasi dalam teks dan daftar pustaka. Seluruh 28 referensi dalam daftar pustaka disitasi dalam teks, dan tidak ada sitasi dalam teks yang tidak memiliki entri di daftar pustaka.

Daftar pustaka mencakup referensi inti: Baevski dkk. 2020 untuk Wav2Vec2, Baevski dkk. 2022 untuk Data2Vec, Manning dkk. 2008 untuk metrik retrieval, Alkanhal dkk. 2023 untuk Aswat, Yang dkk. 2024 untuk evaluasi berskala besar, Pasad dkk. 2023 untuk analisis layer-wise, dan referensi lain yang mendukung landasan teori, metodologi, dan pembahasan.

Jika ada referensi yang tampak tidak langsung disitasi dalam kalimat tertentu, itu karena referensi tersebut mendukung landasan teori atau metodologi secara umum, bukan menjadi objek pembahasan langsung.

---

## Q67 — Lampiran penelitian

**Pertanyaan:** Apa lampiran yang tersedia?

**Jawaban:**

Karena ini eksperimen komputasional, lampiran tidak berupa coding book atau transkrip wawancara seperti dalam penelitian sosial. Lampiran dalam penelitian ini berupa artefak teknis yang mendukung reproduktibilitas.

Artefak yang tersedia meliputi script Python untuk normalisasi audio, segmentasi WhisperX, ekstraksi embedding, perhitungan metrik, dan uji bootstrap. Tersedia juga manifes yang mencatat identitas setiap klip, termasuk nomor surah, ayat, sumber data, dan provenance. Matriks embedding dalam format .npy tersedia untuk setiap layer dan model. Log kemajuan (progress.json) mencatat status pemrosesan setiap berkas.

Artefak hasil agregat seperti tabel MAP per sel dan grafik perbandingan juga tersedia, tetapi daftar peringkat individual per kueri tidak dilampirkan karena ukurannya yang besar. Jika diperlukan, daftar peringkat tersebut dapat dihasilkan ulang dari matriks embedding dan script evaluasi yang tersedia.

---

## Q68 — Mengapa Data2Vec menang MAP tapi tidak metrik lain

**Pertanyaan:** Pada hasil evaluation, kenapa Data2Vec bisa memenangkan MAP, tapi tidak dengan metrik lain?

**Jawaban:**

Pertama, perlu diklarifikasi bahwa Data2Vec tidak selalu menang pada semua metrik. Data2Vec memiliki MAP numerik lebih tinggi pada 9 dari 13 sel, tetapi pada metrik MRR dan Top-K, arahnya tidak selalu sejalan.

Alasannya adalah karena MAP, MRR, dan Top-K menangkap aspek pemeringkatan yang berbeda. MAP merangkum kualitas urutan terhadap seluruh dokumen relevan. MRR berfokus pada posisi dokumen relevan pertama. Top-K menunjukkan proporsi kueri yang memiliki setidaknya satu dokumen relevan dalam K hasil pertama.

Data2Vec mungkin menempatkan dokumen relevan sedikit lebih tinggi secara rata-rata di seluruh peringkat, sehingga MAP-nya lebih tinggi. Tetapi untuk posisi pertama (MRR) atau kehadiran dalam Top-1/Top-5/Top-10, Wav2Vec2 kadang lebih baik. Ini terjadi karena distribusi skor cosine dan geometri embedding yang berbeda antar model.

Selain itu, uji bootstrap dalam penelitian ini hanya dilakukan untuk selisih MAP atau AP per kueri berpasangan. Tidak ada uji inferensial untuk MRR dan Top-K. Jadi perbedaan pada metrik tersebut hanya boleh dibahas secara deskriptif, bukan diklaim signifikan.

---

## Q69 — Owner ratio mana yang lebih baik

**Pertanyaan:** Owner ratio mana yang lebih baik antara keempat varian (60:40, 70:30, 80:20, 90:10)?

**Jawaban:**

Tidak ada satu rasio pemilik yang secara universal lebih baik. Jawaban atas pertanyaan ini bergantung pada skenario, model, dan metrik yang menjadi prioritas.

Pada Skenario B, rasio 60:40 menghasilkan MAP tertinggi untuk kedua model. Ini masuk akal karena rasio 60:40 memiliki basis data terkecil, sehingga tantangan retrieval lebih ringan.

Pada Skenario C, MAP tertinggi Wav2Vec2 tercapai pada rasio 80:20, sedangkan Data2Vec juga tertinggi pada rasio 80:20 untuk MAP, MRR, dan Top-1, tetapi rasio 90:10 tertinggi untuk Top-5 dan Top-10.

Pada Skenario D, Wav2Vec2 tertinggi pada rasio 80:20, sedangkan Data2Vec tertinggi pada rasio 90:10 untuk MAP, Top-5, dan Top-10.

Penting untuk dipahami bahwa perubahan rasio pemilik secara bersamaan mengubah komposisi kueri, komposisi basis data, dan ukuran test set. Oleh karena itu, kenaikan atau penurunan MAP pada rasio tertentu tidak dapat diisolasi sebagai pengaruh rasio semata. Perbandingan antar rasio bersifat deskriptif dan tidak boleh ditafsirkan sebagai hubungan kausal.

Rasio 70:30 bahkan tidak muncul sebagai nilai maksimum pada metrik dan pasangan skenario-model mana pun. Ini bukan berarti rasio 70:30 tidak memiliki hasil evaluasi, melainkan hanya tidak menjadi yang tertinggi dalam kombinasi yang diuji.

---

## Q70 — Arti masing-masing faktor dalam kausal

**Pertanyaan:** Arti masing-masing faktor dalam kausal?

**Jawaban:**

Pertama, perlu diklarifikasi bahwa penelitian ini tidak mengklaim hubungan kausal dalam arti statistika eksperimental. Istilah "kausal" di sini merujuk pada alur logika penelitian dalam Kerangka Pemikiran, bukan bukti kausal antara variabel.

Dalam Kerangka Pemikiran, alur logikanya adalah: Fenomena dan Teori menghasilkan identifikasi Masalah, yaitu minimnya penelitian tentang performa model SSL dalam domain audio retrieval. Masalah ini mengarah pada Solusi, yaitu melakukan studi komparatif untuk mengetahui model SSL mana yang menghasilkan hasil baik dalam audio retrieval. Solusi diimplementasikan melalui Metode CRISP-DM, yang menghasilkan keluaran berupa perbandingan kuantitatif, analisis pengaruh paradigma pretraining, dan rekomendasi arsitektur.

Dalam konteks hasil eksperimen, faktor-faktor yang berasosiasi dengan perbedaan kinerja meliputi skenario (sumber data), rasio pemilik, layer representasi, dan model. Tetapi eksperimen ini tidak mengisolasi pengaruh setiap faktor secara kausal karena faktor-faktor tersebut berubah secara bersamaan. Misalnya, perubahan rasio pemilik juga mengubah komposisi kueri dan basis data, sehingga tidak dapat disimpulkan bahwa rasio tertentu secara kausal menyebabkan kenaikan atau penurunan MAP.

Jadi "faktor kausal" di sini lebih tepat disebut sebagai faktor yang berasosiasi atau faktor yang bervariasi dalam desain eksperimen, bukan variabel independen dalam eksperimen terkontrol yang membuktikan sebab-akibat.

---

## Q71 — Tools visualisasi data

**Pertanyaan:** Dengan tools apa kamu membuat data visualisasi?

**Jawaban:**

Visualisasi data dalam penelitian ini dibuat menggunakan Python, khususnya pustaka matplotlib dan seaborn untuk grafik ilmiah seperti perbandingan MAP per sel, tren MAP terhadap rasio pemilik, dan heatmap rasio terbaik per metrik.

Untuk diagram alur seperti Kerangka Pemikiran dan alur CRISP-DM, saya menggunakan format Mermaid yang dirender menjadi gambar. Mermaid memungkinkan diagram dibuat dalam bentuk teks terstruktur sehingga mudah direvisi.

Seluruh visualisasi dibuat dari data hasil eksperimen yang terdokumentasi dalam matriks embedding dan log evaluasi. Tidak ada visualisasi yang dibuat tanpa dasar data yang dapat ditelusuri.

---

## Q72 — Arti inferensial dalam konteks penelitian

**Pertanyaan:** Apa arti inferensial, dan jelaskan dalam studi kasus penelitian kamu?

**Jawaban:**

Dalam statistika, inferensial merujuk pada proses menarik kesimpulan tentang populasi berdasarkan sampel. Ini berbeda dari statistika deskriptif yang hanya menggambarkan data yang ada.

Dalam konteks penelitian ini, statistika inferensial muncul dalam uji bootstrap berpasangan. Uji ini mengambil selisih AP (Average Precision) Data2Vec dikurangi Wav2Vec2 per kueri berpasangan, kemudian melakukan resampling 10.000 kali untuk membangun interval kepercayaan 95% bagi selisih MAP. Jika interval kepercayaan tidak mencakup nol, perbedaan dianggap signifikan.

Jadi inferensial di sini berarti: berdasarkan selisih AP pada test set yang merupakan sampel dari populasi kueri potensial, kita membuat kesimpulan tentang apakah perbedaan MAP antara kedua model kemungkinan besar bukan karena kebetulan sampling.

Sebaliknya, untuk MRR dan Top-K, penelitian ini hanya melaporkan statistika deskriptif. Tidak ada uji inferensial untuk metrik tersebut, sehingga perbedaan pada MRR dan Top-K tidak boleh diklaim signifikan.

Selain itu, interpretasi mengenai kandungan informasi pada lapisan tertentu juga bersifat inferensial dalam arti yang berbeda. Penelitian ini tidak mengukur secara langsung isi setiap titik representasi, sehingga pernyataan bahwa layer tengah memberikan keseimbangan antara informasi akustik lokal dan konteks yang telah diolah merupakan interpretasi berdasarkan pola hasil, bukan bukti langsung.

---

## Q73 — Alasan saran dalam kesimpulan

**Pertanyaan:** Mengapa anda menyarankan hal demikian dalam kesimpulan? Apakah saran anda akan membawa perbaikan?

**Jawaban:**

Setiap saran dalam kesimpulan didasarkan pada keterbatasan spesifik yang teridentifikasi selama penelitian, dan masing-masing ditujukan untuk mengatasi keterbatasan tersebut.

Saran tentang fine-tuning dan metric learning didasarkan pada temuan bahwa kinerja absolut pada kondisi lintas sumber dan basis data gabungan masih rendah. Top-1 pada Skenario A hanya 5,37%, dan pada Skenario D hanya 24,23%. Frozen embedding saja tidak cukup untuk aplikasi yang menuntut ketepatan tinggi. Fine-tuning pada korpus Al-Qur'an dapat membantu model mempelajari kedekatan antara bacaan dari ayat yang sama dan perbedaan antara ayat yang berlainan. Saran ini akan membawa perbaikan karena model tidak hanya mengekstraksi representasi bawaan, tetapi juga menyesuaikan representasi tersebut untuk tugas retrieval spesifik.

Saran tentang perluasan korpus didasarkan pada batasan masalah yang hanya mencakup Al-Fatihah dan Juz Amma. Generalisasi ke seluruh Al-Qur'an dan pembaca yang lebih beragam perlu diuji. Saran ini akan membawa perbaikan karena memastikan hasil perbandingan tidak hanya berlaku pada cakupan surah dan karakteristik pembaca dalam penelitian ini.

Saran tentang metode agregasi lain didasarkan pada fakta bahwa penelitian ini hanya menggunakan mean pooling. Penggabungan representasi dari beberapa lapisan atau metode pooling lain mungkin menghasilkan kinerja yang berbeda. Saran ini akan membawa perbaikan karena mengeksplorasi ruang konfigurasi yang lebih luas.

Saran tentang analisis kesalahan kualitatif didasarkan pada keterbatasan bahwa hasil agregat tidak menyediakan contoh daftar peringkat individual yang cukup untuk menuliskan studi kasus. Analisis kesalahan secara kualitatif akan membawa perbaikan karena dapat mengidentifikasi pola kegagalan yang tidak terlihat dari metrik agregat, sehingga perbaikan sistem dapat dilakukan secara lebih terarah.

Secara keseluruhan, saran-saran ini tidak menjamin perbaikan otomatis, tetapi memberikan arah yang jelas untuk penelitian lanjutan. Setiap saran terkait langsung dengan temuan atau keterbatasan dalam penelitian ini, bukan saran umum yang tidak berdasar.

---

## Catatan Penggunaan

- Jawaban ini dirancang untuk diucapkan secara lisan, bukan dibaca kata per kata. Pahami intinya, lalu sampaikan dengan bahasa sendiri.
- Jika dosen bertanya lebih detail tentang angka tertentu, merujuk ke BAB IV revisi (Tabel 4.6a–4.6e dan Tabel 4.7).
- Jangan klaim signifikansi untuk MRR dan Top-K karena uji bootstrap hanya dilakukan untuk MAP.
- Jangan klaim satu rasio pemilik secara universal terbaik karena perbandingan antar rasio bersifat deskriptif.
- Jangan klaim hubungan kausal antara faktor dan kinerja karena eksperimen tidak mengisolasi pengaruh setiap faktor.

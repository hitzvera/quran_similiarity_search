# Draft Jawaban Munaqosah: Pertanyaan 1-18

Dokumen ini berisi draf jawaban lisan untuk 18 pertanyaan pertama dari CHECKLIST_MUNAQOSAH.md. Jawaban dirancang untuk disampaikan secara lisan dalam sidang munaqosah, dengan gaya formal namun tetap terdengar alami saat diucapkan. Setiap jawaban merujuk pada bagian spesifik dari Skripsi.md atau script repository yang dapat diverifikasi.

---

## Pertanyaan 1: Mengapa tidak menggunakan model Arab monolingual?

**Jawaban Langsung:**

Penelitian ini tidak menggunakan model Arab monolingual karena dua alasan utama. Pertama, model Arab monolingual yang tersedia saat ini umumnya berbasis arsitektur *contrastive learning* seperti Wav2Vec2, bukan *self-distillation* seperti Data2Vec. Kedua, tujuan penelitian ini adalah membandingkan dua paradigma *pre-training* yang berbeda, yaitu *contrastive learning* pada Wav2Vec2 dan *self-distillation* pada Data2Vec, dalam kondisi *frozen embedding* tanpa *fine-tuning*.

**Penjelasan Lebih Dalam:**

Dalam Latar Belakang (Skripsi.md baris 289-296), disebutkan bahwa model Arab monolingual memang mampu mengungguli model multibahasa untuk tugas pengenalan ucapan Arab. Namun, bukti empiris tersebut terbatas pada arsitektur *contrastive learning*. Data2Vec menggunakan paradigma yang fundamentally berbeda, yaitu *self-distillation* dengan target representasi kontekstual penuh. Perbedaan ini secara teoretis berpotensi menghasilkan representasi fonetik yang lebih umum untuk lintas bahasa, karena target prediksi berupa konteks laten yang melimpah, bukan unit diskrit yang terikat pada distribusi fonetik bahasa *pretraining*.

Penelitian ini memilih Wav2Vec2 dan Data2Vec karena keduanya tersedia dalam versi multibahasa yang telah dilatih pada data audio berskala besar, sehingga perbandingan dapat dilakukan dalam kondisi yang setara. Penggunaan model Arab monolingual akan membatasi perbandingan hanya pada satu paradigma, padahal pertanyaan penelitian justru ingin mengetahui apakah perbedaan paradigma *pre-training* menghasilkan kualitas representasi yang berbeda untuk tugas *retrieval* ayat Al-Qur'an.

**Referensi:**
- Skripsi.md baris 289-310 (Latar Belakang)
- Skripsi.md baris 1195-1217 (Business Understanding)

---

## Pertanyaan 2: Bagaimana proses iteratif dalam CRISP-DM?

**Jawaban Langsung:**

Penelitian mengikuti enam fase CRISP-DM: *business understanding*, *data understanding*, *data preparation*, *modeling*, *evaluation*, dan *deployment*. Alur penelitian bersifat iteratif, artinya temuan pada suatu fase dapat menyebabkan proses kembali ke fase sebelumnya untuk memperbaiki kualitas data atau menyesuaikan proses pengolahannya.

**Penjelasan Lebih Dalam:**

Sebagai contoh konkret, jika pada tahap *data understanding* ditemukan bahwa beberapa berkas audio tidak dapat dibaca atau memiliki format yang tidak konsisten, maka proses kembali ke *data preparation* untuk melakukan normalisasi ulang. Atau jika pada tahap *modeling* ditemukan kegagalan ekstraksi pada beberapa klip, maka dilakukan pemeriksaan ulang terhadap data masukan dan validasi teknis.

Namun, meskipun proses dapat berulang, pemilihan konfigurasi dan pelaporan hasil tetap dipisahkan secara ketat. Titik representasi terbaik dipilih hanya berdasarkan hasil pada *development set*, kemudian dikunci sebelum evaluasi akhir dilakukan pada *test set*. Pemisahan ini menjaga agar *test set* tetap steril dan tidak digunakan untuk mengambil keputusan konfigurasi, sehingga evaluasi akhir tidak bias.

**Referensi:**
- Skripsi.md baris 1167-1187 (BAB III Metodologi, paragraf pembuka)
- Skripsi.md baris 1231-1245 (Gambar 3.1 Adaptasi CRISP-DM)

---

## Pertanyaan 3: Bagaimana pemilihan konfigurasi (development set) dan pelaporan hasil? Mengapa harus dipisah?

**Jawaban Langsung:**

Pemilihan konfigurasi menggunakan *development set*, sedangkan pelaporan hasil akhir menggunakan *test set*. Pemisahan ini diperlukan untuk mencegah *data leakage* dan memastikan evaluasi akhir tidak bias.

**Penjelasan Lebih Dalam:**

*Development set* digunakan untuk menyapu seluruh 13 titik representasi (layer 0 sampai 12) pada masing-masing model. Untuk setiap kombinasi sel dan model, lapisan dengan nilai MAP tertinggi pada *development set* dipilih. Setelah titik representasi ditentukan, konfigurasi tersebut dikunci dan digunakan untuk evaluasi akhir pada *test set*.

*Test set* tidak digunakan selama proses pemilihan titik representasi. Himpunan ini hanya dibuka setelah konfigurasi final ditetapkan, untuk menghasilkan evaluasi akhir berupa MAP, MRR, Top-1, Top-5, dan Top-10. Pemisahan tersebut menjaga agar data pengujian tetap berfungsi sebagai evaluasi yang independen, bukan sebagai bagian dari proses pengambilan keputusan.

Jika *test set* digunakan untuk memilih konfigurasi, maka model akan secara tidak langsung "belajar" dari data uji, dan hasil evaluasi menjadi tidak valid karena tidak lagi mencerminkan kemampuan generalisasi model pada data yang benar-benar baru.

**Referensi:**
- Skripsi.md baris 1898-1913 (Pembagian Development dan Test)
- Skripsi.md baris 2052-2080 (Pembagian Development dan Test di BAB IV)
- Skripsi.md baris 2099-2162 (Pemilihan lapisan representasi per Sel)

---

## Pertanyaan 4: Apa arti korpus?

**Jawaban Langsung:**

Korpus dalam penelitian ini merujuk pada kumpulan data audio yang digunakan untuk eksperimen, yaitu rekaman mahasiswa dan Quran-MD. Korpus mencakup seluruh berkas audio yang telah melalui proses validasi, normalisasi, dan segmentasi hingga siap digunakan dalam evaluasi.

**Penjelasan Lebih Dalam:**

Dalam konteks penelitian, istilah "korpus" digunakan untuk membedakan antara data mentah awal dan data yang telah diproses. Misalnya, korpus mahasiswa awal terdiri dari 81 folder rekaman, tetapi setelah validasi teknis hanya 60 mahasiswa yang lolos. Hasil segmentasi menghasilkan 25.945 kandidat klip, tetapi 116 berkas berukuran nol dikeluarkan, sehingga tersisa 25.829 klip mahasiswa final yang membentuk korpus akhir.

Demikian pula, Quran-MD memiliki 17.130 baris referensi awal, tetapi tiga baris mengalami kegagalan pemrosesan dan dikeluarkan melalui *union filtering*, sehingga tersisa 17.127 referensi yang dapat digunakan. Korpus akhir inilah yang digunakan dalam seluruh evaluasi.

**Referensi:**
- Skripsi.md baris 1690-1756 (Hasil Data Understanding, Tabel 4.1 Rekonsiliasi validasi)
- Skripsi.md baris 1247-1396 (Data Understanding)

---

## Pertanyaan 5: Apa itu union filtering? Bisakah kamu beri contoh?

**Jawaban Langsung:**

*Union filtering* adalah prosedur yang memastikan bahwa jika satu baris data gagal diekstraksi pada salah satu model, baris yang sama juga dikeluarkan dari model lainnya. Tujuannya adalah menjamin bahwa selisih metrik antara kedua model tidak disebabkan oleh perbedaan *query* atau kandidat yang dinilai.

**Penjelasan Lebih Dalam:**

Sebagai contoh, pada Quran-MD terdapat 17.130 baris referensi awal. Ketika ekstraksi embedding dilakukan, tiga baris mengalami kegagalan pada salah satu model (misalnya Wav2Vec2). Melalui *union filtering*, ketiga indeks tersebut dikeluarkan dari hasil kedua model, baik Wav2Vec2 maupun Data2Vec. Dengan demikian, tersisa 17.127 baris identik yang digunakan untuk evaluasi kedua model.

Prosedur ini diterapkan karena evaluasi perbandingan memerlukan kondisi yang setara. Jika satu model dievaluasi pada 17.130 dokumen dan model lain pada 17.127 dokumen, maka perbedaan metrik bisa saja disebabkan oleh perbedaan komposisi data, bukan oleh perbedaan kualitas representasi. *Union filtering* menghilangkan sumber bias tersebut.

Pada klip mahasiswa, seluruh 25.829 klip berhasil diekstraksi oleh kedua model, sehingga tidak ada baris yang perlu dikeluarkan.

**Referensi:**
- Skripsi.md baris 1891-1896 (Union filtering di BAB IV)
- Skripsi.md baris 1537-1551 (Cleaning hasil ekstraksi)
- Skripsi.md baris 2091-2097 (Hasil Modeling)

---

## Pertanyaan 6: Mengapa konfigurasi pencarian titik representasi perlu dipisahkan dengan data uji?

**Jawaban Langsung:**

Konfigurasi pencarian titik representasi dipisahkan dari data uji untuk mencegah *data leakage* dan memastikan bahwa evaluasi akhir mencerminkan kemampuan generalisasi model, bukan kemampuan model untuk "menghafal" data uji.

**Penjelasan Lebih Dalam:**

Jika *test set* digunakan untuk memilih lapisan representasi terbaik, maka proses pemilihan tersebut secara tidak langsung menggunakan informasi dari data uji. Akibatnya, evaluasi akhir pada *test set* tidak lagi independen, dan hasil yang dilaporkan menjadi bias karena model telah "dioptimalkan" untuk data uji tersebut.

Untuk menghindari hal ini, penelitian menggunakan *development set* yang terpisah untuk pemilihan konfigurasi. *Development set* digunakan untuk menyapu seluruh 13 lapisan dan memilih lapisan dengan MAP tertinggi. Setelah konfigurasi dikunci, *test set* digunakan untuk satu kali evaluasi akhir. Pemisahan ini mengikuti prinsip standar dalam machine learning bahwa data uji harus benar-benar tidak terlihat selama proses pelatihan atau pemilihan hyperparameter.

**Referensi:**
- Skripsi.md baris 1184-1187 (BAB III, paragraf tentang pemisahan konfigurasi)
- Skripsi.md baris 2075-2080 (Pemisahan development dan test)

---

## Pertanyaan 7: Mengapa Skenario A tidak bisa displit menjadi 60:40, 70:30, dst?

**Jawaban Langsung:**

Skenario A tidak menggunakan rasio pemilik karena sumber data *query* dan *database* referensi sudah berbeda secara alami. Skenario A menggunakan seluruh rekaman mahasiswa sebagai *query* dan seluruh Quran-MD sebagai *database* referensi. Tidak ada pemisahan data pada domain yang sama karena sumber data sudah berbeda.

**Penjelasan Lebih Dalam:**

Skenario A mewakili kondisi pencarian lintas sumber, yaitu pencarian bacaan mahasiswa terhadap referensi profesional. Seluruh 25.829 klip mahasiswa ditempatkan sebagai *query*, dan seluruh 17.127 klip Quran-MD sebagai *database* referensi. Pemisahan berdasarkan rasio pemilik hanya relevan ketika *query* dan *database* berasal dari sumber yang sama, seperti pada Skenario B (Quran-MD vs Quran-MD), Skenario C (mahasiswa vs mahasiswa), dan Skenario D (mahasiswa vs mahasiswa lain + Quran-MD).

Pada Skenario B, C, dan D, rasio pemilik digunakan untuk membagi identitas pembaca (qori atau mahasiswa) menjadi dua kelompok yang saling lepas: satu kelompok sebagai *query*, kelompok lain sebagai *database* referensi. Pemisahan ini diperlukan agar model tidak mencocokkan bacaan dari pembaca yang sama, sehingga evaluasi mencerminkan kemampuan model dalam menemukan ayat yang sama dari pembaca yang berbeda.

**Referensi:**
- Skripsi.md baris 1266-1282 (Gambaran Skenario A, B, C, D)
- Skripsi.md baris 1853-1862 (Pemisahan Dataset dan Peran Sumber)
- Skripsi.md baris 1915-1935 (Ukuran Data Skenario A)

---

## Pertanyaan 8: Apa itu rasio pemilih pada BAB 4?

**Jawaban Langsung:**

Rasio pemilih, atau *owner ratio*, adalah proporsi pemilik (qori atau mahasiswa) yang ditempatkan sebagai *query* versus *database* referensi. Rasio ini diterapkan pada Skenario B, C, dan D untuk membagi identitas pembaca menjadi dua kelompok yang saling lepas.

**Penjelasan Lebih Dalam:**

Sebagai contoh, rasio 70:30 berarti 70% pemilik (diurutkan berdasarkan NIM atau qori) menjadi *database* referensi, sedangkan 30% sisanya menjadi *query*. Pembagian ini dilakukan secara *stratified* dengan seed 42 untuk menjaga konsistensi antar sel.

Pada Skenario B, pemilik adalah qori. Rasio 70:30 berarti 70% qori menjadi *database* referensi dan 30% qori menjadi *query*. Pada Skenario C, pemilik adalah mahasiswa (NIM). Pada Skenario D, pemilik adalah mahasiswa untuk *query*, dan gabungan mahasiswa lain ditambah seluruh Quran-MD untuk *database* referensi.

Variasi rasio yang digunakan adalah 60:40, 70:30, 80:20, dan 90:10. Setiap rasio menghasilkan komposisi *query* dan *database* yang berbeda, sehingga memungkinkan evaluasi kinerja model pada berbagai kondisi ketersediaan data referensi.

**Referensi:**
- Skripsi.md baris 1864-1883 (Skenario B, C, D: Pemisahan Berdasarkan Pemilik)
- Skripsi.md baris 1878-1883 (Owner ratio)

---

## Pertanyaan 9: Apa itu rekonsiliasi dan berikan contoh pada studi kasus skripsi kamu?

**Jawaban Langsung:**

Rekonsiliasi dalam penelitian ini merujuk pada proses penyelarasan dan pelacakan jumlah data dari tahap pengumpulan awal hingga data final yang siap digunakan. Rekonsiliasi memastikan bahwa setiap tahap transformasi data tercatat dengan jelas, sehingga dapat diaudit dan diverifikasi.

**Penjelasan Lebih Dalam:**

Sebagai contoh, rekonsiliasi korpus mahasiswa dimulai dari 81 folder awal pada pengumpulan. Setelah validasi teknis, 60 mahasiswa lolos. Hasil segmentasi menghasilkan 25.945 kandidat klip. Sebanyak 116 berkas berukuran nol dikeluarkan sebelum ekstraksi embedding, sehingga tersisa 25.829 klip mahasiswa final. Seluruh klip ini berhasil diekstraksi oleh kedua model.

Kategori *provenance* mahasiswa juga direkonsiliasi: 24.872 klip berasal dari rekaman dengan catatan audit berhasil, 710 klip hasil alokasi waktu proporsional (fallback), dan 247 klip tanpa baris audit tetapi tetap terlacak pada manifes. Jumlah kategori ini konsisten karena 24.872 + 710 + 247 = 25.829.

Untuk Quran-MD, rekonsiliasi dimulai dari 17.130 baris referensi awal. Tiga baris mengalami kegagalan pemrosesan dan dikeluarkan melalui *union filtering*, sehingga tersisa 17.127 referensi yang dapat digunakan.

**Referensi:**
- Skripsi.md baris 1690-1756 (Hasil Data Understanding, Gambar 4.1 dan Tabel 4.1)
- Skripsi.md baris 1760-1766 (Konsistensi kategori provenance)

---

## Pertanyaan 10: Apa yang dimaksud dengan 116 berkas berukuran nol?

**Jawaban Langsung:**

116 berkas berukuran nol merujuk pada berkas audio yang memiliki ukuran 0 byte setelah proses segmentasi. Berkas-berkas ini dikeluarkan sebelum ekstraksi embedding karena tidak dapat diproses oleh model.

**Penjelasan Lebih Dalam:**

Berkas berukuran nol ini dapat terjadi karena berbagai alasan, misalnya kegagalan proses pemotongan audio, kesalahan dalam penulisan berkas, atau masalah pada sumber data awal. Sebelum ekstraksi embedding dilakukan, pemeriksaan ukuran berkas dilakukan untuk memastikan bahwa hanya berkas yang valid yang diproses.

Dari 25.945 kandidat hasil segmentasi, 116 berkas berukuran nol dikeluarkan, sehingga tersisa 25.829 klip mahasiswa final. Seluruh klip final ini berhasil diekstraksi oleh Wav2Vec2 dan Data2Vec tanpa kegagalan.

Pemeriksaan dan pengeluaran berkas berukuran nol merupakan bagian dari validasi teknis yang bertujuan memastikan kualitas data masukan sebelum ekstraksi embedding.

**Referensi:**
- Skripsi.md baris 1706-1707 (Gambar 4.1 dan penjelasan)
- Skripsi.md baris 1728-1729 (Tabel 4.1: Berkas nol byte)
- Skripsi.md baris 1478-1483 (Validasi dan pembentukan manifes)

---

## Pertanyaan 11: Tujuan mean pooling itu untuk apa?

**Jawaban Langsung:**

*Mean pooling* digunakan untuk meringkas representasi temporal berdimensi variabel menjadi vektor berdimensi tetap berukuran 768. Tujuannya adalah agar audio dengan durasi berbeda dapat dibandingkan dalam ruang vektor yang sama.

**Penjelasan Lebih Dalam:**

Setiap klip audio menghasilkan urutan representasi kontekstual dari model, dengan panjang temporal yang berbeda-beda tergantung durasi audio. Untuk setiap titik representasi (layer), keluaran temporal berupa matriks dengan dimensi T × 768, di mana T adalah jumlah frame temporal yang bervariasi antar klip.

*Mean pooling* menghitung rata-rata sepanjang dimensi temporal, sehingga menghasilkan vektor berukuran 768 untuk setiap klip. Pendekatan ini memastikan bahwa perbedaan hasil antar titik representasi berasal dari kualitas representasi model, bukan dari perbedaan dimensi embedding.

Satu klip menghasilkan 13 vektor berukuran 768 (satu untuk setiap layer 0-12). Vektor-vektor ini kemudian digunakan untuk perhitungan *cosine similarity* dengan kandidat *database*.

**Referensi:**
- Skripsi.md baris 1500-1518 (Modeling, persamaan 3.1)
- Skripsi.md baris 1514-1518 (Penjelasan mean pooling)

---

## Pertanyaan 12: Apa itu union filtering, bisa diberi contoh?

**Jawaban Langsung:**

*Union filtering* adalah prosedur yang memastikan bahwa jika satu baris data gagal diekstraksi pada salah satu model, baris yang sama juga dikeluarkan dari model lainnya. Tujuannya adalah menjamin bahwa selisih metrik antara kedua model tidak disebabkan oleh perbedaan *query* atau kandidat yang dinilai.

**Penjelasan Lebih Dalam:**

Sebagai contoh, pada Quran-MD terdapat 17.130 baris referensi awal. Ketika ekstraksi embedding dilakukan, tiga baris mengalami kegagalan pada salah satu model. Melalui *union filtering*, ketiga indeks tersebut dikeluarkan dari hasil kedua model, baik Wav2Vec2 maupun Data2Vec. Dengan demikian, tersisa 17.127 baris identik yang digunakan untuk evaluasi kedua model.

Prosedur ini diterapkan karena evaluasi perbandingan memerlukan kondisi yang setara. Jika satu model dievaluasi pada 17.130 dokumen dan model lain pada 17.127 dokumen, maka perbedaan metrik bisa saja disebabkan oleh perbedaan komposisi data, bukan oleh perbedaan kualitas representasi. *Union filtering* menghilangkan sumber bias tersebut.

**Referensi:**
- Skripsi.md baris 1891-1896 (Union filtering di BAB IV)
- Skripsi.md baris 1537-1551 (Cleaning hasil ekstraksi)

---

## Pertanyaan 13: Apa itu provenance dalam konteks skripsi kamu, tolong beri contoh?

**Jawaban Langsung:**

*Provenance* dalam penelitian ini merujuk pada informasi keterlacakan yang mencatat asal-usul dan proses yang dialami oleh setiap klip audio. Informasi ini mencakup rekaman induk, metode segmentasi yang digunakan, dan status audit.

**Penjelasan Lebih Dalam:**

Setiap klip mahasiswa dicatat dalam manifes dengan informasi *provenance* yang mencakup: identitas rekaman induk (folder mahasiswa dan nama berkas sumber), metode segmentasi yang digunakan (apakah berdasarkan penyelarasan kata WhisperX atau alokasi waktu proporsional), dan status audit (berhasil, fallback, atau tanpa baris audit).

Sebagai contoh, dari 25.829 klip mahasiswa final:
- 24.872 klip memiliki *provenance* audit berhasil, artinya berasal dari rekaman dengan catatan audit yang menunjukkan proses segmentasi berbasis penyelarasan kata berhasil.
- 710 klip memiliki *provenance* fallback, artinya hasil alokasi waktu proporsional karena penyelarasan kata tidak tersedia atau tidak dapat digunakan.
- 247 klip tidak memiliki baris audit tetapi tetap terlacak pada manifes.

Informasi *provenance* ini penting untuk audit kualitas data dan untuk memahami ketidakpastian yang melekat pada hasil segmentasi. Kategori *provenance* juga menunjukkan bahwa penelitian tidak menyembunyikan ketidakpastian, melainkan mencatatnya secara eksplisit.

**Referensi:**
- Skripsi.md baris 1808-1810 (Setiap klip dikaitkan dengan informasi provenance)
- Skripsi.md baris 1825-1827 (Rekaman mahasiswa memiliki informasi tambahan mengenai rekaman induk dan metode segmentasi)
- Skripsi.md baris 1735-1745 (Tabel 4.1: Kategori provenance)

---

## Pertanyaan 14: Maksud dari "Penelitian tidak memiliki anotasi batas waktu manual untuk seluruh klip, sehingga ketepatan segmentasi tetap menjadi salah satu sumber kepastian data."

**Jawaban Langsung:**

Pernyataan ini berarti bahwa penelitian tidak melakukan validasi manual terhadap batas-batas ayat pada setiap klip hasil segmentasi. Ketepatan segmentasi bergantung pada prosedur otomatis (WhisperX dan alokasi proporsional), sehingga ketidakpastian tetap ada.

**Penjelasan Lebih Dalam:**

Segmentasi rekaman mahasiswa dilakukan secara otomatis menggunakan WhisperX untuk menghasilkan stempel waktu tingkat kata, kemudian batas ayat dibentuk dengan mengalokasikan urutan kata berdasarkan jumlah kata setiap ayat. Ketika penyelarasan kata tidak tersedia, sistem menggunakan alokasi waktu proporsional sebagai *fallback*.

Meskipun prosedur ini telah dirancang dengan hati-hati, penelitian tidak memiliki anotasi batas waktu manual yang diverifikasi oleh manusia untuk seluruh 25.829 klip. Oleh karena itu, ketepatan batas hasil segmentasi tetap menjadi sumber ketidakpastian data. Status audit berhasil menunjukkan keberhasilan prosedur teknis, bukan jaminan manual bahwa batas klip tepat pada setiap ayat secara fonetik.

Pernyataan ini penting untuk memberikan konteks yang tepat terhadap validasi yang dilakukan. Pemeriksaan folder, ukuran berkas, keterbacaan audio, konsistensi metadata, dan keberadaan catatan audit merupakan validasi teknis dan validasi *provenance*, bukan pembuktian manual bahwa seluruh batas ayat akurat secara fonetik.

**Referensi:**
- Skripsi.md baris 1389-1396 (Validasi yang dilakukan harus dibatasi maknanya)
- Skripsi.md baris 1764-1766 (Penelitian tidak memiliki anotasi batas waktu manual)
- Skripsi.md baris 1468-1474 (Pembagian berdasarkan jumlah kata dan ketidakpastian)

---

## Pertanyaan 15: Apa itu manifest file, dan tujuannya dalam skripsi kamu buat apa? Berikan contoh.

**Jawaban Langsung:**

*Manifest file* adalah berkas CSV yang mencatat informasi dasar setiap klip audio secara terstruktur, dengan urutan yang konsisten. Tujuannya adalah menjaga keterlacakan dan keselarasan antara data audio dan embedding yang dihasilkan.

**Penjelasan Lebih Dalam:**

Manifes menyimpan informasi: lokasi audio, identitas pembaca, nomor surah, nomor ayat, sumber data, dan informasi *provenance* segmentasi. Untuk rekaman mahasiswa, manifes juga menyimpan informasi rekaman induk dan metode segmentasi.

Hubungan satu banding satu antara baris manifes dan baris embedding menjadi dasar keterlacakan sepanjang eksperimen. Baris ke-i pada matriks embedding selalu merujuk ke baris ke-i pada manifes. Dengan demikian, setiap vektor embedding dapat ditelusuri kembali ke klip audio asalnya.

Sebagai contoh, manifes untuk klip mahasiswa memiliki kolom: `audio_path`, `reader_id`, `surah`, `ayah`, `source`, `parent_recording`, `segmentation_method`. Untuk Quran-MD, manifes memiliki kolom serupa tetapi tanpa informasi rekaman induk karena audio sudah berupa tingkat ayat.

Manifes juga digunakan untuk mencatat kegagalan ekstraksi. Jika satu baris gagal diekstraksi, kolom `failed` pada manifes ditandai sebagai `True`, dan baris tersebut dikeluarkan melalui *union filtering*.

**Referensi:**
- Skripsi.md baris 1476-1483 (Validasi dan pembentukan manifes)
- Skripsi.md baris 1529-1535 (Checkpoint dan manifest)
- Skripsi.md baris 2086-2089 (Baris ke-i pada matriks selalu bersesuaian dengan baris ke-i pada manifes)

---

## Pertanyaan 16: Bisakah beri contoh untuk statement "Setiap klip dikaitkan dengan identitas pembaca, nomor surah, nomor ayat, sumber data, lokasi audio, dan informasi provenance."

**Jawaban Langsung:**

Setiap klip audio dalam penelitian ini dicatat dalam manifes dengan informasi lengkap yang mencakup identitas pembaca, nomor surah, nomor ayat, sumber data, lokasi berkas audio, dan informasi *provenance*.

**Penjelasan Lebih Dalam:**

Sebagai contoh, untuk klip mahasiswa:
- `reader_id`: "1197050001" (NIM mahasiswa)
- `surah`: 112 (nomor surah Al-Ikhlas)
- `ayah`: 1 (nomor ayat)
- `source`: "mahasiswa" (sumber data)
- `audio_path`: "data/segmented/1197050001/al-ikhlas/ayah_01.mp3" (lokasi berkas audio)
- `parent_recording`: "1197050001_At-Takatsur.mp4" (rekaman induk)
- `segmentation_method`: "whisperx_alignment" atau "time_proportional" (metode segmentasi)

Untuk Quran-MD:
- `reader_id`: "al-afasy" (identitas qori)
- `surah`: 112
- `ayah`: 1
- `source`: "quran_md"
- `audio_path`: "data/quran_md/112/ayah_01.mp3"

Informasi ini memungkinkan penelusuran penuh dari embedding kembali ke audio asal, serta memungkinkan audit kualitas dan reproduksi eksperimen.

**Referensi:**
- Skripsi.md baris 1808-1810 (Setiap klip dikaitkan dengan identitas...)
- Skripsi.md baris 1823-1827 (Informasi dalam manifes)

---

## Pertanyaan 17: Bisakah beri contoh data bagaimana proses segmentasi menggunakan WhisperX?

**Jawaban Langsung:**

Proses segmentasi menggunakan WhisperX dilakukan dalam beberapa tahap: transkripsi dan penyelarasan kata, alokasi kata ke ayat berdasarkan jumlah kata referensi, dan pemotongan audio berdasarkan stempel waktu.

**Penjelasan Lebih Dalam:**

Sebagai contoh, untuk rekaman mahasiswa dengan NIM 1197050001 yang merekam Surah Al-Fatihah (7 ayat):

1. **Transkripsi dan penyelarasan**: WhisperX memproses audio dan menghasilkan daftar kata dengan stempel waktu:
   ```
   [
     {"word": "بِسْمِ", "start": 0.5, "end": 0.9},
     {"word": "اللَّهِ", "start": 0.9, "end": 1.2},
     {"word": "الرَّحْمَٰنِ", "start": 1.2, "end": 1.6},
     {"word": "الرَّحِيمِ", "start": 1.6, "end": 2.0},
     {"word": "الْحَمْدُ", "start": 2.5, "end": 2.9},
     ...
   ]
   ```

2. **Alokasi kata ke ayat**: Jumlah kata setiap ayat dari referensi digunakan untuk membagi urutan kata. Al-Fatihah ayat 1 memiliki 4 kata (بسم الله الرحمن الرحيم), ayat 2 memiliki 5 kata, dst. Fungsi `allocate_ayah_spans` membagi urutan kata terdeteksi sesuai proporsi ini.

3. **Pemotongan audio**: Batas mulai klip diambil dari awal kata pertama yang dialokasikan, batas akhir dari akhir kata terakhir. Fungsi `build_ayah_cuts` menghasilkan daftar potongan dengan stempel waktu mulai dan selesai untuk setiap ayat.

4. **Ekstraksi audio**: Fungsi `slice_many` memotong audio menggunakan ffmpeg berdasarkan stempel waktu tersebut, menghasilkan berkas `ayah_01.mp3`, `ayah_02.mp3`, dst.

Jika penyelarasan kata tidak tersedia atau jumlah kata terdeteksi terlalu sedikit untuk jumlah ayat, sistem menggunakan mekanisme *fallback* dengan alokasi waktu proporsional.

**Referensi:**
- Skripsi.md baris 1442-1466 (Segmentasi rekaman mahasiswa)
- Script: `backend-skripsi/pre-processing/remote_segment.py` (fungsi `process_entry`)
- Script: `backend-skripsi/pre-processing/ayah_align.py` (fungsi `allocate_ayah_spans`, `build_ayah_cuts`)

---

## Pertanyaan 18: Apa itu mekanisme fallback dalam segmentasi data kamu, dan berikan contoh dalam script python yang ada di skripsi kamu?

**Jawaban Langsung:**

Mekanisme *fallback* adalah prosedur cadangan yang digunakan ketika penyelarasan kata dari WhisperX tidak tersedia atau tidak dapat digunakan untuk membagi audio menjadi klip per ayat. Sistem menggunakan alokasi waktu proporsional berdasarkan jumlah kata referensi.

**Penjelasan Lebih Dalam:**

*Fallback* dipicu dalam dua kondisi utama:
1. **intro_only**: Tidak ada kata recitasi yang terdeteksi setelah pengenalan intro (ta'awwudh/basmala).
2. **insufficient_recitation_words**: Jumlah kata recitasi yang terdeteksi terlalu sedikit untuk jumlah ayat yang diharapkan.

Dalam kedua kasus tersebut, fungsi `build_time_proportional_cuts` digunakan untuk membagi durasi audio secara proporsional berdasarkan jumlah kata setiap ayat dari referensi.

**Contoh dalam script Python:**

Dari `backend-skripsi/pre-processing/remote_segment.py` baris 114-142:

```python
spans = allocate_ayah_spans(len(recited), reference_counts)
if spans is None:
    fallback_start = float(words[start_index - 1]['end']) if start_index else float(recited[0]['start'])
    audio_duration = get_audio_duration(audio_path)
    cuts = build_time_proportional_cuts(fallback_start, audio_duration, reference_counts)
    if not cuts:
        return {
            'status': 'failed',
            'reason': f'only {len(recited)} words for {expected} ayahs',
            'words_detected': len(words),
        }
    # ... proses pemotongan audio ...
    return {
        'status': 'ok',
        'ayahs': len(cuts),
        'words_detected': len(words),
        'words_recited': 0,
        'intro_words_skipped': start_index,
        'fallback': 'time_proportional',
        'fallback_trigger': 'insufficient_recitation_words',
        'fallback_reason': (
            f'only {len(recited)} recitation words for {expected} ayahs; using '
            'reference word-count proportions over remaining audio duration '
            '(deterministic approximation)'
        ),
        'segments': cuts,
    }
```

Fungsi `build_time_proportional_cuts` dari `ayah_align.py` baris 116-163:

```python
def build_time_proportional_cuts(start_sec, end_sec, reference_word_counts):
    """Build ayah cuts proportional to reference word counts over a time span."""
    ayah_count = len(reference_word_counts)
    if ayah_count == 0:
        return []
    
    total_reference = sum(reference_word_counts)
    duration = end_sec - start_sec
    if duration <= 0:
        return []
    
    cuts = []
    cumulative_reference = 0
    
    for index, reference_count in enumerate(reference_word_counts):
        cumulative_reference += reference_count
        remaining_ayahs = ayah_count - index - 1
        
        if remaining_ayahs == 0:
            ayah_end = end_sec
        else:
            proportion = cumulative_reference / total_reference
            ayah_end = start_sec + proportion * duration
            # ... batasan minimum durasi ...
        
        # ... buat potongan ...
        cuts.append({
            'ayah': index + 1,
            'start': round(ayah_start, 3),
            'end': round(ayah_end, 3),
            'words': 0,
        })
    
    return cuts
```

Mekanisme *fallback* ini memastikan bahwa segmentasi tetap dapat dilakukan meskipun WhisperX tidak menghasilkan penyelarasan kata yang memadai. Status *fallback* dicatat dalam metadata agar ketidakpastian tidak disembunyikan.

**Referensi:**
- Skripsi.md baris 1461-1463 (Mekanisme cadangan)
- Skripsi.md baris 1415-1421 (Diagram segmentasi dan fallback)
- Script: `backend-skripsi/pre-processing/remote_segment.py` (logika fallback)
- Script: `backend-skripsi/pre-processing/ayah_align.py` (fungsi `build_time_proportional_cuts`)

---

## Catatan Penggunaan

Dokumen ini berisi draf jawaban untuk pertanyaan 1-18. Jawaban dirancang untuk disampaikan secara lisan dengan gaya formal namun tetap terdengar alami. Setiap jawaban merujuk pada bagian spesifik dari Skripsi.md atau script repository yang dapat diverifikasi.

Untuk pertanyaan yang memiliki duplikasi (seperti Q5 dan Q12 tentang union filtering), jawaban yang sama dapat digunakan dengan penyesuaian minimal.

Jawaban untuk pertanyaan 19-73 tidak termasuk dalam dokumen ini dan akan disiapkan secara terpisah jika diperlukan.

**PERBANDINGAN WAV2VEC2 DAN DATA2VEC TERHADAP FITUR LATEN AUDIO UNTUK
TILAWAH AL-QURAN**

**TUGAS AKHIR**

Diajukan sebagai salah satu syarat untuk memperoleh gelar Sarjana
Teknik\
pada jurusan Teknik Informatika Fakultas Sains dan Teknologi\
Universitas Islam Negeri Sunan Gunung Djati Bandung

**Oleh\
MUJAHID ANSORI MAJID\
1197050093~­~**

**­**

![Logo Description automatically generated with medium
confidence](media/image1.png){width="1.7048611111111112in"
height="2.4569444444444444in"}

> **PROGRAM STUDI TEKNIK INFORMATIKA**
>
> **FAKULTAS SAINS DAN TEKNOLOGI**
>
> **UNIVERSITAS ISLAM NEGERI SUNAN GUNUNG DJATI**
>
> **BANDUNG**
>
> **­**

# LEMBAR PERSETUJUAN

**PERBANDINGAN WAV2VEC2 DAN DATA2VEC TERHADAP FITUR LATEN AUDIO UNTUK
TILAWAH AL-QURAN**

Disusun oleh:

**Mujahid Ansori Majid**

**1197050093**

**Menyetujui,**

+----------------------------------+-----------------------------------+
| **Pembimbing I**                 | **Pembimbing II**                 |
|                                  |                                   |
| **[Wisnu Uriawan, M.Kom., Ph.    | **[Dr. Dian Sa\'adillah           |
| D]{.underline}**\                | Maylawati, M.T.]{.underline}**\   |
| NIP. 197811182009121002          | NIP. 198905262019032023           |
+==================================+===================================+
| **Dekan Fakultas**               | **Ketua Jurusan**                 |
|                                  |                                   |
| **Sains dan Teknologi**          | **Teknik Informatika**            |
|                                  |                                   |
| **[Prof. Dr. Hasniah Aliah,      | **[Dr. Dian Sa\'adillah           |
| M.Si.]{.underline}**             | Maylawati, M.T.]{.underline}**    |
|                                  |                                   |
| NIP. 197806132005012014          | NIP. 198905262019032023           |
+----------------------------------+-----------------------------------+

**LEMBAR PENGESAHAN**

Tugas akhir yang berjudul PERBANDINGAN WAV2VEC2 DAN DATA2VEC TERHADAP
FITUR LATEN AUDIO UNTUK TILAWAH AL-QURAN dinyatakan sah dan telah
disidangkan dalam siding MUNAQOSYAH Fakultas Sains dan Teknologi,
Universitas Islam Negeri Sunan Gunung Djati Bandung Pada Tanggal ...
oleh Majelis siding yang terdiri dari:

  -----------------------------------------------------------------------
                                          Bandung,
                                          .......................20..
  --------------------------------------- -------------------------------

  -----------------------------------------------------------------------

+------------------------------------+---------------------------------+
| Ketua Majelis                      | Sekretaris Majelis              |
|                                    |                                 |
| **[Wisnu Uriawan, M.Kom., Ph.      | **[Dr. Dian Sa\'adillah         |
| D]{.underline}**                   | Maylawati, M.T.]{.underline}**  |
|                                    |                                 |
| NIP. 197811182009121002            | NIP. 198905262019032023         |
+====================================+=================================+

**Mengetahui,**

+------------------------------------+---------------------------------+
| Penguji I                          | Penguji II                      |
|                                    |                                 |
| **[Bapak/Ibu Penguji               | **[Bapak/Ibu Penguji            |
| I]{.underline}**                   | II]{.underline}**               |
|                                    |                                 |
| NIP.                               | NIP.                            |
+====================================+=================================+

# LEMBAR PERNYATAAN KARYA

Penulis yang bertanda tangan di bawah ini,

Nama : Mujahid Ansori Majid

Templat/Tanggal Lahir : : Kota Bandung, 08 April 2002

NIM : 1197050093

Jurusan/Prodia : Teknik Informatika

Judul Skripsi : PERBANDINGAN WAV2VEC2 DAN\
Judul Skripsi : DATA2VEC TERHADAP FITUR LATEN\
Judul Skripsi : AUDIO UNTUK TILAWAH AL-QURAN

Dengan ini menyatakan bahwa:

1.  Skripsi ini adalah asli dan belum pernah diajukan untuk mendapatkan
    gelar akademik, baik di UIN Sunan Gunung Djati Bandung maupun di
    Perguruan Tinggi lain.

2.  Karya tulis ini adalah murni gagasan, rumusan dan penelitian
    penulis, tanpa bantuan pihak lain, kecuali Tim Pembimbing dan
    masukan Tim Penelaah.

3.  Dalam karya tulis ini tidak terdapat karya atau pendapat yang telah
    ditulis atau dipublikasikan orang lain, kecuali secara tertulis
    dengan jelas dicantumkan dalam daftar pustaka sebagai acuan dalam
    naskah dengan menyebutkan nama pengarangnya

4.  Pernyataan ini penulis buat dengan sesungguhnya, dan apabila
    dikemudian hari terdapat penyimpangan dan ketidakbenaran dalam
    pernyataan ini, maka penulis bersedia menerima sanksi akademik
    berupa pencabutan gelar yang telah diperoleh karena karya ini, serta
    sanksi lainnya sesuai dengan norma yang berlaku di Perguruan Tinggi
    ini.

+---------------------------------------+-------------------------------+
|                                       | Bandung,                      |
|                                       | ........................20... |
|                                       |                               |
|                                       | Materai 10000                 |
|                                       |                               |
|                                       | [Mujahid Ansori               |
|                                       | Majid]{.underline}            |
|                                       |                               |
|                                       | NIM. 1197050093               |
+=======================================+===============================+

# LEMBAR PERSEMBAHAN

Berlandaskan rasa syukur dan keikhlasan atas seluruh perjuangan yang
telah dilalui sepanjang masa perkuliahan, tugas akhir ini dipersembahkan
sebagai puncak dari usaha panjang untuk meraih gelar sarjana yang telah
lama dinantikan. Di atas segalanya, karya ini saya persembahkan kepada
ﷲ, sebab tanpa kehendak dan pertolongan-Nya, rasanya mustahil bagi saya
untuk dapat menuntaskan usaha terakhir ini. Kehadiran kedua orang tua
tercinta pun menjadi bukti nyata dari kasih sayang dan ridho-Nya, yang
senantiasa menumbuhkan keyakinan dalam diri saya untuk terus melangkah
hingga sampai pada titik ini.

Ucapan terima kasih dan rasa hormat yang tulus saya sampaikan kepada
Bapak Wisnu Uriawan, M.Kom., Ph. D, selaku dosen pembimbing I, serta Ibu
Dr. Dian Sa\'adillah Maylawati, M.T., selaku dosen pembimbing II, atas
kesabaran dalam memberikan arahan, bimbingan, dan ilmu yang sangat
berharga selama proses penyusunan tugas akhir ini. Tak lupa, rasa syukur
juga saya sampaikan kepada seluruh sahabat di lingkungan perkuliahan,
yang senantiasa menemani proses pendewasaan diri dan tak henti
memberikan dukungan moral sebagai tempat berbagi keluh kesah di tengah
rumitnya kehidupan.

Saya menyadari sepenuhnya bahwa dunia ini tidak lain hanyalah tempat
bagi manusia untuk bertahan menghadapi berbagai ujian demi meraih
ridho-Nya. Semoga seluruh ilmu yang telah diperoleh dan dituangkan dalam
tugas akhir ini dapat membawa manfaat bagi sesama dan bagi semua pihak.

# ABSTRAK

Pencarian berbasis audio untuk menemukan ayat Al-Qur'an memerlukan
representasi ucapan yang mempertahankan informasi fonetik tanpa
bergantung pada transkripsi. Penelitian ini bertujuan membandingkan
Wav2Vec2 dan Data2Vec sebagai representasi vektor tanpa *fine-tuning*
untuk pencarian kemiripan ayat. Dataset mencakup Al-Fatihah dan Juz
Amma, terdiri atas 25.829 klip bacaan mahasiswa dan 17.127 referensi
Quran-MD. Representasi vektor dihasilkan pada 13 titik lapisan dari
setiap model. Kemiripan antara kueri dan *database* referensi dihitung
menggunakan *cosine similarity* untuk menghasilkan skor kemiripan.
Kinerja dievaluasi menggunakan *Mean Average Precision* (MAP), *Mean
Reciprocal Rank* (MRR), dan Top-K dengan pemisahan himpunan pengembangan
dan himpunan pengujian. Pada 13 lapisan evaluasi, Data2Vec menghasilkan
nilai MAP lebih tinggi pada 9 lapisan, sedangkan Wav2Vec2 lebih tinggi
pada 4 lapisan. Uji *bootstrap* berpasangan menunjukkan perbedaan MAP
yang signifikan pada 11 dari 13 perbandingan, sementara 2 perbandingan
lainnya tidak menunjukkan perbedaan yang signifikan. MRR dan Top-K
digunakan sebagai analisis deskriptif untuk melengkapi evaluasi. Temuan
menunjukkan bahwa kinerja kedua model bervariasi antar lapisan dan
skenario, sehingga tidak terdapat model yang unggul secara universal dan
pemilihan model serta lapisan representasi perlu mempertimbangkan
karakteristik skenario pencarian.

**Kata Kunci**: pencarian kemiripan audio, vektor embedding, Wav2vec2,
Data2vec, Al-Qur'an

# *ABSTRACT*

*Voice-based search for finding Quranic verses requires speech
representations that preserve phonetic information without relying on
transcription. This study aims to compare Wav2Vec2 and Data2Vec as
vector representations without fine-tuning for verse similarity
retrieval. The dataset includes Al-Fatihah and Juz Amma, consisting of
25,829 student recitation clips and 17,127 Quran-MD reference clips.
Vector representations were extracted at 13 layer points from each
model. The similarity between queries and reference database clips was
computed using cosine similarity to obtain similarity scores.
Performance was evaluated using Mean Average Precision (MAP), Mean
Reciprocal Rank (MRR), and Top-K, with separate development and test
sets. Across the 13 evaluated layers, Data2Vec achieved higher MAP
values at 9 layers, while Wav2Vec2 achieved higher values at 4 layers. A
paired bootstrap test showed statistically significant differences in
MAP for 11 of the 13 comparisons, while the remaining 2 comparisons did
not show significant differences. MRR and Top-K were used as descriptive
analyses to complement the evaluation. The findings show that the
performance of both models varies across layers and scenarios,
indicating that neither model consistently outperforms the other and
that model and representation-layer selection should consider the
characteristics of the retrieval scenario.*

***Keywords**: audio similarity search, vectors embedding, W*av2*vec2,
Data2vec, Al-Qur'an*

# KATA PENGANTAR

Puji syukur ke hadirat Allah SWT. yang Maha Pengasih lagi Maha Penyayang
atas segala rahmat, karunia, dan hidayah-Nya, sehingga penulis dapat
menyelesaikan Tugas Akhir yang berjudul \"Perbandingan Wav2Vec2 dan
Data2Vec terhadap Fitur Laten Audio untuk Tilawah Al-Quran\" sebagai
salah satu syarat memperoleh gelar Sarjana Teknik pada Jurusan Teknik
Informatika, Fakultas Sains dan Teknologi, Universitas Islam Negeri
Sunan Gunung Djati Bandung.

Shalawat serta salam semoga senantiasa tercurah kepada Nabi Muhammad
SAW., beserta keluarga, sahabat, dan seluruh umatnya hingga akhir zaman.

Penelitian ini berfokus pada perbandingan performa model Wav2Vec2 dan
Data2Vec dalam merepresentasikan fitur laten audio untuk keperluan
pencarian (*retrieval*) ayat tilawah Al-Quran. Melalui pendekatan
self-supervised learning dan evaluasi berbasis *frozen embedding* dengan
pengukuran *cosine similarity*, penelitian ini dirancang untuk
mengidentifikasi model representasi audio yang paling optimal dalam
mengenali dan mencocokkan ayat-ayat Al-Quran berdasarkan input suara.
Diharapkan hasil penelitian ini dapat memberikan kontribusi nyata bagi
pengembangan ilmu pengetahuan di bidang pemrosesan sinyal audio dan
machine learning, khususnya dalam konteks aplikasi keislaman seperti
pembelajaran dan verifikasi tilawah Al-Quran.

Penyusunan Tugas Akhir ini tidak terlepas dari doa, bantuan, bimbingan,
dukungan, serta motivasi dari berbagai pihak. Oleh karena itu, pada
kesempatan ini penulis menyampaikan ucapan terima kasih yang
sebesar-besarnya kepada:

1.  Bapak Wisnu Uriawan, M.Kom., Ph. D., selaku Dosen Pembimbing I yang
    telah meluangkan waktu, memberikan arahan, bimbingan, motivasi,
    kritik, dan saran yang sangat berarti selama proses penyusunan Tugas
    Akhir ini.

2.  Ibu Dr. Dian Sa\'adillah Maylawati, M.T., selaku Dosen Pembimbing II
    dan juga ketua jurusan teknik informatika yang telah memberikan
    bimbingan, arahan, kritik, dan saran yang membangun selama proses
    penyusunan Tugas Akhir ini hingga dapat diselesaikan dengan baik.

3.  Seluruh dosen dan staf Jurusan Teknik Informatika Fakultas Sains dan
    Teknologi Universitas Islam Negeri Sunan Gunung Djati Bandung yang
    telah memberikan ilmu pengetahuan, pengalaman, serta pelayanan
    akademik selama masa perkuliahan.

4.  Kedua orang tua tercinta beserta seluruh keluarga yang selalu
    memberikan doa, kasih sayang, dukungan, semangat, dan motivasi yang
    tiada henti kepada penulis selama menempuh pendidikan hingga
    penyusunan Tugas Akhir ini.

5.  Farhan Nurzaman teman dekat saya yang telah mengasih arahan teknis
    untuk pengerjaan penelitian ini

6.  Seluruh teman-teman Teknik Informatika angkatan 2019 yang telah
    memberikan dukungan, semangat, serta menjadi rekan belajar dan
    berdiskusi selama proses perkuliahan maupun penyusunan Tugas Akhir.

7.  Semua pihak yang tidak dapat penulis sebutkan satu per satu yang
    telah memberikan bantuan, doa, dan dukungan sehingga Tugas Akhir ini
    dapat diselesaikan.

Penulis menyadari bahwa Tugas Akhir ini masih memiliki keterbatasan dan
jauh dari kata sempurna. Oleh karena itu, kritik dan saran yang
membangun sangat diharapkan demi penyempurnaan penelitian di masa yang
akan datang. Penulis berharap Tugas Akhir ini dapat memberikan manfaat,
menambah wawasan, serta menjadi referensi bagi pengembangan ilmu
pengetahuan, khususnya dalam bidang pemrosesan audio dan machine
learning serta penerapannya pada bidang keislaman.

Akhir kata, penulis memohon maaf apabila terdapat kekurangan dalam
penyusunan Tugas Akhir ini. Semoga Allah SWT. senantiasa melimpahkan
rahmat, hidayah, dan ridha-Nya kepada kita semua.

+-----------------------------------+-----------------------------------+
|                                   | Bandung,                          |
|                                   | ............................20... |
|                                   |                                   |
|                                   | Hormat saya,                      |
|                                   |                                   |
|                                   | Mujahid Ansori Majid              |
+===================================+===================================+

# DAFTAR ISI

# Hlm. {#hlm. .TOC-Heading}

[LEMBAR PERSETUJUAN [i](#lembar-persetujuan)](#lembar-persetujuan)

[LEMBAR PERNYATAAN KARYA
[iii](#lembar-pernyataan-karya)](#lembar-pernyataan-karya)

[LEMBAR PERSEMBAHAN [iv](#lembar-persembahan)](#lembar-persembahan)

[ABSTRAK [v](#abstrak)](#abstrak)

[*ABSTRACT* [vi](#abstract)](#abstract)

[KATA PENGANTAR [vii](#kata-pengantar)](#kata-pengantar)

[DAFTAR ISI [ix](#daftar-isi)](#daftar-isi)

[DAFTAR GAMBAR [xi](#daftar-gambar)](#daftar-gambar)

[DAFTAR TABEL [xii](#daftar-tabel)](#daftar-tabel)

[BAB I PENDAHULUAN [1](#bab-i-pendahuluan)](#bab-i-pendahuluan)

[1.1. Latar Belakang [1](#latar-belakang)](#latar-belakang)

[1.2. Rumusan Masalah [4](#rumusan-masalah)](#rumusan-masalah)

[1.3. Tujuan Penelitian [4](#tujuan-penelitian)](#tujuan-penelitian)

[1.4. Batasan Masalah [5](#batasan-masalah)](#batasan-masalah)

[1.5. Manfaat Penelitian [5](#manfaat-penelitian)](#manfaat-penelitian)

[1.6. Kerangka Pemikiran [5](#kerangka-pemikiran)](#kerangka-pemikiran)

[1.7. Sistematika Penulisan
[8](#sistematika-penulisan)](#sistematika-penulisan)

[BAB II KAJIAN LITERATUR
[11](#bab-ii-kajian-literatur)](#bab-ii-kajian-literatur)

[2.1. Tinjauan Pustaka [11](#tinjauan-pustaka)](#tinjauan-pustaka)

[2.2. Landasan Teori [15](#landasan-teori)](#landasan-teori)

[2.2.1. Kecerdasan buatan (Artificial Intelegence)
[15](#kecerdasan-buatan-artificial-intelegence)](#kecerdasan-buatan-artificial-intelegence)

[2.2.2. Deep Learning [15](#deep-learning)](#deep-learning)

[2.2.3. Similarity Search [16](#similarity-search)](#similarity-search)

[2.2.4. Vector Embedding [17](#vector-embedding)](#vector-embedding)

[2.2.5. Self-Supervised Learning (SSL)
[17](#self-supervised-learning-ssl)](#self-supervised-learning-ssl)

[2.2.6. Wav2vec 2.0 [17](#wav2vec-2.0)](#wav2vec-2.0)

[2.2.7. Data2vec [20](#data2vec)](#data2vec)

[2.2.8. Cosine Similarity [22](#cosine-similarity)](#cosine-similarity)

[2.2.9. Metriks Evaluasi Retrieval
[23](#metriks-evaluasi-retrieval)](#metriks-evaluasi-retrieval)

[2.2.10. Dataset Quran-MD [25](#dataset-quran-md)](#dataset-quran-md)

[2.2.11. Metode Penelitian CRISP-DM
[25](#metode-penelitian-crisp-dm)](#metode-penelitian-crisp-dm)

[BAB III METODOLOGI PENELITIAN
[27](#bab-iii-metodologi-penelitian)](#bab-iii-metodologi-penelitian)

[*3.1.* *Business Understanding*
[27](#business-understanding)](#business-understanding)

[*3.2.* *Data Understanding*
[29](#data-understanding)](#data-understanding)

[*3.3.* *Data Preparation* [33](#data-preparation)](#data-preparation)

[3.3.1 Seleksi dan normalisasi identitas
[35](#seleksi-dan-normalisasi-identitas)](#seleksi-dan-normalisasi-identitas)

[3.3.2. Segmentasi rekaman mahasiswa
[35](#segmentasi-rekaman-mahasiswa)](#segmentasi-rekaman-mahasiswa)

[3.3.3. Validasi dan pembentukan manifes
[36](#validasi-dan-pembentukan-manifest)](#validasi-dan-pembentukan-manifest)

[*3.4.* *Modeling* [36](#modeling)](#modeling)

[3.3.1. Cleaning hasil ekstraksi
[38](#pembersihan-data-hasil-ekstraksi)](#pembersihan-data-hasil-ekstraksi)

[3.4.2. Skor dan pemeringkatan
[39](#skor-dan-pemeringkatan)](#skor-dan-pemeringkatan)

[*3.5.* *Evaluation* [39](#_Toc237480653)](#_Toc237480653)

[*3.6.* *Deployment* [40](#deployment)](#deployment)

[BAB IV HASIL DAN PEMBAHASAN
[42](#bab-iv-hasil-dan-pembahasan)](#bab-iv-hasil-dan-pembahasan)

[4.1. Hasil Business Understanding
[42](#hasil-business-understanding)](#hasil-business-understanding)

[4.2. Hasil Data Understanding
[43](#hasil-data-understanding)](#hasil-data-understanding)

[4.3. Hasil Data Preparation
[47](#hasil-data-preparation)](#hasil-data-preparation)

[4.3.1. Permisahan Dataset dan Peran Sumber
[48](#hasil-pemisahan-dataset-dan-peran-sumber-data)](#hasil-pemisahan-dataset-dan-peran-sumber-data)

[4.4. Hasil Modeling [53](#hasil-modeling)](#hasil-modeling)

[4.4.1. Pemilihan lapisan representasi per Sel
[54](#_Toc237480661)](#_Toc237480661)

[4.4.2. Evaluasi MAP Development per Skenario
[55](#evaluasi-map-development-per-skenario-1)](#evaluasi-map-development-per-skenario-1)

[4.4.3. Ringkasan Pemilihan Layer
[60](#ringkasan-pemilihan-layer-1)](#ringkasan-pemilihan-layer-1)

[4.5. Hasil Evaluation dan Pembahasan
[63](#hasil-evaluation-dan-pembahasan)](#hasil-evaluation-dan-pembahasan)

[4.5.1. Perbandingan bootstrap per sel
[71](#perbandingan-bootstrap-per-sel)](#perbandingan-bootstrap-per-sel)

[4.6. Deployment Konseptual
[72](#deployment-konseptual)](#deployment-konseptual)

[BAB V KESIMPULAN DAN SARAN
[75](#bab-v-kesimpulan-dan-saran)](#bab-v-kesimpulan-dan-saran)

[5.1. Kesimpulan [75](#kesimpulan)](#kesimpulan)

[5.2. Saran [76](#saran)](#saran)

[DAFTAR PUSTAKA [77](#daftar-pustaka)](#daftar-pustaka)

# DAFTAR GAMBAR

[*Gambar 1.1 Kerangka Pemikiran [7](#_Toc236767899)*](#_Toc236767899)

[*Gambar 2.1 Arsitektur Wav2vec2 [22](#_Toc236767908)*](#_Toc236767908)

[*Gambar 2.2 Arsitektur Data2vec [24](#_Toc236767909)*](#_Toc236767909)

[*Gambar 3.1 Adaptasi CRISP-DM [31](#_Toc237464121)*](#_Toc237464121)

[*Gambar 3.2 Persiapan data Quran-MD dan mahasiswa
[36](#_Toc237464122)*](#_Toc237464122)

[*Gambar 3.4 Ekstraksi identitas pembaca dan slug
[40](#_Toc239590310)*](#_Toc239590310)

[*Gambar 3.5 Penanganan anomali [41](#_Toc239590311)*](#_Toc239590311)

[*Gambar 3.6 Normalisasi audio di kedua sumber data
[42](#_Toc239590312)*](#_Toc239590312)

[*Gambar 3.7 Ekstraksi layerwise, checkpoint, manifest dan matriks akhir
[47](#_Toc237464123)*](#_Toc237464123)

[*Gambar 3.8 Evaluasi skenario A, B, C
[49](#_Toc237464124)*](#_Toc237464124)

[*Gambar 4.1 Rekonsiliasi data [53](#_Toc236984558)*](#_Toc236984558)

[*Gambar 4.2 Ukuran query dan database pada scenario A, B, C
[55](#_Toc236984559)*](#_Toc236984559)

[*Gambar 4.3 Layer representasi terpilih
[71](#_Toc239590298)*](#_Toc239590298)

[*Gambar 4.4 Perbandingan MAP Final per Sel
[76](#_Toc239590299)*](#_Toc239590299)

[*Gambar 4.5 Tren MAP final terhadap owner ration database
[77](#_Toc239590300)*](#_Toc239590300)

[*Gambar 4.6 Ringkasan rasio terbaik per metrik
[78](#_Toc239590301)*](#_Toc239590301)

[*Gambar 4.7 Rancangan konseptual deployment retrieval
[80](#_Toc236984561)*](#_Toc236984561)

# DAFTAR TABEL

[Tabel 2.1 *State of the art* [11](#_Toc236748112)](#_Toc236748112)

[Tabel 2.2 Konfigurasi Blok Konvolusi Temporal pada *Encoder* *Audio*
[20](#_Toc236748113)](#_Toc236748113)

[Tabel 3.1 Gambaran peran sumber data pada Skenario A, B, C, dan D
[33](#_Toc239590163)](#_Toc239590163)

[Tabel 3.2 Peran sumber data dalam rancangan eksperimen
[34](#_Toc236984596)](#_Toc236984596)

[Tabel 3.3 Contoh pemetaan nama berkas mahasiswa menuju identitas surah
terstandarisasi [38](#_Toc239590165)](#_Toc239590165)

[Tabel 3.4 Variasi alias terdaftar untuk surah terpilih
[38](#_Toc239590166)](#_Toc239590166)

[Tabel 3.5 Ilustrasi alokasi kata terdeteksi menjadi batas ayat
[43](#_Toc239590167)](#_Toc239590167)

[Tabel 4.1 Rekonsiliasi validasi [53](#_Toc236984570)](#_Toc236984570)

[Tabel 4.2 Ukuran final skenario *retrieval*
[55](#_Toc236984571)](#_Toc236984571)

[Tabel 4.3 Ukuran Query dan *database* referensi pada Skenario A
[59](#_Toc239590203)](#_Toc239590203)

[Tabel 4.4 Ukuran *query* dan *database* referensi pada Skenario B
[59](#_Toc239590204)](#_Toc239590204)

[Tabel 4.5 Ukuran *query* dan *database* referensi pada Skenario C
[60](#_Toc239590205)](#_Toc239590205)

[Tabel 4.6 Ukuran *query* dan *database* referensi pada Skenario D
[60](#_Toc239590206)](#_Toc239590206)

[Tabel 4.7 Pembagian *development set* dan *test set* pada setiap sel
[63](#_Toc239590207)](#_Toc239590207)

[Tabel 4.8 Evluasi layer dengan MAP Skenario A (n_dev = 17.839)
[64](#_Toc239590208)](#_Toc239590208)

[Tabel 4.9 Evaluasi layer dengan MAP Skenario B Wav2vec2
[65](#_Toc239590209)](#_Toc239590209)

[Tabel 4.10 Evaluasi layer dengan MAP Skenario B Data2vec
[66](#_Toc239590210)](#_Toc239590210)

[Tabel 4.11 Evaluasi layer dengan MAP Skenario C Wav2vec2
[67](#_Toc239590211)](#_Toc239590211)

[Tabel 4.12 Evaluasi layer dengan MAP Skenario C Data2vec
[67](#_Toc239590212)](#_Toc239590212)

[Tabel 4.13 Evaluasi layer dengan MAP Skenario D Wav2vec2
[68](#_Toc239590213)](#_Toc239590213)

[Tabel 4.14 Evaluation layer dengan MAP Skenario D Data2vec
[69](#_Toc239590214)](#_Toc239590214)

[Tabel 4.15 Ringkasan layer terpilih dan MAP development
[70](#_Toc239590215)](#_Toc239590215)

[Tabel 4.16 Hasil Evaluasi Skenario A
[72](#_Toc239590216)](#_Toc239590216)

[Tabel 4.17 Hasil Evaluasi *Owner Ratio* 60:40 pada Skenario B, C, dan D
[73](#_Toc239590217)](#_Toc239590217)

[Tabel 4.18 Hasil Evaluasi *owner ratio* 70:30 pada skenario B, C, dan D
[73](#_Toc239590218)](#_Toc239590218)

[Tabel 4.19 Hasil Evaluasi *owner ration* 80:20 pada skenario B, C, dan
D [74](#_Toc239590219)](#_Toc239590219)

[Tabel 4.20 Hasil Evaluasi owner ration 90:10 pada skenario B, C, dan D
[75](#_Toc239590220)](#_Toc239590220)

[Tabel 4.21 Perbandingan *bootstrap* selisih MAP
[79](#_Toc236984574)](#_Toc236984574)

**DAFTAR RUMUS**

[Rumus (2.1) [18](#_Toc239590104)](#_Toc239590104)

[Rumus (2.2) [21](#_Toc239590105)](#_Toc239590105)

[Rumus (2.3) [21](#_Toc239590106)](#_Toc239590106)

[Rumus (2.4) [21](#_Toc239590107)](#_Toc239590107)

[Rumus (2.5) [22](#_Toc239590108)](#_Toc239590108)

[Rumus (2.6) [23](#_Toc239590109)](#_Toc239590109)

[Rumus (2.7) [24](#_Toc239590110)](#_Toc239590110)

[Rumus (2.8) [25](#_Toc239590111)](#_Toc239590111)

[Rumus (2.9) [25](#_Toc239590112)](#_Toc239590112)

[Rumus (2.10) [26](#_Toc239590113)](#_Toc239590113)

[Rumus (2.11) [26](#_Toc239590114)](#_Toc239590114)

[Rumus (2.12) [26](#_Toc239590115)](#_Toc239590115)

# BAB I PENDAHULUAN

##  Latar Belakang

Perkembangan *deep learning* dalam pemrosesan audio telah mendorong
penggunaan model *Self-Supervised Learning* (SSL) untuk mempelajari
representasi laten dari data audio mentah tanpa memerlukan label
transkripsi dalam pemrosesan *pretraining*. Wav2Vec 2.0 menjadi salah
satu model yang berpengaruh dalam perkembangan pembelajaran representasi
berbasis SSL. Model ini mampu mempelajari representasi laten (*latent
embeddings*) langsung dari data audio mentah, yang kemudian dapat
dimanfaatkan dan disesuaikan untuk berbagai tugas pemrosesan audio
dengan kebutuhan data berlabel yang lebih sedikit. \[1\], \[2\].
Kemampuan menghasilkan representasi laten tersebut tidak hanya dapat
dimanfaatkan untuk tugas *Automatic Speech Recognition* (ASR), tetapi
juga untuk tugas non-transkripsi, termasuk audio *retrieval*. Pada audio
*retrieval*, sistem tidak bertujuan menghasilkan teks dari masukan
audio, melainkan menemukan entri dalam *database* yang memiliki hubungan
atau kandungan yang sama berdasarkan kemiripan representasi audio.
Dengan demikian, kualitas representasi menjadi faktor penting karena
*embedding* yang dihasilkan harus mampu mempertahankan karakteristik
audio yang relevan untuk membedakan satu unit bacaan dengan unit
lainnya.

Kebutuhan tersebut menjadi semakin penting ketika audio yang
merepresentasikan unit yang sama memiliki variasi akustik yang tinggi.
Salah satu domain yang menyediakan kondisi tersebut adalah korpus
pembacaan Al-Qur\'an. Rekaman ayat yang sama dapat dihasilkan oleh
pembaca yang berbeda sehingga memiliki karakteristik suara dan gaya
pembacaan yang beragam, sementara kandungan ayat yang direpresentasikan
tetap sama. Kondisi tersebut tercermin pada Ar-DAD, sebuah dataset audio
bahasa Arab yang menyediakan 15.810 klip dari 30 pembaca yang mencakup
37 surah, sehingga memungkinkan variasi antar pembaca pada konten bacaan
yang sama untuk dipelajari \[3\]. Dataset QURAN-MD juga menyediakan
rekaman pembacaan Al-Qur\'an pada tingkat ayat dari berbagai reciter
dengan beragam karakteristik pembacaan, serta menyediakan data pada
tingkat kata yang dapat digunakan untuk analisis pelafalan dan fonologi
\[4\]. Selain variasi antar pembaca, pembacaan Al-Qur\'an memiliki
karakteristik fonetik yang berkaitan dengan penerapan kaidah tajwid.
Penelitian mengenai pengenalan fonem pada bacaan Al-Qur\'an menunjukkan
bahwa karakteristik seperti pemanjangan fonem memiliki durasi yang perlu
diperhatikan secara khusus dan tidak sepenuhnya dapat direpresentasikan
hanya menggunakan karakteristik fonetik bahasa Arab secara umum \[5\].
Dengan demikian, korpus pembacaan Al-Qur\'an menyediakan kondisi
pengujian yang relevan karena sistem perlu mempertahankan informasi
fonetik yang membedakan antar ayat sekaligus menghadapi variasi
karakteristik suara antar pembaca. Karakteristik tersebut menjadikan
pembacaan Al-Qur\'an sebagai domain yang relevan untuk menguji kemampuan
representasi audio dalam melakukan pencocokan berbasis *embedding*.

Relevansi penggunaan representasi audio *pretrained* pada domain
tersebut juga berkaitan dengan permasalahan transfer lintas bahasa.
Model SSL umumnya mempelajari karakteristik fonetik berdasarkan
distribusi data yang digunakan selama *pretraining*, sehingga
representasi yang dihasilkan belum tentu memiliki karakteristik yang
sama ketika diterapkan pada bahasa atau domain yang berbeda. Dalam
eksperimen yang dilakukan oleh Toyin dkk., model berbasis bahasa Arab
yang berukuran lebih kecil mampu mengungguli model multibahasa yang
berukuran lebih besar dalam tugas pengenalan ucapan bahasa Arab \[6\].
Temuan tersebut menunjukkan bahwa kesesuaian karakteristik data
*pretraining* terhadap bahasa target dapat memengaruhi kualitas
representasi yang dihasilkan. Temuan ini diperkuat oleh Li dkk. yang
secara kuantitatif menunjukkan bahwa *encoder* SSL yang dilatih
menggunakan data berbahasa Inggris menghasilkan fitur fonetik yang
kurang informatif ketika diterapkan pada input lintas bahasa tanpa
adaptasi domain \[7\]. Namun, bukti mengenai keterbatasan lintas bahasa
tersebut masih banyak diperoleh dari arsitektur SSL yang menggunakan
paradigma *contrastive learning*, khususnya Wav2Vec 2.0.

Sementara itu, Data2Vec menggunakan paradigma *self-distillation* yang
secara fundamental berbeda dari pendekatan *contrastive learning*.
Alih-alih mempelajari representasi dengan membandingkan pasangan positif
dan negatif, Data2Vec melatih jaringan student untuk memprediksi
representasi kontekstual yang dihasilkan oleh jaringan *teacher* dari
keseluruhan masukan \[1\], \[8\]. Perbedaan mekanisme *pretraining*
tersebut secara teoretis dapat menghasilkan karakteristik representasi
yang berbeda. Data2Vec menggunakan target kontinu berupa representasi
kontekstual sehingga pembelajaran tidak secara langsung bergantung pada
unit target diskrit yang dibentuk dari karakteristik data *pretraining*.
Temuan Xue dkk. menunjukkan bahwa penggunaan target kontinu seperti pada
Data2Vec dapat menurunkan *phoneme error rate* pada skenario lintas
bahasa sebesar 5,3% dibandingkan pendekatan dengan target diskrit \[9\].
Temuan tersebut menunjukkan bahwa paradigma *pretraining* dapat
memengaruhi kemampuan model dalam mempertahankan informasi fonetik
ketika diterapkan pada bahasa yang berbeda. Namun demikian, temuan
tersebut diperoleh dalam konteks tugas pengenalan fonem dan belum secara
langsung menjawab apakah perbedaan karakteristik representasi tersebut
juga memberikan keuntungan pada tugas audio *retrieval*.

Kesenjangan tersebut menjadi semakin penting karena kinerja representasi
pada tugas transkripsi tidak dapat secara langsung dianggap setara
dengan kinerja pada tugas *retrieval*. Pada tugas ASR, representasi
audio digunakan sebagai dasar untuk menghasilkan transkripsi dan umumnya
diikuti oleh proses penyesuaian model terhadap tugas tertentu. Oleh
karena itu, metrik seperti *Word Error Rate* (WER) atau *Character Error
Rate* (CER) tidak hanya mencerminkan kualitas representasi awal, tetapi
juga dipengaruhi oleh proses pelatihan dan penyesuaian model terhadap
data dan tugas transkripsi. Dengan demikian, model yang menghasilkan
kinerja transkripsi lebih baik belum tentu menghasilkan representasi
yang lebih baik untuk kebutuhan *retrieval*. *Retrieval* menguji
kemampuan yang berbeda, yaitu kemampuan representasi untuk menempatkan
sampel yang memiliki identitas atau kandungan yang sama dalam ruang
*embedding* yang berdekatan, sementara sampel yang berbeda ditempatkan
lebih jauh.

Perbedaan tersebut menjadi semakin jelas pada sistem *retrieval*
berbasis *frozen embedding*. Dalam pendekatan ini, parameter model tidak
disesuaikan menggunakan data Al-Qur\'an sehingga representasi yang
dihasilkan oleh masing-masing model digunakan secara langsung sebagai
*vector embedding*. Kemudian, kemiripan antar embedding dihitung
menggunakan *cosine similarity* untuk menentukan urutan hasil pencarian.
Kinerja model dengan demikian ditentukan oleh kemampuan representasinya
dalam menempatkan audio yang berasal dari ayat yang sama pada posisi
yang berdekatan dan membedakannya dari audio yang berasal dari ayat yang
berbeda. Evaluasi dilakukan menggunakan metrik *retrieval*, yaitu *Mean
Average Precision* (MAP), *Mean Reciprocal Rank* (MRR), dan *Top-K
Accuracy*. Kondisi *frozen embedding* tersebut memungkinkan perbandingan
difokuskan pada kemampuan representasi *pretrained* masing-masing model,
tanpa pengaruh tambahan dari proses fine-tuning terhadap korpus
pembacaan Al-Qur\'an.

Berdasarkan uraian tersebut, masih terdapat celah penelitian mengenai
apakah perbedaan paradigma *pretraining* antara Wav2Vec 2.0 dan Data2Vec
menghasilkan karakteristik representasi yang berbeda ketika digunakan
secara langsung untuk audio *retrieval* ayat Al-Qur\'an. Penelitian
terdahulu mengenai transfer lintas bahasa telah memberikan bukti
mengenai perbedaan kualitas representasi pada tugas fonetik atau
transkripsi, tetapi perbandingan langsung antara Wav2Vec 2.0 dan
Data2Vec dalam kondisi *frozen embedding* untuk tugas *retrieval* ayat
Al-Qur\'an, khususnya yang dievaluasi menggunakan MAP, MRR, dan Top-K
Accuracy, masih terbatas. Oleh karena itu, penelitian ini berfokus pada
implementasi Wav2Vec 2.0 dan Data2Vec sebagai penghasil representasi
laten audio, kemudian membandingkan kinerja kedua model dalam sistem
retrieval ayat Al-Qur\'an. Setiap model digunakan tanpa *fine-tuning*
pada data Al-Qur\'an sehingga perbandingan dilakukan dalam kondisi yang
sama dan diarahkan untuk mengukur kemampuan representasi *pretrained*
secara langsung. Selain membandingkan kedua model, analisis dilakukan
pada setiap lapisan representasi untuk menentukan lapisan yang
menghasilkan kinerja *retrieval* terbaik bagi masing-masing model.
Dengan demikian, penelitian ini diharapkan dapat memberikan gambaran
empiris mengenai karakteristik representasi Wav2Vec 2.0 dan Data2Vec
pada tugas *retrieval* audio ayat Al-Qur\'an serta mengetahui apakah
perbedaan paradigma *pretraining* menghasilkan pola kinerja *retrieval*
yang berbeda.

##  Rumusan Masalah

Berdasarkan Rumusan masalah terbagi menjadi beberapa point

1.  Bagaimana implementasi model Wav2vec2 dan Data2vec dalam tugas
    *retrieval* ayat Al-Quran?

2.  Bagaimana perbandingan kinerja *frozen-embedding* Wav2vec2 dan
    Data2vec dalam tugas *retrieval* ayat Al-Qur'an?

##  Tujuan Penelitian

Tujuan dari penelitian ini sebagai berikut:

1.  Mengimplementasikan model Wav2vec2 dan Data2vec dalam tugas
    retrieval ayat Al-Quran.

2.  Mengevaluasi perbandingan kinerja frozen embedding Wav2vec2 dan
    Data2vec dalam tugas *retrieval* ayat Al-Qur'an

##  Batasan Masalah

Batasan masalah dari penelitian ini adalah sebagai berikut:

1.  Penelitian dibatasi pada penggunaan model *pre-trained* Data2Vec dan
    Wav2Vec2 tanpa proses *fine-tuning.*

2.  Eksperimen dilakukan pada teks Al-Qur'an, dengan objek penelitian
    berupa surah Al -- Fatihah dan Juz Amma.

3.  Pengujian input audio hanya dapat dilakukan per-ayat.

##  Manfaat Penelitian

Penelitian ini diharapkan memberikan manfaat yang dipaparkan sebagai
berikut:

1.  Manfaat teoritis, penelitian ini memberikan kontribusi terhadap
    pemahaman mengenai perbedaan karakteristik representasi yang
    dihasilkan oleh dua paradigma *self-supervised learning* yang
    berbeda, yaitu *contrastive learning* pada Wav2Vec 2.0 dan
    *self-distillation* pada Data2Vec, ketika diterapkan tanpa
    *fine-tuning* pada domain pembacaan Al-Qur\'an. Temuan mengenai
    lapisan representasi yang menghasilkan kinerja terbaik pada
    masing-masing model dapat menjadi referensi bagi kajian teoretis
    tentang distribusi informasi pada arsitektur *Transformer* berlapis
    dalam konteks tugas *retrieval* audio.

2.  Manfaat praktis, Secara praktis, penelitian ini menyediakan bukti
    empiris yang dapat dijadikan dasar bagi pengembang perangkat lunak
    dalam memilih model representasi dan lapisan model yang sesuai untuk
    sistem pencarian ayat Al-Qur\'an berbasis audio. Penelitian ini
    tidak hanya menghasilkan temuan numerik, tetapi juga meletakkan
    prosedur metodologis yang dapat direplikasi dan dikembangkan oleh
    penelitian selanjutnya.

3.  Manfaat di bidang audio, Di bidang pemrosesan audio, penelitian ini
    menunjukkan kelayakan penggunaan representasi laten dari model
    *self-supervised* sebagai dasar sistem *retrieval* audio tanpa
    memerlukan tahap transkripsi. Pendekatan ini menghindari
    ketergantungan pada sistem *Automatic Speech Recognition* (ASR) yang
    pada umumnya belum tersedia secara memadai untuk pembacaan
    Al-Qur\'an dengan gaya tartil maupun mujawwad. Temuan penelitian ini
    dapat menjadi landasan bagi pengembangan sistem pencarian audio yang
    bekerja langsung pada tingkat representasi akustik, sehingga
    berpotensi mendukung teknologi pembelajaran Al-Qur\'an yang
    memungkinkan pengguna menemukan ayat dari rekaman bacaan tanpa
    memasukkan teks secara manual.

##  Kerangka Pemikiran

Kerangkan pemikiran dipetakan dalam gambar sebagai berikut

![[]{#_Toc236767899 .anchor}Gambar 1.1 Kerangka
Pemikiran](media/image2.png){width="4.951388888888889in"
height="2.7141502624671916in"}

Penelitian ini berangkat dari sebuah fenomena dalam bidang pemrosessan
audio, yaitu kemampuan model *Self-Supervised Learning* (SSL) seperti
Wav2vec2 dan Data2vec dalam menghasilkan representasi laten, langsung
dari audio mentah tanpa memerlukan label transkripsi. Kemampuan ini
membuka peluang bagi tugas penelurusan kemiripan audio (*audio
retrieval*). Di mana sistem *retrieval* yang bekerja langsung pada
*latent embedding* dapat mengungguli pendekatan dua tahap berbasis ASR.
Keunggulan ini bersumber dari kemampuannya menghindari kesalahan
propagasi (*error propagation),* yakin kesalahan transkripsi pada tahap
ASR yang akan merambat dan mendegradasi kualitas pencocokan pada tahap
selanjutnya.

Fenomena tersebut ditopang oleh sejumlah landasan teori dan studi
literatur. Pada tataran mekanisme model, Wav2vec2 mempelajari
representasi melalui *contrastive learning* yang membedakan sampel
positif dan negatif menggunakan *quntized discrete units.* Sementara
Data2vec mengadopsi paradigma *self-distillation* yang melatih jaringan
student untuk memprediksi representasi kontekstual penuh dari jaringan
*teacher* dengan target berupa konteks latent yang kontinu. Perbedaan
paragima inilah yang menjadi dasar perbandingan dalam penelitian ini.
Pada tataran tugas *retrieval*, penelitian Öberg menunjukan bahwa
pencarian audio dapat dilakukan langsung dari audio tanpa tahap enkripsi
\[10\]. Teori ini didukung pula oleh bukti empiris *Gemini embedding 2*,
secara eksplisit menunjukkan bahwa pendekatan *cascaded* ASR ke
*retrieval* mengalami *error propagation* \[11\]. Sedangkan *WavRag*
membuktikan bahwa *retrieval* langsung dari audio mampu menyamai akurasi
pipeline ASR dengan efisiensi yang jauh lebih tinggi \[12\].

Meskipun demikian, keunggulan tersebut belum tentu berlaku ketika model
diterapkan pada domain fonteik yang spesifik seperti bahasa Arab dalam
bacaan Al-Qur'an. Persoalan muncul karena model SSL umumnya
di-*pretrain* pada korpus Bahasa inggris dan kurang informatikf untuk
fonetik Arab. Penelitian Toyin dkk dan Li dkk telah menunjukan
keterbatasan ini \[6\], \[7\]. namun bukti tersebut hanya terbatas pada
arsitektur berbasis constrastive learing seperti Wav2vec2. Sementara
itu, ptotensi paradigma self-distillation yang dimiliki Data2Vec untuk
tugas audio retrieval Al-Quran masih sangat minim dikaji, dan
perbandingan langsung antara kedua model menggunakan metrik *similarity
search*, bukan metrik transkripsi seperti *word error rate* (WER) hingga
kini masih terbatas dalam literatur.

Untuk menjawab persoalan tersebut, penelitian ini menawarkan pendekatan
berupa implementasi dan perbandingan Wav2Vec2 dan Data2Vec dalam
menghasilkan representasi laten untuk tugas retrieval ayat Al-Qur\'an.
Fitur laten dari kedua model *pretrained* diekstraksi dalam kondisi beku
atau *frozen*, kemudian digunakan untuk membentuk representasi vektor
audio. Setiap vektor *query* dibandingkan dengan vektor pada database
(*ground truth)* menggunakan *cosine similarity* sebagai *scoring
function*, lalu hasilnya diurutkan untuk menghasilkan daftar *retrieval*
mulai dari ayat yang memiliki skor paling tinggi menuju yang terendah.

Evaluasi dilakukan menggunakan metrik *Mean Average Precision* (MAP),
*Mean Reciprocal Rank* (MRR), dan *Top-K Accuracy*. Selain membandingkan
kinerja kedua model, evaluasi juga dilakukan pada beberapa lapisan untuk
mengetahui lapisan yang menghasilkan kinerja *retrieval* paling baik.
Hasil lapisan terbaik tersebut digunakan sebagai konfigurasi dalam
perbandingan utama antara Wav2Vec2 dan Data2Vec.

Proses penelitian dilaksanakan mengikuti kerangka kerja atau *framework*
*Cross-Industry Standard Process for Data Mining* (CRISP-DM) yang
diadaptasi ke dalam beberapa tahap, yaitu *Business Understanding*,
*Data Understanding*, *Data Preparation*, *Modeling*, *Evaluation*, dan
*Deployment*. Tahapan tersebut digunakan untuk mengarahkan proses
pengumpulan data, persiapan *audio*, ekstraksi *frozen embedding*,
penerapan *similarity search*, evaluasi kinerja, serta analisis hasil.

Hasil akhir penelitian berupa implementasi sistem *retrieval* berbasis
*frozen embedding* menggunakan Wav2Vec2 dan Data2Vec, serta perbandingan
kinerja kedua model pada data *audio* ayat Al-Qur\'an. Hasil tersebut
diharapkan dapat memberikan gambaran mengenai model dan konfigurasi
lapisan yang paling sesuai untuk mendukung pengembangan sistem *audio
retrieval* pada domain bacaan Al-Qur\'an.

##  Sistematika Penulisan

Sistematika penulisan laporan tugas akhir ini disusun ke dalam lima bab
yang saling berkait secara berurutan, dengan gambaran kandungan setiap
bab sebagai berikut.

**BAB I PENDAHULUAN**

Bab ini menguraikan dasar pemikiran yang melatarbelakangi dilakukannya
penelitian, mencakup latar belakang yang memaparkan fenomena dan celah
penelitian, rumusan masalah sebagai pertanyaan penelitian yang hendak
dijawab, batasan masalah yang menetapkan ruang lingkup penelitian,
manfaat penelitian, *state of the art*, kerangka pemikiran, serta
sistematika penulisan. Bab ini berfungsi sebagai fondasi mengarahkan
keseluruhan penelitian.

**BAB II TINJAUAN PUSTAKA**

Bab ini memaparkan landasan teori dan konsep-konsep fundamental yang
menjadi dasar ilmiah penelitian, meliputi teori mengenai Self-Supervised
Learning (SSL), arsitektur dan mekanisme kerja model Waw2vec2 dan
Data2Vec, konsep representasi latent (*latent embedding*), tugas *audio
retrieval*, serta metrik evaluasi *similarity search*. Berbeda dengan
Bab I yang menyoroti mengapa penelitian mesti dilakukan, bab ini
menyediakan kerangka teoritis apa saja konsep yang digunakan untuk
memahami dan menyelesaikan permasalahan.

**BAB III METODOLOGI PENELITIAN**

Bab ini menjelaskan tahapan prosedur pelaksanaan penelitian secara
sistematis mengikuti kerangka kerja *Cross-Industry Standard Process for
Data Mining* (CRISP-DM) yang diadaptasi, mencakup pengumpulan data,
*pre-processing*, ekstrasi representasi laten, implementasi *similarity
search*, hingga skenario evaluasi. Jika Bab II menjelaskan apa landasan
teorinya, bab ini menguraikan bagaimana teori tersebut diterapkan secara
teknis dan operasional dalam penelitian.

**BAB IV HASIL DAN PENELITIAN**

Bab ini menyajikan hasil implementasi dan evaluasi sistem *retrieval
audio* ayat Al-Qur\'an menggunakan *frozen embedding* Wav2Vec2 dan
Data2Vec. Pembahasan mencakup hasil persiapan data, ekstraksi
representasi laten pada setiap lapisan model, penerapan *cosine
similarity*, penyusunan peringkat hasil *retrieval*, serta evaluasi
menggunakan metrik *Top-K Accuracy*, *Mean Reciprocal Rank* (MRR), dan
*Mean Average Precision* (MAP). Bab ini juga menyajikan perbandingan
kinerja kedua model berdasarkan konfigurasi lapisan yang optimal dan
membahas implikasi hasil tersebut terhadap penerapan sistem *retrieval*
berbasis suara.

**BAB V KESIMPULAN DAN SARAN**

Bab ini memuat kesimpulan yang menjawab secara ringkas dan tegas rumusan
masalah berdasarkan temuan pada Bab IV, disertai saran untuk
pengembangan penelitian selanjutnya. Berbeda dengan Bab IV yang
menyajikan analisis rinci, bab ini merangkum intisari penelitian secara
menyeluruh dan menawarkan arah bagi penelitian di masa mendatang.

# 

# BAB II KAJIAN LITERATUR

##  Tinjauan Pustaka

Analisis literatur terhadap artikel maupun jurnal terdahulu yang relevan
dilakukan untuk memperdalam pemahaman mengenai permasalahan utama dalam
penerapan *Self-Supervised Learning* (SSL) pada representasi ucapan
(*speech representation*) serta pemanfaatannya untuk pencarian audio
berbasis kemiripan (*audio similarity search*) dengan pendekatan
*query-by-example*. Kajian ini sekaligus bertujuan untuk
mengindentifikasi keterbatasan penelitian sebelumnya. Berikut rangkuman
dari penelitian terdahulu dalam bentuk tabel.

  ----------------------------------------------------------------------------------------------------
  **No**   **Peneliti**   **Metode**            **Dataset**                  **Hasil**
  -------- -------------- --------------------- ---------------------------- -------------------------
  1        Baevski dkk    *contrastive masked   LibriSpeech, Libri-light     WER 1,8%/3,3%
           (2020) \[1\]   prediction of                                      (clean/other) dengan 10
                          quantized units*                                   menit data berlabel + 53k
                                                                             jam pra-pelatihan WER
                                                                             4,8%/8,2%.

  2        Baevski dkk    *self-distillation,   LibriSpeech/ Libri-light,    Pada setup 100 jam data
           (2022) \[8\]   teacher-student* EMA, ImageNet-1K, BooksCorpus +   berlabel: WER 2,8%/6,8%
                                                Wikipedia                    (test clean/other),
                                                                             mengungguli Wav2Vec2 Base
                                                                             (3,4%/8,0%) dan HuBERT
                                                                             Base (3,4%/8,1%).

  3\.      Alkanhal dkk   Wav2vec2 dan Data2vec Aswat, Common Voice (AR),    Mencapai WER SOTA 11,7%
           (2021) \[13\]                        MGB-2                        pada Common Voice dan
                                                                             10,3% pada MGB-2 berkat
                                                                             pra-pelatihan pada Aswat.

  4\.      Öberg (2025)   Projection Network +  Google FLEURS, Common Voice  embedding hasil
           \[10\]         contrastive triplet                                transformasi mencapai MRR
                          loss pada embedding                                0,811 dan Recall@20
                          Wav2Vec2                                           0,931, jauh meningkat
                                                                             dari embedding asli.

  5\.      San dkk (2021) QbE-STD dengan frozen Gronings, Besemah,           Peningkatan relatif
           \[14\]         Wav2Vec2              Mavir/Mboshi; LibriSpeech,   56--86% atas SOTA;
                          (English-mono &       XLSR-53.                     English-mono mengungguli
                          XLSR-53).                                          XLSR multibahasa pada 4
                                                                             dataset.

  6\.      Zhaoqi Li dkk  Neural acoustic word  Korpus QbE-STD               Average Precision (AP)
           (2021) \[15\]  embeddings + Wav2Vec,                              meningkat relatif 11,1%
                          circle loss                                        dibanding sistem MFCC;
                          menggantikan triplet                               konvergensi ±40% epoch
                          loss                                               lebih cepat dengan circle
                                                                             loss.

  7\.      Xie dkk (2022) Two-phase: Siamese    ASVspoof 2019 Logical Access EER turun dari SOTA
           \[16\]         Network + contrastive (LA)                         sebelumnya 4,07% menjadi
                          loss, lalu MLP                                     1,15% pada evaluation
                          classifier                                         set.

  8\.      Tang dkk       Perbandingan HuBERT   SUPERB benchmark (task SID & Data2Vec Base: SID 70,21%
           (2022) \[17\]  vs Data2Vec pada SID  ASR; basis                   / WER 4,94%; HuBERT Base:
                          & ASR                 LibriSpeech/VoxCeleb-line)   SID 81,42% / WER 6,42%
                                                                             HuBERT unggul di SID,
                                                                             Data2Vec unggul di ASR.

  9\.      Yoon dkk       Data2Vec 2.0 Base,    LibriSpeech 960 jam, 11      Mengungguli Data2Vec 2.0
           (2023) \[18\]  regularisasi          SUPERB subtasks              di hampir semua tugas
                          konsistensi,                                       kecuali QbE; mencapai
                          fine-tune parsial.                                 SOTA pada PR, ASR, KS,
                                                                             IC, SF, dan ER (mis. WER
                                                                             4,68%, SID 82,40%).

  10\.     Yang dkk       *Frozen embedding*    LibriSpeech, QUESST,         Data2Vec Large mencapai
           (2024) \[19\]  Data2Vec Base/Large + VoxCeleb, LibriTTS           speaker-invariance
                          *lightweight                                       terbaik (7,38 ACC), PER
                          prediction head*                                   terbaik (2,58%), dan
                                                                             performa VC terbaik (6,75
                                                                             MCD, 100% akurasi
                                                                             kemiripan target
                                                                             speaker).

  11\.     Zhang dkk      X-ARES                Berbagai *dataset* yang      X-ARES mengevaluasi
           (2025) \[20\]                        dikategorikan menjadi 3      performa audio encoder
                                                bagian, *speech* dengan 11   melalui evaluasi linear
                                                *datasets*, *sound* dengan 7 dan evaluasi tanpa
                                                *datasets*, dan *music* 4    parameter menggunakan
                                                *datasets*                   k-nearest neighbor.
                                                                             Data2vec dan Wav2Vec2
                                                                             menunjukkan performa yang
                                                                             bergantung pada jenis
                                                                             tugas dan domain. Studi
                                                                             ini relevan untuk
                                                                             evaluasi frozen
                                                                             embedding, tetapi bukan
                                                                             perbandingan retrieval
                                                                             ayat menggunakan MAP,
                                                                             MRR, dan Top-K.

  12\.     Zhang dkk      *Audio Encoder        *Dataset benchmark* audio    Penelitian ini
           (2025) \[21\]  Capability Challenge* pada domain ucapan, musik,   menyediakan protokol
                          untuk mengevaluasi    dan suara lingkungan.        seragam untuk
                          kemampuan berbagai                                 membandingkan berbagai
                          audio *encoder*                                    audio *encoder*. Wav2Vec2
                          dengan evaluasi                                    dan Data2vec termasuk
                          *parameter free                                    dalam kelompok *encoder*
                          k-nearest neighbor*.                               yang dievaluasi. Namun,
                                                                             penelitian ini berfokus
                                                                             pada evaluasi kemampuan
                                                                             *encoder* secara umum,
                                                                             bukan *retrieval* dengan
                                                                             kondisi eksperimen yang
                                                                             sama.

  13\.     Singh dkk      BEST-STD, yaitu       LibriSpeech dan TIMIT.       BEST-STD mengubah
           (2024) \[22\]  *bidirectional                                     representasi ucapan
                          Mamba-enhanced speech                              menjadi token diskret
                          tokenization* untuk                                untuk mendukung
                          *spoken-term                                       *spoken-term detection*.
                          detection.*                                        Model yang diusulkan
                          Representasi                                       mengungguli beberapa
                          *frame-level*                                      baseline seperti HuBERT,
                          dikonversi menjadi                                 WavLM, SpeechTokenizer,
                          token diskret dan                                  dan EnCodec berdasarkan
                          dicari melalui indeks                              MAP, MRR, dan MTWV.
                          terbalik*.*                                        Metode ini relevan untuk
                                                                             efisiensi indeks
                                                                             *retrieval*, tetapi tidak
                                                                             membandingkan *frozen*
                                                                             Wav2Vec2 dengan Data2vec.

  14\.     Singh dkk      Language-agnostic     Kathbath, yaitu korpus       Tokenizer yang diusulkan
           (2025) \[23\]  speech tokenizer      ucapan multibahasa India.    menghasilkan representasi
                          dengan kuantisasi                                  token untuk spoken-term
                          vektor dan retrieval                               detection lintas bahasa.
                          bertahap berbasis                                  Sistem ini diuji pada
                          TF-IDF.                                            kondisi *monolingual*,
                                                                             *multilingual*, dan
                                                                             *cross-lingual*.
                                                                             Penelitian ini mendukung
                                                                             konsep *retrieval* tanpa
                                                                             transkripsi,

  15\.     El Assadi dkk. MAEB (*Massive Audio  Berbagai dataset audio yang  MAEB membandingkan lebih
           (2026) \[24\]  Embedding             mencakup domain ucapan,      dari 50 audio *encoder*
                          Benchmark*), yaitu    musik, suara lingkungan,     pada berbagai tugas,
                          benchmark *embedding* klasifikasi, dan             termasuk *clustering*,
                          audio berskala besar. *retrieval*.                 *pair classification*,
                          *Embedding* audio                                  dan *retrieval*. Hasil
                          diekstraksi dari                                   evaluasi menunjukkan
                          berbagai model pada                                bahwa tidak ada satu
                          berbagai domain.                                   *encoder* yang unggul
                                                                             pada seluruh domain.
                                                                             Wav2Vec2 dan Data2vec
                                                                             lebih relevan untuk
                                                                             karakteristik task ucapan
                                                                             dibandingkan domain audio
                                                                             lainnya.
  ----------------------------------------------------------------------------------------------------

  : []{#_Toc236748112 .anchor}Tabel 2.1 *State of the art*

Penelitian mengenai representasi ucapan yang dilatih sendiri
(*self-supervised*) telah mengalami kemajuan pesat, dimulai dengan
fondasi seperti Wav2vec 2.0 dan Data2vec \[1\], \[8\] yang mampu
mempelajari representasi laten koheren dari *raw audio*. Namun,
mayoritas penelitian awal dan penelitian lanjutan berkonsentrasi pada
peningkatan kinerja ASR (*automatic speech recognition*), termasuk studi
lintas bahasa dan adaptasi domain. Sebagian penelitian lain berfokus
pada non-retrieval yang tetap memanfaatkan kualitas embedding SSL,
seperti deteksi *spoofing* suara dan identifikasi suarat yang
menggabungkan *Siamase Network* pada fitur Wav2vec2 \[16\]. Meskipun
demikian, Sebagian studi mulai menggeser fokus ke kapabilitas
*retrieval* dari *embedding* ini, misalnya penelitian Öberg mengenai
audio search berbasis transformasi *embedding* Wav2vec2 dengan
*contrastive learning* \[10\]*.*

Pada ranah *retrieval* berdasarkan kemiripan, sejumlah penelitian telah
membuktikan keunggulan pendekatan berbasis *embedding* SSL. Nay san dkk.
telah membuktikan bahwa representasi dari Wav2vec2 dapat melakukan
*query-by-example* *spoken term detection* lintas bahasa tanpa
transkripsi maupun fine-tuning dengan peningkatan 56%-86% atas *state of
the art* sebelumnya \[14\]. Sejalan dengan itu, Zhaoqi Li dkk
membuktikan bahwa *Neural Acoustic Word Embeddings* berbasis pretrain
Wav2vec2 mengungguli fitur MFCC untuk retrieval berbasis kemiripan
\[15\]. Temuan-temuan ini menegaskan bahwa kualitas embedding SSL
menjadikannya sangat sesuai dengan untuk tugas *similarity search.*

Dalam konteks bahasa Arab, Alkanhal dkk melakukan perbandingan langsung
antara Wav2vec2 dan Data2vec. Mereka menemukan bahwa Data2vec konsisten
mengungguli Wav2vec2 dalam domain ASR Arab. Namun, perbandingan tersebut
dilakukan dengan *supervised fine-tuning* dan juga dengan pengukuran
metrik transkripsi, bukan *frozen embedding* tanpa *fine-tuning* dan
metrik *retrieval*. Di sisi lain, kajian terhadap Data2vec pada
tugas-tugas non-ASR mulai bermunculan dan justru memperlihatkan pola
yang bergantung pada jenis tugasnya. Tang, dkk menemukan bahwa
efektivitas Data2vec berbeda tergantung tugasnya, Data2vec lebih unggul
dalam tugas yang berbasis konten contohnya ASR, namun kurang optimal
dalam identifikasi pembicara. Yoon dkk mengonfirmasi kecenderungan
serupa, Data2vec 2.0 kuat pada tugas berbasis konten seperti ASR, tetapi
performa *query-by-example* nya justru menurun ketika diberi regulasi
tambahan \[18\]. Evaluasi berskala besar yang dilakukan oleh Yang dkk
mempertegas hal ini, menunjukan bahwa Data2vec mencapai performa terbaik
pada *Voice Conversion* dan *Phone Recognition*, tetapi tertinggal pada
*Speaker Identification* dan ASR *out-of-distribution* dibanding HuBERT
dan WavLM.

Berdasarkan kajian tersebut, terlihat bahwa keunggulan Data2vec tidak
bersifat universal, melainkan spesifik terhadap jenis tugas. Fakta ini
memunculkan pertanyaan fundamental yang belum terjawab dalam literatur.
Apakah keunggulan Data2vec pada ASR bahasa Arab akan sama ketika kedua
model dievaluasi tanpa fine-tuning dan menggunakan metrics dalam domain
similarity search seperti MAP, Top-K, dan MRR.

Penelitian ini menghadirkan perbedaan penting dengan menempatkan kedua
model tersebut dalam konteks *retrieval* audio ayat Al-Qur'an yang mana
memiliki karakteristik fonetik dan aturan tajwid yang unik. Berbeda
dengan penelitian pada umumnya yang menitik beratkan pada pengukuran
performa transkripsi ASR, penelitian ini mengevaluasi kemampuan Wav2vec2
dan Data2vec dalam mencocokkan cuplikan bacaan dengan ayat terkait
melalui pendekatan komparatif berbasis kemiripan *embedding*. Dengan
demikian, penelitian ini menawarkan kontribusi yang masih minim
dieksplorasi, sekaligus menyediakan landasan teknis bagi pengembangan
sistem pendukung hafalan dan pembelajaran Al-Qur'an yang lebih adaptif
dan aplikatif.

##  Landasan Teori

### Kecerdasan buatan (Artificial Intelegence) 

Kecerdasan buatan (*Artificial Intelegence*) merupakan salah satu cabang
ilmu komputer yang mempelajari perancangan sistem yang mampu melakukan
kecerdasan manusia, seperti penalaran, persepsi, pembelajaran, dan
pengambilan keputusan \[25\]. Russel dan Norvig mengklasifikasikan AI ke
dalam empat sudut pandang, yaitu sistem yang berpikir seperti manusia,
bertindak seperti manusia, berpikir secara rasional, dan bertindak
secara rasional. Pendekatan terakhir yakni *rational agent* yang
bertindak untuk mencapai hasil terbaik menurut ukuran kinerja tertentu,
menjadi kerangka dominan dalam AI modern \[25\].

Dalam perkembangannya, *Machine learning* (ML) muncul sebagai sub-bidang
AI yang berfokus pada kemampuan sistem untuk belajar pola dari data
tanpa dirprogram secara eksplisit untuk setiap kasus \[26\]. ML sendiri
terbagi menjadi tiga paradigma utama *supervised learning* (belajar dari
data yang berlabel), *unsupervised* learning (menemukan struktur dari
data tak berlabel), dan *self-supervised learning* yang menjadi fondasi
penelitian ini. Penelitian ini berada dalam ranah AI karena memanfaatkan
model pembelajaraan representasi untuk memahami dan mencocokkan sinyal
ucapan Al-Qur'an secara otomatis.

### Deep Learning 

*Deep learning* adalah cabang dari *machine learning* yang menggunakan
jaringan saraf tiruan berlapis yang jumlahnya sangat banyak (*deep
neural networks*) untuk mempelajari representasi data secara berjenjang
(*hierarchical representation*) \[27\]. Berbeda dari ML tradisional yang
bergantung pada feature engineering manual yang mana, harus membutuhkan
manusia untuk memberikan data yang berlabel, deep learning mempelajari
fitur secara otomatis. Lapisan-lapisan awal menangkap fitur tingkat
rendah (misalnya pola frekuensi pada audio), sedangkan lapisan yang
lebih dalam menyusun fitur tingkat tinggi yang lebih abstrak (misalnya
unit fonetik) \[27\]

Sebuah jaringan saraf tersusun atas neuron-neuron yang
mentransformasikan input melalui bobot this equation **W**, bias **b**.
Dan fungsi aktivasi non-linear *σ,* sehingga keluaran satu lapisan
dinyatakan sebagai berikut

  ---------------------------------------------------------------------------------------
     $$\mathbf{h} = \sigma\left( \mathbf{Wx} + \mathbf{b} \right)$$   []{#_Toc239590104
                                                                      .anchor}(2.1)
  -- ---------------------------------------------------------------- -------------------

  ---------------------------------------------------------------------------------------

Parameter jaringan dioptimasi dengan meminimalkan *loss function*
melalui algoritma *backpropagation* dan *gradient descent* \[27\].
Arsitektur yang relevan bagi penelitian ini adalah jaringan konvolusi
(*Convolutional Neural Network*, CNN) yang efektif mengekstrak fitur
local dari sinyal, dan Transformer \[28\] yang memodelkan ketergantungan
jangka Panjang melalui mekanisme *self-attention*. Kedua arsitektur
inilah yang Menyusun tulang punggung model Wav2vec 2.0 dan varian
*speech* dari Data2Vec.

### Similarity Search 

*Similarity search* adalah proses menemukan objek-objek dalam suatu
database yang paling mirip dengan sebuah objek *query* (data input),
berdasarkan ukuran kemiripan tertentu \[29\]. Dalam paradigma modern,
setiap objek, baik teks, gambar, maupun audio, direpresentasikan sebagai
vector bilangan riil di ruang berdimensi tinggi, sehingga pencarian
kemiripan tereduksi menjadi persoalan *Nearest Neighbor Search* di ruang
vector tersebut \[29\].

Formalnya, diberikan sebuah vector *query* **q** dan himpunan *vector
database* **\**
D={**d**1​,...,**d***N*​} , sistem menghitung skor kemiripan
*s*(**q**,**d***i*​) untuk setiap **d***i* , lalu mengurutkan hasilnya
secara menurun untuk menghasilkan daftar peringkat (*ranked list*).
Pendekatan ini menjadi tulang punggung sistem information *retrieval*
modern \[29\]. Dalam penelitian ini, *similarity search* diterapkan pada
*embedding audio* ayat Al-Qur'an. Data *query* dibandingkan langsung
terhadap *database* embedding ayat tanpa melalui tahap transkripsi,
sehingga menghindari propagasi kesalahan dari tahap ASR (*automatic
speech recognition*).

### Vector Embedding 

*Vector Embedding* adalah representasi suatu objek dalam bentuk vector
yang padat berdimensi tetap $v \in R^{d}\$yang dirancang sedemikian rupa
sehingga kedekatan *geometrics* antar vektor mencerminkan kemiripan
semantik atau perseptual antar objek yang diwakilinya \[27\]. Prinsip
utamanya adalah *distributional hypothesis*, yang mana objek objek yang
serupa akan menempati posisi yang berdekatan di dalam ruang vector
(*embedding space*)

Pada domain audio, *embedding* dihasilkan oleh model *self-supervised
learning* seperti Wav2vec 2.0 dan Data2vec, yang memetakan sinyal audio
mentah menjadi barisan vector kontekstual. Embedding inilah yang menjadi
objek perbandingan dalam penelitian ini. Kualitas sebuah embedding untuk
tugas *retrieval* dapat dilihat dari kemampuannya dalam menempatkan
audio dari ayat yang sama secara berdekatan sekaligus memisahkan audio
dari ayat yang berbeda. Sifat *geometris* tersebut menjadi dasar
pengukuran kinerja dan perbandingan kedua model dalam penelitian ini.

### Self-Supervised Learning (SSL) 

*Self-Supervised Learning (SSL)* adalah paradigma pembelajaran mesin di
mana model dilatih untuk mempelajari representasi yang berguna dari data
yang tidak memiliki *label* dengan menghasilkan labelnya sendiri. SSL
terletak di antara *Supervised Learning* (yang membutuhkan label manual
ekstensif) dan *Unsupervised Learning* (yang mencari pola tanpa label).

Inti dari SSL terletak pada penciptaan \"Tugas Preteks\" (*Pretext
Task*). Tugas ini memaksa model untuk memahami dan memprediksi bagian
tersembunyi (*masked*) atau bagian yang hilang (*corrupted*) dari data
input itu sendiri.

### Wav2vec 2.0 

Wav2vec 2.0 adalah sebuah model *self-supervised learning* (SSL) yang
mempelajari representasi dari ucapan langsung atau audio mentah melalui
pendekatan *constrastive learning*. Arsitekturnya terdiri dari tiga
komponen inti; *feature encoder* konvolusional, jaringan konteks
berbasis *Transformer*, dan modul kuantisasi yang bekerja dalam satu
tugas pretext. Memprediksi representasi diskret yang sesuai dari
representasi konteks pada posisi yang ditutup (masked).

1.  ***Feature encoder***

Sinyal audio mentah $\mathcal{X}$ (16 kHz) diproses oleh tujuh blok
konvolusi temporal, masing-masing berisi 512 kanal dengan *Layer
Normalization* dan aktivasi GELU. Susunan stride $(5,2,2,2,2,2,2)$ dan
lebar kernel $(10,3,3,3,3,2,2)$ menghasilkan barisan representasi laten
$z_{1},\ldots,z_{T}$ pada frekuensi keluaran sekitar 49 Hz (jarak
antar-frame $\approx$`<!-- -->`{=html}20 ms), dengan receptive field
selebar 400 sampel input atau setara 25 ms audio.

  -----------------------------------------------------------------------
         Blok               Stride            Kernel           Kanal
  ------------------- ------------------- --------------- ---------------
           1                   5                10              512

          2-5                  2                 3              512

          6-7                  2                 2              512
  -----------------------------------------------------------------------

  : []{#_Toc236748113 .anchor}Tabel 2.2 Konfigurasi Blok Konvolusi
  Temporal pada *Encoder* *Audio*

2.  **Jaringan konteks**
    $\mathbf{g:}\mathcal{Z}\mathbf{\rightarrow}\mathcal{C}$

Representasi laten $z$ diumpankan ke jaringan *Transformer* N4 yang
memodelkan ketergantungan jangka panjang antar-frame melalui
*self-attention*, menghasilkan representasi kontekstual
$c_{1},\ldots,c_{T}$. Konfigurasi Base memakai 12 blok
($d = 768,\ 8\ Head$, sedangkan Large memakai 24 blok ( $d = 1024$, 16
head). Informasi posisi ditanamkan melalui *relative positional
convolutional embedding* (kernel 128, 16 grup), bukan *positional
encoding* absolut. Representasi laten $z$ diumpankan ke jaringan
*Transformer* N4 yang memodelkan ketergantungan jangka panjang antar
frame melalui *self-attention*, menghasilkan representasi kontekstual
$c_{1},\ldots,c_{T}$ Konfigurasi Base memakai 12 blok ($d = 768$, 8
head), sedangkan *Large* memakai 24 blok ($d = 1028$, 16 head).
Informasi posisi ditanamkan melalui *relative* *positional convolutional
embedding* (kernel 128, 16 grup), bukan *positional encoding* absolut.

3.  **Modul kuantisasi**

Secara paralel, laten $z$ didiskretisasi melalui *product quantization*
N9 dengan $G = 2$ *codebook*, masing-masing $V = 320$ entri. Pemilihan
entri dibuat terdiferensiasi menggunakan Gumbel-Softmax N7, N8

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     $$p_{g,v} = \frac{\exp\left( \left( l_{g,v} + n_{g,v} \right)\text{/}\tau \right)}{\sum_{k = 1}^{V}\exp\left( \left( l_{g,k} + n_{g,k} \right)\text{/}\tau \right)}$$   []{#_Toc239590105
                                                                                                                                                                             .anchor}(2.2)
  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- -------------------

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

dengan $l_{g,v}$ logit entri $v$ pada grup $g$, $\tau$ temperatur
(dijadwalkan turun dari 2 ke 0,5), dan
$n_{v} = - \log\left( - \log\left( u_{v} \right) \right)$,
$u_{v}\mathcal{\sim U}(0,1)\$derau Gumbel. Pada *forward pass* dipilih
entri diskret $i = \arg{\max_{j}p_{g,j}}$; pada *backward pass*
digunakan gradien fungsi Gumbel-Softmax kontinu (*straight-through
estimator* N10) sehingga tetap dapat dilatih.

4.  **Fungsi Objektif**

Pelatihan diarahkan gabungan dua komponen

  ------------------------------------------------------------------------------------------------------------------------------
     $$\mathcal{L =}\mathcal{L}_{\mathcal{m}} + \alpha\,\mathcal{L}_{\mathcal{d}},\quad\quad\alpha = 0,1$$   []{#_Toc239590106
                                                                                                             .anchor}(2.3)
  -- ------------------------------------------------------------------------------------------------------- -------------------

  ------------------------------------------------------------------------------------------------------------------------------

*Constrastive loss* $\mathcal{L}_{\mathcal{m}}$ memaksa model membedakan
target benar $q_{t}$ dari $K = 100$ pengecoh

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     $$\mathcal{L}_{\mathcal{m}} = - \log{}\frac{\exp\left( \frac{\text{sim}\left( c_{t},q_{t} \right)}{\kappa} \right)}{\sum_{\widetilde{q} \sim \mathcal{Q}_{\mathcal{t}}}^{}\exp\left( \frac{\text{sim}\left( c_{t},\widetilde{q} \right)}{\kappa} \right)}$$   []{#_Toc239590107
                                                                                                                                                                                                                                                                   .anchor}(2.4)
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -------------------

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Dengan $\kappa = 0,1$ dan
$sim(a,b) = \frac{a^{T}b}{\text{|}a\text{||}b\text{|}}$ (*cosine
similarity*, berperan sebagai *scoring function*). *Diversity loss*
$\mathcal{L}_{\mathcal{d}}$ mendorong pemakaian merata seluruh entri
*codebook* via maksimalisasi entropi

  ----------------------------------------------------------------------------------------------------------------------------------------------
     $$\mathcal{L}_{\mathcal{d}} = \frac{1}{G}\sum_{g = 1}^{G}{\sum_{v = 1}^{V}\overline{p_{g,v}}\log\overline{p_{g,v}}}$$   []{#_Toc239590108
                                                                                                                             .anchor}(2.5)
  -- ----------------------------------------------------------------------------------------------------------------------- -------------------

  ----------------------------------------------------------------------------------------------------------------------------------------------

Melalui mekanisme kontrastif, Wav2Vec2 membentuk representasi dengan
membedakan unit positif dan unit negatif pada ruang representasi.
Karakteristik tersebut dapat menghasilkan struktur *embedding* yang
lebih terpisah, sehingga dampaknya terhadap kinerja *retrieval* berbasis
*cosine similarity* dapat diamati melalui evaluasi empiris dalam
penelitian ini \[30\].

![[]{#_Toc236767908 .anchor}Gambar 2.1 Arsitektur
Wav2vec2](media/image3.png){width="2.775in"
height="4.464435695538057in"}

### Data2vec 

*Data2vec* merupakan model *Self-Supervised Learning* (SSL) yang
*modality agnostic*, artinya inputan yang dapat diproses dapat berupa
audio, gambar, ataupun text. Perbedaan mendasarnya dengan Wav2vec2 2.0
terletak pada bentuk prediksi, Data2vec tidak mempredisksi unit diskret
terkuantisasi, melainkan representasi laten kontekstual yang bersifat
kontinu. Yang dihasilkan oleh model itu sendiri melalui skema
*self-distillation*.

1.  **Skema *Teacher* dan *Student* (*self-distillation*)**

Satu arsitektur Transformer \[28\] beroperasi dalam dua peran. Jaringan
*student* yang memproses input yang di-*masking* dan menghasilkan
prediksi $f_{t}(x)$ pada tiap posisi ter-*masking*. Jaringan *teacher*
memproses input penuh (tanpa *masking*) dan menyediakan target regresi
$y_{t}$. Student dilatih meregresi target *teacher* hanya pada posisi
ter-*masking*

2.  **Pembaruan *Teacher* via *EMA***

Pembaruan *teacher* tidak dilakukan oleh *gradient*, melainkan
menggunakan *exponential moving average* (EMA) dari parameter *student*
$\Delta \leftarrow \tau\,\Delta + (1 - \tau)\,$dengan $\Delta$ parameter
*teacher*, $\theta$ parameter *student*. Nilai $\tau$ dinaikan linear
dari $\tau_{0}\ ke{\ \tau}_{e}\$langkah awal lalu ditahan konstan, untuk
ucapan $\tau_{0} = 0,,\tau_{e} = 0,,\tau_{n} = 30.0$

3.  **Konstruksi Target Kontekstual**

Target dibentuk dengan merata-ratakan keluaran $K$ blok *Transformer*
teratas *teacher*, setelah masing-masing dinormalisasi

  ----------------------------------------------------------------------------------------------
     $$y_{t} = \frac{1}{K}\sum_{l = L - K + 1}^{L}\widehat{a_{t}^{\, l}}$$   []{#_Toc239590109
                                                                             .anchor}(2.6)
  -- ----------------------------------------------------------------------- -------------------

  ----------------------------------------------------------------------------------------------

Dengan $L$ dengan jumlah total blok (12 pada base), $K = 8$ blok teratas
$\widehat{a_{t}^{\, l}}$ keluaran blok $l$ ternormalisasi. Untuk ucapan
digunakan *instance nomralization* \[31\]. Untuk gambar *layer
normalization* \[32\]. Normalisasi ini mencegah keruntuhan representasi
(*representation collapose*). Karena target menggabungkan berupa lapisan
atas, ia bersifat kontekstual penuh. Target kontekstual penuh tersebut
menjadi salah satu karakteristik Data2Vec yang membedakannya dari
Wav2Vec2. Karakteristik ini telah dikaitkan dengan kemampuan
menghasilkan representasi yang informatif pada sejumlah tugas pemrosesan
suara \[13\], \[19\]. Dalam penelitian ini, karakteristik tersebut
dianalisis melalui penerapan Data2Vec pada tugas *retrieval* audio ayat
Al-Qur\'an

4.  **Fungsi Objektif**

*Student* meregresi target dengan *Smooth L1 loss* (*huber*):

  -------------------------------------------------------------------------------------------------------------------------------------
     $$\mathcal{L}\left( y_{t},f_{t}(x) \right) = \left\{ \begin{matrix}                                            []{#_Toc239590110
     \frac{1}{2}\left( y_{t} - f_{t}(x) \right)^{2}\text{/}\beta, & \left| y_{t} - f_{t}(x) \right| \leq \beta \\   .anchor}(2.7)
     \left| y_{t} - f_{t}(x) \right| - \frac{1}{2}\beta, & \text{selainnya}                                         
     \end{matrix} \right.\ $$                                                                                       
  -- -------------------------------------------------------------------------------------------------------------- -------------------

  -------------------------------------------------------------------------------------------------------------------------------------

Parameter $\beta$ mengatur peralihan wilayah kuadratik
$\left( l_{2} \right)$ dan linear $\left( l_{1} \right)$ untuk
$\beta \rightarrow \infty$ *loss* menjadi MSE murni yang bekerja baik
pada domain ucapan.

![[]{#_Toc236767909 .anchor}Gambar 2.2 Arsitektur
Data2vec](media/image4.png){width="2.4833333333333334in"
height="4.357854330708661in"}

### Cosine Similarity

*Cosine Similarity* mengukur kemiripan antara dua vektor berdasarkan
sudut *cosine* di antara keduanya, tanpa perlu memperhitungkan besar
magnitudonya \[29\]

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     $$\text{sim}\,(q,d) = \cos\theta = \frac{q \cdot d}{\text{|}q\text{|}\,\text{|}d\text{|}} = \frac{\sum_{i = 1}^{d}{q_{i}d_{i}}}{\sqrt{\sum_{i = 1}^{d}q_{i}^{2}}\,\sqrt{\sum_{i = 1}^{d}d_{i}^{2}}}$$   []{#_Toc239590111
                                                                                                                                                                                                             .anchor}(2.8)
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -------------------

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Nilainya berkisar pada \[-1, 1\], dengan nilai mendekati 1 menandakan
arah vektor yang sangat mirip. Perlu ditegakan bahwa dalam bidang
*retrieval*, *cosine similarity* berperan sebagai fungsi penilaian
(*scrolling function*), bukan sebagai metrik evaluasi \[29\]. Manning
dkk. Membedakkan secara tegas antara mekanisme *scoring ranking* dan
metrik evaluasi efektivitas sistem. Selain itu *cosine distance*
$\left( 1 - \cos\theta \right)$ bukan metrik jarak matematis, arena
melanggar pertidaksamaan segitiga \[33\]. oleh karena itu, dalam
penelitian ini *cosine similarity* digunakan sebagai fungsi penilai
kemiripan untuk menghasilkan peringkat, sedangkan efektivitas
*retrieval* diukur menggunakan metrik evaluasi yang dijelaskan pada
sub-bab berikutnya.

### Metriks Evaluasi Retrieval 

Efektivitas sistem *retrieval* dievaluasi menggunakna metrik yang
berbasis pada kualitas peringkat (ranking) hasil pencarian \[29\].
Penelitian ini menggunakan tiga metrik utama berikut.

1.  ***Top K Accuracy***

Metrik ini mengukur proporsi *query* yang dokumen relevannya ditemukan
dalam K peringkat teratas dalam pencarian

  -------------------------------------------------------------------------------------------------------------------------------
     $$Top - K\, Accuracy = \ \ 1\  \div |Q|\sum_{q \in Q}^{}{1\left\lbrack rank(q) \leq K \right\rbrack}$$   []{#_Toc239590112
                                                                                                              .anchor}(2.9)
  -- -------------------------------------------------------------------------------------------------------- -------------------

  -------------------------------------------------------------------------------------------------------------------------------

Dengan ∣Q∣ adalah jumlah total *query*, dan 1\[⋅\] adalah fungsi
indikator yang bernilai 1 apabila dokumen relevan berada pada peringkat
kurang dari atau sama dengan K, dan berniali 0 jika sebaliknya. Metrik
ini mencerminkan kegunaan praktis sistem bagi pengguna yang umumnya
hanya memeriksa hasil teratas.

2.  ***Mean Reciprocal Rank* (MRR)**

MRR mengukur seberapa tinggi peringkat dokumen relevan pertama, dengan
menghitung rata-rata kebalikan peringkatnya \[29\]. MRR memiliki rumus
demikian.

  -------------------------------------------------------------------------------------------
     $$MRR = \frac{1}{|Q|}\sum_{i = 1}^{|Q|}\frac{1}{\text{rank}_{i}}$$   []{#_Toc239590113
                                                                          .anchor}(2.10)
  -- -------------------------------------------------------------------- -------------------

  -------------------------------------------------------------------------------------------

Dengan $\text{rank}_{i}$ posisi peringkat dokumen relevan pertama untuk
*query* ke-$i$. Nilai MRR mendekati 1 menunjukkan dokumen relevan
cenderung berada di peringkat paling atas.

3.  ***Mean Average Precision* (MAP)**

MAP merupakan metrik yang paling komprehensif karena memperhitungkan
presisi pada setiap posisi dokumen relevan sepanjang daftar peringkat
\[29\]. Untuk sebuah *query*, *Average Precision* (AP) dihitung sebagai

  ------------------------------------------------------------------------------------------------
     $$\text{AP} = \frac{1}{|R|}\sum_{k = 1}^{N}{P(k)} \cdot \text{rel}(k)$$   []{#_Toc239590114
                                                                               .anchor}(2.11)
  -- ------------------------------------------------------------------------- -------------------

  ------------------------------------------------------------------------------------------------

Dengan $P(k)$ presisi pada peringkat $k$ $\text{rel}(k)$ fungsi
*indicator* relevansi dokumen di peringkat $k$, $|R|$ jumlah total
dokumen relevan, dan $N$ jumlah dokumen. MAP kemudian adalah rata-rata
AP atas seluruh *query*

  ---------------------------------------------------------------------------------------
     $$\text{MAP} = \frac{1}{|Q|}\sum_{q = 1}^{|Q|}{\text{AP}(q)}$$   []{#_Toc239590115
                                                                      .anchor}(2.12)
  -- ---------------------------------------------------------------- -------------------

  ---------------------------------------------------------------------------------------

Ketiga metrik ini; *Top-K Accuracy*, MRR, dan MAP merupakan metrik
evaluasi yang secara konseptual berbeda dengan *cosine similarity* yang
berperan sebagai *scoring function*

### Dataset Quran-MD 

Penelitian ini menggunakan dataset Quran-MD \[4\], sebuah dataset
multimodal yang tersedia pada platform *HuggingFace*. Dataset ini
mengintegrasikan dimensi teks, *linguistic*, dan audio pada tingkat ayat
maupun kata. Untuk setiap ayat disediakan teks Arab asli serta rekaman
dari 30 pembaca (qari) yang berbeda guna mempresentasikan keragaman gaya
bacaan (qiraat) dan nuansa dialetik.

Secara kuantitatif, koleksi Quran-MD terdiri atas dua sub-dataset
terpisah. sub-dataset ayat (*Buraaq/quran-md-ayahs*) berisi 187.080
sampel rekaman ayat lengkap. Dan sub-dataset kata
(Buraaq/quran-md-words) berisi 77.429 sampel pelafalan kata individual,
sehingga totalnya mencakup sekitar 264.509 audio format berformat mp3
yang meliputi 114 surah dan 6.236 ayat unik \[4\]. Sesuai batasan
masalah penelitian, objek yang digunakan dibatasi pada surah Al-fatihah
dan juz 30 dari sub-dataset tingkat ayat. Pasangan audio teks yang
tersedia pada dataset ini memungkinkan penerapan tugas *similarity
search* berbasis suara tanpa transkripsi.

### Metode Penelitian CRISP-DM 

Penelitian ini mengadopsi kerangka kerja (*framework*) Cross-industry
Standard Process for Data Mining (CRISP-DM) sebagai metodologi (bx8).
CRISP-DM merupakan model process standar yang berisifat iteratif dan
tidak bergantung pada industri maupun teknologi tertentu, sehingga
sesuai untuk penelitian ber*database* dan machine learning. Kerangka ini
terdiri atas enam fase yang saling terikat:

a.  *Business Understanding*. Merumuskan tujuan penelitian, yaitu
    mengimplementasikan Wav2vec2 dan Data2vec sebagai penghasil vector
    embedding untuk retrieval ayat Al-Qur'an serta menetapkan kriteria
    evaluasi dan skema perbandingan kedua model.

b.  *Data understanding*. Mengekplorasi karakteristik dataset Quran-MD,
    mencakup distribusi qari, kualitas audio, dan cakupan surah yang
    digunakan.

c.  *Data preparation*. Menyeleksi Surah Al-Fatihah dan Juz Amma,
    melakukan pre-processing audio dan menyiapkan pasangan *query* dan
    database untuk pengujian

d.  *Modeling*. Mengekstrak embedding dari kedua model SSL tanpa adanya
    *fine-tuning*, lalu menghitung skor kemiripan menggunakan *cosine
    similarity* untuk menghasilkan peringkat *retrieval*.

e.  *Evaluation*. Mengukur kinerja *retrieval* kedua model menggunakan
    metrik MAP, Top-K Accuracy, dan MRR.

f.  *Deployment*. Menyusun kesimpulan dan rekomendasi model serta
    konfigurasi lapisan yang paling sesuai berdasarkan hasil evaluasi
    dan skenario penelitian sebagai acuan bagi pengembangan sistem
    pembelajaran Al-Qur\'an berbasis *audio*

# 

# BAB III METODOLOGI PENELITIAN

Bab ini menjelaskan rancangan penelitian untuk membandingkan
representasi laten Wav2vec2 dan Data2vec pada tugas *retrieval* audio
ayat Al-Qur'an. Penelitian mengikuti enam fase *Cross-Industry Standard
Process for Data Mining* (CRISP-DM). yaitu *business understanding*,
*data understanding*, *data preparation*, *modeling*, *evaluation* dan
*deployment* \[34\]. Dalam penelitian ini modeling tidak membangun
sebuah model baru ataupun *fine-tuning* model yang sudah ada, melainkan
fase modeling berisi ekstraksi representasi dari model *pre-trained*
yang parameternya dalam keadaan frozen. Sedangkan fase *deployment*
membahas rancangan penerapan secara konseptual, bukan pengoperasian
sistem produksi.

Alur penelitian bersifat iteratif. Temuan pada suatu fase dapat
menyebabkan proses kembali ke fase sebelumnya untuk memperbaiki kualitas
data atau menyesuaikan proses pengolahannya. Sebagai contoh,
permasalahan yang ditemukan saat memahami karakteristik data dapat
ditangani kembali pada tahap persiapan data, sedangkan kendala pada
proses pemodelan dapat memerlukan pemeriksaan ulang terhadap data
masukan. Meskipun proses tersebut dapat dilakukan secara berulang,
pemilihan konfigurasi dan pelaporan hasil tetap dipisahkan. Titik
representasi terbaik dipilih hanya berdasarkan hasil pada *development
set*, kemudian dipatenkan sebelum evaluasi akhir dilakukan pada himpunan
*test set*.

##  *Business Understanding* 

Fase *business understanding* menjelaskan mengenai masalah penelitian
yang dihadapi. Masalah utama penelitian adalah belum diketahuinya
representasi bawaan Wav2vec2 dan Data2vec dalam tugas mencocokkan bacaan
ayat Al-Qur\'an secara langsung dari *audio*. Kedua model memiliki
mekanisme *pretraining* yang berbeda. Wav2vec2 menggunakan *constrastive
learning* \[1\], sedangkan Data2vec menggunakan *self-distillation*
dengan target representasi kontekstual \[8\]. Perbedaan tersebut perlu
dinilai pada tugas *retrieval,* sebab keberhasilan pada pengenalan
ucapan tidak dapat langsung dianggap berlaku pada pemeringkatan *audio*.

Tujuan operasional penelitian terdiri atas dua bagian. Pertama,
membangun alur yang dapat mengubah klip *audio* menjadi *frozen
embedding*, menghitung kemiripan, dan menghasilkan list kandidat ayat.
Kedua, membandingkan kualitas daftar peringkat Wav2Vec2 dan Data2Vec
pada empat skenario sumber data, yaitu Skenario A, B, C, dan D yang
secara keseluruhan membentuk 13 evaluasi. Seluruh parameter model
dipertahankan dalam keadaan beku atau *frozen* agar hasil yang
didapatkan mencerminkan kemampuan representasi *pre-trained*, bukan
pengaruh *fine-tuning* pada korpus penelitian.

Wav2vec2 mempelajari representasi ucapan melalui pembelajaran kontrastif
pada ruang *laten* \[1\], sedangkan Data2vec menggunakan pendekatan
*teacher-student self-distillation* dengan target representasi
kontekstual \[8\]. Perbedaan mekanisme tersebut menjadi dasar untuk
membandingkan keduanya secara empiris, tetapi tidak dengan sendirinya
menentukan model yang lebih baik pada korpus bacaan Al-Qur'an.

Kriteria keberhasilan implementasi bukan nilai minimum metrik tertentu.
Implementasi dinyatakan berhasil apabila setiap *embedding* dapat
dilacak ke klip dan label ayatnya, bentuk matriks konsisten, kegagalan
diperlakukan sama pada kedua model, serta peringkat dapat dievaluasi
dengan prosedur yang tetap. Kualitas hasil diukur dengan MAP, MRR,
Top-1, Top-5, dan Top-10 \[29\]. MAP merangkum kualitas urutan terhadap
seluruh dokumen relevan. MRR berfokus pada possisi dokumen relevan
pertama, sedangkan Top-K menunjukkan propersi *query* yang memiliki
sedikitnya satu dokumen relevan dalam *K* hasil pertama \[29\]. *Cosine
similarity* hanya berfungsi sebagai skor untuk menyusun peringkat, bukan
sebagai metrik evaluasi akhir.

Gambar 3.1 memperlihatkan adaptasi CRISP-DM yang digunakan. Terdapat
panah dua arah yang menunjukkan bahwa pemeriksaan kualitas dapat
mengulang tahap sebelumnya (iteratif), tanpa membuka kembali himpunan
pengujian untuk pemilihan konfigurasi.

![[]{#_Toc237464121 .anchor}Gambar 3. Adaptasi
CRISP-DM](media/image5.png){width="2.1666666666666665in"
height="5.00429571303587in"}

Alur tersebut menempatkan validitas data dan prasyarat evaluasi.
Adaptasi ini terdapat penyesuaian, yang terletak pada fase *modeling*.
Yang tidak memuat pembaruan bobot, dan fase *deployment*, yang hanya
menyusun rancangan penggunaan hasil eksperimen. Dengan demikian,
penelitian tetap mengikuti urutan CRISP-DM tanpa menyatakan bahwa
*prototype* dapat dicoba dalam lingkungan produksi.

##  *Data Understanding*

Penelitian menggunakan dua sumber data, yaitu rekaman mahasiswa dan
Quran-MD \[4\]. Rekaman mahasiswa berasal dari pengumpulan tugas
Tahfidz. Satu rekaman dapat berisi bacaan satu surah bahkan satu rekaman
terdapat beberapa surah sehingga perlu diproses untuk menghasilkan
berkas per ayat. Sedangkan untuk sumber data dalam Quran-MD telah
tersedia *audio* pada tingkat ayat beserta identitas qari, surah, dan
ayat. Kedua sumber data hanya diambil untuk surah Al-fatihah dan Surah
dalam juz Amma.

Kedua sumber data tidak selalu ditempatkan pada sisi yang sama dalam
proses *retrieval*. Suatu sumber dapat berperan sebagai data *query*,
*database* referensi, atau keduanya. Peran tersebut diperlukan untuk
mengamati kinerja model ketika sumber *audio*, identitas pembaca, dan
komposisi *database* referensi berubah. Berdasarkan susunan sumber
*query* dan *database,* penelitian membentuk empat skenario utama yang
diberi nama Skenario A, B, C, dan D. penamaan tersebut digunakan sebagai
identitas konfigurasi eksperimen, Gambaran setiap skenario sebagai
berikut.

a.  Skenario A menggunakan rekaman mahasiswa sebagai *query* dan
    Quran-MD sebagai *database* referensi. Skenario ini
    merepresentasikan lintas sumber karena audio *query* dan audio
    referensi berasal dari proses pengambilan yang berbeda.

b.  Skenario B menggunakan Quran-MD sebagai sumber *query* sekaligus
    *database* referensi. Identitas qari pada kedua sisi dipisahkan agar
    model tidak mencocokkan bacaan dari qari yang sama.

c.  Skenario C menggunakan rekaman mahasiswa sebagai sumber *query*
    sekaligus *database* referensi. Identitas mahasiswa pada sisi
    *query* dan *database* referensi juga dipisahkan.

d.  Skenario D menggunakan rekaman mahasiswa sebagai *query* dan
    *database.* Yang mana *database*-nya gabungan antara rekaman
    mahasiswa selain yang digunakan dalam *query* serta seluruh
    referensi Quran-MD.

Skenario A hanya menghasilkan satu sel karena seluruh rekaman mahasiswa
ditempatkan sebagai *query* dan seluruh Quran-MD sebagai *database*
referensi. Skenario B, C, dan D masing-masing dikembangkan menjadi empat
sel berdasarkan rasio 60:40, 70:30, 80:20, dan 90:10. Oleh karena itu,
keseluruhan rancangan menghasilkan 13 sel evaluasi. Pada tahap ini,
skenario hanya diperkenalkan untuk menjelaskan peran sumber data.
Prosedur pembagian pemilik, pembentukan *development set* dan *test
set.* Untuk tiap skenario juga dapatkan digambarkan dalam tabel berikut.

  ----------------------------------------------------------------------------
   **Skenario**  **Sumber       **Sumber        **Pemisahan      **Jumlah sel
                 *query***      *database*      identitas**        (rasio)**
                                referensi**                      
  -------------- -------------- --------------- ---------------- -------------
        A        Mahasiswa      Quran-MD        Terpisah secara        1
                                                alami karena     
                                                berbeda sumber   

        B        Quran-MD       Quran-MD        Qari *query* dan       4
                                                qari *database*  
                                                referensi tidak  
                                                terikat          

        C        Mahasiswa      Mahasiswa       Mahasiswa query        4
                                                dan Mahasiswa    
                                                database         
                                                referensi tidak  
                                                terikat          

        D        Mahasiswa      Mahasiswa       Mahasiswa query        4
                                lain + seluruh  dan mahasiswa    
                                Quran-MD        database refensi 
                                                tidak terikatt   
  ----------------------------------------------------------------------------

  : []{#_Toc239590163 .anchor}Tabel 3. Gambaran peran sumber data pada
  Skenario A, B, C, dan D

Tabel 3.1 memberikan gambaran *high-level* mengenai pembentukan Skenario
A, B, C, dan D berdasarkan hubungan antara sumber *query* dan sumber
*database* referensi. Tabel tersebut belum menjelaskan karakteristik
setiap sumber secara rinci. Oleh karena itu, pembahasan berikutnya
memerinci unit awal, *metadata*, peran, serta perlakuan khusus yang
diterapkan pada rekaman mahasiswa dan Quran-MD.

Penjelasan mengenai sumber data perlu dibedakan dari penjelasan mengenai
skenario. Skenario menunjukkan bagaimana sumber data ditempatkan dalam
suatu eksperimen. Sedangkan sumber data menjelaskan asal, bentuk awal,
*metadata*, dan proses persiapannya. Rincian tersebut disajikan pada
Tabel 3.2

  -------------------------------------------------------------------------
  **Sumber     **Unit awal** ***Metadata*   **Peran dalam  **Perlakuan
  data**                     utama**        penelitian**   khusus**
  ------------ ------------- -------------- -------------- ----------------
  Rekaman      Rekaman surah Identitas      *query* A,     Normalisasi
  mahasiswa    dari folder   mahasiswa,     serta dan      identitas,
               mahasiswa     surah, ayat,   database       ekstraksi audio
                             berkas sumber  referensi C    secara netral
                                            dan bagian     terhadap format,
                                            *database*     segmentasi per
                                            referensi pada ayat, audit asal
                                            D              klip

  Quran-MD     Audio yang    Identitas      Database       Seleksi
  \[28\]       sudah dipecah qari, surah,   referensi A,   Al-Fatihah dan
               per ayat      ayat, audio    serta *query*  Juz Amma,
                                            , database     normalisasi
                                            referensi B,   audio, validasi
                                            dan bagian     baris referensi
                                            dari database  
                                            referensi pada 
                                            D              
  -------------------------------------------------------------------------

  : []{#_Toc236984596 .anchor}Tabel 3. Peran sumber data dalam rancangan
  eksperimen

Berdasarkan Tabel 3.2, rekaman mahasiswa dan Quran-MD memiliki unit awal
serta kebutuhan persiapan yang berbeda. Rekaman mahasiswa pada awalnya
berupa rekaman tingkat surah sehingga perlu melalui proses segmentasi
menjadi klip tingkat ayat. Sebaliknya, Quran-MD telah menyediakan audio
pada tingkat ayat sehingga tidak memerlukan segmentasi dengan prosedur
yang sama. Meskipun demikian, kedua sumber tetap dinormalisasi ke dalam
format yang dapat diterima kedua model, skema *metadata*, dan definisi
relevansi yang konsisten sebelum digunakan dalam pembentukan skenario.

Dengan dasar tersebut, pemahaman data dilanjutkan melalui empat
pemeriksaan. Pertama, struktur direktori dan variasi nama dinormalisasi
menjadi identitas baku. Kedua, berkas diuji keterbacaannya serta
diperiksa ukuran, jumlah kanal, *sample rate*, dan durasinya. Ketiga,
setiap klip dikaitkan dengan identitas pembaca serta pasangan nomor
surah dan ayat. Keempat, ketersediaan sedikitnya satu dokumen relevan
bagi setiap *query* diperiksa setelah setiap skenario dibentuk.

Pemahaman data mahasiswa dilakukan melalui empat pemeriksaan. Pertama,
struktur direktori dan variasi nama direktori dinormalisasi menjadi
identitas baku. Kedua, file audio diuji keterbacaannya serta diperiksa
ukuran, jumlah kanal, *sample rate*, dan durasinya. Ketiga, setiap klip
dikaitkan dengan identitas pembaca serta pasangan nomor surah dan ayat.
Keempat, ketersediaan sedikitnya satu dokumen relevan bagi setiap
*query* diperiksa setelah skenario dibentuk. Pasangan (surah, ayat)
menjadi definisi relevansi. Dokumen dinyatakan relevan apabila pasangan
tersebut sama dengan pasangan pada *query*, tanpa mensyaratkan pembaca
yang sama.

Pemeriksaan kelengkapan awal dilakukan dengan memindai seluruh folder
mahasiswa, menghitung jumlah berkas rekaman per mahasiswa, memetakan
setiap nama berkas ke surah sasaran, dan mencatat surah yang belum
tersedia; hasilnya berupa metadata kelengkapan per mahasiswa yang
menjadi dasar pemantauan proses pengumpulan data. Inventaris ini
bersifat administratif dan tidak menggantikan validasi teknis pada tahap
persiapan data, sebab keberadaan berkas belum menjamin keterbacaan
maupun kesahihan identitasnya.

Validasi yang dilakukan harus dibatasi maknanya secara tepat.
Pemeriksaan folder, ukuran berkas, keterbacaan audio, konsistensi
*metadata*, keberadaan catatan audit, dan hubungan klip dengan rekaman
sumber merupakan validasi teknis dan validasi provenance. Proses
tersebut bukan pembuktian manual bahwa seluruh batas ayat akurat secara
fonetik. Karena tidak ada anotasi batas waktu manual untuk semua klip,
ketepatan batas hasil segmentasi tetap menjadi sumber ketidakpastian
data.

##  *Data Preparation*

Fase persiapan data mengubah dua sumber yang berbeda menjadi audio
tingkat ayat dengan label dan format masukan yang konsisten. Proses
dilaksanakan melalui empat tahap, yaitu seleksi cakupan dan normalisasi
identitas, segmentasi rekaman mahasiswa, validasi teknis beserta
pembentukan manifest, dan pemisahan data menurut peran sumber.

Gambar 3.2 merinci dua cabang persiapan. Cabang Quran-MD tidak melalui
segmentasi karena unit datanya sudah berupa ayat. Cabang mahasiswa
memerlukan stempel waktu kata dan penentuan batas ayat sebelum kedua
cabang dipertemukan pada format audio dan skema *metadata* yang sama.

![[]{#_Toc237464122 .anchor}Gambar 3. Persiapan data Quran-MD dan
mahasiswa](media/image6.png){width="3.2916666666666665in"
height="5.558318022747157in"}

Diagram tersebut memperjelas bahwa WhisperX menghasilkan stempel waktu
pada tingkat kata, bukan batas ayat final. Batas ayat dibentuk
setelahnya dengan mengalokasikan urutan kata menurut jumlah kata setiap
ayat. Jika penyelarasan kata tidak dapat mendukung pembagian tersebut,
waktu audio dialokasikan secara proporsional sebagai fallback. Seluruh
hasil selanjutnya masuk ke pemeriksaan teknis dan keterlacakan yang
sama.

### Seleksi dan normalisasi identitas 

Tahap pertama menetapkan cakupan korpus secara terprogram pada 38 surah,
yang terdiri atas surah Al-Fatihah dan seluruh surah dalam Juz Amma
dengan nomor surah dari 78 sampai 114. Tiap surah yang dipilih akan
ditetapkan sebagai konstanta atau master data yang akan digunakan dalam
proses berikutnya, seperti pemetaan nama berkas, pengecekan kelengkapan,
penentuan jumlah ayat yang sedang diproses, hingga validasi akhir. Yang
akan mengacu pada definisi cakupan tersebut, jumlah ayat yang dipakai
adalah 571 ayat, yaitu 7 surah ayat Al-Fatihah ditambah 564 ayat Juz
Amma.

Variasi penamaan *folder* serta berkas mahasiswa diselesaikan melalui
prosedur dua tahap. Pada tahap pertama, nama berkas dinormalisasi.
Seluruh huruf diubah menjadi huruf kecil, spasi atau simbol yang
memisahkan kata diubah menjadi garis bawah, dan juga *prefix* seperti
QS, Q.S, dan Surah dihilangkan. Deretan digit panjang pada akhir nama
yang menyerupai nomor induk mahasiswa juga diabaikan agar identitas pada
nama berkas tidak mengganggu pencocokan nama surah. Hasil normalisasi
kemudian dicocokan secara tepat terhadap *master data* alias yang memuat
variasi ejaan seperti contohnya sarah satu surah *mutaffifin,
muthaffifin, muthoffifin,* dan *muthaffifiin.* Pencocokan ini dilakukan
agar audio yang diproses yang memiliki variasi penamaan menjadi
terstandarisasi. Jika surah tidak terbaca pada salah satu *master data*
maka akan dilanjutkan pada proses kedua. Pada tahap kedua, nomor surah
diekstrak dari awal nama berkas dan diterima hanya apabila termasuk
dalam cakupan penelitian. Dengan prosedur ini, dua ejaan berbeda dari
surah yang samat tidak diperlakukan sebagai dua kelas yang berbeda.
Tabel 3.3 memperlihatkan contoh pemetaan nama berkas mahasiswa menuju
identitas yang terstandarisasi, sedangkan Tabel 3.4 merangkum variasi
alias terdaftar pada beberapa surah terpilih.

  -------------------------------------------------------------------------------------
  **Nama berkas asli**  **Hasil            **Cara              **Nomor  **Slug
                        normalisasi**      identifikasi**      surah**  kanonis**
  --------------------- ------------------ ------------------ --------- ---------------
  Al-Fatihah.mp4        alfatihah          Alias eksak            1     al-faatiha

  Q.S An-Naba.mp4       annaba             Alias eksak           78     an-naba
                                           (prefiks QS                  
                                           dibuang)                     

  114 - An              114annaas\<NIM\>   Alias eksak (digit    114    an-naas
  Naas\_\<NIM\>.mp4                        awal dan NIM pada            
                                           akhir dibuang)               

  surah al              almutaffifin       Alias eksak           83     al-mutaffifin
  mutaffifin.mp4                           (prefiks Surah               
                                           dibuang)                     

  ALKAUTSAR.mp4         alkautsar          Alias eksak           108    al-kawthar

  102-114.              (tidak dipetakan)  Deteksi rentang       ---    ---
  At-Takatsur -                            multi-surah,                 
  An-Nas.mp4                               dikeluarkan                  
  -------------------------------------------------------------------------------------

  : []{#_Toc239590165 .anchor}Tabel 3. Contoh pemetaan nama berkas
  mahasiswa

  -----------------------------------------------------------------------
  **Nomor surah**   **Nama baku**     **Contoh variasi  **Jumlah alias**
                                      alias terdaftar** 
  ----------------- ----------------- ----------------- -----------------
  83                Al-Mutaffifin     mutaffifin,       14
                                      muthaffifin,      
                                      muthoffifin,      
                                      muthafifin,       
                                      muthaffifiin, ... 

  102               At-Takasur        takasur,          11
                                      takatsur,         
                                      taqasur,          
                                      takastur,         
                                      takaatsur, ...    

  86                At-Tariq          tariq, thoriq,    10
                                      thariq, thaariq,  
                                      atariq, ...       

  108               Al-Kausar         kausar, kautsar,  5
                                      atkautsar         
  -----------------------------------------------------------------------

  : []{#_Toc239590166 .anchor}Tabel 3. Variasi alias terdaftar untuk
  surah terpilih

Identitas pembaca diekstrak dari nama folder yang mengikuti pola NIM dan
diikuti oleh nama mahasiswa, dengan pemisahan hanya pada garis bawah
pertama sehingga garis bawah pada bagian nama tetap dipertahankan.
Bentuk baku kemudian diterapkan pada keluaran proses, setiap surah yang
berhasil diidentifikasi dipetakan ke nomor surah dan penanda *slug* yang
menjadi nama direktori hasil, misalnya surah 1 menjadi *al-faatiha* dan
surah 114 menjadi *an-naas.* Berkas sumber tidak diubah namanya.
Standarisasi terjadi pada identitas numerik surah dan struktur keluaran.
Lebih jelasnya dijelaskan dalam Gambar 3.4

![[]{#_Toc239590310 .anchor}Gambar 3. Ekstraksi identitas pembaca dan
*slug*](media/image7.png){width="2.977002405949256in"
height="2.9938921697287837in"}

Prosedur ini juga menganani tiga anomali penamaan yang ditemukan pada
data. pertama, rekamanan multi surah yang memuat rentang lebih dari dua
surah dideteksi dan dikeluarkan dari segmentasi otomatis agar tidak
dianggap sebagai surah tunggal. Kedua, duplikasi berkas untuk surah yang
sama pada satu mahasiswa diselesaikan dengan mempertahankan satu berkas
menurut urutan nama. Ketiga, berkas yang tetap tidak teridentifikasi
dicatat sebagai data yang tidak terpetakan. Yang dijelaskan dalam Gambar
3.5 berikut

![[]{#_Toc239590311 .anchor}Gambar 3. Penanganan
anomali](media/image8.png){width="5.160667104111986in"
height="3.298611111111111in"}

Normalisasi audio diterapkan pada kedua cabang sumber dengan spesifikasi
keluaran yang sama, yaitu gelombang satu kanal pada laju sampel 16 kHz.
Pada cabang mahasiswa, komponen audio diekstrak dari berkas video tanpa
mengubah isi gelombang, kemudian disimpan sebagai audio antara mono 16
kHz; variasi ekstensi video diterima sepanjang komponen audionya dapat
dibaca. Pada cabang Quran-MD, data sumber berupa dataset bertingkat ayat
dalam format Parquet yang menyertai identitas qari, surah, dan ayat.
Baris di luar cakupan 38 surah disaring terlebih dahulu, kemudian setiap
audio dikonversi menjadi WAV PCM 16-bit mono 16 kHz dan divalidasi ulang
pada kedua properti tersebut. Cabang Quran-MD tidak melalui segmentasi
karena unit datanya telah berupa ayat. Format penyimpanan antara pada
kedua cabang tidak mengubah spesifikasi gelombang yang diberikan kepada
model; hasil segmentasi akhir pada cabang mahasiswa dapat tersimpan
sebagai MP3, tetapi yang dibaca model tetap gelombang mono 16 kHz hasil
pembacaan ulang berkas tersebut. Proses berikut dapat digambarkan dalam
Gambar 3.6 sebagai berikut.

![[]{#_Toc239590312 .anchor}Gambar 3. Normalisasi audio di kedua sumber
data](media/image9.png){width="2.712962598425197in"
height="6.180036089238845in"}

### Segmentasi rekaman mahasiswa 

Sebelum segmentasi dilakukan, teks referensi Al-Qur\'an edisi Utsmani
untuk 38 surah disiapkan sebagai rujukan. Dari teks ini diambil dua
informasi yaitu jumlah kata per ayat (untuk menentukan batas segmentasi)
dan kata pertama isi bacaan (sebagai penanda awal). Pada Surah
Al-Fatihah, basmalah dihitung sebagai ayat pertama sehingga tetap
disertakan. Pada surah lainnya, kata pertama diambil setelah basmalah,
agar pencarian awal bacaan tidak salah berhenti di bagian pembuka.
Segmentasi dijalankan secara bertahap sebagai berikut.

a.  Rekaman tingkat surah dipetakan ke nomor surah baku dan komponen
    audionya dibaca.

b.  WhisperX melakukan transkripsi dan penyelarasan untuk memperoleh
    urutan kata beserta waktu mulai dan selesai pada tingkat kata.

c.  Jumlah kata setiap ayat digunakan untuk membagi urutan stempel
    waktu. Jika ayat pertama berisi sejumlah kata tertentu, sebanyak
    itulah stempel waktu awal dialokasikan kepada ayat pertama, lalu
    proses diteruskan ke ayat berikutnya.

d.  Batas mulai klip diambil dari awal kata pertama yang dialokasikan,
    sedangkan batas akhir diambil dari akhir kata terakhir dalam alokasi
    ayat tersebut.

e.  Apabila hasil audit tidak menyediakan penyelarasan kata yang dapat
    dipakai, durasi rekaman dibagi secara proporsional menurut jumlah
    kata sebagai mekanisme cadangan.

f.  Setiap potongan disimpan bersama identitas rekaman induk, metode
    segmentasi, surah, dan ayat agar asalnya dapat diaudit.

Tabel 3.5 memperlihatkan ilustrasi alokasi kata terdeteksi menjadi batas
ayat pada sebuah rekaman. Kata-kata pada bagian pembuka tidak dipotong
karena bukan bagian ayat. Alokasi dimulai dari kata pertama bacaan isi
surah dan mengikut jumlah kata referensi setiap ayat secara kumulatif.

  -----------------------------------------------------------------------
  **Bagian**        **Jumlah kata     **Indeks kata     **Batas klip**
                    referensi**       terdeteksi**      
  ----------------- ----------------- ----------------- -----------------
  Pembuka (bukan    ---               kata ke-1 s.d.    tidak dipotong
  ayat)                               ke-3              

  Ayat 1            4                 kata ke-4 s.d.    mulai kata ke-4,
                                      ke-7              akhir kata ke-7

  Ayat 2            6                 kata ke-8 s.d.    mulai kata ke-8,
                                      ke-13             akhir kata ke-13

  Ayat 3            5                 kata ke-14 s.d.   mulai kata ke-14,
                                      ke-18             akhir kata ke-18
  -----------------------------------------------------------------------

  : []{#_Toc239590167 .anchor}Tabel 3.5 Ilustrasi alokasi kata
  terdeteksi menjadi batas ayat

Angka pada Tabel 3.5 bersifat ilustratif. Alokasi sebenarnya mengikuti
jumlah kata referensi setiap ayat, dan durasi setiap potongan dipaksa
minimal 200 ms. Pembagian berdasarkan jumlah kata mengasumsikan urutan
kata terdeteksi secara memadai. Kesalahan transkripsi, bagian pembuka,
jeda panjang, pengulangan, atau kata yang terlewat dapat menggeser
batas. Oleh sebab itu, status audit berhasil menunjukkan keberhasilan
prosedur teknis bukan jaminan manual bahwa batas klip tepat pada setiap
ayat.

### Validasi dan pembentukan *manifest* 

Keluaran segmentasi diperiksa ulang oleh validator akhir sebelum masuk
ke tahap ekstraksi representasi. Validator mewajibkan struktur direktori
baku berupa folder mahasiswa, diikuti penanda kanonis surah, diikuti
berkas ayat bernomor dua digit. Identitas folder diurai Kembali dengan
syarat lebih ketat, yaitu NIM harus berupa digit dan nama tidak boleh
kosong. Setiap berkas diuji keterbacaannya melalui pemeriksaan durasi
dan keterbacaan audio, berkas berukuran nol ditolak, dan nomor ayat
harus berada dalam rentang jumlah ayat surah yang sedang diproses.
Rincian prosesnya dirangkum dalam Tabel 3.6

  -----------------------------------------------------------------------
  **Pemeriksaan**         **Kriteria lolos**      **Perlakuan bila
                                                  gagal**
  ----------------------- ----------------------- -----------------------
  Struktur folder         Pola NIM_nama; NIM      Berkas di bawah folder
  mahasiswa               berupa digit; nama      ditolak
                          tidak kosong            

  Penanda kanonis surah   Salah satu dari 38 slug Berkas ditolak
                          terdaftar               

  Nama berkas ayat        Pola ayah_NN.mp3 dengan Berkas ditolak
                          dua digit               

  Rentang nomor ayat      1 sampai jumlah ayat    Berkas ditolak
                          surah                   

  Ukuran berkas           Lebih besar dari nol    Berkas ditolak
                          bita                    

  Keterbacaan audio       Terbaca oleh pemeriksa  Berkas ditolak
                          durasi; durasi positif  

  Keunikan identitas      Kunci NIM-surah-ayat    Seluruh kelompok
                          tidak berulang          tabrakan ditolak

  Rekonsiliasi audit      Jumlah berkas cocok     Dilaporkan sebagai
                          dengan catatan audit    temuan

  Status audit            Hanya informasi         Klip valid tetap
                          provenance, bukan       dimasukkan
                          syarat inklusi          
  -----------------------------------------------------------------------

  : Tabel 3.6 Validasi akhir beserta kriteria dan perlakuan kegagalan

Validasi membedakan dua hal yang berbeda validitas berkas dan
ketertelusuran proses. Kegagalan atau ketiadaan catatan audit segmentasi
tidak serta-merta membuang klip yang terbukti valid secara struktural
dan audio sebaliknya, klip yang tidak valid tetap ditolak meskipun
auditnya menyatakan berhasil. Kebijakan ini menjaga agar keputusan
inklusi didasarkan pada bukti berkas, sementara status audit
dipertahankan sebagai informasi *provenance* pada manifest. Identitas
ganda dideteksi melalui kunci identitas yang dibentuk dari NIM, nomor
surah, dan nomor ayat apabila terjadi tabrakan, seluruh anggota kelompok
tabrakan ditolak tanpa pemilihan arbitrer. Berkas berukuran nol
dikeluarkan sebelum ekstraksi model. Berkas yang lolos dicatat dalam
manifes dengan urutan tetap. Manifes menyimpan lokasi audio, identitas
pembaca, nomor surah, nomor ayat, sumber data, dan informasi
*provenance* segmentasi. Hubungan satu banding satu antara baris manifes
dan baris embedding menjadi dasar keterlacakan sepanjang eksperimen.

Keluaran validasi berupa manifest tervalidasi yang memuat lokasi audio,
identitas pembaca, nomor surah, nomor ayat, sumber data, durasi, dan
status audit setiap klip, disertai artefak rekonsiliasi yang mencocokkan
jumlah berkas pada penyimpanan dengan catatan audit. Skenario validasi
dinyatakan lulus hanya apabila seluruh gerbang terpenuhi: tidak ada
identitas ganda yang lolos, tidak ada berkas kosong atau tak terbaca
yang lolos, dan setiap berkas yang dikeluarkan memiliki alasan yang
tercatat. Tabel 3.7 memperlihatkan skema baris manifest tervalidasi
beserta tiga kelas provenance yang mungkin muncul pada kolom status
audit.

  ----------------------------------------------------------------------------------------------------------------------------
  **query_id**       **NIM**   **Surah**   **Ayat**   **Lokasi berkas**                          **Durasi     ***Provenance*
                                                                                                 (dtk)**      audit**
  ------------------ --------- ----------- ---------- ------------------------------------------ ------------ ----------------
  \<NIM\>\_093_001   \<NIM\>   93          1          \<NIM\>\_\<nama\>/ad-dhuhaa/ayah_01.mp3    \<durasi\>   sukses teraudit

  \<NIM\>\_112_001   \<NIM\>   112         1          \<NIM\>\_\<nama\>/al-ikhlaas/ayah_01.mp3   \<durasi\>   sukses teraudit
                                                                                                              (fallback waktu
                                                                                                              proporsional)

  \<NIM\>\_080_001   \<NIM\>   80          1          \<NIM\>\_\<nama\>/abasa/ayah_01.mp3        \<durasi\>   tanpa baris
                                                                                                              audit
  ----------------------------------------------------------------------------------------------------------------------------

  : Tabel 3. Skema baris *manifest* tervalidasi beserta kelas
  *prevenance*

Tabel 3.7 bersifat skematis NIM, nama, dan durasi ditampilkan sebagai
tempat penanda karena nilainya baru terbentuk setelah validasi
dijalankan. Kolom durasi diisi dari hasil pemeriksaan keterbacaan audio,
sedangkan kolom provenance audio hanya dapat memuat salah satu dari tig
akelas pada table tersebut.

*Manifest* lengkap memuat lima belas kolom, yaitu query_id, *file path*,
*relative path*, NIM, nama, *slug* surah, nomor surah, nama surah, nomor
ayat, durasi, ukuran berkas, status audit, kelas *provenance*, penanda
*fallback*, dan rincian audit. Tabel hanya menampilkan kolom yang
menjelaskan struktur. *query_id* dibentuk dari NIM, nomor surah tiga
digit, dan nomor ayat tiga digit, serta menjadi kunci identitas unik
yang dipakai untuk mendeteksi identitas ganda. Hubungan satu banding
satu antara baris *manifest* dan baris *embedding* selanjutnya menjadi
dasar keterlacakan sepanjang eksperimen.

### Pemisahan data dan peran sumber

Setelah *manifest* terbentuk, data ditempatkan sebagai *query* dan
*database* referensi berdasarkan scenario dalam tiga belas sel
eksperimen. Skenairo A menggunakan seluruh klip mahasiswa sebagai
*query* dan seluruh Quran-MD sebagai *database* referensi, sehingga
pemisahan terjadi karena sumber data yang berbeda. Skenario B, C, dan D
membagi data pada domain yang sama menggunakan rasio pemilik 60:40,
70:30, 80:20, 90:10, dengan rasio didefinisikan sebagai proporsi pemilik
pada sisi *database* referensi. Pemilik didefinisikan sebagai qari pada
skenario B dan mahasiswa pada skenario C, sedangkan skenario D
menggunakan mahasiswa untuk *query* dan gabungan mahasiswa lain dengan
seluruh Quran-MD untuk *database* referensi. Penugasan pemilik dilkaukan
melalui permutasi acak deterministic dengan seed 42, bukan melalui
pengurutan identitas, sehingga pembagian bersifat dapat direproduksi
tanpa mengikuti urutan NIM maupun nama qari. Jumlah pemilik pada setiap
sisi tidak ditetapkan sebagai angka tetap, melainkan dihitung pada saat
pembentukan seld ari jumlah pemilik tervalidasi *N* yang tersedia pada
*manifest*. Aturan perhitungannya dirangkum pada table 3.8.

  -----------------------------------------------------------------------
          **Rasio          **Pemilik database**      **Pemilik query**
    (database:query)**                            
  ----------------------- ----------------------- -----------------------
           60:40               \[0,60 × N\]          N − \[0,60 × N\]

           70:30               \[0,70 × N\]          N − \[0,70 × N\]

           80:20               \[0,80 × N\]          N − \[0,80 × N\]

           90:10               \[0,90 × N\]          N − \[0,90 × N\]
  -----------------------------------------------------------------------

  : Tabel 3.8 Aturan perhitungan jumlah pemilik pada tiap rasio

Notasi \[ \] menyatakan pembulatan ke bilangan bulat terdekat, dan sisa
pemilik seluruhnya dialokasikan ke sisi *query* sehingga kedua sisi
selalu berjumlah *N*. Nilai *N* berbeda menurut definisi pemilik pada
masing-masing skenario, yaitu jumlah mahasiswa tervalidasi pada skenario
C dan D serta jumlah qari pada skenario B. Karena *N* merupakan keluaran
tahap validasi,

Pencegahan kebocoran data diterapkan dan diaudit secara konstruktif pada
skenario B, C, dan D. Pertama, tidak ada pemilik yang muncul pada kedua
sisi *query* dan *database* referensi. Kedua, tidak ada jalur berkas
yang sama antara kedua sisi. Ketiga, duplikasi identitas pada sisi mana
pun menghentikan proses konstruksi sel. Setelah setiap sel terbentuk,
*query* hanya dipertahankan apabila *database* referensi memuat
sedikitnya satu dokumen dengan pasangan surah dan ayat yang sama,
sehingga metrik peringkat tidak menurun semata-mata karena kelas relevan
tidak tersedia.

Setiap sel kemudian membagi *query* menjadi dua himpunan. Pembagian
dilakukan secara terstratifikasi berdasarkan pasangan (surah, ayat)
dengan *seed* 42: pada setiap kelompok pasangan ayat, sekitar tujuh
puluh persen klip, dibulatkan ke bawah dengan minimal satu klip,
dialokasikan ke *development set* dan sisanya ke *test set*.
*Development set* digunakan untuk menyapu seluruh titik representasi dan
memilih konfigurasi terbaik, sedangkan *test set* dikunci dan hanya
digunakan satu kali untuk evaluasi akhir setelah konfigurasi ditetapkan.

## *Modeling* 

Fase *modeling* dirancang melalui empat tahap, yaitu ekstraksi
representasi *layerwise,* pembersihan data hasil ekstraksi, pemilihan
titik representasi per sel, dan penskoran beserta pemeringkatan. Tahap
pemilihan titik representasi diuraikan lebih lanjut menjadi protocol
pemilihan, tata cara penyajian MAP development per scenario, dan
ringkasan titik terpilih. Rancangan setiap tahap diuraikan sebagai
berikut.

### Ekstraksi representasi *layerwise*

Fase *modeling* menggunakan Wav2Vec2 dan Data2Vec dalam keadaan beku.
Tidak ada *fine-tuning*, pembaruan gradien, atau kepala prediksi yang
dilatih dengan label ayat. Kedua model menerima gelombang *mono* 16 kHz
dan menghasilkan urutan representasi kontekstual berdimensi 768.
Perbedaan mekanisme pralatih tetap dipertahankan, tetapi prosedur
masukan, agregasi, penyimpanan, dan pembersihan dibuat setara.

Representasi diambil pada 13 titik. Titik 0 adalah representasi sebelum
keluaran blok Transformer pertama, kemudian titik 1 sampai 12 adalah
keluaran blok Transformer 1 sampai 12. Dengan kata lain, 13 titik
tersebut terdiri atas titik 0 ditambah keluaran dua belas blok
Transformer. Untuk setiap titik $l$, keluaran temporal
$H_{l} \in R^{T_{l} \times \mathbb{768}}$ diringkas dengan *mean
pooling*:

  -----------------------------------------------------------------------------------------------------------------
       $$e_{l} = \frac{1}{T_{l}}\sum_{t = 1}^{T_{l}}{bH_{l,t}},\quad\quad e_{l} \in R^{\mathbb{768}}.$$   (3. 1)
  ---- -------------------------------------------------------------------------------------------------- ---------

  -----------------------------------------------------------------------------------------------------------------

Satu klip menghasilkan 13 vektor berukuran 768. Agregasi rata-rata
dipilih agar audio dengan durasi berbeda dapat dibandingkan dalam ruang
berdimensi tetap. Pendekatan ini juga memastikan bahwa perbedaan hasil
antartitik berasal dari representasi model, bukan dari perbedaan dimensi
embedding.

Gambar 3.3 menunjukkan alur ekstraksi dan penyimpanan. Checkpoint per
baris diperlukan agar proses berskala besar dapat dilanjutkan tanpa
mengulang seluruh korpus serta agar kegagalan tetap tercatat pada posisi
asalnya.

![[]{#_Toc237464123 .anchor}Gambar 3. Ekstraksi *layerwise*,
*checkpoint*, *manifest* dan matriks
akhir](media/image10.png){width="4.417890419947507in"
height="4.811111111111111in"}

Setiap checkpoint sementara .npz menyimpan hasil satu baris dengan
bentuk (13, 768) beserta *metadata* kegagalan. progress.json mencatat
kemajuan penyelesaian sehingga proses dapat dilanjutkan secara
deterministik. Setelah ekstraksi selesai, checkpoint dirakit menjadi
layer_00.npy sampai layer_12.npy; setiap berkas mempunyai bentuk (N,
768). Baris ke-*i* pada seluruh matriks selalu merujuk ke baris ke-i
pada *manifest*.

### Pembersihan data hasil ekstraksi 

Kegagalan ekstraksi dan nilai bukan bilangan diperiksa sebelum evaluasi.
Ekstraktor yang gagal dalam mengekstraksi dapat menghasilkan
representasi yang bentuknya tidak sesuai atau mengandung nilai
*non-finite*, lalu menandai baris tersebut sebagai gagal pada manifes.
Indeks gagal dari Wav2Vec2 dan Data2Vec digabung melalui *union
filtering*. Jika satu baris gagal pada salah satu model, baris yang sama
dikeluarkan dari keduanya. Prosedur ini menjamin bahwa selisih metrik
tidak disebabkan oleh perbedaan *query* atau kandidat yang dinilai.

Sesudah filtering, keselarasan manifes dan matriks diperiksa kembali.
Jumlah baris setiap layer_XX.npy harus sama dengan jumlah baris manifes,
tipe data harus float32, dan urutan identitas tidak boleh berubah.
Aturan metodologis ini diterapkan sama pada kedua model.

### Pemilihan lapisan representasi per sel

Titik representasi tidak ditentukan sebelumnya atau dipilih secara acak,
melainkan dipilih melalui percobaan pada *development set* untuk setiap
kombinasi sel dan model. Prosedur ini menjaga agar data pengujian tidak
ikut menentukan konfigurasi, sehingga evaluasi akhir tetap tidak bias.
Protokol pemilihan lapisan representasi ditetapkan sebagai berikut.

a.  Untuk setiap sel dari 13 sel dan setiap model (Wav2vec2, Data2vec)
    dilakukan evaluasi lapisan pada 13 titik (layer 0-12)

b.  Hitung MAP pada *development set* untuk setiap kombinasi sel, model
    dan lapisan

c.  Pilih lapisan dengan MAP development tertinggi

d.  Jika terjadi hasil yang sama, pilih lapisan dengan indeks lebih
    kecil

e.  Gunakan titik terpilih untuk evaluasi akhir pada *test set*

Dengan protokol tersebut, jumlah evaluasi penyapuan pada *development
set* adalah 13 sel × 2 model × 13 titik = 338 kombinasi. Perbandingan
antarlapisan dilakukan langsung pada nilai MAP mentah bertipe *float*,
bukan pada nilai persen yang telah dibulatkan untuk pelaporan.
Konsekuensinya, pembulatan pada penyajian tabel tidak pernah memengaruhi
lapisan yang terpilih, apabila dua lapisan tampak bernilai sama setelah
dibulatkan, lapisan dengan nilai mentah lebih tinggi tetap yang
terpilih. Jika terjadi hasil MAP seri antar lapisan, aturan indeks
terkecil hanya berlaku pada seri sempurna, yaitu ketika dua lapisan
memiliki nilai MAP yang identik, dan terpenuhi secara otomatis karena
penelusuran kandidat dilakukan menurut urutan indeks lapisan 0 sampai
12. Dengan demikian keputusan pemilihan bersifat deterministik dan dapat
direproduksi.

### Evaluasi MAP *development* per skenario

Hasil penyapuan disusun per skenario agar pola MAP antarlapisan dapat
dibandingkan pada setiap *owner ratio*. Setiap skenario disajikan
menggunakan lapisan sebagai baris dan rasio sebagai kolom, dengan nilai
MAP dinyatkaan dalam persen dan dilaporkan terpisah untuk kedua model.
Penyajian terpisah tersebut diperlukan karena kedua model dapat memuncak
pada lapisan yang berbeda meskipun diberi data yang sama.

### Ringkasan pemilihan *layer*

Seluruh titik terpilih beserta MAP *development*-Nya dirangkum dalam
satu tabel ringkasan dan satu diagram agar pola pemilihan lapisan pada
13 sel dapat dibaca sekaligus. Ringkasan ini menjadi dasar konfigurasi
yang dikunci sebelum himpunan digunakan.

### Skor dan pemeringkatan 

Untuk setiap *query* $q$ dan kandidat *database* $d_{j}$ pada titik
representasi yang sama, tingkat kemiripan dihitung menggunakan cosine
similarity \[29\]. Ukuran ini membandingkan arah kedua vektor tanpa
bergantung secara langsung pada besar atau panjang vektornya. Nilai
*cosine similarity* dihitung menggunakan persamaan berikut.

  ----------------------------------------------------------------------------------------------------------------
       $$s\left( q,d_{j} \right) = \frac{q^{T}d_{j}}{\text{|}q\text{|}_{2}\text{|}d_{j}\text{|}_{2}}$$   (3. 2)
  ---- ------------------------------------------------------------------------------------------------- ---------

  ----------------------------------------------------------------------------------------------------------------

Seluruh kandidat diurutkan dari skor terbesar ke terkecil. Hasilnya
adalah daftar peringkat, bukan keputusan akhir mengenai benar atau
salah. Daftar ini kemudian dinilai terhadap label relevansi (surah,
ayat). Pemisahan fungsi tersebut penting *cosine similarity* menentukan
urutan, sedangkan MAP, MRR, dan Top-K mengukur kualitas urutan.

## Evaluasi

Evaluasi dirancang untuk membandingkan model pada tiga kondisi yang
berbeda. Skenario A menggunakan rekaman mahasiswa sebagai *query* dan
Quran-MD sebagai *database*. Skenario B menggunakan Quran-MD pada kedua
sisi dengan qari *query* dan *database* yang saling lepas. Skenario C
menggunakan rekaman mahasiswa pada kedua sisi dengan mahasiswa yang
saling lepas. Ketiganya menggunakan definisi relevansi, ekstraksi,
fungsi skor, serta metrik yang sama.

Gambar 3.8 menyajikan hubungan sumber data dan protokol
development/test. Diagram ini menekankan bahwa penyapuan titik
representasi berhenti pada himpunan pengembangan.

![[]{#_Toc237464124 .anchor}Gambar 3. Evaluasi skenario A, B,
C](media/image11.png){width="4.8597222222222225in"
height="4.840729440069992in"}

## *Deployment*

Fase deployment merumuskan cara hasil penelitian dapat ditempatkan dalam
alur pencarian audio. Tahap ini bersifat konseptual. Penelitian tidak
menguji layanan produksi, waktu respons pengguna, keamanan API,
kapasitas serentak, pemantauan, atau pemeliharaan indeks.

Secara konseptual, sistem terdiri atas tahap persiapan *database*
referensi dan tahap pencarian *query*. Pada tahap persiapan *database*
referensi, seluruh audio referensi diproses terlebih dahulu menggunakan
model dan titik representasi yang telah dipilih. Hasil ekstraksi
embedding kemudian disimpan bersama *metadata* ayat dan digunakan
sebagai indeks pencarian. Pada tahap pencarian, audio baru dari pengguna
dinormalisasi dan diubah menjadi embedding menggunakan konfigurasi yang
sama. Embedding *query* tersebut kemudian dibandingkan dengan embedding
pada indeks menggunakan *cosine similarity*. Kandidat dengan skor
tertinggi dikembalikan sebagai hasil pencarian. Karena kinerja model
berbeda pada setiap skenario, pemilihan model dan titik representasi
harus disesuaikan dengan karakteristik domain serta metrik yang menjadi
prioritas. Tidak terdapat dasar untuk menganggap bahwa satu model selalu
unggul pada seluruh kondisi.

Kelayakan konseptual dinilai dari tiga syarat. Pertama, konfigurasi
model dan titik representasi harus dikunci dari bukti pengembangan yang
relevan. Kedua, *database* harus memakai normalisasi, model, dan dimensi
yang sama dengan *query*. Ketiga, hasil peringkat perlu diperlakukan
sebagai kandidat, terutama pada domain lintas sumber yang akurasinya
rendah. Rancangan lengkap dan implikasi hasilnya dibahas kembali pada
BAB IV.

# BAB IV HASIL DAN PEMBAHASAN

Bab ini menyajikan hasil pelaksanaan penelitian berdasarkan tahapan
*business understanding*, *data understanding*, *data preparation*,
*modeling*, *evaluation*, dan *deployment*. Hasil yang dibahas mencakup
rekonsiliasi korpus, validasi data, ekstraksi representasi, pemilihan
titik representasi, evaluasi akhir, dan perbandingan statistic antara
Wav2vec2 dan Data2vec.

Evaluasi dilakukan terhadap 13 sel yang berasal dari empat skenario.
Skenario A terdiri atas satu sel tanpa rasio pemilik, sedangkan Skenario
B, C, dan D masing-masing terdiri atas empat rasio pemilik, yaitu 60:40,
70:30, 80:20, dan 90:10. Dengan duan model pada setiap sel, hasil akhir
terdiri atas 26 baris model sel. Seluruh konfiguarasi model dan layer
representasi dipiilh menggunakan himpungan pengembangan (*development
set*). Setelah konfigurasi dipilih, himpunan pengujian (*test set*)
digunakan untuk satu kali evaluasi akhir. Dengan demikian, hasil
pengujian tidak digunakan untuk memilih ulang konfigurasi.

##  Hasil *Business Understanding*

Fase *business understanding* menghasilkan definisi keberhasilan yang
berfokus pada validitas eksperimen dan kemampuan menjawab pertanyaan
penelitian. Alur Wav2Vec2 dan Data2Vec berhasil diimplementasikan untuk
menghasilkan data *embedding* per ayat pada tiap-tiap *layer*
representasi, membentuk daftar peringkat menggunakan *cosine
similarity*, serta mengevaluasi peringkat tersebut menggunakan MAP, MRR,
Top-1, Top-5, dan Top-10.

Keberhasilan implementasi tidak berarti sistem harus memenuhi ambang
batas tertentu. Penelitian tidak menetapkan nilai minimum MAP atau Top-K
sebagai syarat kesiapan untuk digunakan dalam lingkungan produksi.
Keberhasilan pada tahap ini berarti bahwa data dapat ditelusuri kembali
ke sumbernya, kedua model menerima data yang sama, konfigurasi dipilih
tanpa menggunakan data yang terpisah dari data yang akan digunakan untuk
evaluasi, dan seluruh daftar peringkat dapat dievaluasi dengan prosuder
yang sama antara kedua model.

Perbandingan kedua model diatur melalui empat mekanisme. Pertama,
Wav2vec2 dan Data2vec menerima baris audio yang sama setelah proses
*union filtering*. Kedua, *embedding* dari kedua model memiliki dimensi
yang sama dan diringkas menggunakan *mean pooling*. Ketiga, dokumen
relevan ditentukan berdasarkan kesamaan pasangan (surah, ayat). Keempat,
*layer* representasi terbaik dipilih secara terpisah untuk setiap model
dan setiap sel menggunakan MAP pada data yang dikumpulkan sebagai
himpunan pengembangan (*development set*).

Empat skenario digunakan untuk memberikan konteks terhadap hasil
evaluasi.

a.  Skenario A menggunakan seluruh rekaman mahasiswa sebagai *query* dan
    seluruh Quran-MD yang dapat digunakan sebagai *database* referensi

b.  Skenario B menggunakan Quran-MD sebagai sumber *query* dan
    *database* referensi. Yang dipisahkan berdasarkan qari yang membaca
    Al-Qur'annya.

c.  Skenario C menggunakan rekaman mahasiswa sebagai sumber *query* dan
    *database* referensi. Yang mana dipisahkan berdasarkan pembacanya.

d.  Skenario D menggunakan rekaman mahasiswa sebagai *query* dan
    *database* referensi gabungan yang terdiri atas mahasiswa lain dan
    seluruh referensi Quran-MD. Identitas mahasiswa query tidak muncul
    pada bagian mahasiswa database referensi.

Susunan tersebut mengarahkan pembahasan dari pertanyaan umum mengenai
model terbaik menuju pertanyaan yang lebih spesifik, yaitu model mana
yang memberikan hasil lebih tinggi pada skenario, rasio pemilik dan
metrik-metrik yang digunakan.

1.  
2.  
3.  
4.  1.  

##  Hasil *Data Understanding* 

Penyelarasan jumlah pada tahap pengumpulan, segmentasi, validasi, dan
ekstraksi. Pemisahan ini mencegah angka dari tahap yang berbeda
disajikan seolah-olah merujuk pada populasi yang sama. Gambar 4.1
merangkum perubahan korpus mahasiswa dan Quran-MD sampai siap dipakai.
Diagram juga menampilkan tiga kategori provenance klip mahasiswa yang
jumlahnya tepat sama dengan total *query* final.

![[]{#_Toc236984558 .anchor}Gambar 4. Rekonsiliasi
data](media/image12.png){width="5.508333333333334in"
height="4.1930555555555555in"}

Gambar 4.1 menunjukkan bahwa 81 folder adalah jumlah folder mahasiswa
pada pengumpulan awal, sedangkan 60 adalah mahasiswa yang lolos validasi
teknis untuk eksperimen. Sebanyak 116 berkas nol dikeluarkan sebelum
ekstraksi embedding sehingga diperoleh 25.829 klip mahasiswa final. Pada
Quran-MD terdapat 17.130 baris referensi awal. Tiga baris mengalami
kegagalan pemrosesan. Indeks yang gagal tersebut dikeluarkan secara rata
dari hasil kedua model melalui *union filtering*, sehingga tersisa
17.127 referensi Quran-MD yang dapat digunakan.

  -----------------------------------------------------------------------
  **Tahap atau kategori**       **Jumlah**        **Status dan
                                                  penjelasan**
  ----------------------- ----------------------- -----------------------
  Folder mahasiswa awal             81            Populasi folder pada
                                                  awal pengumpulan

  Mahasiswa tervalidasi             60            Memenuhi pemeriksaan
                                                  teknis dan struktur
                                                  data

  Kandidat hasil                  25.945          Seluruh berkas kandidat
  segmentasi                                      sebelum pemeriksaan
                                                  ukuran

  Berkas nol byte                   116           Dikeluarkan sebelum
                                                  ekstraksi embedding

  Klip mahasiswa final            25.829          Seluruhnya berhasil
                                                  diekstraksi oleh kedua
                                                  model

  *Provenance* audit              24.872          Berasal dari rekaman
  berhasil                                        dengan catatan audit
                                                  berhasil

  *Provenance* fallback             710           Hasil alokasi waktu
  proporsional                                    proporsional

  Tanpa baris audit                 247           Tetap terlacak pada
  pasangan                                        manifes, tetapi tidak
                                                  memiliki pasangan baris
                                                  audit

  Baris referensi                 17.130          Kandidat ekstraksi
  Quran-MD awal                                   referensi

  Baris referensi gagal              3            Dikeluarkan dengan
                                                  union filtering

  Referensi Quran-MD              17.127          Baris final yang
  dapat digunakan                                 identik untuk kedua
                                                  model
  -----------------------------------------------------------------------

  : []{#_Toc236984570 .anchor}Tabel 4. Rekonsiliasi validasi

Jumlah kategori *provenance* mahasiswa konsisten karena
$24.872\  + \ 710\  + \ 247\  = \ 25.829$. Kategori tersebut menjelaskan
jalur proses dan ketersediaan catatan audio. Kategori audio berhasil
tidak boleh ditafsirkan sebagai bukti bahwa selruuh batas ayat telah
tepat secara fonetik. Penelitian tidak memiliki anotasi batas waktu
manual untuk seluruh klip, sehingga ketepatan segmentasi tetap menjadi
salah satu sumber kepastian data.

Gambar 4.2 memperlihatkan ukuran tiga skenario setelah seluruh validasi.
Angka pada sisi *query* dan *database* adalah ukuran yang benar-benar
digunakan untuk membentuk peringkat.

![[]{#_Toc236984559 .anchor}Gambar 4. Ukuran *query* dan *database* pada
scenario A, B, C](media/image13.png){width="5.508333333333334in"
height="2.8340277777777776in"}

Skenario A memakai seluruh 25.829 klip mahasiswa sebagai *query*
terhadap 17.127 referensi Quran-MD. Pada B, pembagian qari menghasilkan
5.139 *query* dan 11.988 dokumen. Pada C, pembagian mahasiswa
menghasilkan 7.940 *query* dan 17.889 dokumen. Seluruh *query* memiliki
sedikitnya satu dokumen relevan, sehingga AP nol tidak muncul hanya
karena kelas relevan tidak tersedia.

  ----------------------------------------------------------------------------------
   **Skenario**  **Sumber       **Jumlah **Sumber          **Jumlah    **Pembaca**
                 *query***     *query*** *database***    *database***  
  -------------- ----------- ----------- -------------- -------------- -------------
        A        Mahasiswa        25.829 Quran-MD           17.127     Lintas sumber

        B        Quran-MD          5.139 Quran-MD           11.988     9 qari
                                                                       berbanding 21
                                                                       qari

        C        Mahasiswa         7.940 Mahasiswa          17.889     18 mahasiswa
                                                                       berbanding 42
                                                                       mahasiswa
  ----------------------------------------------------------------------------------

  : []{#_Toc236984571 .anchor}Tabel 4. Ukuran final skenario *retrieval*

##  Hasil *Data Preparation*

Fase data preparation dilaksanakan melalui empat tahap, yaitu seleksi
cakupan dan normalisasi identitas, segmentasi rekaman mahasiswa,
validasi teknis beserta pembentukan *manifest*, dan pemisahan data
menurut peran sumber. Bagian ini menyajikan hasil setiap tahap secara
berurutan beserta angka korpus aktualnya.

### Hasil seleksi dan normalisasi identitas

Seleksi cakupan menghasilkan dua kelompok surah, yaitu surah Al-Fatiahh
dan surah-surah dalam Juz Amma dengan total 38 surah dan 571 ayat. Data
awal memindai 81 folder mahasiswa dan memetakan setiap nama berkas ke
surah sasaran. Hasilnya menunjukkan 25 mahasiswa menggungah rekaman
untuk seluruh 38 surah, sedangkan 56 mahasiswa belum lengkap yang mana
surah yang tidak lengkapnya bervariasi.

Pemetaan identitas surah dijalankan menggunakan tabel alias yang
dideklarasikan pada bab sebelumnya, yang seluruhnya memuat 200 entri
alias bagi 38 surah sasaran, disertai *fallback* penomoran. Pemindaian
menemukan 1.803 berkas rekaman pada 81 folder mahasiswa. Seluruh berkas
kandidat berhasil dipetakan tanpa penerkaan terhadap variasi yang tidak
dikenal. Variasi yang tertangani mencakup perbedaan kapitalisasi,
penomoran awal, simbol pemisah, prefiks label seperti QS dan Surah,
deretan digit NIM pada akhir nama, serta variasi transliterasi seperti
*mutaffifin* dengan *muthoffifin* dan *kausar* dengan *kautsar*.

Ketiga kelas anomali penamaan yang diantisipasi pada BAB III muncul
dengan frekuensi yang berbeda. Anomali multi-surah terjadi satu kali,
yaitu satu berkas rekaman gabungan yang memuat rentang 13 surah dari
surah 102 sampai 114; berkas tersebut dikeluarkan dari segmentasi
otomatis agar tidak diperlakukan sebagai surah tunggal. Anomali
duplikasi berkas untuk surah yang sama pada satu mahasiswa tidak
terjadi, sehingga aturan mempertahankan satu berkas menurut urutan nama
tidak pernah dipakai. Demikian pula, tidak terdapat berkas yang tetap
tidak teridentifikasi, sehingga kategori tidak terpetakan tetap kosong.
Selain itu, 21 dari 81 folder mahasiswa tidak memuat berkas rekaman sama
sekali. Setelah pengurangan satu berkas multi-surah, 1.802 rekaman lolos
pemetaan dan dikonversi menjadi audio antara mono 16 kHz tanpa mengubah
nama berkas sumber; pembakuan diterapkan pada identitas numerik surah
dan nama direktori keluaran yang mengikuti penanda kanonis.

Pada cabang Quran-MD, penyaringan cakupan 38 surah atas dataset
bertingkat ayat menghasilkan 17.130 baris referensi kandidat. Setiap
baris telah menyertai identitas qori, surah, dan ayat, serta lolos
validasi properti audio satu kanal 16 kHz setelah konversi. Cabang ini
tidak melalui segmentasi karena unit datanya telah berupa ayat.

### Hasil segmentasi rekaman mahasiswa

Segmentasi memproses 1.802 rekaman tingkat surah menjadi klip tingkat
ayat. WhisperX menghasilkan stempel waktu tingkat kata, kemudian batas
ayat dibentuk dengan mengalokasikan urutan kata menurut jumlah kata
referensi setiap ayat; durasi setiap potongan dipaksa minimal 200 ms.
Proses ini menghasilkan 25.945 berkas klip kandidat pada penyimpanan.

Berdasarkan catatan audit yang ditempelkan sebagai metadata, 24.872 klip
berasal dari rekaman dengan penyelarasan kata yang berhasil, 710 klip
dihasilkan melalui alokasi waktu proporsional, dan 247 klip tidak
memiliki baris audit namun tetap terlacak pada manifest; ketiganya
berjumlah tepat 25.829 klip. Alokasi waktu proporsional terjadi pada 91
rekaman dengan dua pemicu yang berbeda, sebagaimana dirinci pada Gambar
4.2: 53 rekaman tidak memiliki kata bacaan setelah bagian pembuka (intro
only) sehingga menghasilkan 285 klip, sedangkan 38 rekaman memiliki kata
terdeteksi lebih sedikit daripada jumlah ayat sehingga menghasilkan 425
klip.

![Gambar 4. Pemicu *fallback* alokasi waktu
proporsional](media/image14.png){width="5.508333333333334in"
height="2.457638888888889in"}

Klip hasil segmentasi berdurasi rata-rata 3,59 detik dengan durasi
minimum 0,2 detik sesuai batas minimal yang dipaksakan, durasi maksimum
36,68 detik, dan total durasi seluruh klip 92.650 detik atau sekitar
25,7 jam. Perlu ditegaskan kembali bahwa keberhasilan audit mencerminkan
keberhasilan prosedur teknis, bukan jaminan manual bahwa setiap batas
ayat tepat secara fonetik; kategori fallback tetap dipertahankan pada
metadata agar ketidakpastian ini tidak disembunyikan.

### Hasil validasi dan pembentukan *manifest*

Validasi akhir menerapkan seluruh gerbang yang dirancang pada BAB III
terhadap 25.945 berkas kandidat, sebagaimana dirangkum pada Gambar 4.3.
Sebanyak 116 berkas berukuran nol dikeluarkan sehingga 25.829 klip
dinyatakan lolos. Seluruh klip yang lolos memenuhi struktur direktori
baku, terbaca oleh pemeriksa durasi dan keterbacaan audio, berada pada
rentang jumlah ayat yang sah, dan tidak memuat identitas ganda setiap
penolakan dicatat alasannya.

![Gambar 4. Alur validasi dan pembentukan
*manifest*](media/image15.png){width="5.508333333333334in"
height="1.6909722222222223in"}

Konsekuensi kebijakan inklusi terhadap korpus mahasiswa adalah
berkurangnya populasi dari 81 folder awal menjadi 60 mahasiswa yang
memiliki klip lolos validasi; 21 mahasiswa sisanya tidak memiliki satu
pun klip yang lolos. Distribusi klip per mahasiswa pada korpus final
memiliki rata-rata 430 klip dan median 561 klip, dengan 23 mahasiswa
memiliki klip untuk seluruh 38 surah. Dari sisi rekaman, 1.795 rekaman
menyumbang klip tervalidasi. Manifest tervalidasi yang memuat lima belas
kolom identitas, lokasi, durasi, dan provenance audit menjadi
satu-satunya sumber kebenaran bagi pembentukan sel eksperimen;
rekonsiliasi lengkap antara tahap pengumpulan, segmentasi, validasi, dan
ekstraksi disajikan pada Tabel 4.1. Pada cabang referensi, 17.130 baris
Quran-MD menjadi kandidat ekstraksi; tiga baris di antaranya kemudian
gagal pada ekstraksi representasi dan dikeluarkan secara identik dari
kedua model melalui union filtering, sehingga dibahas pada sub-bab Hasil
*Modeling*.

### Hasil Pemisahan Dataset dan Peran Sumber data 

Eksperimen ini menggunakan beberapa jenis pemisahan data yang berbeda
tergantung dengan skenarionya. Pemisahan ini menentukan interpretasi
hasil dan temuan penelitian. Skenario A menggunakan data lintas domain.
Skenario A menggunakan seluruh 25.829 klip mahasiswa sebagai *query* dan
seluruh 17.127 klip Quran-MD sebagai *database* referensi. Tidak ada
pemisahan data pada domain yang sama karena sumber data sudah berbeda.
Skenario ini mewakili kondisi pencarian bacaan mahasiswa terhadap
referensi profesional.

Skenario B, C, D: Pemisahan Berdasarkan Pemilik Tiga skenario ini
menggunakan rasio pemilik (*owner ratio*) untuk membagi data menjadi
*query* dan *database* pada domain yang sama. Pemilik didefinisikan
sebagai:

a.  Skenario B: Qari (pembaca Quran-MD). Setiap qari memiliki beberapa
    klip ayat.

b.  Skenario C: Mahasiswa (NIM). Setiap mahasiswa memiliki beberapa klip
    hasil segmentasi.

c.  Skenario D: Mahasiswa untuk *query*, gabungan mahasiswa lain yang
    ditambahkan dengan seluruh Quran-MD untuk *database* referensi.

*Owner ratio* yang diterapkan terdapat beberapa jenis variasi 60:40,
70:30, 80:20, 90:10. menentukan proporsi pemilik yang menjadi *query*
versus *database* referensi. Misalnya, rasio 70:30 berarti 70% pemilik
(diurutkan berdasarkan NIM/qari) menjadi *database* referensi, sedangkan
30% sisanya menjadi *query*. Pembagian ini dilakukan secara stratified
dengan seed 42 untuk menjaga konsistensi antar sel.

Pencegahan Kebocoran Data Untuk skenario B, C, dan D, audit kebocoran
memastikan. Pertama, tidak ada pemilik yang muncul di kedua sisi
(*query* dan *database*). Kedua, Tidak ada jalur berkas yang sama antara
*query* dan *database*. Dan Terakhir untuk skenario D, *database* adalah
gabungan mahasiswa lain (yang bukan *query*) ditambah seluruh Quran-MD

*Union filtering* ketika ekstraksi embedding gagal untuk beberapa baris
pada salah satu model, baris tersebut dikeluarkan dari kedua model
menggunakan union filtering. Pada Quran-MD, 3 dari 17.130 baris gagal
diekstraksi dan dikeluarkan secara identik dari kedua model, menyisakan
17.127 baris final. Pada klip mahasiswa, seluruh 25.829 klip berhasil
diekstraksi oleh kedua model.

Pembagian Development dan Test Setiap sel membagi *query* menjadi dua
himpunan:

a.  *Development set*: Digunakan untuk penyapuan 13 titik (layer 0-12)
    dan memilih titik terbaik berdasarkan MAP tertinggi

b.  *Test set*: Terkunci selama pemilihan, hanya digunakan untuk
    evaluasi final setelah titik terbaik ditetapkan

Pembagian ini dilakukan secara *stratified* per pasangan surah-ayat
dengan seed 42: pada setiap kelompok ayat, sekitar 70% klip (dibulatkan
ke bawah, minimal satu klip) masuk development set dan sisanya masuk
test set. Karena pembulatan dilakukan pada tingkat kelompok ayat,
proporsi klip development terhadap total *query* bervariasi antar sel
(57,9%--69,1%). Jumlah development dan test pada setiap sel tercantum
pada Tabel 4.3.

####  Ukuran Data Skenario A 

Skenario A ttidak menggunakan rasio pemilik. Seluruh klip mahasiswa
ditempatkan sebagai *query,* sedangkan seluruh *database* referensi dari
Quran-MD. Skenario A menghasilkan kondisi lintas sumber karena query dan
database referensi berasal dari jalur akuisisi serta persiapan yang
berbeda. Skenario ini teridi atas satu sel dan tidak menggunakan
pembagian rasio pemilik.

  -----------------------------------------------------------------------------------
  **Sel**   **Sumber     **Jumlah    **Sumber        **Jumlah   **Pemisahan data**
            *Query***    *Query***   *database*     *database*  
                                     referensi**    refensi**   
  --------- ------------ ----------- ------------- ------------ ---------------------
  A         Mahasiswa    25.829      Quran-MD         17.127    Terpisah oleh sumber
                                                                data

  -----------------------------------------------------------------------------------

  : []{#_Toc239590203 .anchor}Tabel 4. Ukuran Query dan *database*
  referensi pada Skenario A

####  Ukuran Data Skenario B 

Skenario B menggunakan Quran-MD pada sisi query dan *database*
referensi. Identitas qari dibagi secara disjoint berdasarkan rasio
pemilik. Peningkatan rasio pemilik database referensi menyebabkan jumlah
qari dan klip pada *database* bertambah, sedangkan jumlah qari dan klip
*query* berkurang. Kedua sisi tetap berasal dari Quran-MD, tetapi tidak
ada qari yang muncul pada kedua sisi.

  --------------------------------------------------------------------------
     **Sel**     **Qari query**     **Qari        **Jumlah       **Jumlah
                                 *database***    *query***     *database***
  -------------- -------------- -------------- -------------- --------------
     B-60:40           12             18           6.849          10.278

     B-70:30           9              21           5.139          11.988

     B-80:20           6              24           3.426          13.701

     B-90:10           3              27           1.713          15.414
  --------------------------------------------------------------------------

  : []{#_Toc239590204 .anchor}Tabel 4. Ukuran *query* dan *database*
  referensi pada Skenario B

####  Ukuran Data Skenario C 

Skenario C menggunakan rekaman mahasiswa pada sisi *query* dan
*database* referensi. Identitas mahasiswa pada kedua sisi dipisahkan
berdasarkan rasio pemilik. Skenario C mengevaluasi generalisasi lintas
mahasiswa dalam sumber data yang sama. Meskipun *query* dan *database*
sama-sama berasal dari rekaman mahasiswa, identitas mahasiswa pada kedua
sisi tidak beririsan.

  --------------------------------------------------------------------------
     **Sel**      **Mahasiswa    **Mahasiswa      **Jumlah       **Jumlah
                   *query***     *database***    *query***     *database***
  -------------- -------------- -------------- -------------- --------------
     C-60:40           24             36           9.386          16.443

     C-70:30           18             42           6.995          18.834

     C-80:20           12             48           4.118          21.711

     C-90:10           6              54           1.833          23.996
  --------------------------------------------------------------------------

  : []{#_Toc239590205 .anchor}Tabel 4. Ukuran *query* dan *database*
  referensi pada Skenario C

####  Ukuran Data Skenario D 

Skenario D menggunakan *query* mahasiswa yang sama dengan Skenario C
pada rasio terkait. Perbedaanya terletak pada *database.* Database
referensi skenario D terdiri atas mahasiswa lain yang identitasnya
*disjoint* dari query dan seluruh 17.127 referensi Quran-MD. *Database*
referensi D selalu lebih besar daripada *database* C pada rasio yang
sama karena menambahkan seluruh Quran-MD. Sebagai contoh, pada Tabel 4.5
C-60:40 memiliki 16.443 dokumen dalam *database*, sedangkan dalam Tabel
4.6 D-60:40 memiliki 33.570 dokumen. Selisihnya adalah 17.127 dokumen,
sesuai dengan jumlah referensi Quran-MD.

Seluruh *query* pada Skenario A, B, C, dan D memiliki sedikitnya satu
dokumen relevan dalam *database*. Dengan demikian, AP bernilai nol tidak
muncul hanya karena pasangan (surah, ayat) yang relevan tidak tersedia.

  ----------------------------------------------------------------------------------
    **Sel**    **Mahasiswa   **Mahasiswa    **Jumlah   **Sumber          **Jumlah
                *query***    *database***   *query***  *database***    *database***
  ----------- ------------- -------------- ----------- -------------- --------------
    D-60:40        24             36          9.386    Mahasiswa          33.570
                                                       lain +         
                                                       Quran-MD       

    D-70:30        18             42          6.995    Mahasiswa          35.961
                                                       lain +         
                                                       Quran-MD       

    D-80:20        12             48          4.118    Mahasiswa          38.838
                                                       lain +         
                                                       Quran-MD       

    D-90:10         6             54          1.833    Mahasiswa          41.123
                                                       lain +         
                                                       Quran-MD       
  ----------------------------------------------------------------------------------

  : []{#_Toc239590206 .anchor}Tabel 4. Ukuran *query* dan *database*
  referensi pada Skenario D

####  Penyaringan data *query* 

Setelah data berhasil dikelompokkan berdasarkan *database* referensi dan
*query.* Dilakukan pemeriksaan kelas relevan pada sisi *database*
referensi. Sebuah *query* harus memiliki sedikitnya satu dokumen relevan
di dalam *database* referensi, agar dapat digunakan untuk menghitung
metrik *retrieval*. Dokumen relevan ditentukan berdasarkan pasangan
label (surah, ayat), bukan berdasarkan qari atau mahasiswa.

*query* yang tidak memiliki dokumen dengan pasangan (surah, ayat) yang
sesuai di dalam *database* referensi dikeluarkan. Penyaringan ini
diterapkan setelah pembentukan *query* dan *database* referensi karena
ketersediaan dokumen relevan bergantung pada komposisi *database*
referensi di setiap skenario dan *owner ratio*. Setelah penyaringan
dilakukan, seluruh *query* yang masuk ke tahap evaluasi memiliki
sedikitnya satu dokumen relevan.

Dengan prosedur tersebut, nilai AP nol tidak muncul semata-mata karena
dokumen relevan tidak tersedia di dalam *database* referensi. Proses ini
juga memastikan bahwa metrik yang dihitung merefleksikan kemampuan model
dalam menempatkan dokumen relevan pada peringkat tinggi, bukan ketiadaan
dokumen pembanding yang diperlukan.

####  Pembagian Development dan Test 

Pada proses ini *query* dibagi menjadi *development set* dan *test set*.
Pembagian ini dilakukan secara terstratifikasi berdasarkan pasangan
(surah, ayat) menggunakan seed 42. Tujuannya adalah menjaga agar
pasangan ayat yang tersedia pada data *query* tetap terwakili pada kedua
himpunan sebanyak mungkin.

Pada setiap kelompok pasangan (surah, ayat), indeks *query* diacak
secara deterministik. Sekitar 70% klip pada setiap kelompok dimasukkan
ke dalam *development set*, sedangkan indeks yang tersisa dimasukkan ke
dalam test set. Karena pembulatan diterapkan secara terpisah pada setiap
kelompok ayat, proporsi akhir klip pada *development set* tidak selalu
tepat 70% dari total *query*. Pada seluruh sel, proporsi aktual
development set berada pada kisaran 57,9% hingga 69,1%.

Development set digunakan untuk menyapu mencoba seluruh 13 *layer*
representasi, yaitu titik 0 sampai 12, pada masing-masing model. Titik
representasi dengan nilai MAP development tertinggi dipilih secara
terpisah untuk setiap kombinasi sel dan model. Setelah titik
representasi ditentukan, konfigurasi tersebut akan digunakan untuk
mengevaluasi pada himpunan data *test set*.

Sebaliknya, test set tidak digunakan selama proses pemilihan titik
representasi. Himpunan ini hanya dibuka setelah konfigurasi final
ditetapkan untuk menghasilkan evaluasi akhir berupa MAP, MRR, Top-1,
Top-5, dan Top-10. Pemisahan tersebut menjaga agar data pengujian tetap
berfungsi sebagai evaluasi yang tidak digunakan untuk mengambil
keputusan konfigurasi.

##  Hasil *Modeling* 

Fase *Modeling* dijalankan sesuai dilakukan dalam empat tahap yaitu
ekstraksi *layerwise,* pembersihan data hasil ekstraksi, pemilihan titik
representasi per sel, dan penskoran beserta pemeringkatan. Sejalan
dengan rancangan yang dibuat pada bab sebelumnya, hasil tahap pemilihan
titik representasi disajikan dalam tiga bagian, yaitu hasil penyapuan
lapisan, hasil MAP *development* per skenario, dan ringkasan titik
terpilih. Bagian ini menyajikan setiap tahap beserta angka artefak
aktualnya.

### Hasil ekstraksi representasi *layerwise*

Ekstraksi representasi layerwise dilaksanakan menggunakan dua model
self-supervised learning, yaitu Wav2vec2 dan Data2vec dalam keadaan
*frozen.* Kedua model memiliki arsitektur serupa yang teridiri atas
tujuh lapisan konvolusi sebagai *feature extractor* dan dua belas blok
Transformer, sehingga masing-masing menghasilkan 13 titk representasi
dengan dimensi tersembunyi 768. Proses ekstraksi diterapkan pada dua
himpunan data secara terpisah, yaitu himpunan referensi yang bersumber
dari Quran-MD dan himpunan *query* yang bersumber dari rekaman
mahasiswa.

Alur ekstraksi untuk setiap klip audio dimulai dengan pemuatan berkas
audio menjadi gelombang mono pada *sample rate* 16 kHz menggunakna
*library* *librosa.* Audio tersebut kemudian dinormalisasikan oleh
feature extractor bawaan model dan diteruskan dengan parameter yang
menampilkan *hidden states* agar model mengembalikan keluaran dari
seluruh 13 lapisan, bukan hanya lapisan terkahir. Keluaran setiap
lapisan berupa tensor tiga dimensi berukuran (1, *T*\_enc, 768), di mana
*T\_*enc merupakan jumlah *frame* tempral yang bergantung pada durasi
audio masukan. untuk mengubah representasi panjang *variable* tersebut
menjadi vektor berdimensi tetap, diterapkan mean pooling terhadap sumbu
temporal sebagaimana telah diformulasikan pada persamaan 3.1. Operasi
ini menghasilkan satu vektor berukuran 768 untuk setiap lapisan,
sehingga satu klip audio menghasilkan 13 vektor yang disusun menjadi
matriks *checkpoint* berukuran 13 x 768 Gambar 4.5 mengilustrasikan alur
lengkap proses ini.

![Gambar 4. Alur
Ekstraksi](media/image16.png){width="5.291666666666667in"
height="3.5844838145231845in"}

Gambar 4.5 Alur ekstraksi per klip dan assembly berkas per-layer setiap
checkpoint sementara disimpan secara atomis dalam format terkompresi
.npz sebelum proses melanjutkan ke klip berikutnya. Mekanisme penulisan
atomis menggunakan strategi *write-to-temporary-then-replace*, yaitu
data dituliskan terlebih dahulu ke berkas sementara dan dipindahkan ke
lokasi akhir melalui operasi *os.replace()* yang bersifat atomis pada
file system. Strategi ini menjamin bahwa berkas tidak pernah berada
dalam kondisi setengah tertulis, sehingga kegagalan di tengah proses
tidak merusak data yang telah berhasil diekstrak sebelumnya. Berkas
progress.json mencatatat kemajuan penyelesaian secara berkalasa dan
berfungsi sebagai penanda otoritatif, sehingga proses yang terputus
dapat dilanjutkan tanpa mengulang seluruh korpus.

Setelah seluruh klip dalam satu himpunan data selesai diekstrak, tahap
setelah seluruh klip dalam satu himpunan data selesai diekstrak, tahap
assembly merakit checkpoint-checkpoint per klip mejnadi berkas
per-lapisan. Pada tahap ini dilakukan transposisi logis, data yang
semual terorganisasi per klip dikelompokkan berdasarkan indeks lapisan,
sehingga *embedding* dari lapisan yang sama pada seluruh klip
digabungkan menjadi satu matriks berukuran N x 768. Hasil akhir berupa
13 berkas NumPy Bernama layer_00.npy hingga layer_12.npy, di mana setiap
berkas menyimpan matriks float32 dengan N baris dan 768 kolom. Baris
ke-I pada setiap matriks selalu bersesuaian dengan baris ke-i pada
*manifest* keluaran manifest.csv, sehingga identitas klip (surah, ayat
dan qari atau mahasiswa) dapat dilacak secara konsisten. Gambar di bawah
mengilustrasikan transformasi dimensi tensor pada setiap tahap.

![Gambar 4. Transformasi dimensi tensor pada setiap tahap
ekstraksi](media/image17.png){width="5.508333333333334in"
height="1.15in"}

Validasi artefak akhir dilakukan secara otomatis sebelum status ditandai
selesai. Prosedur validasi memeriksa keberadaan ke-13 berkas lapisan,
memverifikasi bahwa setiap berkas memiliki bentuk (N, 768) dengan tipe
data float32, serta memastikan jumlah baris konsisten dengan *manifest*.
Hanya setelah seluruh proses tersebut lolos, berkas progress.json di
update menjadi *complete* dan proses dinyatakan selesai. Tabel 4.7
merangkum hasil actual ekstraksi pada kedua himpunan data untuk kedua
model.

  -----------------------------------------------------------------------------------------
  **Himpunan    **Model**   **Jumlah   **Jumlah   **Bentuk   **Tipe    **Ukuran   **Baris
  Data**                    Klip (N)** Berkas     Per        Data**    Per        Gagal**
                                       Layer**    Berkas**             Berkas**   
  ------------- ----------- ---------- ---------- ---------- --------- ---------- ---------
  Referensi     Wav2Vec2    17.130     13         (17130,    float32   ≈50,2 MB   3
  (Quran-MD)                                      768)                            

  Referensi     Data2Vec    17.130     13         (17130,    float32   ≈50,2 MB   3
  (Quran-MD)                                      768)                            

  Query         Wav2Vec2    25.829     13         (25829,    float32   ≈75,7 MB   0
  (Mahasiswa)                                     768)                            

  Query         Data2Vec    25.829     13         (25829,    Float32   ≈75,7 MB   0
  (Mahasiswa)                                     768)                            
  -----------------------------------------------------------------------------------------

  : Tabel 4. Hasil ekstraksi representasi *layerwise* pada kedua
  himpunan data

Sebagaimana disajikan pada Tabel 4.7, himpunan referensi menghasilkan
17.130 baris per model dengan tiga baris gagal, sedangkan himpunan query
menghasilkan 25.829 baris per model tanpa kegagalan. Secara keseluruhan,
proses ekstraksi menghasilkan 52 berkas .npy (4 kombinasi himpunan-model
× 13 lapisan) dengan total kapasitas penyimpanan sekitar 3,2 GB.
Ekstraksi himpunan *query* yang berjumlah 25.829 klip pada model
Wav2Vec2. Tiga belas titik representasi yang diekstrak memiliki
karakteristik yang berbeda berdasarkan kedalamannya dalam arsitektur
model. Lapisan 0 merupakan keluaran *feature projection*, yaitu proyeksi
linear dari fitur konvolusi, yang menangkap informasi akustik.
Karakteristik ini sejalan dengan temuan Pasad, dkk \[35\] yang
menunjukkan bahwa lapisan-lapisan awal Wav2vec2 memiliki kemiripan
tinggi dengan fitur akustik. Lapisan 1 hingga 3 merupakan lapisan
Transformer awal yang mulai menggabungkan konteks lokal dan
merepresentasikan pola fonetik dasar. Lapisan 4 hingga 6 berada pada
tingkat menengah dan menangkap pola prosodi serta intonasi bacaan, hal
ini sejalan dengan hasil Yang, dkk \[36\]. Lapisan 7 hingga 9 sering
dianggap sebagai zona transisi yang menyeimbangkan informasi akustik dan
representasi abstrak \[35\]. Lapisan 10 hingga 12 merupakan lapisan
akhir yang menghasilkan representasi paling kontekstual dan abstrak,
dioptimalkan untuk *objective* pelatihan masing-masing model.
Karakteristik ini menjadi dasar bagi penyapuan lapisan pada tahap
selanjutnya, di mana setiap sel eksperimen memilih titik representasi
yang menghasilkan MAP development tertinggi.

![Gambar 4. Spektrum abstraksi representasi per
layer](media/image18.png){width="5.508333333333334in"
height="2.3826388888888888in"}

Ekstraksi final menghasilkan 13 layers representasi untuk setiaip model
pada setiap himpunan data. Setiap klip mula-mula menghasilkan checkpoint
berukuran 13 X 768, kemudian embedding dari lapisan yang sama dirakit
menjadi berkas layer_00.npy hingga layer_12.npy. setiap matriks akhir
berukuran N x 768, dan baris ke-I pada matriks selalu bersesuaian dengan
baris ke -1 pada *manifest*. Ekstraksi final menghasilkan 13 *layers*
representasi untuk setiap model pada setiap himpunan data. Setiap klip
mula-mula menghasilkan *checkpoint* berukuran 13 × 768, kemudian
embedding dari lapisan yang sama dirakit menjadi berkas layer_00.npy
hingga layer_12.npy. Setiap matriks akhir berukuran N × 768, dan baris
ke-i pada matriks selalu bersesuaian dengan baris ke-i pada *manifest*.

Seluruh 25.829 klip mahasiswa berhasil diproses oleh Wav2Vec2 dan
Data2Vec. Dengan demikian, ekstraksi *query* final tidak memiliki baris
yang ditandai gagal. Pada Quran-MD, tiga dari 17.130 baris ditandai
gagal. *Union filtering* mengeluarkan ketiga indeks tersebut dari hasil
kedua model dan menyisakan 17.127 baris identik. Sedangkan validasi
artefak akhir memastikan keberadaan 13 *layers*, bentuk (N, 768), tipe
float32, dan keselarasan jumlah baris dengan *manifest*.

### Hasil pembersihan data hasil ekstraksi

Pembersihan data dilakukan dalam tiga tahap yaitu deteksi pada saat
ekstraksi, penyaringan gabungan (*union filtering)* lintas kedua model
yang dibandingkan pada saat evaluasi dan terakhir validasi artefak
akhir. Ketiga tahap tersebut diterapkan secara identik pada Wav2Vec2 dan
Data2Vec guna menjamin bahwa perbedaan hasil evaluasi semata-mata
berasal dari perbedaan representasi, bukan dari perbedaan prosedur
prapemrosesan.

Pada tahap pertama, setiap klip audio yang telah diekstrak diperiksa
terhadap dua syarat sebelum disimpan. Syarat pertama adalah kesesuaian
bentuk tensor keluaran, yaitu model harus menghasilkan tepat 13 lapisan
dengan dimensi 768 pada setiap lapisan. Syarat kedua adalah keabsahan
numerik, yaitu seluruh nilai pada vektor harus bersifat finite dan tidak
mengandung nilai tak-terdefinisi. Apabila salah satu syarat tidak
terpenuhi misalnya karena berkas audio rusak atau anomali komputasi
baris tersebut ditandai sebagai gagal pada manifes. Vektor representasi
untuk baris yang gagal diisi dengan vektor nol berdimensi 13 × 768,
sehingga posisi indeks pada matriks akhir tidak bergeser. Keputusan
untuk mempertahankan baris gagal dengan vektor nol, alih-alih
menghapusnya dari matriks, dipilih agar korespondensi satu-ke-satu
antara baris matriks dan baris manifes tetap terjaga. Pendekatan
penghapusan akan menggeser seluruh indeks di bawah baris yang dihapus
dan berpotensi memutus keterkaitan antara data representasi dan metadata
identitas klip.

Pada tahap kedua, penyaringan gabungan (union filtering) diterapkan
sebelum evaluasi dimulai. Prosedur ini membangun mask boolean berdimensi
N dengan menggabungkan kolom kegagalan dari manifes Wav2Vec2 dan manifes
Data2Vec melalui operasi konjungsi logis. Dengan pendekatan ini, apabila
suatu baris ditandai gagal pada salah satu model, baris yang sama turut
dikeluarkan dari model lainnya. Tujuan penyaringan gabungan adalah
menyetarakan himpunan klip yang dievaluasi pada kedua model, sehingga
selisih metrik retrieval tidak tercemari oleh perbedaan jumlah atau
komposisi query maupun kandidat yang dinilai. Tanpa prosedur ini, satu
model dapat dievaluasi pada lebih banyak klip dibandingkan model
lainnya, dan perbandingan metrik menjadi tidak setara.

Hasil kuantitatif pembersihan disajikan pada Tabel 4.X. Pada himpunan
referensi Quran-MD, tiga baris dari total 17.130 baris ditandai gagal.
Ketiga baris tersebut berasal dari qari yang sama pada surah 82 ayat 10,
surah 88 ayat 24, dan surah 88 ayat 26. Kegagalan terjadi secara
konsisten pada kedua model, sehingga penyaringan gabungan mengeluarkan
tepat tiga baris dan menyisakan 17.127 baris yang dilibatkan dalam
evaluasi. Pada himpunan query mahasiswa, satu baris dari total 7.119
baris ditandai gagal pada surah 105 ayat 1, yang juga gagal pada kedua
model. Setelah penyaringan gabungan diterapkan, tersisa 7.118 baris yang
dievaluasi. Rasio kegagalan pada kedua himpunan data sangat kecil, yaitu
masing-masing 0,018% dan 0,014%, sehingga tidak memengaruhi
representativitas hasil evaluasi.

Selama proses ekstraksi *layerwise,* setiap klip audio diproses secara
individual melalui model dan hasilnya langsung diverifikasi sebelum
disimpan. Pemeriksaan mencakup dua aspek yaitu kelengkapan bentuk tensor
dan keabsahan numerik. Kelengkapan bentuk tensor mengecek apakah model
mengembalikan tepat 13 lapisan dengan dimensi 768 pada setiap lapisan,
sedangkan keabsahan numerik yaitu apakah seluruh nilai pada vektor tidak
bernilai NaN atau tak hingga. Jika salah satu pemeriksaan gagal baik
karena audio rusak, format tidak didukung atau anomali pada proses
komputasi baris tersebut ditandai sebagai klip yang gagal. Pada
*checkpoint* sementara pada berkas dengan ekstensi *npz*. Vektor
representasi baris yang gagal diisi dengan vector nol berdimensi 768
sebagai *placeholder* agar posisi indeks baris pada matriks akhir tetap
terjaga. Mekanisme ini memastikan bahwa baris ke-*i* pada matriks
layer_XX.npy selalu bersesuaian dengan baris ke-*i* pada manifes,
terlepas dari keberhasilan atau kegagalan ekstraksi.

Tabel 4.8 Statistik pembersihan data hasil ekstraksi

  --------------------------------------------------------------------------------
  **Himpunan    **Total     **Gagal      **Gagal      **Union     **Dievaluasi**
  Data**        Baris**     Wav2Vec2**   Data2Vec**   Gagal**     
  ------------- ----------- ------------ ------------ ----------- ----------------
  Referensi     17.130      3            3            3           17.127
  (Quran-MD)                                                      

  Query         7.119       1            1            1           7.118
  (Mahasiswa)                                                     
  --------------------------------------------------------------------------------

[]{#_Toc237480661 .anchor}

Pada tahap ketiga, validasi artefak akhir dijalankan untuk memastikan
integritas data sebelum evaluasi dimulai. Validasi ini mencakup empat
pemeriksaan, yaitu keberadaan 13 berkas lapisan representasi, kesesuaian
bentuk setiap berkas dengan dimensi (N, 768) sesuai jumlah baris pada
manifes, kesesuaian tipe data seluruh elemen pada presisi float 32-bit,
serta ketiadaan nilai tak-terdefinisi atau tak-hingga pada seluruh
elemen matriks. Apabila salah satu pemeriksaan tidak terpenuhi, proses
evaluasi dihentikan. Validasi ini berfungsi sebagai pemeriksaan terakhir
yang menjamin bahwa seluruh komputasi jarak dan metrik retrieval
beroperasi pada data yang bersih dan konsisten.

Secara keseluruhan, prosedur pembersihan mempertahankan invarian bahwa
baris ke-i pada setiap berkas lapisan representasi selalu merujuk pada
klip yang sama dengan baris ke-i pada manifes. Invarian tersebut dijaga
pada tiga titik, yaitu pada saat ekstraksi melalui penggunaan vektor nol
yang mempertahankan posisi indeks, pada saat penyaringan gabungan
melalui penerapan mask secara sinkron pada matriks dan manifes, dan pada
saat validasi melalui pemeriksaan kesesuaian jumlah baris. Keselarasan
ini diperlukan karena matriks relevansi pada evaluasi retrieval
diturunkan dari pasangan identitas surah dan ayat pada manifes. Apabila
urutan baris antara matriks representasi dan manifes tidak selaras,
seluruh perhitungan MAP dan MRR akan menghasilkan angka yang tidak
mencerminkan kinerja sesungguhnya.

### Pemilihan lapisan representasi per Sel 

Pemilihan titik representasi dilakukan melalui Evaluasi tiap lapisan
yang dilakukan dengan sistematis pada *development set* untuk setiap sel
dan model. Protokol ini memastikan bahwa pemilihan konfigurasi tidak
menggunakan data pengujian, sehingga evaluasi akhir tetap tidak bias.
Protocol pemilihan lapisan representasi sebagai berikut:

a.  Untuk setiap sel yang memiliki jumlah 13 sel dan setiap model
    (Wav2Vec2, Data2Vec), lakukan evaluasi lapisan pada 13 titik (layer
    0-12)

b.  Hitung MAP pada *development set* untuk setiap kombinasi
    sel-model-lapisan

c.  Pilih lapisan dengan MAP tertinggi pada *development set*

d.  Jika terjadi seri, pilih layer lebih kecil

e.  Gunakan titik terpilih untuk evaluasi akhir pada *test set*

Total evaluasi akan sebanyak
$13\ sel\  \times \ 2\ model\  \times \ 13\ titik\  = \ 338.$ untuk
pembagian jumlah *development set* dan *test set* pada setiap sel
terdapat pada tabel 4.7.

  -----------------------------------------------------------------------
         **Sel**                        **n_dev**              **n_test**
  ---------------------- ------------------------ -----------------------
            A                              17.839                   7.990

         B-60:40                            4.565                   2.284

         B-70:30                            3.426                   1.713

         B-80:20                            2.284                   1.142

         B-90:10                            1.142                     571

         C-60:40                            6.357                   3.029

         C-70:30                            4.613                   2.382

         C-80:20                            2.616                   1.502

         C-90:10                            1.062                     771

         D-60:40                            6.357                   3.029

         D-70:30                            4.613                   2.382

         D-80:20                            2.616                   1.502

         D-90:10                            1.062                     771
  -----------------------------------------------------------------------

  : []{#_Toc239590207 .anchor}Tabel 4. Pembagian *development set* dan
  *test set* pada setiap sel

Jumlah *development* dan *test* tepat menjumlah ke total *query* pada
setiap sel. Development test digunakan untuk mengevaluasi 13 layer
representasi, sedangkan *test set* data tetap steril sampai titik
terbaik ditetapkan. Karena itu, lapisan akhir tidak menjadi dasar
pemilihan konfigurasi.

### Evaluasi MAP Development per Skenario 

Bagian ini menyajikan tabel untuk keseluruhan hasil uji pencarian layer
representasi yang paling optimal dengan MAP pada *development set* untuk
setiap skenario. Tabel-tabel ini menunjukkan bagaimana MAP berubah antar
layer 0-12 dan juga untuk setiap *owner ratio*. Setiap tabel menggunakan
layer sebagai baris dan rasio sebagai kolom, dengan nilai MAP dalam
persen.

####  Skenario A 

Skenario A hanya memiliki satu konfigurasi karena tidak ada variasi
rasio pemilik (seluruh mahasiswa vs. seluruh Quran-MD).

  -----------------------------------------------------------------------
         **Layer**             **Wav2Vec2**            **Data2Vec**
  ----------------------- ----------------------- -----------------------
             0                     0,31%                   0,39%

             1                     0,39%                   0,51%

             2                     0,48%                   0,80%

             3                     0,55%                   1,18%

             4                     0,72%                   1,48%

             5                     1,11%                 **1,58%**

             6                     1,31%                   1,41%

             7                   **1,40%**                 1,18%

             8                     1,27%                   1,03%

             9                     0,93%                   0,91%

            10                     0,80%                   0,92%

            11                     0,58%                   0,71%

            12                     0,48%                   0,56%
  -----------------------------------------------------------------------

  : []{#_Toc239590208 .anchor}Tabel 4. Evluasi layer dengan MAP Skenario
  A (n_dev = 17.839)

Pada Skenario A, Wav2Vec2 mencapai MAP tertinggi pada layer 7 (1,40%),
sedangkan Data2Vec mencapai MAP tertinggi pada layer 5 (1,58%). Data2Vec
unggul pada layer rendah hingga tengah (layer 3-6), sedangkan Wav2Vec2
unggul pada layer tengah hingga tinggi (layer 7-8).

####  Skenario B 

Skenario B menggunakan Quran-MD untuk kedua sisi (*query* dan
*database*) dengan pemisahan berdasarkan qari pada empat rasio berbeda.

  --------------------------------------------------------------------------
    **Layer**           **60:40        **70:30        **80:20        **90:10
                    (n=4.565)**    (n=3.426)**    (n=2.284)**    (n=1.142)**
  -------------- -------------- -------------- -------------- --------------
        0                 0,91%          0,88%          0,74%          0,68%

        1                 1,65%          1,59%          1,35%          1,20%

        2                 2,76%          2,62%          2,27%          1,98%

        3                 3,82%          3,56%          3,10%          2,74%

        4                 5,53%          5,21%          4,50%          3,80%

        5                 9,30%          8,93%          7,67%          5,94%

        6                11,78%         11,11%          9,66%          7,44%

        7            **13,53%**     **12,95%**     **11,49%**          8,98%

        8                12,97%         12,58%         11,15%      **9,00%**

        9                 9,39%          9,03%          7,99%          6,55%

        10                7,26%          7,15%          6,46%          5,52%

        11                4,35%          4,32%          4,05%          3,60%

        12                3,18%          3,19%          3,06%          2,63%
  --------------------------------------------------------------------------

  : []{#_Toc239590209 .anchor}Tabel 4. Evaluasi layer dengan MAP
  Skenario B Wav2vec2

  --------------------------------------------------------------------------
    **Layer**           **60:40        **70:30        **80:20        **90:10
                    (n=4.565)**    (n=3.426)**    (n=2.284)**    (n=1.142)**
  -------------- -------------- -------------- -------------- --------------
        0                 1,17%          1,16%          1,01%          0,93%

        1                 2,35%          2,04%          1,75%          1,65%

        2                 4,26%          3,79%          3,24%          3,04%

        3                 7,14%          6,58%          5,37%          4,60%

        4                 9,95%          9,38%          7,92%          6,57%

        5                11,02%         10,38%          9,08%          7,40%

        6            **11,42%**     **10,87%**      **9,55%**      **7,93%**

        7                10,10%          9,53%          8,32%          6,55%

        8                 8,53%          8,10%          7,12%          5,63%

        9                 7,30%          7,01%          6,24%          5,13%

        10                7,34%          7,15%          6,31%          5,12%

        11                5,06%          4,99%          4,51%          3,63%

        12                3,12%          3,15%          2,83%          2,49%
  --------------------------------------------------------------------------

  : []{#_Toc239590210 .anchor}Tabel 4. Evaluasi layer dengan MAP
  Skenario B Data2vec

Pada Skenario B, Wav2Vec2 secara konsisten memilih layer 7 untuk tiga
rasio pertama (60:40, 70:30, 80:20) dan layer 8 untuk rasio 90:10.
Data2Vec secara konsisten memilih layer 6 untuk seluruh empat rasio.
Pola ini menunjukkan bahwa Wav2Vec2 memerlukan layer sedikit lebih
tinggi untuk mencapai performa optimal pada rasio *database* yang sangat
besar (90:10), sedangkan Data2Vec lebih stabil pada layer 6. Pada rasio
90:10, Wav2Vec2 memilih layer 8 dengan MAP development 9,00%, sedikit
lebih tinggi daripada layer 7 (8,98%).

####  Skenario C 

Skenario C menggunakan klip mahasiswa untuk kedua sisi dengan pemisahan
berdasarkan identitas mahasiswa (NIM).

  --------------------------------------------------------------------------
    **Layer**           **60:40        **70:30        **80:20        **90:10
                    (n=6.357)**    (n=4.613)**    (n=2.616)**    (n=1.062)**
  -------------- -------------- -------------- -------------- --------------
        0                 0,38%          0,35%          0,33%          0,33%

        1                 0,49%          0,47%          0,45%          0,44%

        2                 0,60%          0,57%          0,56%          0,53%

        3                 0,77%          0,73%          0,73%          0,68%

        4                 1,18%          1,12%          1,14%          1,04%

        5                 2,52%          2,40%          2,51%          2,12%

        6                 3,36%          3,20%          3,36%          2,86%

        7             **3,49%**      **3,37%**      **3,55%**      **3,03%**

        8                 2,85%          2,71%          2,83%          2,35%

        9                 2,05%          1,96%          2,02%          1,72%

        10                1,67%          1,63%          1,65%          1,42%

        11                1,04%          1,00%          1,00%          0,90%

        12                0,84%          0,80%          0,79%          0,70%
  --------------------------------------------------------------------------

  : []{#_Toc239590211 .anchor}Tabel 4. Evaluasi layer dengan MAP
  Skenario C Wav2vec2

  --------------------------------------------------------------------------
    **Layer**           **60:40        **70:30        **80:20        **90:10
                    (n=6.357)**    (n=4.613)**    (n=2.616)**    (n=1.062)**
  -------------- -------------- -------------- -------------- --------------
        0                 0,45%          0,43%          0,41%          0,38%

        1                 0,79%          0,74%          0,71%          0,66%

        2                 1,39%          1,32%          1,31%          1,10%

        3                 2,31%          2,25%          2,28%          1,81%

        4                 3,35%          3,25%          3,33%          2,58%

        5                 3,72%          3,63%          3,77%          2,87%

        6             **3,73%**      **3,65%**      **3,83%**      **3,11%**

        7                 2,95%          2,86%          2,98%          2,47%

        8                 2,43%          2,32%          2,44%          2,07%

        9                 2,16%          2,06%          2,21%          1,85%

        10                2,20%          2,09%          2,25%          1,91%

        11                1,68%          1,59%          1,67%          1,45%

        12                1,12%          1,05%          1,07%          0,93%
  --------------------------------------------------------------------------

  : []{#_Toc239590212 .anchor}Tabel 4. Evaluasi layer dengan MAP
  Skenario C Data2vec

Pada Skenario C, Wav2Vec2 secara konsisten memilih layer 7 untuk seluruh
empat rasio. Data2Vec secara konsisten memilih layer 6 untuk seluruh
empat rasio. Pola ini sangat stabil dan menunjukkan bahwa karakteristik
data mahasiswa cocok dengan representasi pada layer tengah.

####  Skenario D 

Skenario D menggunakan klip mahasiswa sebagai *query* dan *database*
gabungan (mahasiswa lain + seluruh Quran-MD). Skenario ini menguji
performa pada *database* yang lebih besar dan lebih heterogen.

  --------------------------------------------------------------------------
    **Layer**           **60:40        **70:30        **80:20        **90:10
                    (n=6.357)**    (n=4.613)**    (n=2.616)**    (n=1.062)**
  -------------- -------------- -------------- -------------- --------------
        0                 0,29%          0,28%          0,28%          0,28%

        1                 0,37%          0,36%          0,36%          0,36%

        2                 0,44%          0,44%          0,44%          0,45%

        3                 0,54%          0,53%          0,55%          0,55%

        4                 0,77%          0,77%          0,80%          0,79%

        5                 1,50%          1,49%          1,63%          1,47%

        6                 1,97%          1,96%          2,14%          1,91%

        7             **2,06%**      **2,08%**      **2,26%**      **2,00%**

        8                 1,73%          1,72%          1,85%          1,57%

        9                 1,25%          1,25%          1,33%          1,14%

        10                1,03%          1,04%          1,11%          0,95%

        11                0,65%          0,66%          0,68%          0,62%

        12                0,54%          0,53%          0,55%          0,50%
  --------------------------------------------------------------------------

  : []{#_Toc239590213 .anchor}Tabel 4. Evaluasi layer dengan MAP
  Skenario D Wav2vec2

  --------------------------------------------------------------------------
    **Layer**           **60:40        **70:30        **80:20        **90:10
                    (n=6.357)**    (n=4.613)**    (n=2.616)**    (n=1.062)**
  -------------- -------------- -------------- -------------- --------------
        0                 0,34%          0,34%          0,33%          0,33%

        1                 0,53%          0,53%          0,52%          0,51%

        2                 0,90%          0,90%          0,92%          0,83%

        3                 1,46%          1,47%          1,53%          1,30%

        4                 2,05%          2,06%          2,20%          1,78%

        5             **2,27%**      **2,31%**      **2,47%**          1,96%

        6                 2,23%          2,27%      **2,47%**      **2,10%**

        7                 1,77%          1,80%          1,92%          1,67%

        8                 1,48%          1,48%          1,59%          1,42%

        9                 1,30%          1,30%          1,42%          1,26%

        10                1,32%          1,31%          1,44%          1,29%

        11                1,01%          1,00%          1,08%          0,99%

        12                0,70%          0,68%          0,71%          0,65%
  --------------------------------------------------------------------------

  : []{#_Toc239590214 .anchor}Tabel 4. Evaluation layer dengan MAP
  Skenario D Data2vec

Pada Skenario D, Wav2Vec2 secara konsisten memilih layer 7 untuk seluruh
empat rasio. Data2Vec memilih layer 5 untuk rasio 60:40 dan 70:30, serta
layer 6 untuk rasio 80:20 dan 90:10. Pola ini menunjukkan bahwa pada
*database* gabungan yang lebih besar, Data2Vec memerlukan layer sedikit
lebih tinggi untuk rasio *database* yang lebih besar.

Pada D-80:20, Data2Vec layer 5 dan layer 6 menampilkan nilai MAP yang
sama (2,47% vs 2,47%), tetapi nilai tidak-terbulat layer 6 sedikit lebih
tinggi, sehingga layer 6 terpilih. Karena tidak terjadi seri sempurna
pada nilai tidak-terbulat, aturan seri tidak terpakai pada kasus ini.

### Ringkasan Pemilihan Layer 

Tabel berikut merangkum *layer* terpilih untuk setiap kombinasi sel dan
model, beserta MAP development yang dicapai.

  -----------------------------------------------------------------------
       **Sel**          **Model**          **Layer            **MAP Dev**
                                         Terpilih**     
  ----------------- ----------------- ----------------- -----------------
          A             Wav2Vec2              7                     1,40%

          A             Data2Vec              5                     1,58%

       B-60:40          Wav2Vec2              7                    13,53%

       B-60:40          Data2Vec              6                    11,42%

       B-70:30          Wav2Vec2              7                    12,95%

       B-70:30          Data2Vec              6                    10,87%

       B-80:20          Wav2Vec2              7                    11,49%

       B-80:20          Data2Vec              6                     9,55%

       B-90:10          Wav2Vec2              8                     9,00%

       B-90:10          Data2Vec              6                     7,93%

       C-60:40          Wav2Vec2              7                     3,49%

       C-60:40          Data2Vec              6                     3,73%

       C-70:30          Wav2Vec2              7                     3,37%

       C-70:30          Data2Vec              6                     3,65%

       C-80:20          Wav2Vec2              7                     3,55%

       C-80:20          Data2Vec              6                     3,83%

       C-90:10          Wav2Vec2              7                     3,03%

       C-90:10          Data2Vec              6                     3,11%

       D-60:40          Wav2Vec2              7                     2,06%

       D-60:40          Data2Vec              5                     2,27%

       D-70:30          Wav2Vec2              7                     2,08%

       D-70:30          Data2Vec              5                     2,31%

       D-80:20          Wav2Vec2              7                     2,26%

       D-80:20          Data2Vec              6                     2,47%

       D-90:10          Wav2Vec2              7                     2,00%

       D-90:10          Data2Vec              6                     2,10%
  -----------------------------------------------------------------------

  : []{#_Toc239590215 .anchor}Tabel 4. Ringkasan layer terpilih dan MAP
  development

Pola pemilihan layer menunjukkan konsistensi yang kuat

a.  Wav2vec2, 12 dari 13 sel memilih layer 7, hanya B-90:10 yang memilih
    layer 8

b.  Data2vec, 3 sel memilih layer 5 dan 10 sel memilih layer 6

Konsistensi pemilihan titik pada seluruh 13 sel menunjukkan pola yang
terstruktur. Wav2Vec2 memilih titik 7 pada 12 sel dan titik 8 hanya pada
B-90:10, sedangkan Data2Vec memilih titik 5 pada tiga sel (A, D-60:40,
D-70:30) dan titik 6 pada sepuluh sel sisanya. Seluruh titik terpilih
berada pada keluaran blok Transformer (titik 1 sampai 12), sedangkan
titik 0 yang merupakan proyeksi fitur awal sebelum pemrosesan
Transformer tidak terpilih pada sel mana pun. Temuan ini konsisten
dengan pandangan bahwa performa representasi berlapis bergantung pada
tugas dan domain, bukan pada kedalaman lapisan secara universal \[19\].
Khusus untuk Wav2Vec2, hasil ini sejalan dengan analisis berlapis oleh
Pasad dkk. \[35\] yang menunjukkan bahwa konten fonetik dan kata pada
wav2vec 2.0 cenderung memuncak pada satu atau lebih lapisan tengah dan
menurun pada lapisan tertinggi. Pemilihan titik 5 hingga 6 pada Data2Vec
merupakan temuan empiris penelitian ini dan tidak dapat digeneralisasi
di luar konfigurasi yang diuji. Perlu ditegaskan bahwa eksperimen ini
tidak mengukur secara langsung isi setiap titik representasi, sehingga
interpretasi mengenai kandungan informasi pada lapisan tertentu bersifat
inferensial.

![[]{#_Toc239590298 .anchor}Gambar 4. Layer representasi
terpilih](media/image19.png){width="3.6818186789151355in"
height="3.124811898512686in"}

Gambar 4.3 merangkum nomor layer yang dipilih berdasarkan MAP pada
*development set* untuk setiap sel dan model. Pola visualnya konsisten
dengan Tabel 4.15. Wav2Vec2 lebih banyak memilih layer 7, sedangkan
Data2Vec memilih layer 5 atau 6.

### Hasil Skor dan pemeringkatan

Setelah titik representasi untuk setiap kombinasi sel dan model
ditentukan pada tahap sebelumnya, selanjutnya yaitu proses skor dan
pemeringkatan hasil dari kedua model. Setiap *query* diwakili oleh satu
vektor berdimensi 768 yang berasal dari titik representasi terpilih pada
model yang bersangkutan. Vektor *query* tersebut kemudian dibandingkan
terhadap seluruh vektor kandidat pada *database* referensi menggunakan
*cosine similarity* sebagaimana didefinisikan pada persamaan 3.2. Ukuran
ini mengukur kemiripan arah antara dua vektor tanpa bergantung pada
panjang vektor, sehingga perbedaan amplitude yang mungkin timbul akibat
variasi durasi audio tidak memengaruhi perhitungan skor. Karena kedua
model telah melewati *union filtering* yang identik dan menghasilkan
*embedding* berdimensi sama, setiap *query* dinilai terhadap himpunan
kandidat yang persis sama pada Wav2vec2 dan Data2Vec. Kesetaraan ini
menjamin bahwa selisih skor dan posisi peringkat antara kedua model
murni mencerminkan perbedaan representasi, bukan perbedaan himpunan
kandidat.

Penskoran menghasilkan menghasilkan satu matriks kemiripan untuk setiap
kombinasi sel dan model, dengan dimensi sejumlah *query* pada baris dan
jumlah kandididat *database* pada kolom. Pada skenario A, setiap query
dinilai terhadap 17.127 kandidat Quran-MD yang telah dilakukan *union
filtering*. Pada scenario B, C, dan D, jumlah kandidat mengikuti ukuran
database setiap sel sebagaimana disajikan pada Tabel 4.2 sampai 4.5,
yang bervarias sesuai dengan *owner ratio* masing-masing. Seluruh nilai
pada matriks kemiripan berada pada rentang \[-1, 1\].

Berdasarkan matriks kemiripan tersebut, setiap baris diurutkan dari skor
terbesar ke terkecil sehingga menghasilkan satu daftar peringkat untuk
setiap *query*. Kandidat dengan skor tertinggi menempati peringkat
pertama dan seterusnya. Prosedur pengurutan ini bersifat deterministic
dan tidak melibatkan ambang batas skor minimum, seluruh kandidat pada
database turut diperingkatkan tanpa pemotongan. Dengan demikian,
keluaran tahap ini berupa daftar peringkat lengkap, bukan keputusan
klasifikasi benar atau salah. Pemisahan fungsi antara penskoran dan
evaluasi merupakan hal yang perlu ditegaskan. *Cosine similarity*
berperan semata-mata sebagai fungsi penilaian yang menghasilkan urutan,
sedangkan kualitas urutan diukur pada tahap evaluasi menggunakan metrik
MAP, MRR, dan Top-K. Daftar peringkat dinilai terhadap label relevansi
yang ditentukan oleh pasangan (surah, ayat), bukan oleh skor kemiripan
itu sendiri. Dengan kata lain, skor *cosine similarity* yang tinggi
belum tentu menunjukkan bahwa kandidat tersebut adalah ayat yang benar.
Kesesuaian dengan label relevansi baru dapat diketahui setelah metrik
evaluasi dihitung.

##  Hasil Evaluation dan Pembahasan

Bagian ini menyajikan hasil evaluasi akhir pada *test set* menggunakan
layer yang telah dipilih pada *development set.* Hasil MAP test akan
berbeda dengan MAP development yang sebelumnya sudah dijelaskan. MAP
development digunakan untuk pemilihan layer, sedangkan MAP test
digunakan untuk evaluasi performa final hasil dari tiap-tiap model.

Tabel 4.16 menyajikan hasil final pada *test set* untuk seluruh 13 sel.
Dipecah menjadi lima tabel agar lebih memudahkan untuk dibaca. Tabel
Seluruh nilai dinyatakan dalam persen. Setiap baris memakai titik yang
telah dipilih pada development set, sehingga tidak ada pencarian titik
tambahan pada data pengujian.

  ------------------------------------------------------------------------------------------
    Sel      Model     Layer           MAP         MRR       Top-1        Top-5       Top-10
  ------- ----------- -------- ----------- ----------- ----------- ------------ ------------
     A     Wav2Vec2      7           1,36%       8,92%       5,26%       11,34%       15,81%

     A     Data2Vec      5       **1,61%**   **9,01%**   **5,37%**   **11,66%**   **15,87%**
  ------------------------------------------------------------------------------------------

  : []{#_Toc239590216 .anchor}Tabel 4. Hasil Evaluasi Skenario A

Skenario A merupakan kondisi lintas sumber dataset, sehingga ditampilkan
terpisah dari skenario B, C, dan D yang memiliki variasi *owner ratio*.
Pada Skenario A, Data2Vec menunjukkan kinerja yang sedikit lebih tinggi
daripada Wav2Vec2 pada seluruh metrik. Data2Vec memperoleh MAP sebesar
1,61%, MRR 9,01%, Top-1 5,37%, Top-5 11,66%, dan Top-10 15,87%,
sedangkan Wav2Vec2 masing-masing memperoleh 1,36%, 8,92%, 5,26%, 11,34%,
dan 15,81%. Selisih terbesar terdapat pada MAP, yaitu 0,25 poin
persentase, sedangkan pada metrik lainnya perbedaannya kurang dari 0,32
poin persentase.

  --------------------------------------------------------------------------------------------------
  **Sel**   **Model**    **Layer**       **MAP**      **MRR**    **Top-1**    **Top-5**   **Top-10**
  --------- ----------- ----------- ------------ ------------ ------------ ------------ ------------
  B-60:40   Wav2Vec2         7        **13,78%**   **61,53%**   **52,67%**   **72,42%**   **78,28%**

  B-60:40   Data2Vec         6            11,42%       55,13%       45,36%       66,86%       73,86%

  C-60:40   Wav2Vec2         7             3,39%   **27,24%**   **19,91%**   **34,50%**   **41,43%**

  C-60:40   Data2Vec         6         **3,46%**       25,59%       18,39%       32,65%       39,15%

  D-60:40   Wav2Vec2         7             2,03%   **27,21%**   **19,81%**   **34,47%**   **41,73%**

  D-60:40   Data2Vec         5         **2,18%**       26,09%       18,72%       33,28%       40,28%
  --------------------------------------------------------------------------------------------------

  : []{#_Toc239590217 .anchor}Tabel 4. Hasil Evaluasi *Owner Ratio*
  60:40 pada Skenario B, C, dan D

Pada rasio 60:40, Wav2Vec2 unggul pada seluruh metrik di Skenario B,
dengan MAP sebesar 13,78% dibandingkan 11,42% pada Data2Vec. Sebaliknya,
pada Skenario C dan D, Data2Vec memperoleh MAP yang sedikit lebih
tinggi, masing-masing sebesar 3,46% dan 2,18%, dibandingkan 3,39% dan
2,03% pada Wav2Vec2. Namun, keunggulan tersebut hanya terdapat pada MAP.
Pada MRR dan seluruh metrik Top-K, Wav2Vec2 tetap memperoleh nilai yang
lebih tinggi daripada Data2Vec pada kedua skenario tersebut.

  --------------------------------------------------------------------------------------------------
  **Sel**   **Model**    **Layer**       **MAP**      **MRR**    **Top-1**    **Top-5**   **Top-10**
  --------- ----------- ----------- ------------ ------------ ------------ ------------ ------------
  B-70:30   Wav2Vec2         7        **12,59%**   **58,74%**   **49,56%**   **69,12%**   **75,66%**

  B-70:30   Data2Vec         6            10,45%       53,43%       44,54%       63,16%       70,05%

  C-70:30   Wav2Vec2         7             3,35%   **28,66%**   **21,16%**   **36,40%**       43,03%

  C-70:30   Data2Vec         6         **3,54%**       27,83%       20,03%       35,64%   **43,62%**

  D-70:30   Wav2Vec2         7             2,08%   **28,77%**   **21,33%**   **35,98%**   **43,16%**

  D-70:30   Data2Vec         5         **2,24%**       26,63%       18,93%       33,59%       41,65%
  --------------------------------------------------------------------------------------------------

  : []{#_Toc239590218 .anchor}Tabel 4. Hasil Evaluasi *owner ratio*
  70:30 pada skenario B, C, dan D

Pada rasio 70:30, Wav2Vec2 kembali unggul pada seluruh metrik di
Skenario B, dengan MAP sebesar 12,59% dibandingkan 10,45% pada Data2Vec.
Pada Skenario C dan D, Data2Vec memperoleh MAP yang sedikit lebih
tinggi, masing-masing sebesar 3,54% dan 2,24%, dibandingkan 3,35% dan
2,08% pada Wav2Vec2. Pada Skenario C, Wav2Vec2 memperoleh nilai lebih
tinggi pada MRR, Top-1, dan Top-5, sedangkan Data2Vec sedikit lebih
tinggi pada Top-10. Pada Skenario D, Wav2Vec2 kembali lebih tinggi pada
MRR dan seluruh metrik Top-K.

  --------------------------------------------------------------------------------------------------
  **Sel**   **Model**    **Layer**       **MAP**      **MRR**    **Top-1**    **Top-5**   **Top-10**
  --------- ----------- ----------- ------------ ------------ ------------ ------------ ------------
  B-80:20   Wav2Vec2         7        **11,36%**   **56,76%**   **48,07%**   **67,08%**   **72,42%**

  B-80:20   Data2Vec         6             9,45%       53,50%       44,75%       63,49%       69,18%

  C-80:20   Wav2Vec2         7             3,53%   **31,71%**   **23,97%**   **39,75%**   **47,40%**

  C-80:20   Data2Vec         6         **3,70%**       29,49%       21,97%       38,08%       44,27%

  D-80:20   Wav2Vec2         7             2,25%   **32,03%**   **24,23%**   **40,15%**   **47,07%**

  D-80:20   Data2Vec         6         **2,38%**       29,39%       21,90%       37,08%       43,81%
  --------------------------------------------------------------------------------------------------

  : []{#_Toc239590219 .anchor}Tabel 4. Hasil Evaluasi *owner ration*
  80:20 pada skenario B, C, dan D

Pada rasio 80:20, Wav2Vec2 tetap unggul pada seluruh metrik di Skenario
B, dengan MAP sebesar 11,36% dibandingkan 9,45% pada Data2Vec. Pada
Skenario C, Data2Vec memperoleh MAP sebesar 3,70%, sedikit lebih tinggi
dibandingkan 3,53% pada Wav2Vec2, sedangkan Wav2Vec2 lebih tinggi pada
MRR dan seluruh metrik Top-K. Perbedaan MAP tersebut belum signifikan
berdasarkan uji bootstrap. Pada Skenario D, Data2Vec juga memperoleh MAP
yang lebih tinggi, yaitu 2,38% dibandingkan 2,25% pada Wav2Vec2,
sedangkan Wav2Vec2 kembali lebih tinggi pada MRR dan seluruh metrik
Top-K.

  -------------------------------------------------------------------------------------------------
  **Sel**   **Model**   **Layer**       **MAP**      **MRR**    **Top-1**    **Top-5**   **Top-10**
  --------- ----------- ----------- ----------- ------------ ------------ ------------ ------------
  B-90:10   Wav2Vec2    8             **9,75%**   **52,92%**   **46,41%**   **60,95%**   **63,57%**

  B-90:10   Data2Vec    6                 8,45%       48,84%       41,33%       58,32%       62,00%

  C-90:10   Wav2Vec2    7                 3,16%       28,25%       19,46%       36,32%       45,53%

  C-90:10   Data2Vec    6             **3,57%**   **29,15%**   **20,36%**   **38,91%**   **45,91%**

  D-90:10   Wav2Vec2    7                 2,16%       28,40%       19,71%       36,58%       44,88%

  D-90:10   Data2Vec    6             **2,41%**   **28,67%**   **20,23%**   **38,13%**   **46,30%**
  -------------------------------------------------------------------------------------------------

  : []{#_Toc239590220 .anchor}Tabel 4. Hasil Evaluasi owner ration 90:10
  pada skenario B, C, dan D

Pada rasio 90:10, Wav2Vec2 unggul pada seluruh metrik di Skenario B,
dengan MAP sebesar 9,75% dibandingkan 8,45% pada Data2Vec. Sebaliknya,
pada Skenario C, Data2Vec unggul pada seluruh metrik, dengan MAP sebesar
3,57% dibandingkan 3,16% pada Wav2Vec2, serta MRR dan seluruh metrik
Top-K yang juga lebih tinggi. Pola yang sama terjadi pada Skenario D,
dengan Data2Vec memperoleh MAP sebesar 2,41% dibandingkan 2,16% pada
Wav2Vec2, serta nilai MRR dan seluruh metrik Top-K yang lebih tinggi.

Ditinjau dari beberapa lintas rasio, Wav2Vec2 unggul pada seluruh sel B
dan seluruh metrik. MAP Wav2Vec2 menurun dari 13,78% pada rasio 60:40
menjadi 9,75% pada rasio 90:10. Perubahan ini berasosiasi dengan
bertambahnya ukuran *database* dan berkurangnya jumlah *query*, tetapi
eksperimen tidak mengisolasi pengaruh masing-masing faktor secara
kausal. Walaupun identitas qari dipisahkan, sisi *query* dan *database*
sama-sama berasal dari Quran-MD sehingga ketidakcocokan domain lebih
kecil daripada pada Skenario A.

Pada Skenario C, Data2Vec memiliki MAP numerik lebih tinggi pada keempat
rasio, tetapi hanya C-70:30 dan C-90:10 yang menunjukkan perbedaan
signifikan. Pada Skenario D, Data2Vec juga memiliki MAP lebih tinggi
pada keempat rasio dan seluruh selisihnya signifikan. Nilai MAP absolut
pada D lebih rendah daripada B dan berada pada rentang yang sebanding
dengan C. *Database* D yang lebih besar karena penambahan seluruh
Quran-MD berasosiasi dengan tantangan retrieval yang lebih besar, tetapi
hubungan ini bukan bukti kausal karena komposisi *database* dan ukuran
kandidat berubah secara bersamaan. Pada beberapa sel C dan D, Wav2Vec2
lebih tinggi pada MRR dan Top-K meskipun MAP Data2Vec lebih tinggi. Pola
tersebut menunjukkan bahwa MAP, MRR, dan Top-K menangkap aspek
pemeringkatan yang berbeda dan tidak selalu bergerak dalam arah yang
sama.

![[]{#_Toc239590299 .anchor}Gambar 4. Perbandingan MAP Final per
Sel](media/image20.png){width="5.508333333333334in"
height="4.399305555555555in"}

Gambar 4.4 memperlihatkan MAP pada test set untuk dua model di seluruh
13 sel. Tanda bintang menunjukkan perbedaan signifikan berdasarkan
bootstrap berpasangan, sedangkan n.s. menunjukkan bahwa perbedaan belum
signifikan. Visualisasi ini memperlihatkan pola utama evaluasi: Wav2Vec2
lebih tinggi pada seluruh sel B, sementara Data2Vec memiliki MAP numerik
lebih tinggi pada A serta seluruh sel C dan D.

Tidak ada pemenang universal. Data2Vec memiliki MAP numerik lebih tinggi
pada 9 dari 13 sel (A, keempat C, keempat D). Wav2Vec2 memiliki MAP
numerik lebih tinggi pada 4 sel (keempat B). Namun, dua sel C (C-60:40
dan C-80:20) tidak menunjukkan perbedaan signifikan. Pola ini
menunjukkan bahwa keunggulan model bergantung pada kondisi evaluasi yang
didefinisikan oleh skenario dan rasio pemilik, bukan pada arsitektur
secara universal. Literatur yang ada tentang model audio pralatih tidak
menetapkan pemenang untuk korpus Al-Qur\'an. Temuan ini konsisten dengan
pandangan bahwa performa layerwise bergantung pada tugas dan domain,
sebagaimana ditunjukkan oleh Yang dkk. (2024) secara lintas model dan
oleh Pasad dkk. (2023) khusus untuk Wav2Vec2.

![[]{#_Toc239590300 .anchor}Gambar 4. Tren MAP final terhadap *owner
ration* *database*](media/image21.png){width="5.508333333333334in"
height="5.952938538932633in"}

Gambar 4.5 menampilkan MAP *test set* sebagai fungsi *owner ration
database* untuk skenario B, C, dan D. Skenario A tidak disertakan karena
tidak memiliki variasi *owner ratio*. Pada Skenario B, MAP tertinggi
kedua model tercapai pada rasio 60:40, yaitu rasio dengan *database*
terkecil. Pada Skenario C, MAP tertinggi kedua model tercapai pada rasio
80:20. Pada Skenario D, Wav2Vec2 mencapai MAP tertinggi pada rasio
80:20, sedangkan Data2Vec mencapai MAP tertinggi pada rasio 90:10.

Pola rasio ini bersifat deskriptif dan tidak boleh ditafsirkan sebagai
hubungan kausal. Perubahan rasio pemilik secara bersamaan mengubah
komposisi *query*, komposisi *database*, dan jumlah \_test set\_ (Tabel
4.2 dan Tabel 4.3), sehingga kenaikan atau penurunan MAP pada rasio
tertentu tidak dapat diisolasi sebagai pengaruh rasio semata. Istilah
rasio terbaik pada Gambar 4.5 terbatas pada MAP. MRR dan Top-K dapat
mencapai nilai tertinggi pada rasio yang berbeda, terutama pada Data2Vec
di skenario C dan D, sehingga pemilihan rasio operasional tetap harus
mengikuti metrik prioritas sistem.

![[]{#_Toc239590301 .anchor}Gambar 4. Ringkasan rasio terbaik per
metrik](media/image22.png){width="5.508333333333334in"
height="4.120398075240595in"}

Gambar 4.6 merangkum rasio yang menghasilkan nilai tertinggi untuk MAP,
MRR, Top-1, Top-5, dan Top-10 pada setiap pasangan skenario-model. Rasio
60:40 menjadi yang tertinggi pada seluruh metrik di Skenario B untuk
kedua model. Pada Wav2Vec2, rasio 80:20 juga menjadi yang tertinggi pada
seluruh metrik di Skenario C dan D. Pola Data2Vec lebih beragam: pada
Skenario C, rasio 80:20 tertinggi untuk MAP, MRR, dan Top-1, sedangkan
90:10 tertinggi untuk Top-5 dan Top-10; pada Skenario D, rasio 80:20
tertinggi untuk MRR dan Top-1, sedangkan 90:10 tertinggi untuk MAP,
Top-5, dan Top-10.

Rasio 70:30 tidak muncul sebagai nilai maksimum pada metrik dan pasangan
skenario-model mana pun, sehingga tidak ditampilkan sebagai kategori
warna pada heatmap. Ketiadaan tersebut berarti rasio itu tidak menjadi
yang tertinggi dalam kombinasi yang diuji, bukan berarti rasio 70:30
tidak memiliki hasil evaluasi.

Ringkasan tersebut dibentuk secara deskriptif dengan mengambil nilai
maksimum setiap metrik pada *test set*. Penanda signifikansi tidak
diterapkan pada MRR dan Top-K karena uji *bootstrap* penelitian hanya
dilakukan terhadap selisih MAP atau AP per *query*. Oleh sebab itu,
perbedaan rasio terbaik pada metrik selain MAP tidak boleh dibaca
sebagai perbedaan yang telah terbukti secara inferensial.

### Perbandingan bootstrap per sel 

Uji bootstrap memakai selisih AP Data2Vec dikurangi Wav2Vec2. Tabel 4.5
menyajikan selisih rata-rata dalam poin persentase, interval kepercayaan
95%, dan jumlah kemenangan AP per *query*. Jumlah kemenangan tidak
menentukan signifikansi secara mandiri, sebab besarnya selisih AP pada
setiap *query* juga memengaruhi rata-rata dan interval.

  ----------------------------------------------------------------------------------------
  **Skenario**   **Selisih    **Interval    **Kemenangan   **Kemenangan   **Interpretasi
                 MAP Data2Vec kepercayaan   Data2Vec**     Wav2Vec2**     MAP**
                 dikurangi    95%**                                       
                 Wav2Vec2**                                               
  -------------- ------------ ------------- -------------- -------------- ----------------
  A              +0,17 poin   \[+0,10; +    4.078          3.912          Data2Vec lebih
                 persentase   0,24\]                                      tinggi

  B              -1,51 poin   \[-1,97; -    671            1.042          Wav2Vec2 lebih
                 persentase   1,04\]                                      tinggi

  C              +0,32 poin   \[+0,18; +    1.269          1.340          Data2Vec lebih
                 persentase   0,47\]                                      tinggi
  ----------------------------------------------------------------------------------------

  : []{#_Toc236984574 .anchor}Tabel 4. Perbandingan *bootstrap* selisih
  MAP

Ketiga interval kepercayaan tidak mencakup nol. Dengan aturan inferensi
yang ditetapkan, perbedaan MAP mendapat dukungan statistik pada ketiga
skenario. Arah perbedaannya positif untuk A dan C sehingga mendukung MAP
Data2Vec yang lebih tinggi, sedangkan arah negatif pada B mendukung MAP
Wav2Vec2 yang lebih tinggi.

Makna statistik tersebut terbatas secara tegas pada selisih MAP yang
dibentuk dari AP per *query*. Penelitian tidak melakukan uji inferensial
untuk MRR, Top-1, Top-5, atau Top-10. Perbedaan pada metrik tersebut
hanya boleh dibahas secara deskriptif. Signifikansi juga tidak sama
dengan kepentingan praktis, terutama pada A yang memiliki selisih MAP
kecil dan kinerja absolut rendah.

##  Deployment Konseptual

Hasil eksperimen mendukung rancangan retrieval sebagai proses dua tahap,
yaitu penyiapan indeks secara luring dan pencarian secara daring.
Rancangan ini belum diimplementasikan atau diuji sebagai sistem
produksi. Gambar 4.4 menampilkan batas tersebut secara eksplisit.

![[]{#_Toc236984561 .anchor}Gambar 4. Rancangan konseptual deployment
*retrieval*](media/image23.png){width="3.0015594925634295in"
height="3.3633213035870515in"}

Pada tahap persiapan audio referensi, seluruh audio dinormalisasi dan
diekstraksi satu kali menggunakan model serta titik representasi yang
dipilih untuk domain sasaran. Embedding yang dihasilkan kemudian
disimpan bersama *metadata* surah, ayat, dan sumber audio sebagai basis
pencarian. Ketika terdapat *query* baru, audio *query* dinormalisasi dan
diekstraksi menggunakan model serta titik representasi yang sama dengan
audio referensi. Embedding *query* kemudian dibandingkan dengan seluruh
embedding referensi menggunakan cosine similarity untuk menghasilkan
daftar kandidat berperingkat. Tahap verifikasi tambahan ditempatkan
setelah pengambilan kandidat Top-K karena hasil eksperimen, khususnya
pada skenario A, belum menunjukkan ketepatan yang memadai untuk
menentukan satu ayat secara otomatis.

Pemilihan konfigurasi bergantung pada karakteristik data dan tujuan
pencarian. Untuk *query* dan audio referensi yang sama-sama berasal dari
Quran-MD, terutama ketika prioritas sistem adalah menempatkan hasil
relevan pada peringkat awal, Wav2Vec2 titik 7 memberikan hasil terkuat
dalam eksperimen ini. Sementara itu, berdasarkan metrik MAP, Data2Vec
titik 5 memberikan hasil lebih tinggi ketika rekaman mahasiswa digunakan
sebagai *query* terhadap Quran-MD maupun ketika *query* dan audio
referensi berasal dari mahasiswa yang berbeda. Rekomendasi tersebut
hanya berlaku pada data dan skenario yang diuji. Konfigurasi yang sama
tidak dapat langsung dianggap optimal untuk korpus lain tanpa melalui
evaluasi pada himpunan pengembangan yang sesuai.

Kinerja pada skenario A menjadi keterbatasan utama dalam penerapan
sistem. Nilai Top-1 tertinggi hanya mencapai 5,38%, sedangkan Top-10
tertinggi mencapai 15,77%. Hal ini menunjukkan bahwa, ketika rekaman
mahasiswa digunakan sebagai *query* terhadap audio referensi Quran-MD,
sistem lebih tepat digunakan untuk menghasilkan sejumlah kandidat ayat
yang selanjutnya diperiksa kembali, bukan untuk menentukan satu ayat
secara otomatis. Skenario B menunjukkan hasil yang lebih baik karena
*query* dan audio referensi sama-sama berasal dari Quran-MD. Meskipun
demikian, nilai Top-1 tertinggi sebesar 50,85% menunjukkan bahwa hasil
peringkat pertama belum selalu benar. Oleh karena itu, verifikasi
tambahan tetap diperlukan pada penggunaan yang menuntut tingkat
ketepatan tinggi.

Sebelum sistem diterapkan pada lingkungan produksi, penelitian lanjutan
perlu menguji ketepatan batas ayat melalui pemeriksaan manual, kemampuan
model menghadapi perbedaan karakteristik data, alternatif strategi
temporal pooling, efisiensi pencarian pada indeks embedding, waktu
respons, skalabilitas, keamanan data, dan mekanisme pemantauan sistem.
Aspek-aspek tersebut berada di luar ruang lingkup penelitian ini. Oleh
karena itu, kontribusi pada fase ini dibatasi pada rancangan teknis
pencarian yang konsisten dengan hasil eksperimen serta identifikasi
terhadap berbagai persyaratan yang masih perlu diuji sebelum sistem
dapat digunakan secara operasional.

# BAB V KESIMPULAN DAN SARAN

##  Kesimpulan 

Wav2Vec2 dan Data2Vec berhasil diimplementasikan sebagai pengekstrak
representasi laten dalam keadaan *frozen* untuk tugas retrieval audio
ayat Al-Qur\'an tanpa melalui tahap transkripsi. Implementasi mencakup
persiapan dan segmentasi audio, ekstraksi representasi dari beberapa
lapisan, pembentukan vektor embedding per ayat, serta pemeringkatan
kandidat berdasarkan kemiripan representasi. Pemisahan data berdasarkan
pemilik, pemilihan konfigurasi pada *development set*, dan pengujian
pada *test set* yang terpisah memungkinkan proses evaluasi dilakukan
tanpa menggunakan data pengujian untuk menentukan konfigurasi. Kedua
model mampu membentuk peringkat kandidat yang lebih baik daripada
peringkat acak. Dengan demikian, representasi akustik dari model
*pretrained* dapat dimanfaatkan sebagai dasar sistem pencarian ayat
berbasis kemiripan audio.

Kinerja Wav2Vec2 dan Data2Vec bergantung pada kondisi retrieval, sumber
data, lapisan representasi, komposisi *database*, dan metrik evaluasi.
Wav2Vec2 menunjukkan keunggulan yang konsisten ketika *query* dan
*database* referensi sama-sama berasal dari Quran-MD dengan qari yang
dipisahkan. Data2Vec cenderung lebih baik berdasarkan MAP ketika *query*
berasal dari rekaman mahasiswa, baik pada kondisi lintas sumber,
antar-mahasiswa, maupun *database* referensi gabungan, meskipun pada
sebagian kondisi antar-mahasiswa perbedaannya belum cukup kuat untuk
membedakan kedua model secara statistik. Arah MRR dan Top-K juga tidak
selalu sejalan dengan MAP, sehingga tidak terdapat satu model yang dapat
dinyatakan paling baik untuk seluruh kondisi dan metrik. Representasi
yang paling sesuai bagi kedua model berasal dari lapisan tengah, bukan
lapisan terakhir. Perubahan *owner ratio* memperlihatkan pola hasil yang
berbeda menurut skenario, model, dan metrik, sehingga tidak terdapat
satu rasio yang dapat direkomendasikan sebagai pilihan terbaik secara
umum. Perbandingan antar rasio hanya bersifat deskriptif karena
perubahan rasio juga mengubah komposisi *query*, *database* referensi,
dan data pengujian. Secara keseluruhan, sistem telah menunjukkan
kemampuan dasar untuk melakukan retrieval ayat, tetapi kinerjanya pada
kondisi lintas sumber dan *database* gabungan masih memerlukan
pengembangan serta verifikasi tambahan sebelum diterapkan pada
penggunaan yang menuntut ketepatan tinggi.

##  Saran

Penelitian selanjutnya disarankan memusatkan pengembangan pada adaptasi
model terhadap domain bacaan Al-Qur'an. Wav2vec2 dan Data2vec dapat
diuji setelah melalui *fine-tuning* pada korpus audio Al-Qur'an dan
dikombinasikan dengan *metric learning*. Pengembangan ini merupakan
tahap lanjutan dari penggunaan frozen embedding karena model tidak hanya
mengekstraksi representasi bawaan, tetapi juga mempelajari kedekatan
antara bacaan dari ayat ayng sama dan perbedaan antara bacaan dari ayat
yang berlainan.

Cakupan korpus juga perlu diperluas dengan melibatkan seluruh surah
Al-Qur'an dan pembaca yang lebih beragam. Variasi tersebut dapat
mencakup asal daerah, kemampuan memabca, perangkat perekaman, dan
kondisi lingkungan yang berbeda. Perluasan korpus diperlukan untuk
menguji kemampuan generalisasi model sekaligus memastikan bahwa hasil
perbandingan tidak hanya berlaku pada cakupan surah dan karakteristik
pembaca dalam penelitian ini.

Selain adaptasi model dan perluasan korpus, penelitian berikutnya dapat
mengembangkan pembentukan representasi per ayat dengan membandingkan
*mean pooling* terhadap metode agregasi lain, termasuk penggabungan
representasi dari beberapa lapisan. Pengembangan tersebut sebaiknya
dilengkapi dengan analisis kesalah secara kualitatif terhadap *query*
yang memperoleh peringkat rendah atau tertukar dengan ayat lain.
Analisis ini dapat membantu mengidentifikasi apakah kegagalan retrieval
terutama berkaitan dengan kualitas segmentasi, variasi pembaca, kondisi
audio, atau kemiripan fonteik antar ayat, sehingga perbaikan sistem
dapat dilakuakn secara lebih terarah.

# DAFTAR PUSTAKA

\[1\] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A
Framework for Self-Supervised Learning of Speech Representations," Oct.
22, 2020, *arXiv*: arXiv:2006.11477. doi: 10.48550/arXiv.2006.11477.

\[2\] P. C. English, J. D. Kelleher, and J. Carson-Berndsen, "Searching
for Structure: Appraising the Organisation of Speech Features in wav2vec
2.0 Embeddings," in *Interspeech 2024*, ISCA, Sep. 2024, pp. 4613--4617.
doi: 10.21437/Interspeech.2024-2047.

\[3\] M. Lataifeh and A. Elnagar, "Ar-DAD: Arabic diversified audio
dataset," *Data Brief*, vol. 33, p. 106503, Dec. 2020, doi:
10.1016/j.dib.2020.106503.

\[4\] M. U. Salman, M. A. Qazi, and M. T. Alam, "Quran-MD: A
Fine-Grained Multilingual Multimodal Dataset of the Quran," Jan. 25,
2026, *arXiv*: arXiv:2601.17880. doi: 10.48550/arXiv.2601.17880.

\[5\] A. M. A. Alqadasi *et al.*, "Rule-Based Embedded HMMs Phoneme
Classification to Improve Qur'anic Recitation Recognition,"
*Electronics*, vol. 12, no. 1, p. 176, Dec. 2022, doi:
10.3390/electronics12010176.

\[6\] H. Toyin, A. Djanibekov, A. Kulkarni, and H. Aldarmaki, "ArTST:
Arabic Text and Speech Transformer," in *Proceedings of ArabicNLP 2023*,
Singapore (Hybrid): Association for Computational Linguistics, 2023, pp.
41--51. doi: 10.18653/v1/2023.arabicnlp-1.5.

\[7\] S. S. Li, B. Xu, X. Zhang, H. Liu, W. Chao, and L. P. Garcia, "A
Quantitative Approach to Understand Self-Supervised Models as
Cross-lingual Feature Extractors," 2023.

\[8\] A. Baevski, W.-N. Hsu, Q. Xu, A. Babu, J. Gu, and M. Auli,
"data2vec: A General Framework for Self-supervised Learning in Speech,
Vision and Language," Oct. 25, 2022, *arXiv*: arXiv:2202.03555. doi:
10.48550/arXiv.2202.03555.

\[9\] H. Xue, Q. Shao, P. Chen, P. Guo, L. Xie, and J. Liu, "TranUSR:
Phoneme-to-word Transcoder Based Unified Speech Representation Learning
for Cross-lingual Speech Recognition," in *INTERSPEECH 2023*, Aug. 2023,
pp. 216--220. doi: 10.21437/Interspeech.2023-746.

\[10\] W. Öberg, "Query-by-Example Audio Search using Acoustic Word
Embeddings: Transforming wav2vec 2.0 Embeddings using Contrastive
Learning," 2025.

\[11\] M. Shanbhogue *et al.*, "Gemini Embedding 2: A Native Multimodal
Embedding Model from Gemini," May 26, 2026, *arXiv*: arXiv:2605.27295.
doi: 10.48550/arXiv.2605.27295.

\[12\] Y. Chen *et al.*, "WavRAG: Audio-Integrated Retrieval Augmented
Generation for Spoken Dialogue Models," Feb. 20, 2025, *arXiv*:
arXiv:2502.14727. doi: 10.48550/arXiv.2502.14727.

\[13\] L. Alkanhal, A. Alessa, E. Almahmoud, and R. Alaqil, "Aswat:
Arabic Audio Dataset for Automatic Speech Recognition Using
Speech-Representation Learning," in *Proceedings of ArabicNLP 2023*,
Singapore (Hybrid): Association for Computational Linguistics, 2023, pp.
120--127. doi: 10.18653/v1/2023.arabicnlp-1.10.

\[14\] N. San *et al.*, "Leveraging pre-trained representations to
improve access to untranscribed speech from endangered languages," Sep.
14, 2021, *arXiv*: arXiv:2103.14583. doi: 10.48550/arXiv.2103.14583.

\[15\] Z. Li, L. Wu, T. Li, and Y. Yan, "Improves Neural Acoustic Word
Embeddings Query by Example Spoken Term Detection with Wav2vec
Pretraining and Circle Loss," in *2021 12th International Symposium on
Chinese Spoken Language Processing (ISCSLP)*, Hong Kong: IEEE, Jan.
2021, pp. 1--5. doi: 10.1109/ISCSLP49672.2021.9362065.

\[16\] Y. Xie, Z. Zhang, and Y. Yang, "Siamese Network with wav2vec
Feature for Spoofing Speech Detection," in *Interspeech 2021*, ISCA,
Aug. 2021, pp. 4269--4273. doi: 10.21437/Interspeech.2021-847.

\[17\] C. Tang, Y. Wang, X. Chen, and W.-Q. Zhang, "Exploring Effective
Fusion Algorithms for Speech Based Self-Supervised Learning Models,"
Dec. 20, 2022, *arXiv*: arXiv:2212.10092. doi:
10.48550/arXiv.2212.10092.

\[18\] J. W. Yoon, S. M. Kim, and N. S. Kim, "MCR-Data2vec 2.0:
Improving Self-supervised Speech Pre-training via Model-level
Consistency Regularization," Jun. 14, 2023, *arXiv*: arXiv:2306.08463.
doi: 10.48550/arXiv.2306.08463.

\[19\] S. Yang *et al.*, "A Large-Scale Evaluation of Speech Foundation
Models," May 29, 2024, *arXiv*: arXiv:2404.09385. doi:
10.48550/arXiv.2404.09385.

\[20\] J. Zhang *et al.*, "X-ARES: A Comprehensive Framework for
Assessing Audio Encoder Performance," 2025, *arXiv*. doi:
10.48550/ARXIV.2505.16369.

\[21\] J. Zhang *et al.*, "The ICME 2025 Audio Encoder Capability
Challenge," 2025, *arXiv*. doi: 10.48550/ARXIV.2501.15302.

\[22\] A. Singh, K. Demuynck, and V. Arora, "BEST-STD: Bidirectional
Mamba-Enhanced Speech Tokenization for Spoken Term Detection," 2024,
*arXiv*. doi: 10.48550/ARXIV.2411.14100.

\[23\] A. Singh, K. Demuynck, and V. Arora, "Language-Agnostic Speech
Tokenizer for Spoken Term Detection with Efficient Retrieval," in
*Interspeech 2025*, ISCA, Aug. 2025, pp. 2630--2634. doi:
10.21437/Interspeech.2025-2722.

\[24\] A. E. Assadi *et al.*, "MAEB: Massive Audio Embedding Benchmark,"
2026, *arXiv*. doi: 10.48550/ARXIV.2602.16008.

\[25\] S. J. Russell, S. Russell, and P. Norvig, *Artificial
Intelligence: A Modern Approach*. in Pearson series in artificial
intelligence. Pearson, 2020. \[Online\]. Available:
https://books.google.co.id/books?id=koFptAEACAAJ

\[26\] T. M. Mitchell, *Machine learning*, Nachdr. in McGraw-Hill series
in Computer Science. New York: McGraw-Hill, 2013.

\[27\] I. Goodfellow, Y. Bengio, and A. Courville, *Deep learning*. in
Adaptive computation and machine learning. Cambridge, Mass: The MIT
press, 2016.

\[28\] A. Vaswani *et al.*, "Attention Is All You Need," Aug. 02, 2023,
*arXiv*: arXiv:1706.03762. doi: 10.48550/arXiv.1706.03762.

\[29\] C. D. Manning, P. Raghavan, and H. Schütze, *Introduction to
Information Retrieval*, 1st ed. Cambridge University Press, 2008. doi:
10.1017/CBO9780511809071.

\[30\] T. Wang and P. Isola, "Understanding Contrastive Representation
Learning through Alignment and Uniformity on the Hypersphere," Aug. 15,
2022, *arXiv*: arXiv:2005.10242. doi: 10.48550/arXiv.2005.10242.

\[31\] D. Ulyanov, A. Vedaldi, and V. Lempitsky, "Instance
Normalization: The Missing Ingredient for Fast Stylization," Nov. 06,
2017, *arXiv*: arXiv:1607.08022. doi: 10.48550/arXiv.1607.08022.

\[32\] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer Normalization,"
Jul. 21, 2016, *arXiv*: arXiv:1607.06450. doi:
10.48550/arXiv.1607.06450.

\[33\] E. Schubert, "A Triangle Inequality for Cosine Similarity," vol.
13058, 2021, pp. 32--44. doi: 10.1007/978-3-030-89657-7_3.

\[34\] C. Schröer, F. Kruse, and J. M. Gómez, "A Systematic Literature
Review on Applying CRISP-DM Process Model," *Procedia Comput. Sci.*,
vol. 181, pp. 526--534, 2021, doi: 10.1016/j.procs.2021.01.199.

\[35\] A. Pasad, B. Shi, and K. Livescu, "Comparative Layer-Wise
Analysis of Self-Supervised Speech Models," in *ICASSP 2023 - 2023 IEEE
International Conference on Acoustics, Speech and Signal Processing
(ICASSP)*, Rhodes Island, Greece: IEEE, Jun. 2023, pp. 1--5. doi:
10.1109/ICASSP49357.2023.10096149.

\[36\] M. Yang, R. C. M. C. Shekar, O. Kang, and J. H. L. Hansen, "What
Can an Accent Identifier Learn? Probing Phonetic and Prosodic
Information in a Wav2vec2-based Accent Identification Model," 2023,
*arXiv*. doi: 10.48550/ARXIV.2306.06524.

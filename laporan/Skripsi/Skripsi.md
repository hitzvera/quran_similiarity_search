**=PERBANDINGAN WAV2VEC2 DAN DAN DATA2VEC TERHADAP FITUR LATEN AUDIO
UNTUK TILAWAH\
AL-QURAN**

**TUGAS AKHIR**

Diajukan sebagai salah satu syarat untuk memnperoleh gelar Sarjana
Teknik\
pada jurusan Teknik Informatika Fakultas Sains dan Teknologi\
Universitas Islam Negeri Sunan Gunung Djati Bandung

**Oleh\
MUJAHID ANSORI MAJID\
1197050093~­~**

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
> **2026**

# DAFTAR ISI

#  {#section .TOC-Heading}

[DAFTAR ISI [i](#daftar-isi)](#daftar-isi)

[DAFTAR GAMBAR [ii](#daftar-gambar)](#daftar-gambar)

[DAFTAR TABEL [iii](#daftar-tabel)](#daftar-tabel)

[PENDAHULUAN [1](#bab-i-pendahuluan)](#bab-i-pendahuluan)

[1.1 Latar Belakang [1](#latar-belakang)](#latar-belakang)

[1.2 Rumusan Masalah [2](#rumusan-masalah)](#rumusan-masalah)

[1.3 Tujuan Penelitian [2](#tujuan-penelitian)](#tujuan-penelitian)

[1.4 Batasan Masalah [2](#_Toc234440433)](#_Toc234440433)

[2 Manfaat Penelitian [3](#_Toc234440434)](#_Toc234440434)

[3 *The State of The Art*
[3](#the-state-of-the-art)](#the-state-of-the-art)

[4 Studi Literatur [17](#studi-literatur)](#studi-literatur)

[**7.1 Model Representasi Laten *Self-Supervised* (SSL)**
[17](#model-representasi-laten-self-supervised-ssl)](#model-representasi-laten-self-supervised-ssl)

[**7.1.1 Wav2vec 2.0: Representasi Kontekstual Melalui Contrastive
Learning**
[17](#wav2vec-2.0-representasi-kontekstual-melalui-contrastive-learning)](#wav2vec-2.0-representasi-kontekstual-melalui-contrastive-learning)

[**7.1.2 Data2Vec: *Embedding* generalis melalui Self-Distillation**
[17](#data2vec-embedding-generalis-melalui-self-distillation)](#data2vec-embedding-generalis-melalui-self-distillation)

[5 Metode Penelitian [20](#metode-penelitian)](#metode-penelitian)

[1. Pengumpulan data [21](#pengumpulan-data)](#pengumpulan-data)

[2. Tahap *Pre-processing* Data
[21](#tahap-pre-processing-data)](#tahap-pre-processing-data)

[3. Tahap Ekstraksi Representasi Laten (Vector *Embedding*)
[22](#tahap-ekstraksi-representasi-laten-vector-embedding)](#tahap-ekstraksi-representasi-laten-vector-embedding)

[4. Tahap Implementasi *Similarity Search*
[22](#tahap-implementasi-similarity-search)](#tahap-implementasi-similarity-search)

[5. Tahap Evaluasi Kinerja (Perbandingan)
[23](#tahap-evaluasi-kinerja-perbandingan)](#tahap-evaluasi-kinerja-perbandingan)

[6. Tahap Analisis Hasil
[23](#tahap-analisis-hasil)](#tahap-analisis-hasil)

[DAFTAR PUSTAKA [26](#daftar-pustaka)](#daftar-pustaka)

# DAFTAR GAMBAR

[Gambar 1 Kerangka Pemikiran [16](#_Toc213132313)](#_Toc213132313)

[Gambar 2 Metode Penelitian [20](#_Toc213132314)](#_Toc213132314)

# 

# DAFTAR TABEL

[Tabel 1 State of the art [7](#_Toc213131565)](#_Toc213131565)

# BAB I PENDAHULUAN

## Latar Belakang

> Perkembangan *deep learning* dalam pemrosesan audio telah didominasi
> oleh model Self-Supervised Learning (SSL), dengan Wav2vec2 menjadi
> arsitektur pionir yang sangat berpengaruh. Model-model ini mampu
> mempelajari representasi audio mendalam (*latent embeddings*) dari
> data audio mentah tanpa memerlukan label transkripsi, dan telah
> menunjukkan efektivitas tinggi pada berbagai tugas pemrosesan suara
> \[1\], \[2\]. Kemampuan model-model ini dalam menghasilkan fitur laten
> menjadi fondasi bagi tugas-tugas non-transkripsi, termasuk penelusuran
> kemiripan audio, yang mencocokkan potongan bacaan dengan entri yang
> paling mirip di *database* berdasarkan kemiripan *vector embedding*,
> bukan mencocokkan melalui teks hasil enkripsi \[3\]. Pendekatan ini
> telah didukung oleh temuan empiris yang menunjukkan bahwa sistem
> retrieval berbasis latent embedding yang dihasilkan langsung dari
> audio secara konsisten mengungguli pendekatan dua tahap dari ASR ke
> retrieval, karena kesalahan transkripsi pada tahap ASR akan
> berpropagasi dan mendegradasi kualitas pencocokan pada tahap
> selanjutnya \[4\], \[5\]. Karakteristik inilah yang menjadikan
> pendekatan berbasis embedding lebih sesuai untuk konteks pencocokan
> bacaan Quran, yang menuntut presisi tinggi pada aspek fonetik dan
> tajwid.
>
> Al-Qur\'an berisi teks berbahasa Arab dengan kaidah pelafalan (tajwid)
> yang ketat dan presisi tinggi, tidak hanya memiliki signifikansi
> religius bagi umat Muslim, tetapi juga merupakan *testbed* fonetik
> yang menantang secara teknis untuk sistem pemrosesan suara.
>
> Persoalan mendasar muncul ketika model SSL yang di-*pretrain* pada
> data yang berbahasa Inggris diterapkan pada fonetik Arab, yang
> memiliki karakteristik berbeda seperti contohnya *emphatic consonants*
> dan *pharyngeal sounds* yang tidak dimiliki oleh bahasa Inggris \[6\].
> Dalam eksperimen yang dilakukan oleh Toyin model berbasis
> *Arab-monolingual* mampu mengungguli model multibahasa yang memiliki
> ukuran dua kali lipat dalam tugas pengenalan ucapan Arab\[6\]. Temuan
> ini diperkuat oleh Li et al yang secara kuantitatif menunjukkan bahwa
> encoder SSL yang dilatih menggunakan teks berbahasa inggris
> menghasilkan fitur fonetik yang kurang informatif ketika diterapkan
> pada inputan lintas-bahasa tanpa adaptasi domain \[7\]. Namun
> demikian, bukti-bukti tersebut terbatas pada arsitektur berbasis
> *contrastive learning* seperti Wav2Vec2. Sementara itu, Data2Vec
> menggunakan paradigma *self-distillation* yang secara fundamental
> berbeda. alih-alih membandingkan representasi positif versus negatif,
> Data2Vec melatih jaringan *student* untuk memprediksi representasi
> kontekstual penuh yang dihasilkan oleh jaringan *teacher* atas seluruh
> masukan \[1\], \[8\]. Perbedaan ini secara teoretis berpotensi
> menghasilkan representasi fonetik yang lebih umum untuk lintas bahasa,
> karena target prediksi berupa konteks laten yang kaya bukan unit
> diskrit yang terikat pada distribusi fonetik bahasa *pretraining*.
> Dugaan ini didukung oleh temuan Xue et al. yang menunjukkan bahwa
> penggantian target diskrit dengan target kontinu ala Data2Vec
> menurunkan *phoneme* *error* rate lintas bahasa sebesar 5,3% \[9\].
>
> Temuan-temuan di atas menuntun pada satu celah riset yang belum
> terjawab: seluruh bukti kegagalan lintas-bahasa yang tersedia hanya
> terbatas pada arsitektur contrastive learning, sementara kajian
> terhadap paradigma self-distillation yang dimiliki Data2Vec dalam
> skenario serupa masih sangat terbatas. Lebih jauh lagi, perbandingan
> langsung antara Wav2Vec2 dan Data2Vec untuk tugas audio
> retrieval---yang dievaluasi melalui metrik similarity search seperti
> Top-K Accuracy, Mean Reciprocal Rank (MRR), dan Mean Average Precision
> (MAP), bukan metrik transkripsi (WER/CER)---hingga kini masih minim di
> literatur. Ketiadaan perbandingan ini menyisakan pertanyaan
> fundamental: apakah perbedaan paradigma pretraining antara kedua model
> menghasilkan kualitas representasi fonetik yang berbeda pula untuk
> tugas retrieval ayat Al-Qur\'an.
>
> Namun, potensi keunggulan *self-distillation* tersebut tidak dapat
> diasumsikan berlaku pada tugas *retrieval*, sebab *retrieval* dan
> transkripsi mengukur properti representasi yang berbeda. Pada ASR,
> fitur diteruskan ke kepala klasifikasi yang ikut dilatih secara
> tersupervisi., Akibatnya, metrik seperti WER tidak murni mengukur
> kualitas embedding awal, melainkan separabilitas fitur setelah
> disesuaikan oleh proses pelatihan tersebut. Sebaliknya, pada retrieval
> berbasis *frozen embedding* tidak ada komponen yang dilatih, sehingga
> metrik seperti MAP secara intrinsic mengukur koherensi geometris ruang
> vector itu sendiri, tanpa bantuan optimasi tambahan.
>
> Perbedaan ini memunculkan dua hipotesis yang saling bersaing.
> Hipotesis pertama (H1) memperkirakan Data2Vec tetap unggul: target
> prediksinya yang berupa representasi kontekstual laten penuh
> berpotensi menghasilkan fitur fonetik yang lebih kaya dan umum,
> sehingga keunggulannya diperkirakan bertahan pada retrieval. Hipotesis
> kedua (H2) memperkirakan sebaliknya, bahwa Wav2Vec2 justru lebih
> unggul pada retrieval: *contrastive loss*-nya secara eksplisit menarik
> pasangan positif dan menolak kandidat negatif, sehingga cenderung
> membentuk ruang embedding yang lebih *clusterable* dan lebih selaras
> dengan pengukuran jarak cosine --- properti yang justru menentukan
> pada retrieval, bukan pada ASR.
>
> Pengujian empiris atas kedua hipotesis inilah yang menjadi inti
> kontribusi penelitian ini. Jika H1 terkonfirmasi, temuan ini
> memperkuat pandangan bahwa kualitas embedding bersifat lintas-tugas
> --- unggul di satu tugas cenderung unggul pula di tugas lain. Jika H2
> yang terkonfirmasi, retrieval justru terbukti mengukur dimensi
> kualitas representasi yang berbeda sama sekali dari ASR (ortogonal),
> bukan sekadar versi lain dari hal yang sama.
>
> Oleh karena itu, penelitian ini diarahkan untuk melakukan perbandingan
> komprehensif antara Wav2Vec2 dan Data2Vec guna mengukur efektivitas
> fitur laten audio kedua model dalam tugas retrieval ayat Al-Qur\'an.
> Melalui perbandingan ini, penelitian ini juga bertujuan menganalisis
> bagaimana perbedaan arsitektur SSL memengaruhi kualitas representasi
> fonetik yang dihasilkan. Hasil analisis ini diharapkan dapat
> memberikan rekomendasi arsitektur yang paling optimal sebagai landasan
> teknis bagi pengembangan sistem *audio retrieval* pada domain bahasa
> Arab khususnya yang memiliki karakteristik fonetik khas seperti
> *emphatic consonants* dan *pharyngeal sounds*.
>
> .

## Rumusan Masalah

> Berdasarkan Rumusan masalah terbagi menjadi beberapa point

1.  Bagaimana kinjera *retrieval* *frozen-embedding* Wav2vec2 diukur
    dengan MAP dan Top-K Accuracy?

2.  Bagaimana kinerja retrieval *frozen-embedding* Data2vec pada metriks
    yang sama?

3.  Apakah layer yang optimal untuk retrieval fonetik berbeda dari layer
    yang dilaporkan optimal untuk ASR, dan apa implikasinya terhadap
    pemilihan arsitektur?

## 1.3 Tujuan Penelitian

> Tujuan dari penelitian ini sebagai berikut:

1.  []{#_Toc234440433 .anchor}Mengevaluasi kinerja retrieval ayat
    Al-Qur\'an berbasis audio menggunakan *frozen embedding* Wav2Vec2
    (MAP, Top-K, cosine similarity).

2.  Mengevaluasi kinerja retrieval ayat Al-Qur\'an berbasis audio
    menggunakan *frozen embedding* Data2vec pada metrik yang sama.

3.  Membandingkan kedua model dan menganalisis apakah performa
    *retrieval* sejalan dengan performa ASR bersifat konsisten atau
    sebaliknya, untuk menentukan model yang paling sesuai untuk sistem
    pencarian ayat Al-Qur\'an berbasis suara.

4.  Menganalisis distribusi kinerja retrieval pada tiap lapisan kedua
    model untuk mengidentifikasi lapisan optimal bagi tugas *retrieval*
    fonetik.

## 1.4 Batasan Masalah

> Batasan masalah dari penelitian ini adalah sebagai berikut:

1.  Penelitian dibatasi pada penggunaan model pretrained Data2Vec dan
    Wav2Vec2, tanpa pelatihan ulang (fine-tuning) pada data khusus
    Al-Qur'an

2.  Eksperimen dilakukan pada teks Al-Qur'an, dengan objek penelitian
    berupa surah Al -- Fatihah dan surah-surah dalam Juz Amma.

3.  Pengujian input audio hanya dapat dilakukan perayat.

## 1.5 Manfaat Penelitian

> Manfaat yang diharapkan dari hasil penelitian ini adalah sebagai
> berikut:

1.  Memberikan pemahaman ilmiah yang lebih dalam tentang efektivitas
    model representasi audio seperti Data2Vec dan Wav2Vec2 dalam domain
    bahasa Arab, khususnya untuk tugas pencocokan kemiripan teks
    berbasis suara.

2.  Memberikan acuan bagi peneliti dan pengembang aplikasi keislaman
    dalam memilih model *embedding* audio terbaik untuk diterapkan pada
    sistem pembelajaran Al-Qur'an berbasis suara

## 1.6 Kerangkan Pemikiran

> Berdasarkan uraian tersebut kerangkan pemikiran dapat diuraikan
> sebagai berikut

##  ![](media/image2.png){width="4.066666666666666in" height="8.041117672790902in"}

Gambar 1 Kerangka Penelitian

> Penelitian ini berangkat dari sebuah fenomena dalam bidang pemrosessan
> audio, yaitu kemampuan model *Self-Supervised Learning* (SSL) seperti
> Wav2vec2 dan Data2vec dalam menghasilkan representasi laten, langsung
> dari audio mentah tanpa memerlukan label transkripsi. Kemampuan ini
> membuka peluang bagi tugas penelurusan kemiripan audio (*audio
> retrieval*). Di mana sistem *retrieval* yang bekerja langsung pada
> *latent embedding* dapat mengungguli pendekatan dua tahap berbasis
> ASR. Keunggulan ini bersumber dari kemampuannya menghindari kesalahan
> propagasi (*error propagation),* yakin kesalahan transkripsi pada
> tahap ASR yang akan merambat dan mendegradasi kualitas pencocokan pada
> tahap selanjutnya.
>
> Fenomena tersebut ditopang oleh sejumlah landasan teori dan studi
> literatur. Pada tataran mekanisme model, Wav2vec2 mempelajari
> representasi melalui *contrastive learning* yang membedakan sampel
> positif dan negatif menggunakan *quntized discrete units.* Sementara
> Data2vec mengadopsi paradigma *self-distillation* yang melatih
> jaringan student untuk memprediksi representasi kontekstual penuh dari
> jaringan *teacher* dengan target berupa konteks latent yang kontinu.
> Perbedaan paragima inilah yang menjadi dasar perbandingan dalam
> penelitian ini. Pada tataran tugas *retrieval*, penelitian Öberg
> menunjukan bahwa pencarian audio dapat dilakukan langsung dari audio
> tanpa tahap enkripsi \[3\]. Teori ini didukung pula oleh bukti empiris
> *Gemini embedding 2*, secara eksplisit menunjukkan bahwa pendekatan
> *cascaded* ASR ke *retrieval* mengalami *error propagation* \[4\].
> Sedangkan *WavRag* membuktikan bahwa *retrieval* langsung dari audio
> mampu menyamai akurasi pipeline ASR dengan efisiensi yang jauh lebih
> tinggi \[5\].
>
> Meskipun demikian, keunggulan tersebut belum tentu berlaku ketika
> model diterapkan pada domain fonteik yang spesifik seperti bahasa Arab
> dalam bacaan Al-Qur'an. Persoalan muncul karena model SSL umumnya
> di-*pretrain* pada korpus Bahasa inggris dan kurang informatikf untuk
> fonetik Arab. Penelitian Toyin et al dan Li et al telah menunjukan
> keterbatasan ini \[6\], \[7\]. namun bukti tersebut hanya terbatas
> pada arsitektur berbasis constrastive learing seperti Wav2vec2.
> Sementara itu, ptotensi paradigma self-distillation yang dimiliki
> Data2Vec untuk tugas audio retrieval Al-Quran masih sangat minim
> dikaji, dan perbandingan langsung antara kedua model menggunakan
> metrik similarity search, bukan metrik transkripsi seperti *word error
> rate* (WER) hingga kini masih terbatas dalam literatur.
>
> Untuk menjawab persoalan tersebut, penelitian ini menawarkan solusi
> berupa perbandingan komprehensif antara Wav2Vec2 dan Data2Vec dalam
> menghasilkan representasi laten untuk tugas retrieval ayat Al-Qur\'an.
> Fitur laten dari kedua model pretrained diekstraksi dalam kondisi beku
> (frozen), kemudian dievaluasi secara objektif menggunakan metrik
> similarity search seperti Top-K Accuracy, Mean Reciprocal Rank (MRR),
> dan Mean Average Precision (MAP). Pendekatan ini bertujuan mengukur
> secara langsung pengaruh perbedaan arsitektur SSL terhadap kualitas
> representasi fonetik Arab.
>
> Proses penelitian dilaksanakan mengikuti kerangka kerja Cross-Industry
> Standard Process for Data Mining (CRISP-DM) yang diadaptasi ke dalam
> enam tahap, yaitu pengumpulan data, pre-processing data, ekstraksi
> representasi laten (vector embedding), implementasi similarity search,
> evaluasi kinerja, serta analisis hasil dan penarikan kesimpulan.
> Melalui tahapan yang sistematis ini, penelitian diharapkan
> menghasilkan perbandingan kuantitatif kinerja retrieval antara
> Wav2Vec2 dan Data2Vec, sekaligus memberikan analisis mengenai pengaruh
> paradigma pretraining terhadap kualitas representasi fonetik. Hasil
> akhir penelitian ini berupa rekomendasi arsitektur SSL yang paling
> optimal sebagai landasan teknis bagi pengembangan sistem audio
> retrieval pada domain berbahasa Arab.

## 1.7 Sistematika Penulisan

> Sistematika penulisan laporan tugas akhir ini disusun ke dalam lima
> bab yang saling berkait secara berurutan, dengan gambaran kandungan
> setiap bab sebagai berikut.

**BAB I: PENDAHULUAN**

> Bab ini menguraikan dasar pemikiran yang melatarbelakangi dilakukannya
> penelitian, mencakup latar belakang yang memaparkan fenomena dan celah
> penelitian, rumusan masalah sebagai pertanyaan penelitian yang hendak
> dijawab, batasan masalah yang menetapkan ruang lingkup penelitian,
> manfaat penelitian, state of the art, kerangka pemikiran, serta
> sistematika penulisan. Bab ini berfungsi sebagai fondasi mengarahkan
> keseluruhan penelitian.

**BAB II: TINJAUAN PUSTAKA**

> Bab ini memaparkan landasan teori dan konsep-konsep fundamental yang
> menjadi dasar ilmiah penelitian, meliputi teori mengenai
> Self-Supervised Learning (SSL), arsitektur dan mekanisme kerja model
> Waw2vec2 dan Data2Vec, konsep representasi latent (*latent
> embedding*), tugas audio retrieval, serta metrik evaluasi similarity
> search. Berbeda dengan Bab I yang menyoroti mengapa penelitian mesti
> dilakukan, bab ini menyediakan kerangka teoritis apa saja konsep yang
> digunakan untuk memahami dan menyelesaikan permalahan.

**BAB III: METODOLOGI PENELITIAN**

> Bab ini menjelaskan tahapan prosedur pelaksanaan penelitian secara
> sistematis mengikuti kerangka kerja *Cross-Industry Standard Process
> for Data Mining* (CRISP-DM) yang diadaptasi, mencakup pengumpulan
> data, pre-processing, ekstrasi representasi laten, implementasi
> similarity search, hingga skenario evaluasi. Jika Bab II menjelaskan
> apa landasan teorinya, bab ini menguraikan bagaimana teori tersebut
> diterapkan secara teknis dan operasional dalam penelitian.

**BAB IV: HASIL DAN PENELITIAN**

> Bab ini menyajikan hasil yang diperoleh dari pelaksanaan metodologi
> pada Bab III, berupa data kuantitaif kinerja *retrieval* kedua model
> berdasarkan metrik *Top-K Accuracy*, *Mean Reciprocal Rank* (MRR), dan
> *Mean Average Precision* (MAP), disertai analisis dan interpretasi
> mendalam terhadap perbandingan Wav2vec2 dan Data2vec. Bab ini
> merupakan inti temuan penelitian yang menghubungkan hasil eksperimen
> dengan pertanyaan penelitian pada Bab I.

**BAB V: KESIMPULAN DAN SARAN**

> Bab ini memuat kesimpulan yang menjawab secara ringkas dan tegas
> rumusan masalah berdasarkan temuan pada Bab IV, disertai saran dan
> untuk pengembangan penelitian selanjutnya. Berbeda dengan Bab IV yang
> menyajikan analisis rinci, bab ini merangkum intisari penelitian
> secara menyeluruh dan menawarkan arah bagi penelitian di masa
> mendatang.

# BAB II KAJIAN LITERATUR

## *The State of The Art*

Analisis literatur terhadap artikel maupun jurnal terdahulu yang relevan
dilakukan untuk memperdalam pemahaman mengenai permasalahan utama dalam
penerapan *Self-Supervised Learning* (SSL) pada representasi ucapan
(*speech representation*) serta pemanfaatannya untuk pencarian audio
berbasis kemiripan (*audio similarity search*) dengan pendekatan
*query-by-example*. Kajian ini sekaligus bertujuan untuk
mengindentifikasi keterbatasan penelitian sebelumnya. Berikut rangkuman
dari penelitian terdahulu dalam bentuk table.

[]{#_Toc213131565 .anchor}Tabel 1 State of the art

+--------+-----------------------+--------------------+----------------------------+-------------------------+
| **No** | **Judul Jurnal dan    | **Metode**         | **Dataset**                | **Hasil**               |
|        | Peneliti**            |                    |                            |                         |
+:=======+:======================+:===================+:===========================+:========================+
| 1      | wav2vec 2.0: A        | Wav2vec2,          | LS-960 (960 jam), LV-60k   | Wav2Vec2 mempelajari    |
|        | Framework for         | *contrastive       | (53200 jam), Fine-tune:    | representasi suara dari |
|        | Self-Supervised       | masked prediction  | Libri-light 10 min-960     | raw audio melalui       |
|        | Learning of Speech    | of quantized       |                            | pendekatan              |
|        | Representations       | units*             |                            | Self-Supervised         |
|        |                       |                    |                            | Learning (SSL) berbasis |
|        | Peneliti:             |                    |                            | contrastive learning.   |
|        |                       |                    |                            | Dengan hanya melakukan  |
|        | Alexei Baevski dkk    |                    |                            | fine-tuning menggunakan |
|        | (2020)                |                    |                            | 10 menit data latih     |
|        |                       |                    |                            | berlabel (setara 48     |
|        |                       |                    |                            | rekaman dengan durasi   |
|        |                       |                    |                            | rata-rata 12.5 detik),  |
|        |                       |                    |                            | model ini mencapai Word |
|        |                       |                    |                            | Error Rate (WER)        |
|        |                       |                    |                            | sebesar 4.8/8.2 pada    |
|        |                       |                    |                            | test-clean/other        |
|        |                       |                    |                            | dataset LibriSpeech.    |
|        |                       |                    |                            | \[1\]                   |
+--------+-----------------------+--------------------+----------------------------+-------------------------+
| 2      | data2vec: A General   | Data2vec,          | LibriSpeech 960h /         | Mengembangkan framework |
|        | Framework for         | *self-distillation | Libri-light 60K jam;       | SSL seragam lintas      |
|        | Self-supervised       | of contextual      | Vision: ImageNet-1K; NLP:  | modalitas dengan        |
|        | Learning in Speech,   | latent             | BooksCorpus + Wikipedia    | memprediksi             |
|        | Vision and Language   | representations,   | (GLUE)                     | representasi laten      |
|        |                       | teacher-student*   |                            | kontekstual (bukan      |
|        | Peneliti:             | EMA, target        |                            | token diskrit); WER     |
|        |                       | kontinu (rata-rata |                            | test-other 5,5 (960h),  |
|        | Alexei Baevski dkk    | top-K layer,       |                            | 86,6% ImageNet, 82,7    |
|        | 2022                  | Smooth L1 loss     |                            | GLUE membuktikan target |
|        |                       |                    |                            | kontinu unggul di       |
|        |                       |                    |                            | low-resource \[8\]      |
+--------+-----------------------+--------------------+----------------------------+-------------------------+
| 3\.    | Aswat: Arabic Audio   | Wav2vec2 dan       | Aswat (732 jam audio       | Membandingkan           |
|        | Dataset For Automatic | Data2vec           | Arab), evaluasi pada       | efektivitas Wav2vec2    |
|        | Speech Recognition    |                    | Common Voice (AR) dan      | dan Data2vec pada       |
|        | Using                 |                    | MGB-2                      | pengenalan ucapan Arab. |
|        | Speech-Representation |                    |                            | Hasilnya SOTA WER 11,7% |
|        | Learning              |                    |                            | (Common Voice) & 10,3%  |
|        |                       |                    |                            | (MGB-2); data2vec       |
|        | Peneliti:             |                    |                            | konsisten mengungguli   |
|        |                       |                    |                            | wav2vec untuk Arab      |
|        | Lamya Alkanhal, dkk   |                    |                            | \[14\]                  |
|        | (2021)                |                    |                            |                         |
+--------+-----------------------+--------------------+----------------------------+-------------------------+
| 4\.    | Query-by-Example      | *Embedding         | Google FLEURS + Common     | Penelitian ini          |
|        | Audio Search using    | Transformation*    | Voice (word-segmented);    | menginvestigasi         |
|        | Acoustic Word         | menggunakan        | evaluasi SWE-Hard & X-Hard | peningkatan kinerja     |
|        | Embeddings:           | *Projection        |                            | Pencarian Audio         |
|        | Transforming wav2vec  | Network* dengan    |                            | *Query-by-Example*      |
|        | 2.0 Embeddings using  | Contrastive        |                            | (QbE) melalui           |
|        | Contrastive Learning  | Learning (Triplet  |                            | transformasi            |
|        |                       | Loss) pada         |                            | *embedding* Wav2Vec 2.0 |
|        | Peneliti:             | *Embedding*        |                            | menggunakan Contrastive |
|        |                       | Wav2Vec 2.0.       |                            | Learning (khususnya     |
|        | Wilhelm Öberg (2025)  |                    |                            | *Triplet Loss*).        |
|        |                       |                    |                            | Tujuannya adalah        |
|        |                       |                    |                            | melatih jaringan        |
|        |                       |                    |                            | proyeksi untuk          |
|        |                       |                    |                            | menyelaraskan           |
|        |                       |                    |                            | *embedding* kata yang   |
|        |                       |                    |                            | serupa dan memfilter    |
|        |                       |                    |                            | karakteristik *speaker* |
|        |                       |                    |                            | dan kebisingan,         |
|        |                       |                    |                            | sehingga menghasilkan   |
|        |                       |                    |                            | ruang *embedding* yang  |
|        |                       |                    |                            | lebih terstruktur.      |
|        |                       |                    |                            | Metode ini              |
|        |                       |                    |                            | berkontribusi pada      |
|        |                       |                    |                            | pencarian audio yang    |
|        |                       |                    |                            | sepenuhnya bebas        |
|        |                       |                    |                            | transkripsi dan sangat  |
|        |                       |                    |                            | relevan untuk tugas     |
|        |                       |                    |                            | *retrieval* berbasis    |
|        |                       |                    |                            | kemiripan fitur laten.  |
|        |                       |                    |                            | \[3\]                   |
+--------+-----------------------+--------------------+----------------------------+-------------------------+
| 5\.    | LEVERAGING            | Melakukan          | Bahasa low-resource:       | Menguji kemampuan       |
|        | PRE-TRAINED           | Query-by-Example   | Gronings, Besemah, dan     | retrieval lintas-bahasa |
|        | REPRESENTATIONS TO    | Spoken Term        | dataset Mavir/Mboshi-line; | menggunakan embedding   |
|        | IMPROVE ACCESS TO     | Detection          | pra-pelatihan English      | SSL yang dibekukan      |
|        | UNTRANSCRIBED SPEECH  | menggunakan        | (LibriSpeech) &            | (frozen), tanpa         |
|        | FROM ENDANGERED       | representasi       | multibahasa (XLSR-53, 53   | bergantung pada         |
|        | LANGUAGES             | frozen dari        | bahasa)                    | transkripsi, yang       |
|        |                       | wav2vec 2.0 (versi |                            | hasilnya Peningkatan    |
|        | Penulis:              | English-mono dan   |                            | relatif 56--86% atas    |
|        |                       | XLSR-53), tanpa    |                            | SOTA; English-mono      |
|        | Nay San, dkk (2021)   | transkripsi maupun |                            | mengungguli XLSR        |
|        |                       | fine-tuning        |                            | multibahasa di 4        |
|        |                       |                    |                            | dataset \[15\]          |
+--------+-----------------------+--------------------+----------------------------+-------------------------+
| 6\.    | Improves Neural       | Menggabungkan      | Korpus QbE-STD             | Membuktikan bahwa       |
|        | Acoustic Word         | Neural Acoustic    |                            | embedding SSL           |
|        | Embeddings Query by   | Word Embeddings    |                            | mengungguli fitur MFCC  |
|        | Example Spoken Term   | dengan             |                            | untuk retrieval         |
|        | Detection with        | pra-pelatihan      |                            | berbasis kemiripan      |
|        | Wav2vec Pretraining   | wav2vec untuk      |                            | (similarity-based       |
|        | and Circle Loss       | QbE-STD, dilatih   |                            | retrieval). Hasilnya AP |
|        |                       | menggunakan        |                            | +11,1% vs MFCC; sistem  |
|        | Penulis:              | cosine/triplet     |                            | terbaik +19,7% atas     |
|        |                       | (circle) loss      |                            | baseline MFCC           |
|        | Zhaoqi Li, dkk (2021) |                    |                            |                         |
+--------+-----------------------+--------------------+----------------------------+-------------------------+
| 7\.    | Siamese Network with  | Two-Phase          | ASVspoof 2019 Logical      | Mengusulkan sistem dua  |
|        | Wav2vec Feature for   | Learning: 1.       | Access (LA)                | fase untuk mengatasi    |
|        | Spoofing Speech       | Siamese Network +  |                            | masalah generalisasi    |
|        | Detection             | Contrastive Loss   |                            | terhadap serangan       |
|        |                       | pada fitur         |                            | *spoofing* ucapan       |
|        | Peneliti:             | Wav2vec. 2. MLP    |                            | (*out-of-distribution*) |
|        |                       | Classifier pada    |                            | yang tidak dikenal.     |
|        | Yang Xie, dkk (2022)  | *embedding* hasil  |                            | Sistem ini memanfaatkan |
|        |                       | Phase 1.           |                            | fitur Wav2vec yang      |
|        |                       |                    |                            | sudah dilatih           |
|        |                       |                    |                            | (*pretrained*) sebagai  |
|        |                       |                    |                            | fitur input             |
|        |                       |                    |                            | diskriminatif.          |
|        |                       |                    |                            | Pendekatan ini secara   |
|        |                       |                    |                            | signifikan meningkatkan |
|        |                       |                    |                            | kinerja, mengurangi     |
|        |                       |                    |                            | *Equal Error Rate*      |
|        |                       |                    |                            | (EER) dari 4.07% (SOTA  |
|        |                       |                    |                            | sebelumnya) menjadi     |
|        |                       |                    |                            | 1.15% pada *benchmark*  |
|        |                       |                    |                            | ASVspoof 2019.          |
+--------+-----------------------+--------------------+----------------------------+-------------------------+
| 8\.    | Exploring Effective   | Membandingkan dan  | SUPERB benchmark (task SID | Membuktikan bahwa       |
|        | Fusion Algorithms for | menggabungkan      | & ASR; basis               | penggabungan model      |
|        | Speech Based          | HuBERT dan         | LibriSpeech/VoxCeleb-line) | dalam domain SID        |
|        | Self-Supervised       | Data2vec pada task |                            | (speaker                |
|        | Learning Models       | SUPERB SID         |                            | identification) lebih   |
|        |                       | (*speaker          |                            | buruk, sedangkan dalam  |
|        | Peneliti:             | identification*) & |                            | ASR menghasilkan hasil  |
|        |                       | ASR (*automatic    |                            | yang lebih baik \[16\]  |
|        | Tang C, dkk (2022)    | speech             |                            |                         |
|        |                       | recognition*)      |                            |                         |
+--------+-----------------------+--------------------+----------------------------+-------------------------+
| 9\.    | MCR-Data2vec 2.0:     | Data2Vec 2.0 Base  | LibriSpeech 960 jam, 11    | Membuat novel model     |
|        | Improving             | dengan             | SUPERB subtasks            | pretrained baru         |
|        | Self-supervised       | regularisasi       |                            | (MCR-Data2vec) hasilnya |
|        | Speech Pre-training   | konsistensi;       |                            | MCR meningkatkan        |
|        | via Model-level       | fine-tuning hanya  |                            | performa pada SID, ASV, |
|        | Consistency           | pada lapisan       |                            | dan Speech Enhancement, |
|        | Regularization        | tugas, sedangkan   |                            | namun sedikit           |
|        |                       | model utama tidak  |                            | menurunkan performa     |
|        | Penulis:              | dilakukan          |                            | pada QbE. \[17\]        |
|        |                       | *fine-tune*.       |                            |                         |
|        | Yoon, dkk (2023)      |                    |                            |                         |
+--------+-----------------------+--------------------+----------------------------+-------------------------+
| 10\.   | A Large-Scale         | Evaluasi Data2Vec  | LibriSpeech, QUESST (QbE), | Data2Vec mencapai       |
|        | Evaluation of Speech  | Base dan Large     | VoxCeleb (SID/ASV),        | performa terbaik pada   |
|        | Foundation Models     | pada 15 tugas      | LibriTTS (VC), etc.        | Voice Conversion dan    |
|        |                       | downstream dengan  |                            | Phone Recognition,      |
|        | Peneliti:             | embedding yang     |                            | tetapi kurang optimal   |
|        |                       | dibekukan (frozen  |                            | pada SID dan OOD-ASR    |
|        | Yang, dkk (2024)      | embedding) dan     |                            | dibanding HuBERT dan    |
|        |                       | lightweight        |                            | WavLM \[18\].           |
|        |                       | prediction head.   |                            |                         |
+--------+-----------------------+--------------------+----------------------------+-------------------------+

> Penelitian mengenai representasi ucapan yang dilatih sendiri
> (*self-supervised*) telah mengalami kemajuan pesat, dimulai dengan
> fondasi seperti Wav2vec 2.0 dan Data2vec \[1\], \[8\] yang mampu
> mempelajari representasi laten koheren dari *raw audio*. Namun,
> mayoritas penelitian awal dan penelitian lanjutan berkonsentrasi pada
> peningkatan kinerja ASR (*automatic speech recognition*), termasuk
> studi lintas bahasa dan adaptasi domain. Sebagaian penelitian lain
> berfokus pada non-retrieval yang tetap memanfaatkan kualitas embedding
> SSL, seperti deteksi *spoofing* suara dan identifikasi suarat yang
> menggabungkan *Siamase Network* pada fitur Wav2vec2 \[12\]. Meskipun
> demikian, Sebagian studi mulai menggeser fokus ke kapabilitas
> *retrieval* dari *embedding* ini, misalnya penelitian Öberg mengenai
> audio search berbasis transformasi *embedding* Wav2vec2 dengan
> *contrastive learning* \[3\]
>
> Pada ranah *retrieval* berdasarkan kemiripan, sejumlah penelitian
> telah membuktikan keunggulan pendekatan berbasis *embedding* SSL. Nay
> san, dkk telah membuktikan bahwa representasi dari Wav2vec2 dapat
> melakukan *query-by-example* *spoken term detection* lintas bahasa
> tanpa transkripsi maupun fine-tuning dengan peningkatan 56%-86% atas
> *state of the art* sebelumnya \[15\]. Sejalan dengan itu, Zhaoqi Li,
> dkk membuktikan bahwa *Neural Acoustic Word Embeddings* berbasis
> pretrain Wav2vec2 mengungguli fitur MFCC untuk retrieval berbasis
> kemiripan \[19\]. Temuan-temuan ini menegaskan bahwa kualitas
> embedding SSL menjadikannya sangat sesuai dengan untuk tugas
> *similarity search.*
>
> Dalam konteks bahasa Arab alkanhal, dkk melakukan perbandingan
> langsung antara Wav2vec2 dan Data2vec. Mereka menemukan bahwa Data2vec
> konsisten mengungguli Wav2vec2 dalam domain ASR Arab. Namun,
> perbandingan tersebut dilakukan dengan supervised fine-tuning dan juga
> dengan pengukuran metrik transkripsi, bukan frozen embedding tanpa
> fine-tuning dan metrik retrieval. Di sisi lain, kajian terhadap
> Data2vec pada tugas-tugas non-ASR mulai bermunculan dan justru
> memperlihatkan pola yang bergantung pada jenis tugasnya. Tang, dkk
> menemukan bahwa efektivitas Data2vec berbeda tergantung tugasnya,
> Data2vec lebih unggul dalam tugas yang berbasis konten contohnya ASR,
> namun kurang optimal dalam identifikasi pembicara. Yoon, dkk
> mengonfirmasi kecenderungan serupa, Data2vec 2.0 kuat pada tugas
> berbasis konten seperti ASR, tetapi performa query-by-example nya
> justru menurun ketika diberi regulasi tambahan \[17\]. Evaluasi
> berskala besar yang dilakukan oleh Yang, dkk mempertegas hal ini,
> menunjukan bahwa Data2vec mencapai performa terbaik pada Voice
> Conversion dan Phone Recognition, tetapi tertinggal pada Speaker
> Identification dan ASR out-of-distribution dibanding HuBERT dan WavLM.
>
> Berdasarkan kajian tersebut, terlihat bahwa keunggulan Data2vec tidak
> bersifat universial, melainkan spesifik terhadap jenis tugas. Fakta
> ini memunculkan pertanyaan fundamental yang belum terjawab dalam
> literatur: apakah keunggulan Data2vec pada ASR bahasa Arab akan sama
> ketika kedua model dievaluasi tanpa fine-tuning dan menggunakan
> metrics dalam domain similarity search seperti MAP, Top-K, dan MRR.
>
> Penelitian ini menghadirkan perbedaan penting dengan menempatkan kedua
> model tersebut dalam konteks *retrieval* audio ayat Al-Qur'an yang
> mana memiliki karakteristik fonetik dan aturan tajwid yang unik.
> Berbeda dengan penelitian pada umumnya yang menitik beratkan pada
> pengukuran performa transkripsi ASR, penelitian ini mengevaluasi
> kemampuan Wav2vec2 dan Data2vec dalam mencocokan cuplikan bacaan
> dengan ayat terkait melalui pendekatan komparatif berbasis kemiripan
> embedding. Dengan demikian, penelitian ini menawarkan kontribusi yang
> masih minim dieksplorasi, sekaligus menyediakan landasan teknis bagi
> pengembangan sistem pendukung hafalan dan pembelajaran Al-Qur'an yang
> lebih adaptif dan aplikatif.

## Studi Literatur

### **2.2.1 Model Representasi Laten *Self-Supervised* (SSL)** 

> *Self-Supervised Learning (SSL)* adalah paradigma pembelajaran mesin
> di mana model dilatih untuk mempelajari representasi yang berguna dari
> data yang tidak memiliki *label* dengan menghasilkan labelnya sendiri.
> SSL terletak di antara *Supervised Learning* (yang membutuhkan label
> manual ekstensif) dan *Unsupervised Learning* (yang mencari pola tanpa
> label).
>
> Inti dari SSL terletak pada penciptaan \"Tugas Preteks\" (*Pretext
> Task*). Tugas ini memaksa model untuk memahami dan memprediksi bagian
> tersembunyi (*masked*) atau bagian yang hilang (*corrupted*) dari data
> input itu sendiri.

### **2.2.2 Wav2vec 2.0: Representasi Kontekstual Melalui Contrastive Learning** 

> Wav2vec 2.0 merupakan kerangka self-supervised yang menetapkan fondasi
> kuat dalam pembelajaran representasi ucapan pembelajaran representasi
> ucapan. Arsitektur ini menggunakan *Feature Encoder* konvolusional
> untuk mengekstrak representasi skala waktu rendah dari sinyal audio
> mentah, yang kemudian diumpankan ke jaringan Transformer untuk
> menghasilkan rerpesentasi konteksual. Mekanisme pelatihannya berpusat
> pada Contrastive Loss, yang berfungsi memaksa model untuk membedakan
> segmen ucapan yang di-*masking* dari sejumlah kandidat negatif. Proses
> ini menghasilkan *embedding* sangat efektif dalam mengambil detail
> fonetik dan akustik yang merupakan kunci dalam tugas retrieval
> berbasis *similarity* \[1\]*.*

### **7.1.2 Data2Vec: *Embedding* generalis melalui Self-Distillation** 

> Sebagai model pembanding, Data2Vec memperkenalkan pendekatan
> *self-supervised* yang bersifat *modality-agnostic*, memungkinkan
> pembelajaran representasi yang dapat digunakan pada ucapan, teks
> maupun citra. Perbedaan utama Data2Vec terletak pada mekanisme
> *Self-Distillation*, di mana *Student Model* dilatih untuk memprediksi
> representasi laten kontekstual yang dihasilkan oleh *Teacher Model*.
> Pendekatan ini bertujuan menciptakan *embedding* yang lebih umum dan
> kurang terikat pada suatu modalitas. Karakteristik Data2Vec yang
> generalis ini menjadikannya model yang ideal untuk dikomparasi dengan
> Wav2vec 2.0 yang spesifik ucapan, guna mengevaluasi *trade-off* antara
> generalisi model dan kualitas *embedding* untuk tugas *retrieval*
> fonetik \[8\].
>
> **7.2 Validasi Kualitas Fitur Laten untuk Pencarian Fonetik**
>
> Sub-bab ini berfokus pada pembuktian bahwa representasi laten yang
> diekstrak oleh model SSL tidak hanya bersifat abstrak, tetapi
> terstruktur dan cukup diskriminatif untuk digunakan dalam tugas
> *Similarity Search* berbasis fonetik.
>
> **7.2.1 Bukti Struktur Fitur Fontek dalam *Embedding***
>
> Representasi laten yang dihasilkan oleh Wav2Vec 2.0 terbukti memiliki
> kualitas dan organisasi yang melampaui sekadar representasi akustik
> sederhana. Penelitian *Searching for Structure*. Memberikan
> justifikasi teoretis yang kuat dengan menerapkan *Probing Methods*,
> melatih pengklasifikasi sederhana pada *embedding* yang
> dibekukan---untuk memprediksi fitur-fitur fonetik spesifik (seperti
> *voicing, plosive, dan bilabial*). Hasil analisis asosiasi lebih
> lanjut membuktikan bahwa embedding secara struktural mengorganisasikan
> fitur fonetik dengan cara yang konsisten dan selaras dengan prinsip
> fonologi linguistik teoritis. Pembuktian bahwa *embedding* memiliki
> makna fonetik yang terstruktur ini merupakan landasan ilmiah utama
> untuk hipotesis penelitian ini: bahwa representasi tersebut mampu
> membedakan dan mengukur kemiripan antar bacaan Al-Quran berdasarkan
> kesamaan fonetik dan aturan tajwid yang terinternalisasi.
>
> **7.2.2 Generalisasi dan Adaptasi *Embedding* ke Tugas Kemiripan**
>
> Validitas *embedding* Wav2Vec 2.0 untuk tugas *Similarity Search*
> diperkuat oleh kemampuan generalisasinya di luar *Aumatic Speech
> Recognition* (ASR). Fitur laten ini terbukti efektif dalam memetakan
> kemiripan di berbagai tugas non-ASR lain. Contohnya studi Siamese
> Network menunjukkan bagaimana fitur Wav2Vec, ketika dipasangkan dengan
> Contrastive Loss, dapat secara optimal mempelajari metrik jarak untuk
> tugas diskriminasi. Seperti *anti-spoofing.* Keberhasilan ini
> menegaskan bahwa embedding tersebut secara inheren cocok utnuk
> pemetaan kemiripan.
>
> **7.2.3 Peningkatan *Robustness* Model Terhadap Domain Baru**
>
> Kualitas *embedding* untuk retieval juga bergantung pada stabilitasnya
> di tengah variasi data. Dalam konteks adaptasi model ke domain baru
> seperti audio Al-Quran yang mungkin memiliki keragaman *speaker* dan
> kualitas rekaman, *Continued Pre-training* (CPT) merupakan teknik yang
> relevan menunjukkan bahwa CPT efektif dalam meningkatkan robustness
> model dengan mengadaptasi *embedding* ke kondisi akustik yang berbeda
> (misalnya, kebisingan lingkungan). Pemarapan ini membenarkan bahwa
> *embedding* dapat dipertahankan stabilitasnya melalui teknik adaptasi
> domain, memastikan konsistensi fitur yang diekstrak dari data Al-Quran
> yang bervariasi.
>
> **7.3.1 Prinsip *Similarity Search* Berbasis Vektor**
>
> Metodologi penelitian ini berpusat pada pencarian kemiripan
> (similarity search) di ruang vektor laten, yang bertujuan menemukan
> data dalam database yang memiliki representasi vektor terdekat dengan
> representasi vektor *query* yang diberikan. Dalam konteks penelusuran
> ayat Al-Qur\'an, tugas ini berarti mencari indeks ayat yang paling
> mirip (closest match) berdasarkan embedding audio. Pendekatan ini
> secara fundamental berbeda dari tujuan Automatic Speech Recognition
> (ASR) konvensional karena model tidak dilatih untuk menghasilkan teks,
> melainkan untuk mempertahankan kedekatan fonetik dan kontekstual dalam
> ruang vektor. Keberhasilan dalam tugas ini secara langsung mengukur
> kualitas embedding untuk memetakan kemiripan audio.
>
> **7.3.2 Metrik Kemiripan Vektor: *Cosine Similarity***
>
> Cosine Similarity diimplementasikan sebagai metrik utama untuk
> mengukur kedekatan antara embedding yang diekstrak dari kedua model
> SSL (Wav2Vec 2.0 dan Data2Vec). Cosine Similarity mengukur sudut antar
> dua vektor, sehingga mengukur kesamaan arah dan orientasi fitur sambil
> mengabaikan perbedaan magnitudo vektor. Metrik ini sangat efektif
> dalam konteks *similarity* search karena ia fokus pada kesamaan konten
> fitur yang diwakili oleh arah vektor. Kinerja retrieval kemudian
> dievaluasi menggunakan metrik berbasis pemeringkatan (ranking),
> seperti Top-K Accuracy, untuk menilai seberapa efektif setiap model
> menempatkan ayat target yang benar dalam *K* hasil pencarian teratas.
>
> **7.3.3 Metrik *Baseline* Komparatif ASR (WER/CER)**
>
> Meskipun fokus penelitian adalah *Similarity Search*, metrik ASR
> tradisional, seperti *Word Error Rate* (WER) dan *Character Error
> Rate* (CER), akan diukur sebagai metrik *baseline* komparatif.
> Pengukuran WER/CER bertujuan untuk menilai kapabilitas transkripsi
> awal model dalam domain audio Al-Qur\'an. Data ini akan digunakan
> sebagai konteks untuk menganalisis korelasi antara kinerja transkripsi
> dan kinerja *retrieval* model. Dengan demikian, penekanan utama dan
> klaim kontribusi penelitian diletakkan secara eksklusif pada metrik
> *Similarity Search* (Cosine Similarity dan Top-K Accuracy).

## Metode Penelitian

> Penelitian ini akan dilakukan melalui enam tahapan utama yang
> berurutan, dimulai dari pengumpulan data hingga analisis perbandingan
> hasil *retrieval* kedua model seperti yang dipaparkan dalam Gambar 2.

![[]{#_Toc213132314 .anchor}Gambar 3 Metode
Penelitian](media/image3.png){width="6.268055555555556in"
height="3.3916666666666666in"}

## Pengumpulan data

> Tahap awal penelitian berfokus pada persiapan dan pengumpulan sumber
> data audio dan teks Al-Quran. Untuk sumber audio, penelitian ini akan
> menggunakan dataset audio Al-Quran yang terstruktur dan tersedia untuk
> umum, seperti koleksi Aswat atau dataset serupa yang relevan. Dataset
> audio ini harus dilengkapi dengan data teks Al-Quran yang berpasangan
> secara akurat dengan audio, serta memiliki penandaan (anotasi) ayat
> yang jelas untuk membangun database *retrieval*. Setelah data
> terkumpul, akan dilakukan pembagian data menjadi dua set utama:
> Database (D) dan Query Set (Q). Database (D) akan mencakup keseluruhan
> audio ayat yang berfungsi sebagai target pencarian (*search space*).
> Sementara itu, Query Set (Q) akan terdiri dari kumpulan audio, baik
> dalam bentuk potongan ayat maupun ayat penuh, yang akan digunakan
> sebagai input query untuk menguji dan mengevaluasi kinerja sistem
> *retrieval*.

## Tahap *Pre-processing* Data

> Tahap kedua adalah *pre-processing* data, yang bertujuan untuk
> menstandarisasi seluruh data audio agar siap digunakan untuk ekstraksi
> fitur (*feature extraction*). Proses ini dimulai dengan Normalisasi
> Audio, di mana semua berkas audio dalam *Database* (D) dan *Query Set*
> (Q) akan diseragamkan ke *sampling rate* dan format yang konsisten,
> misalnya 16kHz, mono. Langkah krusial berikutnya adalah Segmentasi
> Ayat, yang memastikan bahwa setiap berkas audio telah tersegmentasi
> secara akurat sesuai dengan batasan ayat Al-Qur'an yang benar. Selain
> itu, sebagai langkah opsional untuk mendukung perhitungan *baseline*
> WER/CER, akan dilakukan Pembersihan Teks Arab, seperti penghilangan
> tanda diakritik (*harakat*) minor untuk menjaga konsistensi data
> transkripsi.

## Tahap Ekstraksi Representasi Laten (Vector *Embedding*)

> Tahap ketiga merupakan inti dari penelitian ini, yaitu menghasilkan
> vector *embedding* dari kedua model *Self-Supervised* (SSL) yang
> penghilangan tanda diakritik (*harakat*) minor untuk menjaga
> konsistensi data transkripsi. Langkah awal adalah pemilihan Model, di
> mana penelitian ini akan menggunakan model Wav2vec2 dan Data2Vec yang
> telah dilatih (*pre-trained*), idealnya menggunakan versi *large* atau
> versi yang telah diadaptasi secara spesifik ke domain bahasa Arab,
> jika tersedia. Proses ekstraksi fitur kemudian dilakukan secara
> pararel untuk kedua model. Untuk Wav2vec2, setiap berkas audio yang
> telah di *pre-processing* dari Database (D) dan *Query Set* (Q)
> diumpankan ke *feature encoder* model. Representasi laten (*hidden
> states*) yang dihasilkan dari lapisan *Transformer* terakhir kemudian
> diekstrak. Setelah itu, dilakukan *Pooling Temporal* (misalnya, *Mean
> Pooling*) pada *hidden states* tersebut untuk mengagregasi seluruh
> urutan fitur menjadi atau vektor tunggal berdimensi tetap per ayat
> (atau per *query*), yang kemudian disimpan sebagai *Embedding*
> Wav2Vec2. Proses yang identik diulangi untuk Model 2 (Data2Vec) guna
> menghasilkan *Embedding* Data2Vec. Hasil akhir tahap ini adalah dua
> database vektor yang komprehensif dan siap dibandingkan.

## Tahap Implementasi *Similarity Search*

> Tahap keempat berfokus pada implementasi mekanisme pencarian kemiripan
> (*Similarity search*) menggunakan *embedding* vektor yang telah
> diekstrak, guna menguji perbandingan kedua model. Untuk setiap audio
> query *q~i~* di *Query Set*. Proses pencarian dilakukan secara
> terpisah untuk setiap model. Pertama, vektor *query V~q,i~* dari model
> terkait diekstrak. Kemudian, dilakukan perhitungan cosine similarity
> antara *V~q,i\ ~*dan seluruh vektor yang tersimpan di dalam *Database
> (D)* yang bersesuaian (*D~Wav2vec2\ ~*atau D~Data2vec~). Hasil
> perhitungan ini menghasilkan daftar peringkat (*ranked list*)
> ayat-ayat di database, yang diturunkan dari skor kemiripan cosine dari
> tertinggi hingga terendah. Penentuan *Retrieval* dilakukan dengan
> menganggap ayat yang memiliki skor kemiripan Cosine tertinggi sebagai
> hasil prediksi model yang paling mendekati *query* audio.

## Tahap Evaluasi Kinerja (Perbandingan)

> Tahap kelima bertujuan untuk membandingkan kedua model SSL secara
> kuantitatif. Evaluasi Kinerja *Retrieval* akan menjadi metrik utama.
> Metrik yang digunakan meliputi *Top-K Accuracy* untuk mengukur
> persentase *query* yang berhasil menempatkan ayat target yang benar di
> dalam *K* peringkat atas (misalnya, Top-1, Top-5, dan Top-10), di mana
> *Top-1 Accuracy* berfungsi sebagai metrik keberhasilan pencarian yang
> paling ketat. Selain itu, Mean Average Precision (MAP) akan dihutung
> sebagai metrik yang lebih komprehensif untuk menilai kualitas urutan
> peringkat dari seluruh hasil *retrieval*.

## Tahap Analisis Hasil

> Tahap akhir penelitian adalah merumuskan temuan. Perbandingan
> Kuantitatif akan dilaukan dengan membandingkan secara langsung metrik
> *Top-K Accuracy* antara Wav2vec2 dan Data2Vec untuk menentukan model
> SSL mana yang unggul dalam tugas *Audio Retrieval* ayat Al-Quran.
> Selain itu, akan dilakukan Analisis Kualitatif melalui studi kasus
> mendalam terhadap *query* yang mengalami kegagalan dan keberhasilan.
> Analisis ini meliputi pengidentifikasian kasus di mana satu model
> unggul (misalnya, Data2Vec mencari ayat dengan konteks yang berbeda
> namun kemiripan fonetik yang kuat), serta menginvestigasi kasus
> kesalah *retrieval* (misalnya, *query* suatu ayat mencari ayat lain
> yang memiliki kesamaan *matras* suara yang sangat tinggi) untuk
> menilai sensitivitas fonetik setiap model. Kesimpulan penelitian yang
> kemudian dirumuskan utuk menentukan model terbaik dan mengaitkan
> temuan empiris dengan justrifikasi teoritis struktur *embedding* yang
> telah dibahas dalam tinjauan pustaka.
>
> **Lokasi Penelitian**
>
> Lokasi penelitian dapat dilakukan dimana saja dikarenakan penelitian
> tidak membutuhkan tempat khusus dalam pengambilan data maupun metode
> yang digunakan.

**Jadwal Penelitian**

+--------+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------+
| **NO** | **KEGIATAN** | **MINGGU**                                                                                                                                                                        | **HASIL          |
|        |              |                                                                                                                                                                                   | KESELURUHAN**    |
|        |              +--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+                  |
|        |              | **1**        | **2**        | **3**        | **4**        | **5**        | **6**        | **7**        | **8**        | **9**        | **10**       | **11**       | **12**       |                  |
+========+==============+==============+==============+==============+==============+==============+==============+==============+==============+==============+==============+==============+==============+==================+
| 1      | Studi        |              |              |              |              |              |              |              |              |              |              |              |              | Landasan teori   |
|        | Literatur    |              |              |              |              |              |              |              |              |              |              |              |              | yang kuat        |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | mengenai         |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | mekanisme        |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | *Self-Supervised |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | Learning*        |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | (Wav2vec2 dan    |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | Data2Vec),       |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | validasi         |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | kualitas *vector |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | embedding* untuk |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | pencarian        |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | fonetik, dan     |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | penentuan metrik |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | evaluasi         |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | *Similarity      |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | Search* (*Top-K  |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | Accuracy* dan    |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | MAP)             |
+--------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+------------------+
| 2      | Pengumpulan  |              |              |              |              |              |              |              |              |              |              |              |              | Dataset siap     |
|        | dataset      |              |              |              |              |              |              |              |              |              |              |              |              | digunakan        |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | (pembersihan,    |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | normalisasi,     |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | tokenisasi)      |
+--------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+------------------+
| 3      | Pembuatan    |              |              |              |              |              |              |              |              |              |              |              |              | Vektor embedding |
|        | Embedding    |              |              |              |              |              |              |              |              |              |              |              |              | untuk setiap     |
|        | (Wav2vec2 &  |              |              |              |              |              |              |              |              |              |              |              |              | ayat/bacaan      |
|        | Data2Vec)    |              |              |              |              |              |              |              |              |              |              |              |              | Qur'an           |
+--------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+------------------+
| 4      | Implementasi |              |              |              |              |              |              |              |              |              |              |              |              | Prototipe sistem |
|        | Sistem       |              |              |              |              |              |              |              |              |              |              |              |              | pencarian        |
|        | Pencarian    |              |              |              |              |              |              |              |              |              |              |              |              | berbasis         |
|        | Semantik     |              |              |              |              |              |              |              |              |              |              |              |              | embedding        |
|        | (Cosine      |              |              |              |              |              |              |              |              |              |              |              |              |                  |
|        | Similarity + |              |              |              |              |              |              |              |              |              |              |              |              |                  |
|        | FAISS)       |              |              |              |              |              |              |              |              |              |              |              |              |                  |
+--------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+------------------+
| 5      | Evaluasi     |              |              |              |              |              |              |              |              |              |              |              |              | Hasil evaluasi   |
|        | Model        |              |              |              |              |              |              |              |              |              |              |              |              | performa model   |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | Wav2Vec2 vs      |
|        |              |              |              |              |              |              |              |              |              |              |              |              |              | Data2Vec         |
+--------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+------------------+
| 6      | Penarikan    |              |              |              |              |              |              |              |              |              |              |              |              | Kesimpulan akhir |
|        | Kesimpulan & |              |              |              |              |              |              |              |              |              |              |              |              | dan naskah       |
|        | penyusunan   |              |              |              |              |              |              |              |              |              |              |              |              | laporan          |
|        | laporan      |              |              |              |              |              |              |              |              |              |              |              |              | penelitian       |
+--------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+------------------+

# DAFTAR PUSTAKA

\[1\] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A
Framework for Self-Supervised Learning of Speech Representations," Oct.
22, 2020, *arXiv*: arXiv:2006.11477. doi: 10.48550/arXiv.2006.11477.

\[2\] P. C. English, J. D. Kelleher, and J. Carson-Berndsen, "Searching
for Structure: Appraising the Organisation of Speech Features in wav2vec
2.0 Embeddings," in *Interspeech 2024*, ISCA, Sep. 2024, pp. 4613--4617.
doi: 10.21437/Interspeech.2024-2047.

\[3\] W. Öberg, "Query-by-Example Audio Search using Acoustic Word
Embeddings: Transforming wav2vec 2.0 Embeddings using Contrastive
Learning," 2025.

\[4\] M. Shanbhogue *et al.*, "Gemini Embedding 2: A Native Multimodal
Embedding Model from Gemini," May 26, 2026, *arXiv*: arXiv:2605.27295.
doi: 10.48550/arXiv.2605.27295.

\[5\] Y. Chen *et al.*, "WavRAG: Audio-Integrated Retrieval Augmented
Generation for Spoken Dialogue Models," Feb. 20, 2025, *arXiv*:
arXiv:2502.14727. doi: 10.48550/arXiv.2502.14727.

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

\[10\] O. Mohamed and S. A. Aly, "Arabic Speech Emotion Recognition
Employing Wav2vec2.0 and HuBERT Based on BAVED Dataset," Oct. 09, 2021,
*arXiv*: arXiv:2110.04425. doi: 10.48550/arXiv.2110.04425.

\[11\] Q. Shao, L. Dong, K. Wei, S. Sun, and L. Xie, "DQ-Data2vec:
Decoupling Quantization for Multilingual Speech Recognition," Jan. 23,
2025, *arXiv*: arXiv:2501.13497. doi: 10.48550/arXiv.2501.13497.

\[12\] Y. Xie, Z. Zhang, and Y. Yang, "Siamese Network with wav2vec
Feature for Spoofing Speech Detection," in *Interspeech 2021*, ISCA,
Aug. 2021, pp. 4269--4273. doi: 10.21437/Interspeech.2021-847.

\[13\] N. Vaessen and D. A. van Leeuwen, "Fine-tuning wav2vec2 for
speaker recognition," in *ICASSP 2022 - 2022 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP)*, May
2022, pp. 7967--7971. doi: 10.1109/ICASSP43922.2022.9746952.

\[14\] L. Alkanhal, A. Alessa, E. Almahmoud, and R. Alaqil, "Aswat:
Arabic Audio Dataset for Automatic Speech Recognition Using
Speech-Representation Learning," in *Proceedings of ArabicNLP 2023*,
Singapore (Hybrid): Association for Computational Linguistics, 2023, pp.
120--127. doi: 10.18653/v1/2023.arabicnlp-1.10.

\[15\] N. San *et al.*, "Leveraging pre-trained representations to
improve access to untranscribed speech from endangered languages," Sep.
14, 2021, *arXiv*: arXiv:2103.14583. doi: 10.48550/arXiv.2103.14583.

\[16\] C. Tang, Y. Wang, X. Chen, and W.-Q. Zhang, "Exploring Effective
Fusion Algorithms for Speech Based Self-Supervised Learning Models,"
Dec. 20, 2022, *arXiv*: arXiv:2212.10092. doi:
10.48550/arXiv.2212.10092.

\[17\] J. W. Yoon, S. M. Kim, and N. S. Kim, "MCR-Data2vec 2.0:
Improving Self-supervised Speech Pre-training via Model-level
Consistency Regularization," Jun. 14, 2023, *arXiv*: arXiv:2306.08463.
doi: 10.48550/arXiv.2306.08463.

\[18\] S. Yang *et al.*, "A Large-Scale Evaluation of Speech Foundation
Models," May 29, 2024, *arXiv*: arXiv:2404.09385. doi:
10.48550/arXiv.2404.09385.

\[19\] Z. Li, L. Wu, T. Li, and Y. Yan, "Improves Neural Acoustic Word
Embeddings Query by Example Spoken Term Detection with Wav2vec
Pretraining and Circle Loss," in *2021 12th International Symposium on
Chinese Spoken Language Processing (ISCSLP)*, Hong Kong: IEEE, Jan.
2021, pp. 1--5. doi: 10.1109/ISCSLP49672.2021.9362065.

\[20\] A. Conneau, A. Baevski, R. Collobert, A. Mohamed, and M. Auli,
"Unsupervised Cross-lingual Representation Learning for Speech
Recognition," Dec. 15, 2020, *arXiv*: arXiv:2006.13979. doi:
10.48550/arXiv.2006.13979.

\[21\] A. A. Attia, D. Demszky, T. Ogunremi, J. Liu, and C. Espy-Wilson,
"CPT-Boosted Wav2vec2.0: Towards Noise Robust Speech Recognition for
Classroom Environments," in *ICASSP 2025 - 2025 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Apr.
2025, pp. 1--5. doi: 10.1109/ICASSP49660.2025.10890830.

\[22\] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, "Librispeech:
An ASR corpus based on public domain audio books," in *2015 IEEE
International Conference on Acoustics, Speech and Signal Processing
(ICASSP)*, South Brisbane, Queensland, Australia: IEEE, Apr. 2015, pp.
5206--5210. doi: 10.1109/ICASSP.2015.7178964.

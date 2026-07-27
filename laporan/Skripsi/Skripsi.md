**PERBANDINGAN WAV2VEC2 DAN DATA2VEC TERHADAP FITUR LATEN AUDIO UNTUK
TILAWAH\
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
[17](#_Toc234440437)](#_Toc234440437)

[**7.1.1 Wav2vec 2.0: Representasi Kontekstual Melalui Contrastive
Learning** [17](#_Toc234440438)](#_Toc234440438)

[**7.1.2 Data2Vec: *Embedding* generalis melalui Self-Distillation**
[17](#_Toc234440439)](#_Toc234440439)

[5 Metode Penelitian [20](#_Toc234440440)](#_Toc234440440)

[1. Pengumpulan data [21](#_Toc234440441)](#_Toc234440441)

[2. Tahap *Pre-processing* Data [21](#_Toc234440442)](#_Toc234440442)

[3. Tahap Ekstraksi Representasi Laten (Vector *Embedding*)
[22](#_Toc234440443)](#_Toc234440443)

[4. Tahap Implementasi *Similarity Search*
[22](#_Toc234440444)](#_Toc234440444)

[5. Tahap Evaluasi Kinerja (Perbandingan)
[23](#_Toc234440445)](#_Toc234440445)

[6. Tahap Analisis Hasil [23](#_Toc234440446)](#_Toc234440446)

[DAFTAR PUSTAKA [26](#_Toc234440447)](#_Toc234440447)

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
|        | Lamya Alkanhal, dkk   |                    |                            | \[10\]                  |
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
|        |                       |                    |                            | dataset \[11\]          |
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
|        |                       | ASR (*automatic    |                            | yang lebih baik \[12\]  |
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
|        | Penulis:              | dilakukan          |                            | pada QbE. \[13\]        |
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
|        |                       | lightweight        |                            | WavLM \[14\].           |
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
> menggabungkan *Siamase Network* pada fitur Wav2vec2 \[15\]. Meskipun
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
> *state of the art* sebelumnya \[11\]. Sejalan dengan itu, Zhaoqi Li,
> dkk membuktikan bahwa *Neural Acoustic Word Embeddings* berbasis
> pretrain Wav2vec2 mengungguli fitur MFCC untuk retrieval berbasis
> kemiripan \[16\]. Temuan-temuan ini menegaskan bahwa kualitas
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
> justru menurun ketika diberi regulasi tambahan \[13\]. Evaluasi
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

1.  **Kecerdasan buatan (Artificial Intelegent)**

> Kecerdasan buatan (Artificial Intelegent) merupakan salah satu cabang
> ilmu komputer yang mempelajari perancangan sistem yang mampu melakukan
> kecerdasan manusia, seperti penalaran, persepsi, pembelajaran, dan
> pengambilan keputusan \[17\]. Russel dan Norvig mengklasifikasikan AI
> ke dalam empat sudut pandang, yaitu sistem yang berpikir seperti
> manusia, bertindak seperti manusia, berpikir secara rasional, dan
> bertindak secara rasional. Pendekatan terakhir yakni rational agent
> yang bertindak untuk mencapai hasil terbaik menurut ukuran kinerja
> tertentu, menjadi kerangka dominan dalam AI modern \[17\].
>
> Dalam perkembangannya, Machine learning (ML) muncul sebagai sub-bidang
> AI yang berfokus pada kemampuan sistem untuk belajar pola dari data
> tanpa dirprogram secara eksplisit untuk setiap kasus \[18\]. ML
> sendiri terbagi menjadi tiga paradigma utama supervised learning
> (belajar dari data yang berlabel), unsupervised learning (menemukan
> struktur dari data tak berlabel), dan self-supervised learning yang
> menjadi fondasi penelitian ini. Penelitian ini berada dalam ranah AI
> karena memanfaatkan model pembelajaraan representasi untuk memahami
> dan mencocokkan sinyal ucapan Al-Qur'an secara otomatis.

2.  **Deep Learning**

> Deep learning adalah cabang dari machine learning yang menggunakan
> jaringan saraf tiruan berlapis yang jumlahnya sangat banyak (*deep
> neural networks*) untuk mempelajari representasi data secara
> berjenjang (*hierarchical representation*) \[19\]. Berbeda dari ML
> tradisional yang bergantung pada feature engineering manual yang mana,
> harus membutuhkan manusia untuk memberikan data yang berlabel, deep
> learning mempelajari fitur secara otomatis. Lapisan-lapisan awal
> menangkap fitur tingkat rendah (misalnya pola frekuensi pada audio),
> sedangkan lapisan yang lebih dalam menyusun fitur tingkat tinggi yang
> lebih abstrak (misalnya unit fonetik) \[19\]
>
> Sebuah jaringan saraf tersusun atas neuron-neuron yang
> mentransformasikan input melalui bobot
> this equation **W**, bias **b**. Dan fungsi aktivasi non-linear *σ,*
> sehingga keluaran satu lapisan dinyatakan sebagai
>
> $$\mathbf{h} = \sigma\left( \mathbf{Wx} + \mathbf{b} \right)$$
>
> Parameter jaringan dioptimasi dengan meminimalkan *loss function*
> melalui algoritma *backpropagation* dan *gradient descent* \[19\].
> Arsitektur yang relevan bagi penelitian ini adalah jaringan konvolusi
> (Convolutional Neural Network, CNN) yang efektif mengekstrak fitur
> local dari sinyal, dan Transformer \[20\] yang memodelkan
> ketergantungan jangka Panjang melalui mekanisme self-attention. Kedua
> arsitektur inilah yang Menyusun tulang punggung model Wav2vec2 2.0 dan
> varian speech dari Data2Vec.

3.  **Similarity Search**

> Similarity search adalah proses menemukan objek-objek dalam suatu
> database yang paling mirip dengan sebuah objek kueri (data input),
> berdasarkan ukuran kemiripan tertentu \[21\]. Dalam paradigma modern,
> setiap objek, baik teks, gambar, maupun audio, direpresentasikan
> sebagai vector bilangan riil di ruang berdimensi tinggi, sehingga
> pencarian kemiripan tereduksi menjadi persoalan *Nearest Neighbor
> Search* di ruang vector tersebut \[21\]
>
> Formalnya, diberikan sebuah vector kueri **q** dan himpunan vector
> database **\**
> D={**d**1​,...,**d***N*​} , sistem menghitung skor kemiripan
> *s*(**q**,**d***i*​) untuk setiap **d***i* , lalu mengurutkan hasilnya
> secara menurun untuk menghasilkan daftar peringkat (*ranked list*).
> Pendekatan ini menjadi tulang punggung sistem information retrieval
> modern \[21\]. Dalam penelitian ini, similarity search diterapkan pada
> embedding audio ayat Al-Qur'an. Data kueri dibandingkan langsung
> terhadap database embedding ayat tanpa melalui tahap transkripsi,
> sehingga menghindari propagasi kesalah dari tahap ASR (*automatic
> speech recognition*).

4.  **Vector Embedding**

> Vector Embedding adalah representasi suatu objek dalam bentuk vector
> yang padat berdimensi tetap $v \in R^{d}\$yang dirancang sedemikian
> rupa sehingga kedekatan geometrics antar-vektor mencerminkan kemiripan
> semantic atau perseptual antar objek yang diwakilinya \[19\]. Prinsip
> utamanya adalah distributional hypothesis, yang mana objek objek yang
> serupa akan menempati posisi yang berdekatan di dalam ruang vector
> (*embedding space*)
>
> Pada domain audio, embedding dihasilkan oleh model self-supervised
> seperti Wav2vec2 2.0 dan Data2vec, yang memetakan sinyal audio mentah
> menjadi barisan vector kontekstual. Embedding inilah yan gmenjadi
> objek perbandingan dalam penelitian ini. Kualitas sebuah embedding
> untuk tugas retrieval ditentukan oleh sejauh mana ia mengelompokkan
> (cluster) ayat yang sama secara berdekatan sekaligus memisahkan ayat
> yang berbeda, suatu sifat geometric yang menjadi dasar hipotesis pada
> penelitian ini.

5.  **Self-Supervised Learning (SSL)**

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

6.  **Wav2vec2 2.0**

> Wav2vec2 2.0 adalah sebuah model self-supervised learning (SSL) yang
> mempelajari representasi dari ucapan langsung atau audio mentah
> melalui pendekatan *constrastive learning*. Arsitekturnya terdiri dari
> tiga komponen inti; feature encoder konvolusional, jaringan konteks
> berbasis Transformer, dan modul kuantisasi yang bekerja dalam satu
> tugas pretext. Memprediksi unit terakuntisasi yang benar dari
> representasi konteks pada posisi yang ditutup (*masked).*

1.  **Feature encoder**

> Sinyal audio mentah $\mathcal{X}$ (16 kHz) diproses oleh tujuh blok
> konvolusi temporal, masing-masing berisi 512 kanal dengan Layer
> Normalization dan aktivasi GELU. Susunan stride $(5,2,2,2,2,2,2)$ dan
> lebar kernel $(10,3,3,3,3,2,2)$ menghasilkan barisan representasi
> laten $z_{1},\ldots,z_{T}$ pada frekuensi keluaran sekitar 49 Hz
> (jarak antar-frame $\approx$`<!-- -->`{=html}20 ms), dengan receptive
> field selebar 400 sampel input atau setara 25 ms audio.

  ------------------------------------------------------------------
  Blok            Stride           Kernel           Kanal
  --------------- ---------------- ---------------- ----------------
  1               5                10               512

  2-5             2                3                512

  6-7             2                2                512
  ------------------------------------------------------------------

2.  **Jaringan konteks**
    $\mathbf{g:}\mathcal{Z}\mathbf{\rightarrow}\mathcal{C}$

> Representasi laten $z$ diumpankan ke jaringan Transformer N4 yang
> memodelkan ketergantungan jangka panjang antar-frame melalui
> self-attention, menghasilkan representasi kontekstual
> $c_{1},\ldots,c_{T}$. Konfigurasi Base memakai 12 blok
> ($d = 768,\ 8\ Head$, sedangkan Large memakai 24 blok ( $d = 1024$, 16
> head). Informasi posisi ditanamkan melalui relative positional
> convolutional embedding (kernel 128, 16 grup), bukan positional
> encoding absolut. Representasi laten $z$ diumpankan ke jaringan
> Transformer N4 yang memodelkan ketergantungan jangka panjang
> antar-frame melalui self-attention, menghasilkan representasi
> kontekstual $c_{1},\ldots,c_{T}$ Konfigurasi Base memakai 12 blok
> ($d = 768$, 8 head), sedangkan Large memakai 24 blok ($d = 1028$, 16
> head). Informasi posisi ditanamkan melalui relative positional
> convolutional embedding (kernel 128, 16 grup), bukan positional
> encoding absolut

3.  **Modul kuantisasi**

> Secara paralel, laten \$\\mathbf{z}\$ didiskretisasi melalui product
> quantization N9 dengan \$G=2\$ codebook, masing-masing \$V=320\$
> entri. Pemilihan entri dibuat terdiferensiasi menggunakan
> Gumbel-Softmax N7, N8:
>
> $$p_{g,v} = \frac{\exp{\text{!}\backslash}big\left( \left( l_{g,v} + n_{v} \right)\text{/}\tau\backslash big \right)}{\sum_{k = 1}^{V}\exp{\text{!}\backslash}big\left( \left( l_{g,k} + n_{k} \right)\text{/}\tau\backslash big \right)}$$
>
> dengan $l_{g,v}$ logit entri $v$ pada grup $g$, $\tau$ temperatur
> (dijadwalkan turun dari 2 ke 0,5), dan
> $n_{v} = - \log\left( - \log\left( u_{v} \right) \right)$,
> $u_{v}\mathcal{\sim U}(0,1)\$derau Gumbel. Pada forward pass dipilih
> entri diskret $i = \arg{\max_{j}p_{g,j}}$; pada backward pass
> digunakan gradien fungsi Gumbel-Softmax kontinu (straight-through
> estimator N10) sehingga tetap dapat dilatih.

4.  **Fungsi Objektif**

> Pelatihan diarahkan gabungan dua komponen
>
> $$\mathcal{L =}\mathcal{L}_{\mathcal{m}} + \alpha\,\mathcal{L}_{\mathcal{d}},\quad\quad\alpha = 0,1$$
>
> Constrastive loss $\mathcal{L}_{\mathcal{m}}$ memaksa model membedakan
> target benar $q_{t}$ dari $K = 100$ pengecoh:
>
> $$\mathcal{L}_{\mathcal{m}} = - \log{}\frac{\exp\left( \frac{\text{sim}\left( c_{t},q_{t} \right)}{\kappa} \right)}{\sum_{\widetilde{q} \sim \mathcal{Q}_{\mathcal{t}}}^{}\exp\left( \frac{\text{sim}\left( c_{t},\widetilde{q} \right)}{\kappa} \right)}$$
>
> Dengan $\kappa = 0,1$ dan
> $\text{sim}(a,b) = \frac{}{a^{\top b}\text{|}a\text{|}\,\text{|}b\text{|}}$
> (*cosine similarity*, berpersan sebagai scoring function). Diversity
> loss $\mathcal{L}_{\mathcal{d}}$ mendorong pemakaian merata seluruh
> entri codebook via maksimalisasi entropi:
>
> $$\mathcal{L}_{\mathcal{d}} = \frac{1}{GV}\sum_{g = 1}^{G}{\sum_{v = 1}^{V}\overline{p_{g,v}\log\overline{p_{g,v}}}}$$
>
> Melalui mekanisme kontrastif inilah Wav2vec2 2.0 dihipotesiskan
> menghasilkan embedding yang lebih clusterable untuk retrieval berbasis
> cosine \[22\]

![Gambar 2 Arsitektur Wav2vec2 2.0](media/image3.png){width="2.775in"
height="4.464435695538057in"}

7.  **Data2vec**

> Data2vec merupakan model Self-Supervised Learning (SSL) yang modality
> agnostic, artinya inputan yang dapat diproses dapat berupa audio,
> gambar ataupun text. Perbedaan mendasarnya dengan Wav2vec2 2.0
> terletak pada bentuk prediksi, Data2vec tidak mempredisksi unit
> diskret terkuantisasi, melainkan representasi laten kontekstual yang
> bersifat kontinu. Yang dihasilkan oleh model itu sendiri melalui skema
> self-distillation

1.  **Skema Teacher dan Student (*self-distillation*)**

> Satu arsitektur Transformer \[20\] beroperasi dalam dua peran.
> Jaringan student yang memproses input yang dimasking dan menghasilkan
> prediksi $f_{t}(x)$ pada tiap posisi termasking. Jaringan teacher
> memproses input penuh (tanpa masking) dan menyediakan target regresi
> $y_{t}$. Student dilatih meregresi target teacher hanya pada posisi
> termasking

2.  **Pembaruan Teacher via EMA**

> Pembaruan teacher tidak dilakukan oleh gradient, melainkan menggunakan
> exponential moving average (EMA) dari parameter student
> $\Delta \leftarrow \tau\,\Delta + (1 - \tau)\,$dengan $\Delta$
> parameter teacher, $\theta$ parameter student. Nilai $\tau$ dinaikan
> linear dari $\tau_{0}\ ke{\ \tau}_{e}\$langkah awal lalu ditahan
> konstan, untuk ucapan $\tau_{0} = 0,,\tau_{e} = 0,,\tau_{n} = 30.0$

3.  **Konstruksi Target Kontekstual**

> Target dibentuk dengan merata-ratakan keluaran $K$ blok Transformer
> teratas teacher, setelah masing-masing dinormalisasi
>
> $$y_{t} = \frac{1}{K}\sum_{l = L - K + 1}^{L}\widehat{a_{t}^{\, l}}$$
>
> Dengan $L$ dengan jumlah total blok (12 pada base), $K = 8$ blok
> teratas $\widehat{a_{t}^{\, l}}$ keluaran blok $l$ ternormalisasi.
> Untuk ucapan digunakan *instance nomralization* \[23\]. Untuk gambar
> layer normalization \[24\]. Normalisasi ini mencegah keruntuhan
> representasi (representation collapose). Karent target menggabungkan
> berupa lapisan atas, ia bersifat kontekstual penuh. Hal ini menjadi
> dasar hipotesis bahwa Data2ve menghasilkan fitur fonetik lebih kaya
> \[10\], \[14\].

4.  **Fungsi Objektif**

> Student meregresi target dengan Smooth L1 loss (huber):
>
> $$\mathcal{L}\left( y_{t},f_{t}(x) \right) = \left\{ \begin{matrix}
> \frac{1}{2}\left( y_{t} - f_{t}(x) \right)^{2}\text{/}\beta, & \left| y_{t} - f_{t}(x) \right| \leq \beta \\
> \left| y_{t} - f_{t}(x) \right| - \frac{1}{2}\beta, & \text{selainnya}
> \end{matrix} \right.\ $$
>
> Parameter $\beta$ mengatur peralihan wilaayah kuadratik
> $\left( l_{2} \right)$ dan linear $\left( l_{1} \right)$ untuk
> $\beta \rightarrow \infty$ loss menjadi MSE murni yang bekerja baik
> pada domain ucapan.

![Gambar 3 Arsitektur
Data2vec](media/image4.png){width="2.4833333333333334in"
height="4.357854330708661in"}

8.  **Cosine Similarity**

> Cosine Similarity mengukur kemiripan antara dua vektor berdasarkan
> sudut cosine di antara keduanya, tanpa perlu memperhitungkan besar
> magnitudonya (bx5)
>
> $$\text{sim}\,(q,d) = \cos\theta = \frac{q \cdot d}{\text{|}q\text{|}\,\text{|}d\text{|}} = \frac{\sum_{i = 1}^{d}{q_{i}d_{i}}}{\sqrt{\sum_{i = 1}^{d}q_{i}^{2}}\,\sqrt{\sum_{i = 1}^{d}d_{i}^{2}}}$$
>
> Nilainya berkisar pada \[-1, 1\], dengan nilai mendekati 1 menandakan
> arah vektor yang sangat mirip. Perlu ditegakan bahwa dalam bidang
> retrieval, cosine similarity berperan sebagai fungsi penilaian
> (*scrolling function*), bukan sebagai metrik evaluasi \[21\]. Manning
> dkk. Membedakkan secara tegas antara mekanisme scoring ranking dan
> metrik evaluasi efektivitas sistem. Selain itu cosine distance
> $\left( 1 - \cos\theta \right)$ bukan metrik jarak matematis, arena
> melanggar pertidaksamaan segitiga \[25\]. oleh karena itu, dalam
> penelitian ini cosine similarity digunakan sebagai fungsi penilai
> kemiripan untuk menghasilkan peringkat, sedangkan efektivitas
> retrieval diukur menggunakan metrik evaluasi yang dijelaskan pada
> sub-bab berikutnya.

9.  **Metriks Evaluasi Retrieval**

> Efektivitas sistem retrieval dievaluasi menggunakna metrik yang
> berbasis pada kualitas peringkat (ranking) hasil pencarian \[21\].
> Penelitian ini menggunakan tiga metrik utama berikut.

1.  **Top K Accuracy**

> Metrik ini mengukur proporsi kueri yang dokumen relevannya ditemukan
> dalam K peringkat teratas dalam pencarian
>
> $$Top - K\, Accuracy = \ \ 1\  \div |Q|\sum_{q \in Q}^{}{1\left\lbrack rank(q) \leq K \right\rbrack}$$
>
> Dengan ∣Q∣ adalah jumlah total kueri, dan 1\[⋅\] adalah fungsi
> indikator yang bernilai 1 apabila dokumen relevan berada pada
> peringkat kurang dari atau sama dengan K, dan berniali 0 jika
> sebaliknya. Metrik ini mencerminkan kegunaan praktis sistem bagi
> pengguna yang umumnya hanya memeriksa hasil teratas

2.  **Mean Reciprocal Rank (MRR)**

> MRR mengukur seberapa tinggi peringkat dokumen relevan pertama, dengan
> menghitung rata-rata kebalikan peringkatnya \[21\]. MRR memiliki rumus
> demikian
>
> $$MRR = \frac{1}{|Q|}\sum_{i = 1}^{|Q|}\frac{1}{\text{rank}_{i}}$$
>
> Dengan $\text{rank}_{i}$ posisi peringkat dokumen relevan pertama
> untuk kueri ke-$i$. Nilai MRR mendekati 1 menunjukkan dokumen relevan
> cenderung berada di peringkat paling atas.

3.  **Mean Average Precision (MAP)**

> MAP merupakan metrik yang paling komprehensif karena memperhitungkan
> presisi pada setiap posisi dokumen relevan sepanjang daftar peringkat
> (bx5). Untuk sebuah kueri, Average Precision (AP) dihitung sebagai
>
> $$\text{AP} = \frac{1}{|R|}\sum_{k = 1}^{N}{P(k)} \cdot \text{rel}(k)$$
>
> Dengan $P(k)$ presisi pada peringkat $k$ $\text{rel}(k)$ fungsi
> indicator relevansi dokumen di peringkat $k$, $|R|$ jumlah total
> dokumen relevan, dan $N$ jumlah dokumen. MAP kemudian adalah rata-rata
> AP atas seluruh kueri
>
> $$\text{MAP} = \frac{1}{|Q|}\sum_{q = 1}^{|Q|}{\text{AP}(q)}$$
>
> Ketiga metrik ini; Top-K Accuracy, MRR, dan MAP merupakan metrik
> evaluasi yang secara konseptual berbeda dengan cosine similarity yang
> berperan sebagai *scoring function*

10. **Dataset Quran-MD**

> Penelitian ini menggunakan dataset Quran-MD (bx7), sebuah dataset
> multimodal yang tersedia pada platform huggingface. Dataset ini
> mengintegrasikan dimensi teks, linguistic, dan audio pada tingkat ayat
> maupun kata. Untuk setiap ayat disediakan teks Arab asli serta rekaman
> dari 30 pembaca (qori) yang berbeda guna mempresentasikan keragaman
> gaya bacaan (qiraat) dan nuansa dialetik.
>
> Secara kuantitatif, koleksi Quran-MD terdiri atas dua sub-dataset
> terpisah. sub-dataset ayat (Buraaq/quran-md-ayahs) berisi 187080
> sampel rekaman ayat lengkap. Dan sub-dataset kata
> (Buraaq/quran-md-words) berisi 77429 sampel pelafalan kata individual,
> sehingga totalnya mencakup sekitar 264509 audio format berformat mp3
> yang meliputi 114 surah dan 6236 ayat unik (bx7). Sesuai batasan
> masalah penelitian, objek yang digunakan dibatasi pada surah
> Al-fatihah dan juz 30 dari sub-dataset tingkat ayat. Pasangan
> audio-teks yang tersedia pada dataset ini memungkinkan penerapan tugas
> similarity search berbasis suara tanpa transkripsi.

11. **Metode Penelitian CRISP-DM**

> Penelitian ini mengadopsi kerangka kerja (*framework*) Cross-industry
> Standard Process for Data Mining (CRISP-DM) sebagai metodologi (bx8).
> CRISP-DM merupakan model process standar yang berisifat iteratif dan
> tidak bergantung pada industri maupun teknologi tertentu, sehingga
> sesuai untuk penelitian berbasis data dan machine learning. Kerangka
> ini terdiri atas enam fase yang saling terikat:

1.  Business Understanding. Merumuskan tujuan penelitian, yaitu
    membandingkan efektivitas embedding Wav2vec2 dan Data2vec untuk
    retrieval ayat Al-Qur'an, beserta hipotesis H1 dan H2.

2.  Data understanding. Mengekplorasi karakteristik dataset Quran-MD,
    mencakup distribusi qari, kualitas audio, dan cakupan surah yang
    digunakan.

3.  Data preparation. Menyeleksi Surah Al-Fatihah dan Juz Amma,
    melakukan pre-processing audio dan menyiapkan pasangan kueri dan
    database untuk pengujian

4.  Modeling. Mengekstrak embedding dari kedua model SSL tanpa adanya
    fine-tuning, lalu menghitung skor kemiripan menggunakan cosine
    similarity untuk menghasilkan peringkat retrieval.

5.  Evaluation. Mengukur kinerja retrieval kedua model menggunakan
    metrik MAP, Top-K Accuracy, dan MRR.

6.  Deployment. Menyusun kesimpulan dan rekomendasi model terbaik
    sebagai acuan bagi pemngembangan system pembelajaran Al-Quran
    berbasis audio.

# BAB III  METODOLOGI PENELITIAN

Metodologi penelitian yang diadaptasi dari penelitian ini yaitu
*framework* *Cross-Industry Standard Process for Data Mining*
(CRISP-DM). Dikarenakan framework ini bersifat sistematis, adaptif serta
iterative dan tidak terikat pada teknologi tertentu, sehingga sesuai
dengan penelitian berbasis data dan *Machine Learing.* Namun demikian,
karena penelitian ini bertujuan untuk membandingkan efektivitas fitur
laten model Self-Supervised Learning tanpa dilakukan *transfer
learning,* maka terdapat dua fase dalam framework CRISP-DM yang perlu
disesuaikan agar selaras dengan tujuan penelitian ini, yaitu *Modeling*
dan *Deployment*. Fase *Modeling* tidak berupa pelatihan model,
melainkan ekstrasi representasi laten. Sementara fase *Deployment*
diarahkan pada sintesis temuan, pengujian hipotesis, dan perumusan
rekomendasi arsitektur, bukan penerapan sistem dalam lingkungan
produksi. Dengan penyesuaian tersebut, proses penelitian dilaksanakan
secara terstruktur melalui enam fase, mulai dari *Business
Understanding*, *Data Understanding*, *Data Preparation*, *Modeling*,
*Evaluation* dam *Deployment* yang diadaptasi.

## 3.1. Business Understanding

Fase *business understanding* berfungsi menerjemahkan permasalahan
penelitian menjadi sasaran analistis yang terukur \[26\]. Tujuan utama
yang ditetapkan dalam penelitian ini adalah mengukur dan membandingkan
efektivitas fitur laten dua model Self-Supervised Learning (SSL) model,
yaitu Wav2vec2 dan Data2vec. Dalam tugas *retrieval* ayat Al-Quran.
Perbandingan dilakukan dalam kondisi *frozen.*Penelitian ini
memanfaatkan model *pre-trained* tanpa pelatihan ulang (fine-tuning).
Sehingga metrik yang dihasilkan merupakan raw performance masing masing
model.

Sasaran tersebut menghasilkan dua hipotesis yang saling berkebalikan
sebagai inti kontribusi penelitian. Hipotesis pertama memperkirakan
Data2vec unggul, karena target prediksinya yang berupa representasi
kontekstual laten penuh berpotensi menghasilkan fitur fonetik yang kaya
dan umum. Hipotesisi kedua memperkirakan justru Wav2vec2 yang unggul
pada retrieval. Karena constrastive-loss-nya secara explisit menari
pasangan positif dan menolak kandidat negatif, sehingga cenderung
menciptakan ruang embedding yang lebih clusterable dan lebih selaras
dengan pengukuran jarak consine-properti yang justru menentukan dalam
kasus retrieval. Penelitian ini tidak hanya membandingkan angka kinerja,
melainkan menguji apakah keunggulan suatu paradigma pretraining bersifat
lintas tugas atau justru spesifik terhadap jenis tugas.

Agar kedua hipotesis dapat diuji secara objektif, pada fase ini
ditetapkan pula definisi relevansi yang menjadi dasar penyusunan ground
truth. Sebuah rekaman dinyatakan relevan terhadap suatu query apabila
keduanya merupakan ayat yang sama meskipun dibacakan oleh pembaca (qori)
yang berbeda. Definisi ini memungkinkan sistem retrieval diuji atas
kemampuannya mengenali kesamaan konten ayat lintas variasi pembaca, gaya
bacaan, dan nuansa dialektik. Kriteria keberhasilan penelitian kemudian
ditetapkan menggunakan metrik evaluasi retrieval berbasis kualitas
peringkat, yaitu Mean Average Precision (MAP), Top-K Accuracy, dan Mean
Reciprocal Rank (MRR) \[21\]. Sejalan dengan hal tersebut, cosine
similarity diposisikan sebagai fungsi penilaian (scoring function) untuk
menghasilkan peringkat, bukan sebagai metrik evaluasi.

Penelitian ini dibatasi pada penggunakan pre-trained model tanpa
fine-tuning , dengan objek teliti berupa surah Al-Fatihah dan Juz Amma,
serta pengujian pada tingkat ayat. Salah satu sasaran analitis yang
turut dirumuskan adalah identifikasi lapisan (layer) yang paling optimal
bagi tugas *retrieval* fonetik. Perumusan sasaran ini dilandasi temuan
bahwa informasi fonetik dan leksikal pada model SSL tidak terdistribusi
secara merata di seluruh lapisan, melainkan terkonsentrasi pada lapisan
tertentu yang bergantung pada paradigma pre-training. Pada model yang
menggunakan constrastive seperti Wav2vec2, informasi fonetik cenderung
meningkat di layer tengah dan menurun di layer akhir \[27\]. Sehingga
lapisan terbaik untuk sebuah studi kasus tidak selalu berada pada
lapisan terkahir. Dengan demikian penelitian ini sejak awal diarahkan
untuk melakukan ekstrasi secara *layer-wise*.

2.  **Data Understanding**

Fase Data Understanding bertujuan mengenali karakteristik data secara
menyeluruh sebelum dilakukan pemrosesan mencakup pengumpulan data awal,
pendeskripsian sifat data, eksplorasi pola, serta verifikasi kualitasnya
\[26\]. Pada fase data understanding, tidak adanya perubahan data. Namun
mengamati data dan menilai data, sehingga data data yang ada masih
mentah. Pemahaman yang ada pada proses ini menjadi landasan untuk proses
pre-processing kedepannya. Sekaligus memastikan bahwa data yang tersedia
memang mampu mendukung tujuan penelitian.

Data yang akan digunakan bersumber dari dataset Quran-MD yang tersedia
pada platform HuggingFace, khususnya sub-dataset pada tingkat ayat
\[28\]. Dataset ini menyediakan pasangan audio-teks untuk setiap ayat
disertai rekaman audio dari 30 qori yang berbeda yang bertujuan untuk
merepresentasikan keragaman gaya bacaan ayat Al-Quran. Pada tahap
pengumpulan awal, akan didokumentasikan struktur dataset yang meliputi
berkas audio, teks Arab yang berpasangan, serta metadata pendanda ayat
berupa nomor surah, nomor ayat, dan identitas qori.

Selanjutnya akan dilakukan analisis file audio. Mencakup format audio,
sampling rate asli dan jumlah kanal. Pendeskripsian ini penting karena
untuk model Wav2vec2 dan Data2vec mengharuskan input audio pada sampling
rate 16kHz dan pada kanal mono. Selain itu juga akan ada perhitungan
terkait distribusi jumlah qori per ayat, distribusi durasi rekaman antar
ayat, serta distribusi jumlah ayat pada tiap surah dalam subset yang
digunakan. Proses ini memiliki peran krusial karena menentukan banyaknya
dokumen relevan bagi setiap query, yang secara langsung memengaruhi
perhitungan metrik Mean Average Precision (MAP) dan Mean Reciprocal Rank
(MRR). Sementara itu , distribusi durasi ayat perlu dipahami karena
ketimpangan durasi antara ayat pendek dan ayat panjang berpotensi
memengaruhi hasil agregasi fitur melalui *temporal pooling* pada tahap
ekstraksi representasi laten. Temuan dari fase Data Understanding ini
akan menjadi landasan bagi penetapan prosedur pre-processing dan
penyususan ground truth pada fase-fase selanjutnya.

Dataset yang digunakan dalam penelitian ini terdiri atas rekaman video
bacaan Juz Amma dan surah Al-Fatihah yang berasal dari mahasiswa Program
Studi Teknik Informatika UIN Sunan Gunung Djati Bandung. Rekaman
tersebut dikumpulkan sebagai bagian dari pemenuhan mata kuliah Tahfidz
dan persyaratan sidang komprehensif. Setiap data disimpan dalam
direktori yang diidentifikasi berdasarkan kombinasi Nomor Induk
Mahasiswa (NIM) dan nama mahasiswa, sedangkan setiap direktori memuat
sejumlah berkas video yang merepresentasikan bacaan surah tertentu.
Meskipun demikian, konvensi penamaan direktori dan berkas tidak bersifat
homogen. Ditemukan berbagai variasi penamaan, seperti perbedaan
kapitalisasi huruf, penggunaan simbol atau karakter khusus yang beragam,
serta penggunaan format penamaan yang tidak secara eksplisit mengacu
pada nomor atau nama surah. Heterogenitas struktur penamaan tersebut
menjadi salah satu tantangan utama dalam tahap prapemrosesan
(*preprocessing*), sehingga diperlukan proses normalisasi untuk
menghasilkan representasi data yang konsisten dan dapat diproses secara
otomatis.

3.  **Data Preparation**

Fase Data Preparation bertujuan mengubah mentah menjadi data yang siap
digunakan untuk proses ekstrasi representasi laten. Fase ini mencakup
empat tahap yaitu seleksi data, normalisasi audio, verifikasi segmentasi
ayat, kontruksi ground truth sebagai dasar evaluasi retrieval.

Proses pertama adalah seleksai data sesuai dengan Batasan masalah, yakni
memilih rekaman dari surah Al-Fatihah dan surah-surah dalam Juz Amma
yang disusun berdasarkan ayat. Dari proses seleksi ini akan disusun
identifier yang terdiri dari qori, surah dan ayat. Identifier ini yang
kelak akan digunakan sebagai referensi yang akan disimpan dalam table
lain pada database.

Proses kedua adalah normalisasi data, normalisasi data adalah
menstandarisasi seluruh data audio yang akan dijadikan *ground truth*
dan juga data input query yang nantinya digunakan sebagai data testing.
Langkah ini untuk menjadi solusi dari pembahasan data understanding yang
mengungkap bahwa data Quran-MD itu memiliki audio sampling rate dan
jumlah kanal yang berbeda. Sehingga perlu dilakukan konversi pada data
yang berbeda untuk sampling rate 16Khz dan jumlah kanal menjadi tunggal
(*mono*). Hal ini bertujuan supaya hasil dari representasi laten antara
model Wav2vec2 dan Data2vec berasal dari performa masing-masing model
tersebut, bukan dari ketidaksesuain data yang ada.

Proses ketiga adalah *pre-processing* diawali dengan proses normalisasi
struktur dataset, yaitu menyeragamkan penamaan direktori dan berkas
berdasarkan nomor serta nama surah yang sesuai dengan urutan dalam
Al-Qur\'an. Proses ini dilakukan untuk mengatasi ketidakkonsistenan
penamaan, seperti perbedaan penggunaan huruf besar dan huruf kecil,
variasi simbol, maupun format penamaan yang tidak mengikuti konvensi
tertentu. Hasil normalisasi tersebut digunakan sebagai dasar dalam
proses identifikasi surah secara otomatis pada tahap selanjutnya.

Setelah proses normalisasi selesai, dilakukan segmentasi ayat dan
verifikasi terhadap data. Data yang digunakan sebagai *input query*
berupa rekaman video mahasiswa yang dalam satu videonya membacakan satu
surah secara utuh, mencakup seluruh surah dalam Juz Amma beserta surah
Al-Fatihah. Oleh karena itu, setiap rekaman terlebih dahulu diekstraksi
menjadi berkas audio, kemudian dilakukan proses segmentasi untuk membagi
audio surah menjadi beberapa segmen yang masing-masing merepresentasikan
satu ayat. Segmentasi dilakukan dengan memanfaatkan hasil *forced
alignment* sehingga setiap ayat memiliki batas waktu (*timestamp*) awal
dan akhir yang akurat. Selanjutnya, setiap hasil segmentasi diverifikasi
untuk memastikan bahwa batas potong audio sesuai dengan ayat yang
dibacakan. Hasil akhir dari tahap ini adalah kumpulan berkas audio per
ayat yang telah terverifikasi dan siap digunakan pada proses ekstraksi
fitur serta pembentukan *embedding*.

4.  **Modeling**

Fase Modeling pada penelitina ini diadaptasi dari makna aslinya dalam
*framework* CRISP-DM. karena penelitian ini menggunakan *frozen
embedding* tanpa fine-tuning atau transfer learning, fase ini tidak
melibatkan pelatihan model, melainkan befokus pada ekstraksi
representasi laten dari mdoel pre-trained serta konstruksi *similarity*
yang menjadi dasar proses retrieval. Fase ini terdiri atas empat
sub-proses yaitu pemilihan model, ekstraksi representasi laten secara
layer-wise, agregasi fitur menjadi vektor embedding, dan perhitungan
skor kemiripan untuk menghasilkan peringkat.

Pada sub proses pertama yaitu pemilihan model, penelitian ini
menggunakna dua model self-supervised Learning (SSL), yaitu Wav2vecc2
dan Data2vec, dalam kondisi *frozen*. Keduanya dipilih karena
menggunakan paradigma pretraining yang berbeda, yaitu pendekatan
contrastive pada Wav2vec2 dan pendekatan self-distillation dengan target
laten kontekstual pada Data2vec. Perbedaan paradigma inilah yang menjadi
objek perbandingan utama dan menjadi dasar bagi pengujian hipotesis H1
dan H2.

Proses selanjutnya adalah ekstrasi representasi laten. Setiap audio yang
telah dinormalisasi, baik pada Database (D) maupun Query set (Q),
diinputkan ke masing-masing model untuk memperoleh representasi laten.
Ekstrasi tidak dibatasi pada lapisan transformer terkahir, melainkan
dilakukan pada seluruh lapisan. Pendekatan layer-wise ini diperlukan
karena informasi fonetika dan leksikal pada model SSL tidak
terdistribusi merata di seluruh lapisan, melainkan terkonsentrasi pada
lapisan tertentu yang bergantung pada paradigma pretraining. Pada model
bertipe bertipe contrastive seperti Wav2vec2, informasi fonetik
cenderung memuncak di lapisan mengengah dan menurun pada lapisan akhir,
sehingga lapisan terbaik untuk guatu tugas belum tentu merupakan lapisan
terakhir \[27\]. Dengan mengekstrasi seluruh lapisan, penelitian dapat
menganalisis distribusi kinerja antar lapisan dan mengindentifikasi
laipsan yang paling optimal bagi tugas retrieval fonetik.

Sub proses ketiga adalah agregasi fitur. Representasi laten pada tiap
lapisan berupa urutan vektor sepanjang durasi audio, sehingga perlu
digaregasi menjadi satu vector berdimensi tetap agar dapat dibandingkan
antar audio. Agregasi dilakukan melalui temporal pooling, dalam hal ini
mean pooling, yang merata-ratakan seluruh vector sepanjang dimensi
waktu. Hasilnya adalah satu vector embedding per rekaman untuk setiap
kombinasi model dan lapisan, yang selnajutnya disusun menjadi basis data
vector terpisah bagi masing-masing kombinasi tersebut.

Sub proses terakhir adalah perhitungan *similarity score*. Untuk setiap
input query, vector embedding nya akan dibandingkan terhadap seluruh
vector pada Database menggunakan cosine similarity, kemudian hasilnya
durutkan dari skor tertinggi hingga terendah untuk membetuk daftar
peringkat (ranked list). Perlu ditegaskan bahwa dalam penelitian ini
cosine similarity berperan sebagai scoring function yang menghasilkan
peringkat, bukan sebagai metrik evaluasi \[21\]. Ayat dengan skor
kemiripan tertinggi dianggap sebagai hasil retrieval yang paling
mendekati query, sedangkan kualitas keseluruhan peringat akan diukur
pada fase valuasi menggunakan metrik yang sesuai.

5.  **Evaluation**

Fase ini bertujuan untuk mengukur efektivitas retrieval dari kedua model
secara kuantitafif dan menguji hipotesis penelitian berdsaarkan hasil
yang diperoleh. Pada fase inilah peringkat yang dihasilkan cosine
similarity akan dinilai menggunakan metrik evaluasi retrieval. Metrik
evaluasi menilai kualitas keseluruhan peringkat terhadap ground truth,
bukan menghitung kemiripan antar vector \[21\]. Fase ini terdiri atas
empat sub proses yaitu perhitungan metrik evaluasi, evaluasi layer-wise,
pengujian hipotesis dan analisis kualitatif.

Sub proses pertama adalah perhitungan metrik evaluasi retrieval.
Penelitian ini menggunakan tiga metrik berbasis kualitas peringkat,
yaitu Top-K Accuracy, Mean Reciprocal Rank (MRR), dan Mean Average
Precision (MAP). Top-K Accuracy mengukur proporsi query yang dokumen
relevannya berhasil ditempatkan dalam K peringkat teratas, dengan
variasi nilai K berupa Top-1, Top-5, dan Top-10, dimana Top-1 berfungsi
sebagai kriteria keberhasilan yang paling ketat. MRR mengukur seberapa
tinggi peringkat dokumen relevan pertama, sedangkan MAP menilai kualitas
peringkat secara meneyrluruh dengna memperhitungkan presisi pada setiap
posisi dokumen relevan sepanjang daftar. Ketiga metrik dihitung secara
terpisah untuk Wav2vec2 dan Data2vec agar dapat dibandingkan secara
langsung.

Sub proses kedua dalah evaluasi layer-wise. Ketiga metrik dihutung pada
setiap lapisan transfoermer dari kedua model. Hasilnya disusun menjadi
kurva kinerja terhadapt indeks lapisan sehingga dapat diindentifikasi
lapisan yang menghasilkan kinerja retrieval tertinggi bagi masing-masing
model. Evaluasi ini menjadi dasar untuk menjawab pertanyaan penelitian
mengenai apakah lapisan optimal bagi tugas retrieval fonetik berbeda
dari lapisan yang selama ini dilaporkan optimal untuk tugas ASR,
sekaligus menilai implikasinya terhadap pemilihan arsitektur.

Sub proses ketiga adalah pengujian hipotesis. Berdasarkan perbandingan
fnilai metrik antara kedua model pada lapisan optimalnya masing-masing,
penelitian menguji dua hipotesis yang saling bersaing. Apabilai Data2vec
menghasilkan kinerja retrieval yang lebih tinggi, maka hipotesisi
pertama (H1) terkonfirmasi dan mendukung pandangan bahwa keunggulan
representasi bersifat lintas tugas. Sebaliknya apabila Wav2vec2 yang
lebih unggul, maka hipotesis kedua (H2) terkonfirmasi dan menunjukan
bahwa tugas retrieval mengukur dimensi kualitas representasi yang
berbeda dari tugas ASR.

Sub proses terakhir adalah analisis kualitatif untuk melengkapi temuan
kuantitatif. Analisis ini dilakukan melalui studi kasus mendalam
terhadap query yang berhasil maupun gagal dalam proses retrieval. Kasus
keberhasilan dan kegagalan ditelaah untuk menilai sensitivitas fonetik
masing-masing model, misalnya mengidentifikasi kondisi ketika suatu
model keliru mengambil ayat lain yamg memiliki kemiripan bunyi atau pola
matra yang tinggi. Hasil analisis ini digunakan untuk mengaitkan temuan
empiris dengan justifikasi teori mengenai perbedaan struktur embedding
yang dihasilkan oleh paradigma contrastive leanring dan
self-distillation.

6.  **Deployment**

Fase Deployment pada penelitian ini diadaptasi dari makna aslinya dalam
CRISP-DM. Karena penelitian bersifat komparatif dan fasi ini tidak
melibatkan penerapan model dalam sistem produksi. Melainkan berfokus
pada sintesis temuan menjadi kesimpulan yang utuh serta perumusan
rekomendasi bagi penelitian lanjutan. Fase ini terdiri atas tiga sub
proses yaitu sintesis temuan, perumusan rekomendasi pemilihan model, dan
identifikasi arah penelitian lanjutan.

Sub proses pertama adalah sintesis temuan. Seluruh hasil dari fase
evaluation, baik opsiperbandingan metrik antar model, kurva kinerja,
layer-wise, hasil pengujian hipotesis, maupun temun dari analisis
kualitatif, dirangkum menjadi satu kesimpulan yang menjawab rumusan
masalah secara menyeluruh. Sintesis ini menegaskan model mana yang lebih
unggul untuk tugas retrieval ayat Al-Quran berbasis frozen embedding,
hipotesis mana yang terkonfirmasi, serta lapisan mana yang paling
optimal bagi masing-masing model.

Sub proses kedua adalah perumusan rekomendasi pemilihan model.
Berdasarkan temuan yang telah disintesis, penelitian merumuskan panduan
praktis mengenai model dan konfigurasi lapisan yang paling sesuai untuk
membangun sistem pencarian kemiripan ayat Al-Quran tanpa proses
transkripsi maupun *fine-tuning.* Rekomendasi ini mempertimbangkan tidak
hanya kualitas retrieval tertinggi, tetapi juga karakteristik
representasi masing-masing paradigma pretraining sebagaimana terungkap
dari hasil evaluasi.

Sub proses terkahir adalah identifikasi arah penelitian lanjutan
berdasarkan keterbatasan yang teridentifikasi selama penelitian,
dirumuskan sejumlah arah pengembangan yang dapat ditempuh pada
penelitian berikutnya. Arah tersebut mencakup, antara lain, penerapan
constrastive fine-tuning untuk menguji apakah adaptasi ringan pada model
pretrain dapat meningkatkan kualitas retrieval, perluasan cakupan data
ke seluruh juz Al-Quran, serta pengujian model Self-Supervised Learning
lain sebagai pembanding tambahan. Arah-arah ini sekaligus menandai batas
ruang linkgup penelitian saat ini dan potensi kontribusinya di masa
mendatang.[]{#_Toc234440447 .anchor}

BAB IV

HASIL DAN PEMBAHASAN

1.  **Hasil Business Understanding**

Tahap Business understanding menghasilkan dua output utama sesuai alur
penelitian, yaitu identifikasi masalah penelitian, yaitu identifikasi
masalah penelitian beserta tujuan penelitian, serta penetapan hipotesis
awal dan indicator keberhasilan. Kedua output ini menjadi fondasi arah
seluruh proses peneltiain yang dilaksanakan, mulai dari pengumpulan data
audio hingga evaluasi kinerja retrieval kedua model yang dibandingkan.

1.  **Identifikasi Masalah**

Proses identifikasi masalah dilakukan melalui kajian literatur terhadap
penelitian terdahulu mengenai model Self-Supervised Learning (SSL) untuk
pemrosesan suara, serta analisis terhadap penerapan model-model tersebut
pada domain Bahasa Arab. Berdasarkan hasil identifikasi tersebut,
ditemukan bahwa model SSL yang di-*pretrain* pada data berbahasa Inggris
cenderung menurun kinerjanya ketika diterapkan pada fonetik Arab, yang
memiliki karakteristik khas seperti *emphatic consonants* dan
*pharyngeal sounds* yang tidak dimiliki Bahasa inggris \[29\]. Al
Qur'an, sebagai teks berbahasa Arab dengan kaidah tajwid yang ketat dan
presisi tinggi, menjadikan tantangan ini semakin nyata karena pencocokan
bacaan menuntu ketepatan pada aspek fonetik, bukan sekedar kecocokan
tekstual.

Analisis lebih lanjut terhadap literatur mengungkapkan tiga kesenjangan
utama. Pertama, seluruh bukti kegagalan lintas bahasa yang tersedia
hanya terbatas pada arsitektur berbasis contrastive learning seperti
wav2vec2, sementara kejiadn terhadap paradigma self-distilation yang
dimiliki Data2vec dalam scenario serupa masih sangat terbatas. Kedua,
meskipun penelitian terdahulu telah membandingkan langsung wav2vec2 dan
Data2vec untuk pengenalan ucapan (ASR) Bahasa Arab, perbandingan
tersebut dilakukan dengan supervised fine-tuning dan diukur menggunakan
metrik transkripsi (WER/CER) bukan dalam kondisi frozen embedding tanpa
fine-tuning dengan metrik retrieval. Ketiga, perbandingan langsung kedua
model utnuk tugas audio retrieval yang dievaluasi melalui metrik
similarity search seperti MAP, Top-K Accuracy, dan MRR hingga kini masih
minim untuk diteliti, sehingga menyisakan pertanyaan fundamental apakah
keunggulan salah satu paradigma pada ASR akan tetap berlaku pada
retrieval.

Kesenjangan ini menjadi penting karena retrieval dan transkripsi
mengukur properti representasi yang berbeda. Pada ASR, fitur diteruskan
ke kepala klasifikasi yang ikut dilatih secara *supervised.* Sehingga
metrik seperti WER tidak murni mengukur kualitas embedding awal,
melainkan separabilitas fitur setelah proses pelatihan tersebut.
Sebaliknya, pada retrieval berbasis frozen embedding tidak ada kompenen
yang dilatih, sehingga metrik seperti MAP secara intrinsik mengukur
koherensi geometris ruang vector itu sendiri. Oleh karena itu,
keunggulan sebuah model pada ASR tidak dapat langsung diasumsikan
berlaku pada retrieval dan inilah celah yang menjadi fokut utama
penelitian ini.

2.  Tujuan Penelitian

Berdasarkan identifikasi masalah tersebut, ditetapkan tujuan penelitina
ini menghasilkan rekomendasi arsitektur SSL yang paling optimal sebagai
landasan teknis bagi pengembagan system audio retrieval ayat Al-Qur'an,
khususnya untuk domain Bahasa Arab yang memiliki karakteristik fonetik
khas. Untuk mencapai rekomendasi yang objektif, penelitian ini
membandingkan dua model dengan paradigma pretraining yang berbeda,
Wav2vec2 dengan paradigma contrastive learning dan Data2vec dengan
paradigma Self-Distillation dalam skenarion retrieval.

Tujuan tersebut dipaparkan dijelaskan dalam lebih detail menjadi tiga
tujuan yang spesifik. Pertama, mengevaluasi kinerja retrieval frozen
embedding wav2vec2 dan Data2vec pada metrik yang sama, yaitu MAP dan
Top-K Accuracy, dengan cosine similarity sebagai fungsi *scoring.*
Kedua, membandingkan kedua model secara statistic untuk menentukan
apakah performa retrieval sejalan dengan performa ASR atau justru
bersifat orthogonal, sebagai dasar penentuan model terbaik. Ketiga,
menganalisis distribusi kinerja retrieval pada tiap lapisan (layer-wise)
dari seluruh 13 lapisan kedua model untuk mengidentifikasi lapisan
optimal bagi tugas retrieval fonetik, dengan skema pemiilhan laipsan
terbaik pada himpunan pengembangan 70% dan pelaporan akhir pada himpunan
uji 30%

3.  **Hipotesis dan kriteria keberhasilan**

Sebagai acuan dalam mengevaluasi pencapaian tujuan penelitian,
ditetapkan hipotesis awal beserta kriteria keberhasilan yang mencakup
dua aspek, yaitu aspek validitas eksperimen dan aspek signifikansi
perbandingan model.

Perbedaan paradigma kedua model memunculkan dua hipotesis yang saling
berkebalikan. Hipotesis pertama (H1) memperkirakan Data2vec tetap
unggul, target prediksinya yang berupa representasi kontekstual laten
penuh berpotensi menghasilkan fitur fonetik yang lebih kaya dan umum,
sehingga keunggulannya diperkirakan bertahan pada domain retrieval.
Hipotesis kedua (H2) memperkirakan sebaliknya, bahwa Wav2vec2 justru
pada domain retrieval, dikarenakan constrastive lossnya secara eksplisit
menarik pasangan positif dan menolak kandidat negatif, sehingga
cenderung membentuk ruang embedding yang lebih clusterable dan lebih
selaras dengan pengukuran jarak cosine. Property yang justru menentukan
pada retrieval, bukan pada ASR. Jika H1 terkonfirasi, kualitas embedding
terbukti bersifat lintas tugas. Jika H2 terkonfirmasi, retrieval
terbukti mengukur dimensi kualitas representasi yang berbeda (ortogonal)
dari ASR.

Pada aspek validitas eksperimen, kriteria keberhasilan meliputi
keberhasilan ekstraksi sleuruh pasangan query dan database menjadi
embedding lapisan-per-lapisan tanpa kegagalan sistematis, serta
terpenuhinya cakupan relevansi penuh, yaitu setiap ayat query memiliki
setidaknya satu dokumen relevan di dalam database, sehingga tidak ada
kueri yang dipaksa bernilai MAP = 0 secara artifisial. Klip yang gagal
diekstraksi pada model manapun ditangani secara konsisten agar tidak
masuk dalam perhitungan.

Pada aspek signifikansi perbandingan model, kriteris keberhasilan
ditetapkan bukan pada ambang metrik absolut, melainkan pada kemampuan
eksperimen untuk membedakan kinerja kedua model secara meyakinkan.
Perbedaan MAP antara Data2vec dan Wav2vec2 pada himpunan uji diniliai
signifikan apabila interval kepercayaan 95% yang dihitung melalui metode
bootstrap (dengan B = 1000 resampling atas nilai average precision
per-kueri) tidak memuat nilai nol. Terpenuhinya kriteria ini
memungkinkan penelitian menyimpulkan secara tegas hipotesis mana H1 atau
H2 yang didukung oleh bukti empiris, sekaligus menjawab tujuan
penelitian.

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

\[10\] L. Alkanhal, A. Alessa, E. Almahmoud, and R. Alaqil, "Aswat:
Arabic Audio Dataset for Automatic Speech Recognition Using
Speech-Representation Learning," in *Proceedings of ArabicNLP 2023*,
Singapore (Hybrid): Association for Computational Linguistics, 2023, pp.
120--127. doi: 10.18653/v1/2023.arabicnlp-1.10.

\[11\] N. San *et al.*, "Leveraging pre-trained representations to
improve access to untranscribed speech from endangered languages," Sep.
14, 2021, *arXiv*: arXiv:2103.14583. doi: 10.48550/arXiv.2103.14583.

\[12\] C. Tang, Y. Wang, X. Chen, and W.-Q. Zhang, "Exploring Effective
Fusion Algorithms for Speech Based Self-Supervised Learning Models,"
Dec. 20, 2022, *arXiv*: arXiv:2212.10092. doi:
10.48550/arXiv.2212.10092.

\[13\] J. W. Yoon, S. M. Kim, and N. S. Kim, "MCR-Data2vec 2.0:
Improving Self-supervised Speech Pre-training via Model-level
Consistency Regularization," Jun. 14, 2023, *arXiv*: arXiv:2306.08463.
doi: 10.48550/arXiv.2306.08463.

\[14\] S. Yang *et al.*, "A Large-Scale Evaluation of Speech Foundation
Models," May 29, 2024, *arXiv*: arXiv:2404.09385. doi:
10.48550/arXiv.2404.09385.

\[15\] Y. Xie, Z. Zhang, and Y. Yang, "Siamese Network with wav2vec
Feature for Spoofing Speech Detection," in *Interspeech 2021*, ISCA,
Aug. 2021, pp. 4269--4273. doi: 10.21437/Interspeech.2021-847.

\[16\] Z. Li, L. Wu, T. Li, and Y. Yan, "Improves Neural Acoustic Word
Embeddings Query by Example Spoken Term Detection with Wav2vec
Pretraining and Circle Loss," in *2021 12th International Symposium on
Chinese Spoken Language Processing (ISCSLP)*, Hong Kong: IEEE, Jan.
2021, pp. 1--5. doi: 10.1109/ISCSLP49672.2021.9362065.

\[17\] S. J. Russell, S. Russell, and P. Norvig, *Artificial
Intelligence: A Modern Approach*. in Pearson series in artificial
intelligence. Pearson, 2020. \[Online\]. Available:
https://books.google.co.id/books?id=koFptAEACAAJ

\[18\] T. M. Mitchell, *Machine learning*, Nachdr. in McGraw-Hill series
in Computer Science. New York: McGraw-Hill, 2013.

\[19\] I. Goodfellow, Y. Bengio, and A. Courville, *Deep learning*. in
Adaptive computation and machine learning. Cambridge, Mass: The MIT
press, 2016.

\[20\] A. Vaswani *et al.*, "Attention Is All You Need," Aug. 02, 2023,
*arXiv*: arXiv:1706.03762. doi: 10.48550/arXiv.1706.03762.

\[21\] C. D. Manning, P. Raghavan, and H. Schütze, *Introduction to
Information Retrieval*, 1st ed. Cambridge University Press, 2008. doi:
10.1017/CBO9780511809071.

\[22\] T. Wang and P. Isola, "Understanding Contrastive Representation
Learning through Alignment and Uniformity on the Hypersphere," Aug. 15,
2022, *arXiv*: arXiv:2005.10242. doi: 10.48550/arXiv.2005.10242.

\[23\] D. Ulyanov, A. Vedaldi, and V. Lempitsky, "Instance
Normalization: The Missing Ingredient for Fast Stylization," Nov. 06,
2017, *arXiv*: arXiv:1607.08022. doi: 10.48550/arXiv.1607.08022.

\[24\] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer Normalization,"
Jul. 21, 2016, *arXiv*: arXiv:1607.06450. doi:
10.48550/arXiv.1607.06450.

\[25\] E. Schubert, "A Triangle Inequality for Cosine Similarity," vol.
13058, 2021, pp. 32--44. doi: 10.1007/978-3-030-89657-7_3.

\[26\] C. Schröer, F. Kruse, and J. M. Gómez, "A Systematic Literature
Review on Applying CRISP-DM Process Model," *Procedia Comput. Sci.*,
vol. 181, pp. 526--534, 2021, doi: 10.1016/j.procs.2021.01.199.

\[27\] A. Pasad, B. Shi, and K. Livescu, "Comparative Layer-Wise
Analysis of Self-Supervised Speech Models," in *ICASSP 2023 - 2023 IEEE
International Conference on Acoustics, Speech and Signal Processing
(ICASSP)*, Rhodes Island, Greece: IEEE, Jun. 2023, pp. 1--5. doi:
10.1109/ICASSP49357.2023.10096149.

\[28\] M. U. Salman, M. A. Qazi, and M. T. Alam, "Quran-MD: A
Fine-Grained Multilingual Multimodal Dataset of the Quran," Jan. 25,
2026, *arXiv*: arXiv:2601.17880. doi: 10.48550/arXiv.2601.17880.

\[29\] X. Chang *et al.*, "An Exploration of Self-Supervised Pretrained
Representations for End-to-End Speech Recognition," Oct. 09, 2021,
*arXiv*: arXiv:2110.04590. doi: 10.48550/arXiv.2110.04590.

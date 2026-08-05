**PERBANDINGAN WAV2VEC2 DAN DATA2VEC TERHADAP FITUR LATEN AUDIO UNTUK
TILAWAH AL-QURAN**

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

# Hlm. {#hlm. .TOC-Heading}

[DAFTAR ISI [i](#daftar-isi)](#daftar-isi)

[DAFTAR GAMBAR [iii](#daftar-gambar)](#daftar-gambar)

[DAFTAR TABEL [iv](#daftar-tabel)](#daftar-tabel)

[BAB I PENDAHULUAN [1](#bab-i-pendahuluan)](#bab-i-pendahuluan)

[1.1. Latar Belakang [1](#latar-belakang)](#latar-belakang)

[1.2. Rumusan Masalah [4](#rumusan-masalah)](#rumusan-masalah)

[1.3. Tujuan Penelitian [4](#tujuan-penelitian)](#tujuan-penelitian)

[1.4. Batasan Masalah [5](#batasan-masalah)](#batasan-masalah)

[1.5. Manfaat Penelitian [5](#manfaat-penelitian)](#manfaat-penelitian)

[1.6. Kerangkan Pemikiran
[6](#kerangkan-pemikiran)](#kerangkan-pemikiran)

[1.7. Sistematika Penulisan
[9](#sistematika-penulisan)](#sistematika-penulisan)

[BAB II KAJIAN LITERATUR
[11](#bab-ii-kajian-literatur)](#bab-ii-kajian-literatur)

[2.1. Tinjauan Pustaka [11](#tinjauan-pustaka)](#tinjauan-pustaka)

[2.2. Landasan Teori [15](#landasan-teori)](#landasan-teori)

[2.2.1. Kecerdasan buatan (Artificial Intelegents)
[15](#kecerdasan-buatan-artificial-intelegents)](#kecerdasan-buatan-artificial-intelegents)

[2.2.2. Deep Learning [15](#deep-learning)](#deep-learning)

[2.2.3. Similarity Search [16](#similarity-search)](#similarity-search)

[2.2.4. Vector Embedding [17](#vector-embedding)](#vector-embedding)

[2.2.5. Self-Supervised Learning (SSL)
[17](#self-supervised-learning-ssl)](#self-supervised-learning-ssl)

[2.2.6. Wav2vec 2.0 [17](#wav2vec-2.0)](#wav2vec-2.0)

[2.2.7. Data2vec [20](#data2vec)](#data2vec)

[2.2.8. Cosine Similarity [23](#cosine-similarity)](#cosine-similarity)

[2.2.9. Metriks Evaluasi Retrieval
[23](#metriks-evaluasi-retrieval)](#metriks-evaluasi-retrieval)

[2.2.10. Dataset Quran-MD [25](#dataset-quran-md)](#dataset-quran-md)

[2.2.11. Metode Penelitian CRISP-DM
[25](#metode-penelitian-crisp-dm)](#metode-penelitian-crisp-dm)

[BAB III METODOLOGI PENELITIAN
[27](#bab-iii-metodologi-penelitian)](#bab-iii-metodologi-penelitian)

[3.1. Business Understanding
[27](#business-understanding)](#business-understanding)

[3.2. Data Understanding [27](#data-understanding)](#data-understanding)

[3.3. Data Preparation [27](#data-preparation)](#data-preparation)

[3.4. Modeling [27](#modeling)](#modeling)

[3.5. Evaluation [27](#evaluation)](#evaluation)

[3.6. Deployment [30](#deployment)](#deployment)

[BAB IV HASIL DAN PEMBAHASAN
[32](#bab-iv-hasil-dan-pembahasan)](#bab-iv-hasil-dan-pembahasan)

[4.1. Hasil Business Understanding
[32](#hasil-business-understanding)](#hasil-business-understanding)

[4.1.1. Identifikasi Masalah
[32](#identifikasi-masalah)](#identifikasi-masalah)

[4.1.2. Tujuan Penelitian
[33](#tujuan-penelitian-1)](#tujuan-penelitian-1)

[4.1.3. Kriteria Evaluasi dan Keberhasilan
[34](#kriteria-evaluasi-dan-keberhasilan)](#kriteria-evaluasi-dan-keberhasilan)

[4.2. Hasil Data Understanding
[36](#hasil-data-understanding)](#hasil-data-understanding)

[4.2.1. Karakteristik Himpunan data Mahasiswa
[37](#karakteristik-himpunan-data-mahasiswa)](#karakteristik-himpunan-data-mahasiswa)

[4.2.2. Karakteristik Himpunan Data Quran-MD
[37](#karakteristik-himpunan-data-quran-md)](#karakteristik-himpunan-data-quran-md)

[4.2.3. Cakupan relevansi dan Ukuran Skenario
[38](#cakupan-relevansi-dan-ukuran-skenario)](#cakupan-relevansi-dan-ukuran-skenario)

[4.2.4. Kualitas Data [38](#kualitas-data)](#kualitas-data)

[4.3. Hasil Data Preparation
[39](#hasil-data-preparation)](#hasil-data-preparation)

[4.3.1. Normalisasi Audio [39](#normalisasi-audio)](#normalisasi-audio)

[*4.3.2.* Pembagian set *dev* dan set *test*
[42](#pembagian-set-dev-dan-set-test)](#pembagian-set-dev-dan-set-test)

[4.4. Modeling [42](#modeling-1)](#modeling-1)

[4.4.1. Ekstraksi Frozen Embedding per-lapisan
[43](#ekstraksi-frozen-embedding-per-lapisan)](#ekstraksi-frozen-embedding-per-lapisan)

[4.4.2. Data Cleaning [43](#data-cleaning)](#data-cleaning)

[4.4.3. Similarity Scoring dan Ranking
[44](#similarity-scoring-dan-ranking)](#similarity-scoring-dan-ranking)

[4.5. Evaluation [44](#evaluation-1)](#evaluation-1)

[4.5.1. Kinerja Retrieval Frozen Embedding
[45](#kinerja-retrieval-frozen-embedding)](#kinerja-retrieval-frozen-embedding)

[4.5.2. Perbandingan Wav2vec2 dan Data2vec
[46](#perbandingan-wav2vec2-dan-data2vec)](#perbandingan-wav2vec2-dan-data2vec)

[4.5.3. Lapisan paling informatif
[46](#lapisan-paling-informatif)](#lapisan-paling-informatif)

[4.6. Deployment (Pertimbangan Penerapan)
[47](#deployment-pertimbangan-penerapan)](#deployment-pertimbangan-penerapan)

[4.6.1. Skema Penerapan Sistem Retrieval
[47](#skema-penerapan-sistem-retrieval)](#skema-penerapan-sistem-retrieval)

[4.6.2. Kelayakan penerapan berdasarkan temuan
[47](#kelayakan-penerapan-berdasarkan-temuan)](#kelayakan-penerapan-berdasarkan-temuan)

[DAFTAR PUSTAKA [48](#daftar-pustaka)](#daftar-pustaka)

# DAFTAR GAMBAR

[Gambar 1.1 Kerangka Pemikiran [6](#_Toc236767899)](#_Toc236767899)

[Gambar 2.1 Arsitektur Wav2vec2 [23](#_Toc236767908)](#_Toc236767908)

[Gambar 2.2 Arsitektur Data2vec [25](#_Toc236767909)](#_Toc236767909)

[Gambar 3.1 Ringkasan Dataset Quran-MD
[31](#_Toc236767923)](#_Toc236767923)

[Gambar 3.2 Karakteristik Audio [32](#_Toc236767924)](#_Toc236767924)

[Gambar 3.3 Total Distribusi Video [33](#_Toc236767925)](#_Toc236767925)

[Gambar 3.4 Box Plot Showing Outlier
[34](#_Toc236767926)](#_Toc236767926)

[Gambar 4.1 Hasil Evaluasi Tiap-tiap Layer
[47](#_Toc236811332)](#_Toc236811332)

# DAFTAR TABEL

Tabel 2.1 State of the art [11](#_Toc236748112)

Tabel 2.2 Konfigurasi Blok Konvolusi Temporal pada Encoder Audio
[20](#_Toc236748113)

Tabel 3.1 Summary Statistic Dataset Query [33](#_Toc236748756)

Tabel 4.1 Atribut dan Tipe Data Quran-MD [45](#_Toc236748707)

Tabel 4.2 Ukuran Data dan Cakupan Relevansi pada Setiap Skenario
Evaluasi [47](#_Toc236748708)

# BAB I PENDAHULUAN

##  Latar Belakang

Perkembangan *deep learning* dalam pemrosesan audio telah didominasi
oleh model Self-Supervised Learning (SSL), dengan Wav2Vec 2.0 menjadi
salah satu model yang berpengaruh dalam perkembangan pembelajaran
representasi ujaran berbasis *self-supervised learning*. Model ini mampu
mempelajari representasi laten (*latent embeddings*) langsung dari data
audio mentah tanpa memerlukan label transkripsi. Representasi yang
dihasilkan kemudian dapat dimanfaatkan dan disesuaikan untuk berbagai
tugas pemrosesan suara, dengan kebutuhan data berlabel yang lebih
sedikit \[1\], \[2\]. Kemampuan model-model ini dalam menghasilkan fitur
laten menjadi fondasi bagi tugas-tugas non-transkripsi, termasuk
penelusuran kemiripan audio, yang mencocokkan potongan bacaan dengan
entri yang paling mirip di *database* berdasarkan kemiripan *vector
embedding*, bukan mencocokkan melalui teks hasil enkripsi \[3\].
Pendekatan ini telah didukung oleh temuan empiris yang menunjukkan bahwa
sistem *retrieval* berbasis *latent embedding* yang dihasilkan langsung
dari audio secara konsisten mengungguli pendekatan dua tahap dari ASR ke
retrieval, karena kesalahan transkripsi pada tahap ASR akan berpropagasi
dan mendegradasi kualitas pencocokan pada tahap selanjutnya \[4\],
\[5\]. Karakteristik inilah yang menjadikan pendekatan berbasis
embedding lebih sesuai untuk konteks pencocokan bacaan Quran, yang
menuntut presisi tinggi pada aspek fonetik dan tajwid.

Al-Qur\'an berisi teks berbahasa Arab dengan kaidah pelafalan (tajwid)
yang ketat dan presisi tinggi, tidak hanya memiliki signifikansi
religius bagi umat Muslim, tetapi juga merupakan *testbed* fonetik yang
menantang secara teknis untuk sistem pemrosesan suara. Dalam eksperimen
yang dilakukan oleh Toyin model berbasis *Arab-monolingual* mampu
mengungguli model multibahasa yang memiliki ukuran dua kali lipat dalam
tugas pengenalan ucapan Arab \[6\]. Temuan ini diperkuat oleh Li et

al yang secara kuantitatif menunjukkan bahwa encoder SSL yang dilatih
menggunakan teks berbahasa inggris menghasilkan fitur fonetik yang
kurang informatif ketika diterapkan pada inputan lintas-bahasa tanpa
adaptasi domain \[7\]. Namun demikian, bukti-bukti tersebut terbatas
pada arsitektur berbasis *contrastive learning* seperti Wav2Vec2.
Sementara itu, Data2Vec menggunakan paradigma *self-distillation* yang
secara fundamental berbeda. alih-alih membandingkan representasi positif
melawan negatif, Data2Vec melatih jaringan *student* untuk memprediksi
representasi kontekstual penuh yang dihasilkan oleh jaringan *teacher*
atas seluruh masukan \[1\], \[8\]. Perbedaan ini secara teoretis
berpotensi menghasilkan representasi fonetik yang lebih umum untuk
lintas bahasa, karena target prediksi berupa konteks laten yang melimpah
bukan unit diskrit yang terikat pada distribusi fonetik bahasa
*pretraining*. Temuan Xue et al. menunjukkan bahwa penggantian target
diskrit dengan target kontinu ala Data2Vec menurunkan *phoneme* *error*
rate lintas bahasa sebesar 5,3% \[9\]. Temuan tersebut memperlihatkan
adanya perbedaan karakteristik representasi yang dihasilkan oleh
masing-masing paradigma *pre-training*, sehingga perbandingan kedua
model pada tugas *retrieval* menjadi relevan untuk dilakukan.

Temuan-temuan di atas membuka peluang untuk dilakukan riset untuk
memberikan jawaban terhadap fenomena yang belum terjawab. Seluruh bukti
kegagalan lintas bahasa yang tersedia hanya terbatas pada arsitektur
*contrastive learning*, sementara kajian terhadap paradigma
*self-distillation* yang dimiliki Data2Vec dalam skenario serupa masih
sangat terbatas. Lebih jauh lagi, perbandingan langsung antara Wav2Vec2
dan Data2Vec untuk tugas audio retrieval yang dievaluasi melalui metrik
*similarity search* seperti Top-K Accuracy, Mean Reciprocal Rank (MRR),
dan Mean Average Precision (MAP), bukan metrik transkripsi (WER/CER)
hingga kini masih minim di literatur. Ketiadaan perbandingan ini
menyisakan pertanyaan fundamental: apakah perbedaan paradigma
pretraining antara kedua model menghasilkan kualitas representasi
fonetik yang berbeda pula untuk tugas retrieval ayat Al-Qur\'an.

Namun, karakteristik representasi yang unggul pada tugas transkripsi
tidak dapat langsung dianggap memberikan hasil yang sama pada tugas
retrieval. Retrieval dan transkripsi mengukur aspek representasi yang
berbeda. Pada tugas *Automatic Speech Recognition* (ASR). Representasi
audio digunakan sebagai dasar untuk menghasilkan transkripsi, kemudian
model disesuaikan melalui proses pelatihan terkontrol agar mampu
menjalankan tugas tersebut. Oleh karena itu, metrik seperti Word Error
Rate (WER) tidak hanya dipengaruhi oleh kualitas representasi awal,
tetapi juga oleh proses penyesuaian model terhadap tugas transkripsi.

Sebaliknya pada sistem retrieval berbasis frozen embedding, parameter
model tidak dilatih untuk pada data Al-Qur'an. Representasi yang
dihasilkan model langsung digunakan untuk membentuk *vector embedding*.
Kemudian hasil representasi nya dibandingkan menggunakan skoring *cosine
similarity*. Dengan demikian, kualitas hasil dari masing-masing model
diuji melalui kemampuan *vector embedding* dalam menempatkan audio yang
berasal dari ayat yang sama pada posisi yang berdekatan serta memisahkan
audio dari ayat yang berbeda. Kinerja sistem kemudian dinilai
menggunakan metrik retrieval, yaitu *Mean Average Precision* (MAP),
*Mean Reciprocal Rank* (MRR), dan Top-K Accuracy.

Perbedaan karakteristik antara tugas ASR dan retrieval tersebut
menunjukkan bahwa perbandingan kinerja Wav2Vec2 dan Data2Vec perlu
dilakukan secara langsung dalam kondisi *frozen embedding* dan dalam
tugas retrieval ayat Al-Qur'an. Perbandingan ini tidak diarahkan untuk
menetapkan bahwa salah satu model selalu lebih baik, tetapi untuk
memperoleh gambaran empiris mengenai kinerja masing-masing model pada
tugas retrieval audio ayat Al-Qur\'an, tetapi untuk memperoleh gambaran
empiris mengenai kinerja masing-masing model pada tugas retrieval audio
ayat Al-Qur\'an. Selain itu, perbandingan dilakukan untuk mengetahui
apakah kedua model menghasilkan pola kinerja yang berbeda ketika
digunakan untuk mencocokkan audio bacaan dengan ayat yang sesuai.

Berdasarkan celah penelitian tersebut, penelitian ini berfokus pada
implementasi Wav2Vec2 dan Data2Vec sebagai penghasil representasi laten
audio, kemudian membandingkan kinerja kedua model dalam sistem retrieval
ayat Al-Qur\'an. Setiap model digunakan tanpa fine-tuning pada data
Al-Qur\'an sehingga perbandingan dapat dilakukan terhadap kemampuan
representasi pretrained dalam kondisi yang sama. Analisis pada setiap
lapisan model juga dilakukan untuk menentukan konfigurasi lapisan yang
memberikan hasil retrieval paling baik bagi masing-masing model.

Dengan pendekatan tersebut, hasil penelitian tidak diarahkan pada
penetapan model yang unggul secara universal. Hasil penelitian digunakan
untuk menjelaskan kinerja kedua model pada ruang lingkup data, metode,
dan skenario evaluasi yang digunakan. Apabila terdapat perbedaan
kinerja, perbedaan tersebut dianalisis berdasarkan nilai metrik
retrieval dan interval kepercayaan. Apabila perbedaan tidak menunjukkan
makna statistik yang kuat, hasil tersebut tetap menjadi temuan penting
karena menunjukkan bahwa kedua model memiliki kinerja yang relatif
sebanding dalam skenario retrieval yang digunakan.

Oleh karena itu, penelitian ini diarahkan untuk mengimplementasikan
Wav2Vec2 dan Data2Vec dalam menghasilkan fitur laten audio untuk
retrieval ayat Al-Qur\'an, serta membandingkan kinerja kedua model
berdasarkan metrik similarity search. Melalui implementasi dan
perbandingan tersebut, penelitian ini diharapkan dapat memberikan
gambaran mengenai kesesuaian masing-masing model untuk tugas retrieval
audio pada domain bacaan Al-Qur\'an. Hasil penelitian juga dapat menjadi
landasan teknis bagi pengembangan sistem pencarian ayat Al-Qur\'an
berbasis suara tanpa melalui tahap transkripsi.

##  Rumusan Masalah

Berdasarkan Rumusan masalah terbagi menjadi beberapa point

1.  Bagaimana implementasi model Wav2vec2 dan Data2vec dalam tugas
    *retrieval* ayat Al-Quran?

2.  Bagaimana kinerja *frozen-embedding* Wav2vec2 dan Data2vec dalam
    tugas *retrieval* ayat Al-Qur'an?

## Tujuan Penelitian

Tujuan dari penelitian ini sebagai berikut:

1.  Mengimplementasikan model Wav2vec2 dan Data2vec dalam tugas
    retrieval ayat Al-Quran.

2.  Mengevaluasi kinerja frozen embedding Wav2vec2 dan Data2vec dalam
    tugas *retrieval* ayat Al-Qur'an

##  Batasan Masalah

Batasan masalah dari penelitian ini adalah sebagai berikut:

1.  Penelitian dibatasi pada penggunaan model *pre-trained* Data2Vec dan
    Wav2Vec2 tanpa proses *fine-tuning.*

2.  Eksperimen dilakukan pada teks Al-Qur'an, dengan objek penelitian
    berupa surah Al -- Fatihah dan surah-surah dalam Juz Amma.

3.  Pengujian input audio hanya dapat dilakukan perayat.

##  Manfaat Penelitian

Penelitian yang dilakukan ini memiliki manfaat dalam memberikan
pemahaman ilmiah yang lebih mengenai seberapa efektif model representasi
audio seperti Data2vec dan Wav2vec2 dalam domain pembacaan ayat suci
Al-Quran. Khususnya dalam tugas retrieval ayat. Juga dapat menjadi
pedoman atau acuan bagi peneliti dan pengembang perangkat lunak dalam
memilih model mana yang cocok dalam tugas retrieval ayat Al-Qur'an

##  Kerangkan Pemikiran

> Berdasarkan uraian tersebut kerangkan pemikiran dapat diuraikan
> sebagai berikut

![[]{#_Toc236767899 .anchor}Gambar 1.1 Kerangka
Pemikiran](media/image2.png){width="3.932082239720035in"
height="7.775in"}

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
\[3\]. Teori ini didukung pula oleh bukti empiris *Gemini embedding 2*,
secara eksplisit menunjukkan bahwa pendekatan *cascaded* ASR ke
*retrieval* mengalami *error propagation* \[4\]. Sedangkan *WavRag*
membuktikan bahwa *retrieval* langsung dari audio mampu menyamai akurasi
pipeline ASR dengan efisiensi yang jauh lebih tinggi \[5\].

Meskipun demikian, keunggulan tersebut belum tentu berlaku ketika model
diterapkan pada domain fonteik yang spesifik seperti bahasa Arab dalam
bacaan Al-Qur'an. Persoalan muncul karena model SSL umumnya
di-*pretrain* pada korpus Bahasa inggris dan kurang informatikf untuk
fonetik Arab. Penelitian Toyin et al dan Li et al telah menunjukan
keterbatasan ini \[6\], \[7\]. namun bukti tersebut hanya terbatas pada
arsitektur berbasis constrastive learing seperti Wav2vec2. Sementara
itu, ptotensi paradigma self-distillation yang dimiliki Data2Vec untuk
tugas audio retrieval Al-Quran masih sangat minim dikaji, dan
perbandingan langsung antara kedua model menggunakan metrik similarity
search, bukan metrik transkripsi seperti *word error rate* (WER) hingga
kini masih terbatas dalam literatur.

Untuk menjawab persoalan tersebut, penelitian ini menawarkan pendekatan
berupa implementasi dan perbandingan Wav2Vec2 dan Data2Vec dalam
menghasilkan representasi laten untuk tugas retrieval ayat Al-Qur\'an.
Fitur laten dari kedua model *pretrained* diekstraksi dalam kondisi beku
atau *frozen*, kemudian digunakan untuk membentuk representasi vektor
audio. Setiap vektor *query* dibandingkan dengan vektor pada database
(*ground truth)* menggunakan *cosine similarity* sebagai *scoring
function*, lalu hasilnya diurutkan untuk menghasilkan daftar retrieval
mulai dari ayat yang memiliki skor paling tinggi menuju yang terendah.

Evaluasi dilakukan menggunakan metrik *Mean Average Precision* (MAP),
*Mean Reciprocal Rank* (MRR), dan Top-K Accuracy. Selain membandingkan
kinerja kedua model, evaluasi juga dilakukan pada beberapa lapisan untuk
mengetahui lapisan yang menghasilkan kinerja retrieval paling baik.
Hasil lapisan terbaik tersebut digunakan sebagai konfigurasi dalam
perbandingan utama antara Wav2Vec2 dan Data2Vec.

Proses penelitian dilaksanakan mengikuti kerangka kerja atau *framework*
Cross-Industry Standard Process for Data Mining (CRISP-DM) yang
diadaptasi ke dalam beberapa tahap, yaitu Business Understanding, Data
Understanding, Data Preparation, Modeling, Evaluation, dan Deployment.
Tahapan tersebut digunakan untuk mengarahkan proses pengumpulan data,
persiapan audio, ekstraksi frozen embedding, penerapan similarity
search, evaluasi kinerja, serta analisis hasil.

Hasil akhir penelitian berupa implementasi sistem retrieval berbasis
frozen embedding menggunakan Wav2Vec2 dan Data2Vec, serta perbandingan
kinerja kedua model pada data audio ayat Al-Qur\'an. Hasil tersebut
diharapkan dapat memberikan gambaran mengenai model dan konfigurasi
lapisan yang paling sesuai untuk mendukung pengembangan sistem audio
retrieval pada domain bacaan Al-Qur\'an.

## Sistematika Penulisan

Sistematika penulisan laporan tugas akhir ini disusun ke dalam lima bab
yang saling berkait secara berurutan, dengan gambaran kandungan setiap
bab sebagai berikut.

**BAB I: PENDAHULUAN**

Bab ini menguraikan dasar pemikiran yang melatarbelakangi dilakukannya
penelitian, mencakup latar belakang yang memaparkan fenomena dan celah
penelitian, rumusan masalah sebagai pertanyaan penelitian yang hendak
dijawab, batasan masalah yang menetapkan ruang lingkup penelitian,
manfaat penelitian, state of the art, kerangka pemikiran, serta
sistematika penulisan. Bab ini berfungsi sebagai fondasi mengarahkan
keseluruhan penelitian.

**BAB II: TINJAUAN PUSTAKA**

Bab ini memaparkan landasan teori dan konsep-konsep fundamental yang
menjadi dasar ilmiah penelitian, meliputi teori mengenai Self-Supervised
Learning (SSL), arsitektur dan mekanisme kerja model Waw2vec2 dan
Data2Vec, konsep representasi latent (*latent embedding*), tugas audio
retrieval, serta metrik evaluasi similarity search. Berbeda dengan Bab I
yang menyoroti mengapa penelitian mesti dilakukan, bab ini menyediakan
kerangka teoritis apa saja konsep yang digunakan untuk memahami dan
menyelesaikan permalahan.

**BAB III: METODOLOGI PENELITIAN**

Bab ini menjelaskan tahapan prosedur pelaksanaan penelitian secara
sistematis mengikuti kerangka kerja *Cross-Industry Standard Process for
Data Mining* (CRISP-DM) yang diadaptasi, mencakup pengumpulan data,
pre-processing, ekstrasi representasi laten, implementasi similarity
search, hingga skenario evaluasi. Jika Bab II menjelaskan apa landasan
teorinya, bab ini menguraikan bagaimana teori tersebut diterapkan secara
teknis dan operasional dalam penelitian.

**BAB IV: HASIL DAN PENELITIAN**

Bab ini menyajikan hasil implementasi dan evaluasi sistem retrieval
audio ayat Al-Qur\'an menggunakan frozen embedding Wav2Vec2 dan
Data2Vec. Pembahasan mencakup hasil persiapan data, ekstraksi
representasi laten pada setiap lapisan model, penerapan cosine
similarity, penyusunan peringkat hasil retrieval, serta evaluasi
menggunakan metrik Top-K Accuracy, Mean Reciprocal Rank (MRR), dan Mean
Average Precision (MAP). Bab ini juga menyajikan perbandingan kinerja
kedua model berdasarkan konfigurasi lapisan yang optimal dan membahas
implikasi hasil tersebut terhadap penerapan sistem retrieval berbasis
suara.

**BAB V: KESIMPULAN DAN SARAN**

Bab ini memuat kesimpulan yang menjawab secara ringkas dan tegas rumusan
masalah berdasarkan temuan pada Bab IV, disertai saran dan untuk
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

  3\.      Alkanhal, dkk  Wav2vec2 dan Data2vec Aswat, Common Voice (AR),    Mencapai WER SOTA 11,7%
           (2021) \[10\]                        MGB-2                        pada Common Voice dan
                                                                             10,3% pada MGB-2 berkat
                                                                             pra-pelatihan pada Aswat.

  4\.      Öberg (2025)   Projection Network +  Google FLEURS, Common Voice  embedding hasil
           \[3\]          contrastive triplet                                transformasi mencapai MRR
                          loss pada embedding                                0,811 dan Recall@20
                          Wav2Vec2                                           0,931, jauh meningkat
                                                                             dari embedding asli.

  5\.      San, dkk       QbE-STD dengan frozen Gronings, Besemah,           Peningkatan relatif
           (2021) \[11\]  Wav2Vec2              Mavir/Mboshi; LibriSpeech,   56--86% atas SOTA;
                          (English-mono &       XLSR-53.                     English-mono mengungguli
                          XLSR-53).                                          XLSR multibahasa pada 4
                                                                             dataset.

  6\.      Zhaoqi Li, dkk Neural acoustic word  Korpus QbE-STD               Average Precision (AP)
           (2021) \[12\]  embeddings + Wav2Vec,                              meningkat relatif 11,1%
                          circle loss                                        dibanding sistem MFCC;
                          menggantikan triplet                               konvergensi ±40% epoch
                          loss                                               lebih cepat dengan circle
                                                                             loss.

  7\.      Xie dkk (2022) Two-phase: Siamese    ASVspoof 2019 Logical Access EER turun dari SOTA
           \[13\]         Network + contrastive (LA)                         sebelumnya 4,07% menjadi
                          loss, lalu MLP                                     1,15% pada evaluation
                          classifier                                         set.

  8\.      Tang, dkk      Perbandingan HuBERT   SUPERB benchmark (task SID & Data2Vec Base: SID 70,21%
           (2022) \[14\]  vs Data2Vec pada SID  ASR; basis                   / WER 4,94%; HuBERT Base:
                          & ASR                 LibriSpeech/VoxCeleb-line)   SID 81,42% / WER 6,42%
                                                                             HuBERT unggul di SID,
                                                                             Data2Vec unggul di ASR.

  9\.      Yoon, dkk      Data2Vec 2.0 Base,    LibriSpeech 960 jam, 11      Mengungguli Data2Vec 2.0
           (2023) \[15\]  regularisasi          SUPERB subtasks              di hampir semua tugas
                          konsistensi,                                       kecuali QbE; mencapai
                          fine-tune parsial.                                 SOTA pada PR, ASR, KS,
                                                                             IC, SF, dan ER (mis. WER
                                                                             4,68%, SID 82,40%).

  10\.     Yang, dkk      Frozen embedding      LibriSpeech, QUESST,         Data2Vec Large mencapai
           (2024) \[16\]  Data2Vec Base/Large + VoxCeleb, LibriTTS           speaker-invariance
                          lightweight                                        terbaik (7,38 ACC), PER
                          prediction head                                    terbaik (2,58%), dan
                                                                             performa VC terbaik (6,75
                                                                             MCD, 100% akurasi
                                                                             kemiripan target
                                                                             speaker).
  ----------------------------------------------------------------------------------------------------

  : []{#_Toc236748112 .anchor}Tabel 2.1 State of the art

Penelitian mengenai representasi ucapan yang dilatih sendiri
(*self-supervised*) telah mengalami kemajuan pesat, dimulai dengan
fondasi seperti Wav2vec 2.0 dan Data2vec \[1\], \[8\] yang mampu
mempelajari representasi laten koheren dari *raw audio*. Namun,
mayoritas penelitian awal dan penelitian lanjutan berkonsentrasi pada
peningkatan kinerja ASR (*automatic speech recognition*), termasuk studi
lintas bahasa dan adaptasi domain. Sebagaian penelitian lain berfokus
pada non-retrieval yang tetap memanfaatkan kualitas embedding SSL,
seperti deteksi *spoofing* suara dan identifikasi suarat yang
menggabungkan *Siamase Network* pada fitur Wav2vec2 \[13\]. Meskipun
demikian, Sebagian studi mulai menggeser fokus ke kapabilitas
*retrieval* dari *embedding* ini, misalnya penelitian Öberg mengenai
audio search berbasis transformasi *embedding* Wav2vec2 dengan
*contrastive learning* \[3\]

Pada ranah *retrieval* berdasarkan kemiripan, sejumlah penelitian telah
membuktikan keunggulan pendekatan berbasis *embedding* SSL. Nay san, dkk
telah membuktikan bahwa representasi dari Wav2vec2 dapat melakukan
*query-by-example* *spoken term detection* lintas bahasa tanpa
transkripsi maupun fine-tuning dengan peningkatan 56%-86% atas *state of
the art* sebelumnya \[11\]. Sejalan dengan itu, Zhaoqi Li, dkk
membuktikan bahwa *Neural Acoustic Word Embeddings* berbasis pretrain
Wav2vec2 mengungguli fitur MFCC untuk retrieval berbasis kemiripan
\[12\]. Temuan-temuan ini menegaskan bahwa kualitas embedding SSL
menjadikannya sangat sesuai dengan untuk tugas *similarity search.*

Dalam konteks bahasa Arab alkanhal, dkk melakukan perbandingan langsung
antara Wav2vec2 dan Data2vec. Mereka menemukan bahwa Data2vec konsisten
mengungguli Wav2vec2 dalam domain ASR Arab. Namun, perbandingan tersebut
dilakukan dengan supervised fine-tuning dan juga dengan pengukuran
metrik transkripsi, bukan frozen embedding tanpa fine-tuning dan metrik
retrieval. Di sisi lain, kajian terhadap Data2vec pada tugas-tugas
non-ASR mulai bermunculan dan justru memperlihatkan pola yang bergantung
pada jenis tugasnya. Tang, dkk menemukan bahwa efektivitas Data2vec
berbeda tergantung tugasnya, Data2vec lebih unggul dalam tugas yang
berbasis konten contohnya ASR, namun kurang optimal dalam identifikasi
pembicara. Yoon, dkk mengonfirmasi kecenderungan serupa, Data2vec 2.0
kuat pada tugas berbasis konten seperti ASR, tetapi performa
query-by-example nya justru menurun ketika diberi regulasi tambahan
\[15\]. Evaluasi berskala besar yang dilakukan oleh Yang, dkk
mempertegas hal ini, menunjukan bahwa Data2vec mencapai performa terbaik
pada Voice Conversion dan Phone Recognition, tetapi tertinggal pada
Speaker Identification dan ASR out-of-distribution dibanding HuBERT dan
WavLM.

Berdasarkan kajian tersebut, terlihat bahwa keunggulan Data2vec tidak
bersifat universial, melainkan spesifik terhadap jenis tugas. Fakta ini
memunculkan pertanyaan fundamental yang belum terjawab dalam literatur:
apakah keunggulan Data2vec pada ASR bahasa Arab akan sama ketika kedua
model dievaluasi tanpa fine-tuning dan menggunakan metrics dalam domain
similarity search seperti MAP, Top-K, dan MRR.

Penelitian ini menghadirkan perbedaan penting dengan menempatkan kedua
model tersebut dalam konteks *retrieval* audio ayat Al-Qur'an yang mana
memiliki karakteristik fonetik dan aturan tajwid yang unik. Berbeda
dengan penelitian pada umumnya yang menitik beratkan pada pengukuran
performa transkripsi ASR, penelitian ini mengevaluasi kemampuan Wav2vec2
dan Data2vec dalam mencocokan cuplikan bacaan dengan ayat terkait
melalui pendekatan komparatif berbasis kemiripan embedding. Dengan
demikian, penelitian ini menawarkan kontribusi yang masih minim
dieksplorasi, sekaligus menyediakan landasan teknis bagi pengembangan
sistem pendukung hafalan dan pembelajaran Al-Qur'an yang lebih adaptif
dan aplikatif.

##  Landasan Teori

### Kecerdasan buatan (Artificial Intelegents) 

Kecerdasan buatan (Artificial Intelegents) merupakan salah satu cabang
ilmu komputer yang mempelajari perancangan sistem yang mampu melakukan
kecerdasan manusia, seperti penalaran, persepsi, pembelajaran, dan
pengambilan keputusan \[17\]. Russel dan Norvig mengklasifikasikan AI ke
dalam empat sudut pandang, yaitu sistem yang berpikir seperti manusia,
bertindak seperti manusia, berpikir secara rasional, dan bertindak
secara rasional. Pendekatan terakhir yakni rational agent yang bertindak
untuk mencapai hasil terbaik menurut ukuran kinerja tertentu, menjadi
kerangka dominan dalam AI modern \[17\].

Dalam perkembangannya, Machine learning (ML) muncul sebagai sub-bidang
AI yang berfokus pada kemampuan sistem untuk belajar pola dari data
tanpa dirprogram secara eksplisit untuk setiap kasus \[18\]. ML sendiri
terbagi menjadi tiga paradigma utama supervised learning (belajar dari
data yang berlabel), unsupervised learning (menemukan struktur dari data
tak berlabel), dan self-supervised learning yang menjadi fondasi
penelitian ini. Penelitian ini berada dalam ranah AI karena memanfaatkan
model pembelajaraan representasi untuk memahami dan mencocokkan sinyal
ucapan Al-Qur'an secara otomatis.

### Deep Learning 

Deep learning adalah cabang dari machine learning yang menggunakan
jaringan saraf tiruan berlapis yang jumlahnya sangat banyak (*deep
neural networks*) untuk mempelajari representasi data secara berjenjang
(*hierarchical representation*) \[19\]. Berbeda dari ML tradisional yang
bergantung pada feature engineering manual yang mana, harus membutuhkan
manusia untuk memberikan data yang berlabel, deep learning mempelajari
fitur secara otomatis. Lapisan-lapisan awal menangkap fitur tingkat
rendah (misalnya pola frekuensi pada audio), sedangkan lapisan yang
lebih dalam menyusun fitur tingkat tinggi yang lebih abstrak (misalnya
unit fonetik) \[19\]

Sebuah jaringan saraf tersusun atas neuron-neuron yang
mentransformasikan input melalui bobot this equation **W**, bias **b**.
Dan fungsi aktivasi non-linear *σ,* sehingga keluaran satu lapisan
dinyatakan sebagai berikut

  -----------------------------------------------------------------------------
     $$\mathbf{h} = \sigma\left( \mathbf{Wx} + \mathbf{b} \right)$$   (2.1)
  -- ---------------------------------------------------------------- ---------

  -----------------------------------------------------------------------------

Parameter jaringan dioptimasi dengan meminimalkan *loss function*
melalui algoritma *backpropagation* dan *gradient descent* \[19\].
Arsitektur yang relevan bagi penelitian ini adalah jaringan konvolusi
(Convolutional Neural Network, CNN) yang efektif mengekstrak fitur local
dari sinyal, dan Transformer \[20\] yang memodelkan ketergantungan
jangka Panjang melalui mekanisme self-attention. Kedua arsitektur inilah
yang Menyusun tulang punggung model Wav2vec2 2.0 dan varian speech dari
Data2Vec.

### Similarity Search 

Similarity search adalah proses menemukan objek-objek dalam suatu
database yang paling mirip dengan sebuah objek kueri (data input),
berdasarkan ukuran kemiripan tertentu \[21\]. Dalam paradigma modern,
setiap objek, baik teks, gambar, maupun audio, direpresentasikan sebagai
vector bilangan riil di ruang berdimensi tinggi, sehingga pencarian
kemiripan tereduksi menjadi persoalan *Nearest Neighbor Search* di ruang
vector tersebut \[21\]

Formalnya, diberikan sebuah vector kueri **q** dan himpunan vector
database **\**
D={**d**1​,...,**d***N*​} , sistem menghitung skor kemiripan
*s*(**q**,**d***i*​) untuk setiap **d***i* , lalu mengurutkan hasilnya
secara menurun untuk menghasilkan daftar peringkat (*ranked list*).
Pendekatan ini menjadi tulang punggung sistem information retrieval
modern \[21\]. Dalam penelitian ini, similarity search diterapkan pada
embedding audio ayat Al-Qur'an. Data kueri dibandingkan langsung
terhadap database embedding ayat tanpa melalui tahap transkripsi,
sehingga menghindari propagasi kesalah dari tahap ASR (*automatic speech
recognition*).

### Vector Embedding 

Vector Embedding adalah representasi suatu objek dalam bentuk vector
yang padat berdimensi tetap $v \in R^{d}\$yang dirancang sedemikian rupa
sehingga kedekatan geometrics antar vektor mencerminkan kemiripan
semantic atau perseptual antar objek yang diwakilinya \[19\]. Prinsip
utamanya adalah distributional hypothesis, yang mana objek objek yang
serupa akan menempati posisi yang berdekatan di dalam ruang vector
(*embedding space*)

Pada domain audio, embedding dihasilkan oleh model self-supervised
seperti Wav2vec2 2.0 dan Data2vec, yang memetakan sinyal audio mentah
menjadi barisan vector kontekstual. Embedding inilah yan gmenjadi objek
perbandingan dalam penelitian ini. Kualitas sebuah embedding untuk tugas
retrieval dapat dilihat dari kemampuannya dalam menempatkan audio dari
ayat yang sama secara berdekatan sekaligus memisahkan audio dari ayat
yang berbeda. Sifat geometris tersebut menjadi dasar pengukuran kinerja
dan perbandingan kedua model dalam penelitian ini.

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

Wav2vec 2.0 adalah sebuah model self-supervised learning (SSL) yang
mempelajari representasi dari ucapan langsung atau audio mentah melalui
pendekatan *constrastive learning*. Arsitekturnya terdiri dari tiga
komponen inti; feature encoder konvolusional, jaringan konteks berbasis
Transformer, dan modul kuantisasi yang bekerja dalam satu tugas pretext.
Memprediksi unit terakuntisasi yang benar dari representasi konteks pada
posisi yang ditutup (*masked).*

1.  **Feature encoder**

Sinyal audio mentah $\mathcal{X}$ (16 kHz) diproses oleh tujuh blok
konvolusi temporal, masing-masing berisi 512 kanal dengan Layer
Normalization dan aktivasi GELU. Susunan stride $(5,2,2,2,2,2,2)$ dan
lebar kernel $(10,3,3,3,3,2,2)$ menghasilkan barisan representasi laten
$z_{1},\ldots,z_{T}$ pada frekuensi keluaran sekitar 49 Hz (jarak
antar-frame $\approx$`<!-- -->`{=html}20 ms), dengan receptive field
selebar 400 sampel input atau setara 25 ms audio.

  -----------------------------------------------------------------------
                 Blok              Stride          Kernel           Kanal
  ------------------- ------------------- --------------- ---------------
                    1                   5              10             512

                  2-5                   2               3             512

                  6-7                   2               2             512
  -----------------------------------------------------------------------

  : []{#_Toc236748113 .anchor}Tabel 2.2 Konfigurasi Blok Konvolusi
  Temporal pada Encoder Audio

2.  **Jaringan konteks**
    $\mathbf{g:}\mathcal{Z}\mathbf{\rightarrow}\mathcal{C}$

Representasi laten $z$ diumpankan ke jaringan Transformer N4 yang
memodelkan ketergantungan jangka panjang antar-frame melalui
self-attention, menghasilkan representasi kontekstual
$c_{1},\ldots,c_{T}$. Konfigurasi Base memakai 12 blok
($d = 768,\ 8\ Head$, sedangkan Large memakai 24 blok ( $d = 1024$, 16
head). Informasi posisi ditanamkan melalui relative positional
convolutional embedding (kernel 128, 16 grup), bukan positional encoding
absolut. Representasi laten $z$ diumpankan ke jaringan Transformer N4
yang memodelkan ketergantungan jangka panjang antar-frame melalui
self-attention, menghasilkan representasi kontekstual
$c_{1},\ldots,c_{T}$ Konfigurasi Base memakai 12 blok ($d = 768$, 8
head), sedangkan Large memakai 24 blok ($d = 1028$, 16 head). Informasi
posisi ditanamkan melalui relative positional convolutional embedding
(kernel 128, 16 grup), bukan positional encoding absolut.

3.  **Modul kuantisasi**

Secara paralel, laten $z$ didiskretisasi melalui product quantization N9
dengan $G = 2$ codebook, masing-masing $V = 320$ entri. Pemilihan entri
dibuat terdiferensiasi menggunakan Gumbel-Softmax N7, N8

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     $$p_{g,v} = \frac{\exp\left( \left( l_{g,v} + n_{g,v} \right)\text{/}\tau \right)}{\sum_{k = 1}^{V}\exp\left( \left( l_{g,k} + n_{g,k} \right)\text{/}\tau \right)}$$   (2.2)
  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- ---------

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

dengan $l_{g,v}$ logit entri $v$ pada grup $g$, $\tau$ temperatur
(dijadwalkan turun dari 2 ke 0,5), dan
$n_{v} = - \log\left( - \log\left( u_{v} \right) \right)$,
$u_{v}\mathcal{\sim U}(0,1)\$derau Gumbel. Pada forward pass dipilih
entri diskret $i = \arg{\max_{j}p_{g,j}}$; pada backward pass digunakan
gradien fungsi Gumbel-Softmax kontinu (straight-through estimator N10)
sehingga tetap dapat dilatih.

4.  **Fungsi Objektif**

Pelatihan diarahkan gabungan dua komponen

  --------------------------------------------------------------------------------------------------------------------
     $$\mathcal{L =}\mathcal{L}_{\mathcal{m}} + \alpha\,\mathcal{L}_{\mathcal{d}},\quad\quad\alpha = 0,1$$   (2.3)
  -- ------------------------------------------------------------------------------------------------------- ---------

  --------------------------------------------------------------------------------------------------------------------

Constrastive loss $\mathcal{L}_{\mathcal{m}}$ memaksa model membedakan
target benar $q_{t}$ dari $K = 100$ pengecoh

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     $$\mathcal{L}_{\mathcal{m}} = - \log{}\frac{\exp\left( \frac{\text{sim}\left( c_{t},q_{t} \right)}{\kappa} \right)}{\sum_{\widetilde{q} \sim \mathcal{Q}_{\mathcal{t}}}^{}\exp\left( \frac{\text{sim}\left( c_{t},\widetilde{q} \right)}{\kappa} \right)}$$   (2.4)
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ---------

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Dengan $\kappa = 0,1$ dan
$sim(a,b) = \frac{a^{T}b}{\text{|}a\text{||}b\text{|}}$ (*cosine
similarity*, berpersan sebagai scoring function). Diversity loss
$\mathcal{L}_{\mathcal{d}}$ mendorong pemakaian merata seluruh entri
codebook via maksimalisasi entropi

  ------------------------------------------------------------------------------------------------------------------------------------
     $$\mathcal{L}_{\mathcal{d}} = \frac{1}{G}\sum_{g = 1}^{G}{\sum_{v = 1}^{V}\overline{p_{g,v}}\log\overline{p_{g,v}}}$$   (2.5)
  -- ----------------------------------------------------------------------------------------------------------------------- ---------

  ------------------------------------------------------------------------------------------------------------------------------------

Melalui mekanisme kontrastif, Wav2Vec2 membentuk representasi dengan
membedakan unit positif dan unit negatif pada ruang representasi.
Karakteristik tersebut dapat menghasilkan struktur embedding yang lebih
terpisah, sehingga dampaknya terhadap kinerja retrieval berbasis cosine
similarity dapat diamati melalui evaluasi empiris dalam penelitian ini
\[22\].

![[]{#_Toc236767908 .anchor}Gambar 2.1 Arsitektur
Wav2vec2](media/image3.png){width="2.775in"
height="4.464435695538057in"}

### Data2vec 

Data2vec merupakan model Self-Supervised Learning (SSL) yang modality
agnostic, artinya inputan yang dapat diproses dapat berupa audio, gambar
ataupun text. Perbedaan mendasarnya dengan Wav2vec2 2.0 terletak pada
bentuk prediksi, Data2vec tidak mempredisksi unit diskret terkuantisasi,
melainkan representasi laten kontekstual yang bersifat kontinu. Yang
dihasilkan oleh model itu sendiri melalui skema self-distillation.

1.  **Skema Teacher dan Student (*self-distillation*)**

Satu arsitektur Transformer \[20\] beroperasi dalam dua peran. Jaringan
student yang memproses input yang dimasking dan menghasilkan prediksi
$f_{t}(x)$ pada tiap posisi ter

masking. Jaringan teacher memproses input penuh (tanpa masking) dan
menyediakan target regresi $y_{t}$. Student dilatih meregresi target
teacher hanya pada posisi termasking

2.  **Pembaruan Teacher via EMA**

Pembaruan teacher tidak dilakukan oleh gradient, melainkan menggunakan
exponential moving average (EMA) dari parameter student
$\Delta \leftarrow \tau\,\Delta + (1 - \tau)\,$dengan $\Delta$ parameter
teacher, $\theta$ parameter student. Nilai $\tau$ dinaikan linear dari
$\tau_{0}\ ke{\ \tau}_{e}\$langkah awal lalu ditahan konstan, untuk
ucapan $\tau_{0} = 0,,\tau_{e} = 0,,\tau_{n} = 30.0$

3.  **Konstruksi Target Kontekstual**

Target dibentuk dengan merata-ratakan keluaran $K$ blok Transformer
teratas teacher, setelah masing-masing dinormalisasi

  ------------------------------------------------------------------------------------
     $$y_{t} = \frac{1}{K}\sum_{l = L - K + 1}^{L}\widehat{a_{t}^{\, l}}$$   (2.6)
  -- ----------------------------------------------------------------------- ---------

  ------------------------------------------------------------------------------------

Dengan $L$ dengan jumlah total blok (12 pada base), $K = 8$ blok teratas
$\widehat{a_{t}^{\, l}}$ keluaran blok $l$ ternormalisasi. Untuk ucapan
digunakan *instance nomralization* \[23\]. Untuk gambar layer
normalization \[24\]. Normalisasi ini mencegah keruntuhan representasi
(representation collapose). Karena target menggabungkan berupa lapisan
atas, ia bersifat kontekstual penuh. Target kontekstual penuh tersebut
menjadi salah satu karakteristik Data2Vec yang membedakannya dari
Wav2Vec2. Karakteristik ini telah dikaitkan dengan kemampuan
menghasilkan representasi yang informatif pada sejumlah tugas pemrosesan
suara \[10\], \[16\]. Dalam penelitian ini, karakteristik tersebut
dianalisis melalui penerapan Data2Vec pada tugas retrieval audio ayat
Al-Qur\'an.

4.  **Fungsi Objektif**

Student meregresi target dengan Smooth L1 loss (huber):

  ---------------------------------------------------------------------------------------------------------------------------
     $$\mathcal{L}\left( y_{t},f_{t}(x) \right) = \left\{ \begin{matrix}                                            (2.7)
     \frac{1}{2}\left( y_{t} - f_{t}(x) \right)^{2}\text{/}\beta, & \left| y_{t} - f_{t}(x) \right| \leq \beta \\   
     \left| y_{t} - f_{t}(x) \right| - \frac{1}{2}\beta, & \text{selainnya}                                         
     \end{matrix} \right.\ $$                                                                                       
  -- -------------------------------------------------------------------------------------------------------------- ---------

  ---------------------------------------------------------------------------------------------------------------------------

Parameter $\beta$ mengatur peralihan wilaayah kuadratik
$\left( l_{2} \right)$ dan linear $\left( l_{1} \right)$ untuk
$\beta \rightarrow \infty$ loss menjadi MSE murni yang bekerja baik pada
domain ucapan.

![[]{#_Toc236767909 .anchor}Gambar 2.2 Arsitektur
Data2vec](media/image4.png){width="2.4833333333333334in"
height="4.357854330708661in"}

### Cosine Similarity

Cosine Similarity mengukur kemiripan antara dua vektor berdasarkan sudut
cosine di antara keduanya, tanpa perlu memperhitungkan besar
magnitudonya \[21\]

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     $$\text{sim}\,(q,d) = \cos\theta = \frac{q \cdot d}{\text{|}q\text{|}\,\text{|}d\text{|}} = \frac{\sum_{i = 1}^{d}{q_{i}d_{i}}}{\sqrt{\sum_{i = 1}^{d}q_{i}^{2}}\,\sqrt{\sum_{i = 1}^{d}d_{i}^{2}}}$$   (2.8)
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ---------

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Nilainya berkisar pada \[-1, 1\], dengan nilai mendekati 1 menandakan
arah vektor yang sangat mirip. Perlu ditegakan bahwa dalam bidang
retrieval, cosine similarity berperan sebagai fungsi penilaian
(*scrolling function*), bukan sebagai metrik evaluasi \[21\]. Manning
dkk. Membedakkan secara tegas antara mekanisme scoring ranking dan
metrik evaluasi efektivitas sistem. Selain itu cosine distance
$\left( 1 - \cos\theta \right)$ bukan metrik jarak matematis, arena
melanggar pertidaksamaan segitiga \[25\]. oleh karena itu, dalam
penelitian ini cosine similarity digunakan sebagai fungsi penilai
kemiripan untuk menghasilkan peringkat, sedangkan efektivitas retrieval
diukur menggunakan metrik evaluasi yang dijelaskan pada sub-bab
berikutnya.

### Metriks Evaluasi Retrieval 

Efektivitas sistem retrieval dievaluasi menggunakna metrik yang berbasis
pada kualitas peringkat (ranking) hasil pencarian \[21\]. Penelitian ini
menggunakan tiga metrik utama berikut.

1.  **Top K Accuracy**

Metrik ini mengukur proporsi kueri yang dokumen relevannya ditemukan
dalam K peringkat teratas dalam pencarian

  ---------------------------------------------------------------------------------------------------------------------
     $$Top - K\, Accuracy = \ \ 1\  \div |Q|\sum_{q \in Q}^{}{1\left\lbrack rank(q) \leq K \right\rbrack}$$   (2.8)
  -- -------------------------------------------------------------------------------------------------------- ---------

  ---------------------------------------------------------------------------------------------------------------------

Dengan ∣Q∣ adalah jumlah total kueri, dan 1\[⋅\] adalah fungsi indikator
yang bernilai 1 apabila dokumen relevan berada pada peringkat kurang
dari atau sama dengan K, dan berniali 0 jika sebaliknya. Metrik ini
mencerminkan kegunaan praktis sistem bagi pengguna yang umumnya hanya
memeriksa hasil teratas.

2.  **Mean Reciprocal Rank (MRR)**

MRR mengukur seberapa tinggi peringkat dokumen relevan pertama, dengan
menghitung rata-rata kebalikan peringkatnya \[21\]. MRR memiliki rumus
demikian

  ---------------------------------------------------------------------------------
     $$MRR = \frac{1}{|Q|}\sum_{i = 1}^{|Q|}\frac{1}{\text{rank}_{i}}$$   (2.9)
  -- -------------------------------------------------------------------- ---------

  ---------------------------------------------------------------------------------

Dengan $\text{rank}_{i}$ posisi peringkat dokumen relevan pertama untuk
kueri ke-$i$. Nilai MRR mendekati 1 menunjukkan dokumen relevan
cenderung berada di peringkat paling atas.

3.  **Mean Average Precision (MAP)**

MAP merupakan metrik yang paling komprehensif karena memperhitungkan
presisi pada setiap posisi dokumen relevan sepanjang daftar peringkat
\[21\]. Untuk sebuah kueri, Average Precision (AP) dihitung sebagai

  --------------------------------------------------------------------------------------
     $$\text{AP} = \frac{1}{|R|}\sum_{k = 1}^{N}{P(k)} \cdot \text{rel}(k)$$   (2.10)
  -- ------------------------------------------------------------------------- ---------

  --------------------------------------------------------------------------------------

Dengan $P(k)$ presisi pada peringkat $k$ $\text{rel}(k)$ fungsi
indicator relevansi dokumen di peringkat $k$, $|R|$ jumlah total dokumen
relevan, dan $N$ jumlah dokumen. MAP kemudian adalah rata-rata AP atas
seluruh kueri

  -----------------------------------------------------------------------------
     $$\text{MAP} = \frac{1}{|Q|}\sum_{q = 1}^{|Q|}{\text{AP}(q)}$$   (2.11)
  -- ---------------------------------------------------------------- ---------

  -----------------------------------------------------------------------------

Ketiga metrik ini; Top-K Accuracy, MRR, dan MAP merupakan metrik
evaluasi yang secara konseptual berbeda dengan cosine similarity yang
berperan sebagai *scoring function*

### Dataset Quran-MD 

Penelitian ini menggunakan dataset Quran-MD (bx7), sebuah dataset
multimodal yang tersedia pada platform huggingface. Dataset ini
mengintegrasikan dimensi teks, linguistic, dan audio pada tingkat ayat
maupun kata. Untuk setiap ayat disediakan teks Arab asli serta rekaman
dari 30 pembaca (qori) yang berbeda guna mempresentasikan keragaman gaya
bacaan (qiraat) dan nuansa dialetik.

Secara kuantitatif, koleksi Quran-MD terdiri atas dua sub-dataset
terpisah. sub-dataset ayat (Buraaq/quran-md-ayahs) berisi 187080 sampel
rekaman ayat lengkap. Dan sub-dataset kata (Buraaq/quran-md-words)
berisi 77429 sampel pelafalan kata individual, sehingga totalnya
mencakup sekitar 264509 audio format berformat mp3 yang meliputi 114
surah dan 6236 ayat unik (bx7). Sesuai batasan masalah penelitian, objek
yang digunakan dibatasi pada surah Al-fatihah dan juz 30 dari
sub-dataset tingkat ayat. Pasangan audio-teks yang tersedia pada dataset
ini memungkinkan penerapan tugas similarity search berbasis suara tanpa
transkripsi.

### Metode Penelitian CRISP-DM 

Penelitian ini mengadopsi kerangka kerja (*framework*) Cross-industry
Standard Process for Data Mining (CRISP-DM) sebagai metodologi (bx8).
CRISP-DM merupakan model process standar yang berisifat iteratif dan
tidak bergantung pada industri maupun teknologi tertentu, sehingga
sesuai untuk penelitian berbasis data dan machine learning. Kerangka ini
terdiri atas enam fase yang saling terikat:

1.  Business Understanding. Merumuskan tujuan penelitian, yaitu
    mengimplementasikan Wav2vec2 dan Data2vec sebagai penghasil vector
    embedding untuk retrieval ayat Al-Qur'an serta menetapkan kriteria
    evaluasi dan skema perbandingan kedua model.

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

6.  Deployment. Menyusun Menyusun kesimpulan dan rekomendasi model serta
    konfigurasi lapisan yang paling sesuai berdasarkan hasil evaluasi
    dan skenario penelitian sebagai acuan bagi pengembangan sistem
    pembelajaran Al-Qur\'an berbasis audio

# 

# BAB III METODOLOGI PENELITIAN

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
melainkan ekstrasi representasi laten. Sementara fase Deployment
diarahkan pada sintesis temuan, penilaian kelayakan penerapan, dan
perumusan rekomendasi arsitektur berdasarkan hasil perbandingan kedua
model, bukan penerapan sistem dalam lingkungan produksi. Dengan
penyesuaian tersebut, proses penelitian dilaksanakan secara terstruktur
melalui enam fase, mulai dari *Business Understanding*, *Data
Understanding*, *Data Preparation*, *Modeling*, *Evaluation* dam
*Deployment* yang diadaptasi.

##  Business Understanding

Fase *business understanding* berfungsi menerjemahkan permasalahan
penelitian menjadi sasaran analistis yang terukur \[26\]. Tujuan utama
yang ditetapkan dalam penelitian ini adalah mengukur dan membandingkan
efektivitas fitur laten dua model Self-Supervised Learning (SSL) model,
yaitu Wav2vec2 dan Data2vec. Dalam tugas *retrieval* ayat Al-Quran.
Perbandingan dilakukan dalam kondisi *frozen.*Penelitian ini
memanfaatkan model *pre-trained* tanpa pelatihan ulang (fine-tuning).
Sehingga metrik yang dihasilkan merupakan *raw* performance masing
masing model.

Sasaran utama penelitian ini adalah mengimplementasikan dan
membandingkan dua model Self-Supervised Learning, yaitu Wav2Vec2 dan
Data2Vec, sebagai penghasil frozen embedding untuk retrieval audio ayat
Al-Qur\'an. Kedua model memiliki mekanisme *pre-training* yang berbeda.
Wav2Vec2 menggunakan pendekatan contrastive learning, sedangkan Data2Vec
menggunakan pendekatan self-distillation dengan target representasi
kontekstual.

Perbedaan mekanisme tersebut menjadi dasar untuk melakukan perbandingan
secara empiris. Perbandingan tidak diarahkan untuk menetapkan bahwa
salah satu paradigma selalu lebih unggul, tetapi untuk mengetahui
bagaimana kinerja masing-masing model ketika digunakan pada tugas
retrieval berbasis cosine similarity. Hasil perbandingan juga digunakan
untuk melihat apakah karakteristik representasi yang diperoleh pada
tugas ASR menunjukkan pola yang sama ketika model digunakan dalam
kondisi frozen embedding untuk retrieval.

Agar perbandingan kedua model dapat dilakukan secara objektif, kedua
model diproses menggunakan data, pembagian dataset, prosedur ekstraksi,
metode *similarity scoring*, dan skenario evaluasi yang sama. Kinerja
sistem dinilai menggunakan metrik retrieval, yaitu Mean Average
Precision (MAP), Mean Reciprocal Rank (MRR), dan Top-K Accuracy \[21\].
MAP digunakan untuk mengukur kualitas peringkat secara keseluruhan, MRR
digunakan untuk mengukur posisi kemunculan hasil relevan pertama,
sedangkan Top-K Accuracy digunakan untuk mengetahui apakah hasil relevan
muncul pada sejumlah posisi teratas. Nilai cosine similarity digunakan
sebagai skor untuk mengurutkan kandidat ayat, bukan sebagai metrik
evaluasi akhir.

Penelitian ini dibatasi pada penggunakan pre-trained model tanpa
fine-tuning , dengan objek teliti berupa surah Al-Fatihah dan Juz Amma,
serta pengujian pada tingkat ayat. Selain evaluasi berdasarkan model,
penelitian ini melakukan analisis terhadap representasi yang dihasilkan
oleh setiap lapisan Wav2Vec2 dan Data2Vec. Analisis tersebut digunakan
untuk menentukan konfigurasi lapisan yang menghasilkan kinerja retrieval
terbaik bagi masing-masing model. Konfigurasi terbaik yang diperoleh
dari data pengembangan kemudian digunakan dalam perbandingan utama
antara Wav2Vec2 dan Data2Vec pada data pengujian. Dengan prosedur
tersebut, perbandingan dilakukan berdasarkan konfigurasi yang paling
sesuai bagi masing-masing model. Pada model yang menggunakan
constrastive seperti Wav2vec2, informasi fonetik cenderung meningkat di
layer tengah dan menurun di layer akhir \[27\]. Sehingga lapisan terbaik
untuk sebuah studi kasus tidak selalu berada pada lapisan terkahir.
Dengan demikian penelitian ini sejak awal diarahkan untuk melakukan
ekstrasi secara *layer-wise*.

##  Data Understanding

Fase Data Understanding bertujuan mengenali karakteristik data secara
menyeluruh sebelum dilakukan pemrosesan, mencakup pengumpulan data,
pendeskripsian sifat data, eksplorasi pola, serta verifikasi kualitasnya
\[26\]. Pada fase data understanding, tidak adanya perubahan data. Namun
mengamati data dan menilai data, sehingga data-data yang ada masih
berupa data mentah. Pemahaman yang ada pada proses ini menjadi landasan
untuk proses *pre-processing* kedepannya. Sekaligus memastikan bahwa
data yang tersedia memang mampu mendukung tujuan penelitian.

Data yang akan digunakan bersumber dari dataset Quran-MD yang tersedia
pada platform *HuggingFace*, khususnya sub-dataset pada tingkat ayat
\[28\]. Dataset ini menyediakan pasangan audio-teks untuk setiap ayat
disertai rekaman audio dari 30 qori yang berbeda yang bertujuan untuk
merepresentasikan keragaman gaya bacaan ayat Al-Quran. Pada tahap
pengumpulan awal, akan didokumentasikan struktur dataset yang meliputi
berkas audio, teks Arab yang berpasangan, serta metadata penanda ayat
berupa nomor surah, nomor ayat, dan identitas qori. Untuk *summary* dari
dataset divisualisasikan dalam Gambar 4. Namun karena batasan masalah
penelitian ini, yang digunakan hanyalah 37 surah yang ada di juz Amma
dan surah Al-Fatihah.

![[]{#_Toc236767923 .anchor}Gambar 3.1 Ringkasan Dataset
Quran-MD](media/image5.png){width="4.725270122484689in"
height="1.4102559055118111in"}

Selanjutnya akan dilakukan analisis file audio. Mencakup format audio,
*sampling rate* asli dan jumlah kanal. Pendeskripsian ini penting karena
untuk model Wav2vec2 dan Data2vec mengharuskan input audio pada sampling
rate 16kHz dan pada kanal mono. Jika dilihat dari data Quran-MD. Selain
itu juga akan ada perhitungan terkait distribusi jumlah qori per ayat,
distribusi durasi rekaman antar ayat, serta distribusi jumlah ayat pada
tiap surah dalam subset yang digunakan. Proses ini memiliki peran
krusial karena menentukan banyaknya dokumen relevan bagi setiap query,
yang secara langsung memengaruhi perhitungan metrik *Mean Average
Precision* (MAP) dan *Mean Reciprocal Rank* (MRR). Sementara itu ,
distribusi durasi ayat perlu dipahami karena ketimpangan durasi antara
ayat pendek dan ayat panjang berpotensi memengaruhi hasil agregasi fitur
melalui *temporal pooling* pada tahap ekstraksi representasi laten.
Analisis karakteristik audio meliputi distribusi *sampling* rate, jumlah
kanal, dan durasi rekaman disajikan dalam Gambar 5

![[]{#_Toc236767924 .anchor}Gambar 3.2 Karakteristik
Audio](media/image6.png){width="4.933333333333334in"
height="3.5029615048118985in"}

Temuan dari fase Data Understanding ini akan menjadi landasan bagi
penetapan prosedur pre-processing dan penyususan ground truth pada
fase-fase selanjutnya.

Dataset testing yang digunakan dalam penelitian ini terdiri atas rekaman
video bacaan Juz Amma dan surah Al-Fatihah dari 81 mahasiswa Program
studio Teknik Informatika UIN Sunan Gunung Djati Bandung. Juz Amma
merupakan bagian dari Al-Qur'an yang mencakup 37 surah, yang dimulai
dari surah ke-78 (An-Naba) sampai dengan surah ke-114 (An-Nas). Dengan
demikian, setiap mahasiswa diharapkan mengumpulkan total 38 video (1
surah Al-Fatihah ditambah 37 surah Juz Amma). Rekaman tersebut
dikumpulkan sebagai bagian dari pemenuhan mata kuliah Tahfidz dan
persyaratan sidang komprehensif.

  -----------------------------------------------------------------------
  Total Mahasiswa                                                      81
  ------------------------------------ ----------------------------------
  Lengkap                                                              25

  Tidak Lengkap                                                        56

  Rata-rata video                                                    22.2

  Median video                                                       28.0

  Min Video                                                             0

  Max Video                                                            38
  -----------------------------------------------------------------------

  : []{#_Toc236748756 .anchor}Tabel 3.1 Summary Statistic Dataset Query

Pada Tabel 3.1 menunjukan ringkasan statistik dataset query. Dari 81
mahasiswa, hanya 25 mahasiswa (30.9%) yang telah melengkapi seluruh 38
video, sedangkan 56 mahasiswa (69.1%) belum melengkapi seluruh 38 video.
Rata-rata jumlah video per mahasiswa adalah 22,2 dengan median 28,0,
menunjukkan bahwa sebagian besar mahasiswa belum mencapai target
kelengkapan.

![[]{#_Toc236767925 .anchor}Gambar 3.3 Total Distribusi
Video](media/image7.png){width="5.508333333333334in"
height="3.2851049868766404in"}

Analisis lebih mendalam terhadap surah yang paling sering tidak lengkap
ditunjukkan pada Gambar 6. Bar chart horizontal menampilkan 15 surah
yang paling sering *missing* dari dataset mahasiswa. Surah-surah ini
menjadi indikator pola kelengkapan yang perlu diperhatikan.

Setiap data mahasiswa disimpan dalam direktori terpisah yang
diidentifikasi berdasarkan kombinasi Nomor Induk Mahasiswa (NIM) dan
nama, misalnya \"NIM_NAMA\". Namun, konvensi penamaan tidak homogen.
Ditemukan variasi separator seperti underscore (\"\_\") dan dash
(\"-\"), serta perbedaan format penamaan berkas video. Beberapa
mahasiswa menggunakan nomor surah asli sebagai prefiks (misalnya
\"78-AnNaba.mp4\"), sementara yang lain menggunakan nomor urut
pengumpulan (misalnya \"1. An-Nas.mp4\" yang sebenarnya adalah surah
ke-114). Variasi lain meliputi perbedaan kapitalisasi (\"Al-Ghasyiyah\"
vs \"al-ghasyiyah\"), penggunaan simbol khusus (\"Al-\_Asr\",
\"Al-A_la\"), dan penambahan kata \"Surah\" atau \"Q.S\" di awal nama
berkas.

![[]{#_Toc236767926 .anchor}Gambar 3.4 Box Plot Showing
Outlier](media/image8.png){width="5.508333333333334in"
height="2.3379593175853017in"}

­­Box plot pada Gambar 7 memperlihatkan adanya outlier dalam distribusi
jumlah video. Beberapa mahasiswa memiliki jumlah video yang sangat
rendah (0-4 video), sementara yang lain mencapai 38 video. Variansi ini
menunjukan tingkat pengumpulan mahasiswa yang beragam. Heterogenitas
struktur penamaan dan ketidaklengkapan data menjadi tantangan utama
dalam tahap prapemrosesan. Diperlukan proses normalisasi untuk
menghasilkan representasi data yang konsisten, serta mekanisme
pencocokan nama surah yang toleran terhadap variasi penamaan untuk
memungkinkan pemrosesan otomatis yang akurat.

##  Data Preparation

Fase Data Preparation bertujuan mengubah data mentah menjadi data yang
siap digunakan untuk proses ekstrasi representasi laten. Fase ini
mencakup empat tahap yaitu seleksi data, normalisasi audio, verifikasi
segmentasi ayat, kontruksi ground truth sebagai dasar evaluasi retrieval
ayat.

Proses pertama adalah seleksai data sesuai dengan Batasan masalah, yakni
memilih rekaman dari surah Al-Fatihah dan surah-surah dalam Juz Amma
yang disusun berdasarkan ayat. Dari proses seleksi ini akan disusun
*identifier* yang terdiri dari gabungan qori, surah dan ayat. Untuk data
pada Quran-MD karena mencakup keseluruhan surah yang kita perlukan
hanyalah surah Al-Fatihah dan surah-surah Juz Amma. Sedangkan pada data
*query* yaitu rekamanan mahasiswa itu sudah terdiri dari surah-surah Juz
Amma, namun perlu diingat untuk tugas pengumpulan tahfiz terdapat video
ayat Al-Quran yang ada kaitannya dengan Sains dan teknologi, itupun
perlu dipisahkan.

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
satu ayat. Segmentasi dilakukan melalui dua tahap. Pertama, model
*forced alignment WhisperX* digunakan untuk memperoleh *timestamp* pada
tingkat kata dari rekaman mahasiswa satu full surat. Kedua, batas antar
ayat ditentukan melalui alokasi proporsional berdasarkan jumlah kata
tiap ayat yang bersumber dari API Al-Quran, kemudian audio dipotong
menggunakan library *pydub*. Perlu dicatat bahwa pembagian durasi
berdasarkan rasio jumlah kata dapat menyebabkan kesalahan kecil pada
setiap batas ayat. Kesalahan tersebut dapat terbawa dan bertambah secara
kumulatif pada batas-batas berikutnya, sehingga posisi segmentasi dapat
semakin melenceng dari batas ayat yang sebenarnya, terutama pada bagian
akhir surah.

##  Modeling

Fase Modeling pada penelitina ini diadaptasi dari makna aslinya dalam
*framework* CRISP-DM. karena penelitian ini menggunakan *frozen
embedding* tanpa fine-tuning atau transfer learning, fase ini tidak
melibatkan pelatihan model, melainkan befokus pada ekstraksi
representasi laten dari mdoel pre-trained serta konstruksi *similarity*
yang menjadi dasar proses retrieval. Fase ini terdiri atas empat
sub-proses yaitu pemilihan model, ekstraksi representasi laten secara
layer-wise, agregasi fitur menjadi vektor embedding, dan perhitungan
skor kemiripan untuk menghasilkan peringkat.

Subproses pertama adalah pemilihan model. Penelitian ini menggunakan dua
model Self-Supervised Learning (SSL), yaitu Wav2Vec2 dan Data2Vec,
sebagai penghasil frozen embedding. Kedua model digunakan tanpa
fine-tuning agar representasi yang dievaluasi berasal langsung dari
model pretrained. Wav2Vec2 mempelajari representasi suara melalui
pendekatan contrastive learning dengan membedakan representasi target
dari sejumlah kandidat negatif \[1\], sedangkan Data2Vec menggunakan
pendekatan self-distillation dengan memprediksi representasi laten
kontekstual yang dihasilkan oleh jaringan teacher \[8\]. Perbedaan
mekanisme pretraining tersebut menjadi dasar pelaksanaan analisis
komparatif untuk mengetahui kinerja representasi yang dihasilkan oleh
kedua model pada tugas retrieval audio ayat Al-Qur\'an. Agar
perbandingan dilakukan secara konsisten, kedua model diproses
menggunakan data, tahapan preprocessing, metode ekstraksi embedding,
fungsi penilaian cosine similarity, dan prosedur evaluasi yang sama.

Proses selanjutnya adalah ekstrasi representasi laten. Setiap audio yang
telah dinormalisasi, baik pada Database (D) maupun Query set (Q),
diinputkan ke masing-masing model untuk memperoleh representasi laten.
Ekstrasi tidak dibatasi pada lapisan Transformer terkahir, melainkan
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

##  Evaluation

Fase ini bertujuan untuk mengukur efektivitas retrieval dari kedua model
secara kuantitafif dan membandingkan kinerja keduanya berdasarkan hasil
evaluasi yang diperoleh. Pada fase inilah peringkat yang dihasilkan
*cosine similarity* dinilai menggunakan metrik evaluasi retrieval.
Metrik evaluasi menilai kualitas keseluruhan peringkat terhadap ground
truth, bukan menghitung kemiripan antar vector \[21\].

Fase ini terdiri atas lima subproses, yaitu perhitungan metrik evaluasi,
evaluasi layer-wise, analisis perbandingan model, perbandingan tiga
skenario evaluasi terkontrol, dan analisis kualitatif.

Sub proses pertama adalah perhitungan metrik evaluasi retrieval.
Penelitian ini menggunakan tiga metrik berbasis kualitas peringkat,
yaitu Top-K Accuracy, Mean Reciprocal Rank (MRR), dan Mean Average
Precision (MAP). Top-K Accuracy mengukur proporsi *query* yang dokumen
relevannya berhasil ditempatkan dalam $K$ peringkat teratas, dengan
variasi nilai $K$ berupa Top-1, Top-5, dan Top-10, dimana Top-1
digunakan sebagai ukuran evaluasi dengan persyaratan posisi hasil yang
paling ketat karena dokumen relevan harus berada pada peringkat pertama.
MRR mengukur seberapa tinggi peringkat dokumen relevan pertama,
sedangkan MAP menilai kualitas peringkat secara menyeluruh dengan
memperhitungkan presisi pada setiap posisi dokumen relevan sepanjang
daftar. Ketiga metrik dihitung secara terpisah untuk Wav2vec2 dan
Data2vec agar dapat dibandingkan secara langsung.

Sub proses kedua adalah evaluasi layer-wise. Ketiga metrik dihutung pada
setiap lapisan Transformer dari kedua model. Hasilnya disusun menjadi
kurva kinerja terhadap indeks lapisan sehingga dapat diidentifikasi
lapisan yang menghasilkan kinerja retrieval tertinggi bagi masing-masing
model. Evaluasi layer-wise digunakan untuk menentukan konfigurasi
lapisan yang menghasilkan nilai MAP tertinggi bagi masing-masing model.
Pemilihan lapisan dilakukan menggunakan himpunan pengembangan agar data
pengujian tidak digunakan dalam proses penentuan konfigurasi. Lapisan
terpilih kemudian digunakan dalam perbandingan akhir Wav2Vec2 dan
Data2Vec pada himpunan pengujian. Analisis ini juga memberikan informasi
tambahan mengenai distribusi kualitas representasi retrieval pada setiap
lapisan model. Pemilihan lapisan menggunakan himpunan pengembangan dan
pelaporan akhir menggunakan himpunan pengujian dilakukan untuk
mengurangi bias pemilihan konfigurasi.

Subproses ketiga adalah analisis perbandingan kinerja kedua model.
Perbandingan dilakukan menggunakan hasil evaluasi Wav2Vec2 dan Data2Vec
pada konfigurasi lapisan terbaik masing-masing. Nilai MAP, MRR, dan
Top-K Accuracy digunakan untuk mendeskripsikan perbedaan kinerja
retrieval, sedangkan interval kepercayaan bootstrap terhadap selisih
Average Precision per kueri digunakan untuk menilai kebermaknaan
perbedaan MAP kedua model.

Apabila interval kepercayaan 95% terhadap selisih MAP tidak mencakup
nilai nol, perbedaan kinerja dinyatakan memiliki dukungan statistik pada
tingkat kepercayaan yang digunakan. Sebaliknya, apabila interval
kepercayaan mencakup nilai nol, belum terdapat bukti yang cukup untuk
menyatakan bahwa salah satu model lebih unggul secara meyakinkan.
Prosedur ini digunakan sebagai aturan interpretasi statistik dalam
penelitian.

Sub proses keempat adalah perbandingan tiga scenario evaluasi dalam
lingkungan yang terkontrol. Untuk memisahkan pengaruh kualitas
representasi model dari pengaruh kualitas data, retrieval dijalankan
pada tiga scenario yang berbagi satu protokol identik. Layer sweep pada
dev set, pelaporan pada test set, serta uji signifikansi *bootstrap*
namun berbeda pada sumber data query dan databasenya. Skenario A
merupakan kondisi penelitian, yaitu query rekamanan mahasiswa terhadap
database Quran-MD (lintas dataset). Skenario B menggunakan Quran-MD pada
kedua sisi dengan pembagian *leave-reciters-out,* sehingga qori pada
query tidak muncul pada database, skenario ini menilai kualitas
embedding pada kondisi akustik terkendali. Skenario C menggunakan data
mahasiswa pada kedua sisi dengan pembagian leave-students-out, sehingga
domain gap akustik dihilnagkan sepenuhnya namun noise segmentasi tetap
dipertahankan. Pada ketiga skenario, relevansi didefinisikan sebagain
kesamaan pasangan (surah, ayat) lintas pembaca, sehingga tidak ada satu
pembaca yang muncul pada kedua sisi. Perbandingan skenario A dengan B
mengisolasi pengaruh perbedaan domain akustik, sedangkan perbandingan
skenario B dengan C mengisolasi pengaruh kualitas akuisisi dan
segmentasi. Karena ketiga skenario memiliki ukuran databbase dan
kerapatan dokumen relevan yang berbeda, nilai MAP dilaporkan bersama
lift terhadap baseline acak agar perbandingan antar skenario bersifat
jujur.

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

##  Deployment 

Fase Deployment pada penelitian ini diadaptasi dari makna aslinya dalam
CRISP-DM. Karena penelitian bersifat komparatif dan fasi ini tidak
melibatkan penerapan model dalam sistem produksi. Melainkan berfokus
pada sintesis temuan menjadi kesimpulan yang utuh serta perumusan
rekomendasi bagi penelitian lanjutan. Fase ini terdiri atas tiga sub
proses yaitu sintesis temuan, perumusan rekomendasi pemilihan model, dan
identifikasi arah penelitian lanjutan.

Sub proses pertama adalah sintesis temuan. Seluruh hasil evaluasi,
meliputi nilai metrik retrieval, kurva kinerja layer-wise, hasil
perbandingan statistik, perbandingan antar skenario, dan temuan analisis
kualitatif, dirangkum untuk memperoleh gambaran menyeluruh mengenai
kinerja kedua model. Sintesis tersebut menjelaskan hasil implementasi
Wav2Vec2 dan Data2Vec, perbedaan kinerja retrieval kedua model, tingkat
kebermaknaan perbedaannya, serta konfigurasi lapisan yang paling sesuai
bagi masing-masing model

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
mendatang.

# 

# BAB IV HASIL DAN PEMBAHASAN

##  Hasil Business Understanding

Tahap Business understanding menghasilkan dua output utama sesuai alur
penelitian, yaitu identifikasi masalah penelitian, yaitu identifikasi
masalah penelitian beserta tujuan penelitian, serta penetapan kriteria
evaluasi dan indikator keberhailan. Kedua output ini menjadi fondasi
arah seluruh proses peneltiain yang dilaksanakan, mulai dari pengumpulan
data audio hingga evaluasi kinerja retrieval kedua model yang
dibandingkan.

### Identifikasi Masalah 

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
wav2vec2 , sementara kejadian terhadap paradigma self-distilation yang
dimiliki Data2vec dalam scenario serupa masih sangat terbatas. Kedua,
meskipun penelitian terdahulu telah membandingkan langsung wav2vec2 dan
Data2vec untuk pengenalan ucapan (ASR) Bahasa Arab \[10\], perbandingan
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
koherensi geometris ruang vector itu sendiri \[30\]. Oleh karena itu,
keunggulan sebuah model pada ASR tidak dapat langsung diasumsikan
berlaku pada retrieval dan inilah celah yang menjadi fokut utama
penelitian ini.

### Tujuan Penelitian 

Berdasarkan identifikasi masalah tersebut, tujuan penelitian ini adalah
mengimplementasikan Wav2Vec2 dan Data2Vec sebagai penghasil frozen
embedding serta mengevaluasi dan membandingkan kinerja retrieval kedua
model pada audio ayat Al-Qur\'an. Hasil evaluasi digunakan untuk
memberikan rekomendasi mengenai model dan konfigurasi lapisan yang
sesuai berdasarkan dataset dan skenario penelitian yang digunakan,
khususnya untuk domain ayat Al-Qur'an yang memiliki karakteristik
fonetik khas. Untuk mencapai rekomendasi yang objektif, penelitian ini
membandingkan dua model dengan paradigma pretraining yang berbeda,
Wav2vec2 dengan paradigma *contrastive learning* dan Data2vec dengan
paradigma *Self-Distillation* dalam skenarion retrieval.

Tujuan penelitian ini terdiri atas dua tujuan utama. Pertama,
mengimplementasikan model Wav2Vec2 dan Data2Vec sebagai penghasil frozen
embedding dalam sistem retrieval audio ayat Al-Qur\'an. Kedua,
mengevaluasi dan membandingkan kinerja retrieval frozen embedding
Wav2Vec2 dan Data2Vec berdasarkan metrik Mean Average Precision (MAP),
Mean Reciprocal Rank (MRR), dan Top-K Accuracy, dengan cosine similarity
sebagai fungsi penilaian*.*

### Kriteria Evaluasi dan Keberhasilan 

Kriteria evaluasi dan validitas eksperimen ditetapkan sebagai acuan
untuk menilai ketercapaian dua tujuan penelitian, yaitu implementasi
Wav2Vec2 dan Data2Vec sebagai penghasil frozen embedding serta evaluasi
dan perbandingan kinerja retrieval kedua model. Kriteria tersebut
mencakup keberhasilan implementasi sistem, validitas data evaluasi, dan
prosedur perbandingan kinerja model.

Pada aspek implementasi, sistem dinyatakan berhasil apabila Wav2Vec2 dan
Data2Vec dapat digunakan untuk mengekstraksi representasi frozen
embedding dari data audio yang memenuhi persyaratan input model.
Embedding yang dihasilkan harus memiliki bentuk dan dimensi yang
konsisten, dapat dikaitkan dengan metadata identitas ayat, serta dapat
digunakan dalam perhitungan cosine similarity dan penyusunan peringkat
hasil retrieval. Tidak terbentuknya embedding pada sebagian audio tidak
secara langsung dikategorikan sebagai kegagalan implementasi. Setiap
kasus tersebut terlebih dahulu dianalisis berdasarkan penyebabnya,
seperti durasi audio yang terlalu pendek, berkas audio yang rusak,
sinyal kosong setelah preprocessing, ketidaksesuaian format input,
keluaran numerik yang tidak valid, keterbatasan komputasi, atau
kesalahan pada tahapan pemrosesan. Kasus yang disebabkan oleh
karakteristik audio yang tidak memenuhi kebutuhan input model dicatat
sebagai keterbatasan data atau keterbatasan model dalam memproses
karakteristik input tertentu. Sementara itu, kasus yang disebabkan oleh
kesalahan kode, konfigurasi, atau proses komputasi dikategorikan sebagai
kegagalan teknis yang harus diperbaiki sebelum evaluasi dilanjutkan.

Untuk menjaga konsistensi perbandingan, evaluasi hanya dilakukan
terhadap audio yang dapat diproses oleh kedua model. Apabila suatu audio
tidak menghasilkan embedding pada salah satu model, audio tersebut tidak
dimasukkan ke dalam evaluasi utama kedua model. Jumlah audio yang
dikeluarkan, karakteristiknya, dan penyebab kegagalan ekstraksinya tetap
didokumentasikan sebagai bagian dari analisis kualitas data dan
keterbatasan penelitian. Prosedur ini memastikan bahwa Wav2Vec2 dan
Data2Vec dibandingkan menggunakan himpunan kueri dan basis data yang
sama.

Implementasi dinilai berhasil apabila pipeline ekstraksi dapat
dijalankan secara konsisten pada seluruh data yang memenuhi persyaratan
input model, sedangkan setiap data yang tidak dapat menghasilkan
embedding telah diidentifikasi penyebabnya, didokumentasikan, dan
ditangani menggunakan prosedur eksklusi yang sama pada kedua model.

Pada aspek validitas data evaluasi, setiap audio kueri harus memiliki
setidaknya satu audio relevan di dalam basis data. Relevansi ditentukan
berdasarkan kesamaan pasangan identitas surah dan ayat antara audio
kueri dan audio pada basis data. Ketentuan tersebut diperlukan agar
tidak terdapat kueri yang memperoleh nilai Average Precision sebesar nol
hanya karena dokumen relevan tidak tersedia di dalam basis data.

Kinerja retrieval dievaluasi menggunakan Mean Average Precision (MAP),
Mean Reciprocal Rank (MRR), dan Top-K Accuracy. MAP digunakan untuk
menilai kualitas peringkat hasil retrieval secara keseluruhan, MRR
digunakan untuk menilai posisi kemunculan dokumen relevan pertama,
sedangkan Top-K Accuracy digunakan untuk mengetahui apakah dokumen
relevan ditemukan pada sejumlah posisi teratas. Cosine similarity
digunakan sebagai fungsi penilaian untuk menghitung kemiripan dan
menghasilkan urutan kandidat ayat, bukan sebagai metrik evaluasi.

Untuk menjaga objektivitas pemilihan konfigurasi, lapisan terbaik
masing-masing model ditentukan menggunakan himpunan pengembangan
berdasarkan nilai MAP tertinggi. Lapisan terpilih kemudian digunakan
untuk mengevaluasi kinerja akhir pada himpunan pengujian yang tidak
digunakan selama proses pemilihan lapisan. Pemisahan tersebut dilakukan
untuk mengurangi bias pemilihan konfigurasi dan menghasilkan estimasi
kinerja yang lebih objektif.

Perbandingan Wav2Vec2 dan Data2Vec dilakukan berdasarkan nilai metrik
yang diperoleh pada himpunan pengujian. Kebermaknaan selisih kinerja
kedua model dianalisis menggunakan interval kepercayaan 95% yang
dihitung melalui metode bootstrap dengan 10.000 resampling terhadap
nilai Average Precision per kueri. Apabila interval kepercayaan selisih
MAP tidak mencakup nilai nol, perbedaan kinerja kedua model dinyatakan
memiliki dukungan statistik. Sebaliknya, apabila interval kepercayaan
mencakup nilai nol, belum terdapat bukti yang cukup untuk menyatakan
bahwa salah satu model lebih unggul secara meyakinkan.

Penelitian tidak menetapkan nilai minimum MAP, MRR, atau Top-K Accuracy
sebagai syarat keberhasilan. Ketercapaian penelitian ditentukan
berdasarkan keberhasilan implementasi kedua model, validitas prosedur
evaluasi, dan tersedianya hasil perbandingan empiris yang dapat menjawab
rumusan masalah. Dengan demikian, hasil berupa perbedaan yang signifikan
maupun tidak signifikan tetap menjadi temuan penelitian selama diperoleh
melalui prosedur evaluasi yang konsisten dan dapat
dipertanggungjawabkan.

##  Hasil Data Understanding

Data dalam skenario ini data terbagi menjadi dua kelompok. *ground
truth* dan data testing, yang mana ground truth didapatkan dari Quran-MD
\[28\], dan data testing didapatkan dari pengumpulan rekaman bacaan
Al-Quran dari mahasiswa. Eksplorasi ini menghasilkan pemahaman mengenai
cakupan, distribusi, serta kualitas data. Termasuk temuan yang
berimplikasi pada validitas valuasi. Quran-MD berupa parquet file yang
berisikan file berikut.

  -----------------------------------------------------------------------
  Nama Column                          Tipe Data
  ------------------------------------ ----------------------------------
  surah_id                             Int32

  ayah_id                              Int32

  Surah_name_ar                        String

  Surah_name_en                        String

  Surah_name_tr                        String

  Ayah_count                           Int32

  Ayah_ar                              String

  Ayah_tr                              String

  Reciter_id                           String

  Reciter_name                         String

  Audio                                Audio File (MP3 format)
  -----------------------------------------------------------------------

  : []{#_Toc236748707 .anchor}Tabel 4.1 Atribut dan Tipe Data Quran-MD

Adapun data query berbentuk berkas video yang masing-masing merupakan
pembacaan satu surah Al-Qur'an secara penuh oleh seorang mahasiswa.

Karena evaluasi pada penelitian ini dijalankan pada skenario terkontrol
(A, B, dan C) yang berbeda pada sumber data query dan databasenya,
pemahaman data berikut mencakup kedua himpunan sekaligus, yaitu himpunan
mahasiswa dan himpunan Quran-MD, beserta pembagiannya pada masing-msaing
skenario.

### Karakteristik Himpunan data Mahasiswa 

Himpunan data mahasiswa terdiri atas 25.945 berkas audio ayat hasil
segmentasi yang ditemukan pada direktoriyang direkam oleh 42 mahasiswa,
tersebar pada 396 pasangan (mahasiswa, surah) dan mencakup 20 surah dari
Juz Amma. Setiap berkas merepresentasikan bacaan satu surat oleh satu
mahasiswa. Seluruh nama folder, meskipun memiliki variasi format
penulisan, berhasil dinormalisasi dan dipetakan ke nomor surah standar
yang sesuai. Tidak ada folder yang gagal dikenali atau tidak dapat
ditentukan nomor surahnya.

Distribusi cakupan surah pada himpunan kueri sangat tidak seimbang.
Beberapa surah direkam oleh mayoritas mahasiswa, misalnya 'Abasa dan
Al-Balad, masing-masing oleh 32 mahasiswa sementara sejumlah surah
lainnya (Al-Bayyina, Al-Humaza, dan Al-Masad) hanya direkam oleh satu
mahasiswa, ketidakseimbangan ini penting dicatat karena metrik retrieval
pada surah dengan cakupan sangat rendah akan memiliki variansi tinggi
dan sensitif terhadap kesalahan tunggal, sebagaimana dibahas lebih
lanjut pada evaluasi kelemahan penelitian.

### Karakteristik Himpunan Data Quran-MD 

Himpunan Quran-MD memuat 17.130 bacaan yang dibacakan oleh 30 qori
mencakup surah Al-Fatihah dan Juz Amma (38 surah). Verifikasi integritas
manifest menunjukkan 0 berkas hilang dan 0 identitas ayat kosong,
sehingga seluruh entri valid sebagai target retrieval. Setiap klip
beranotasi identitas ayat (nomor surah dan nomor ayat) secara akurat,
yang menjadi dasar penentuan relevansi. Setiap klip beranotasi identitas
ayat (nomor surah dan nomor ayat) secara akurat, yang menjadi dasar
penentuan relevansi./

Himpunan Quran-MD berperan sebagai database pada scenario A dan sebagai
query sekaligus database pada scenario B. pada scenario B, ke-30 qori
dibagi secara leave-reciters-out dengan pembagian 9 qori sebagai query
dan 21 qori sebagai database, sehingga tidak ada qori yang muncul pada
kedua sisi.

### Cakupan relevansi dan Ukuran Skenario 

Aspek terpenting bagi validitas evaluasi adalah cakupan relevansi, yaitu
jaminan bahwa setiap query memiliki setidaknya satu dokumen relevan pada
database. Ukuran dan cakupan relevansi ketiga skenario setelah
penyingkiran klip gagal dirangkum pada tabel berikut.

  ------------------------------------------------------------------------------
  **Skenario**                **Klip Query** **Klip Database**         **Cakupan
                                                                     Relevansi**
  ------------------------- ---------------- ----------------- -----------------
  A (Mahasiswa-\> Quran-MD)            7.118            17.127              100%

  B (Quran-MD-\>Quran-MD)              5.139            11.988              100%

  C (Mahasiswa-\>Mahasiswa)            2.452             4.666            99,31%
  ------------------------------------------------------------------------------

  : []{#_Toc236748708 .anchor}Tabel 4.2 Ukuran Data dan Cakupan
  Relevansi pada Setiap Skenario Evaluasi

Pada scenario A dan B, seluruh ayat unik yang muncul sebagai query
tercakup 100% oleh setidaknya satu klip pada database, sehingga tidak
ada query yang secara artifisial dipaksa bernilai MAP = 0 akibat
ketiadaan dokumen relevan. Pada scenario C, cakupan relevansi mencapai
99,31% sebanyak 17 query tidak memiliki dokumen relevan pada database
dan dikeluarkan dari perhitungan metrik, sehingga dari 2.452 klip query
terdapat 2.435 query yang tercakup dan dievaluasi.

### Kualitas Data 

Eksplorasi mengungkap dua temuan kualitas yang haarus dilaporkan secara
transparan. Pertama karena segmentasi ayat pada himpunan query dilakukan
secara manual, audit menemukan bahwa dari 396 pasangan (mahasiswa,
surah), 353 berstatus lengkap sesuai dengan jumlah ayat pada surah,
sedangkan 43 berstatus tidak lengkap yaitu jumlah berkas hasil
segmentasi kurang dari jumlah ayat sebernarnya. Kasus paling ekstrem
adalah satu mahasiswa pada surah Al-Fil yang hanya menghasilkan 1 dari 5
ayat. Ketidaklengkapan ini mengindikasikan potensi kesalahan segmentasi
pada sebagian kecil data.

Kedua, pada tahap ekstraksi embedding teridentifikasi sejumlah klip yang
gagal diproses karena durasi audio sangat pendek. 1 Kilp pada himpunan
query dan 3 klip pada database. Klip-klip ini ditangani secara khusus
agar tidak menjadi data yang kurang bagus dalam perhitungan kemiripan.
Sebagaimana yang telah diuraikan dalam data preparation.

##  Hasil Data Preparation

### Normalisasi Audio 

Seluruh berkas audio pada himupnan query dan database diseragamkan ke
sampling rate 16 khz dengan kanal mono, sesuai dengan spesifikasi
masukan yang disyaratkan oleh kedua model pretrained Wav2vec2 dan
Data2vec. Normalisasi ini memastikan bahwa perbedaan format perekaman
antara rekaman query, yang diupload mahasiswa dan rekaman pada Quran-MD
tidak menjadi sumber variasi yang ditakutkan malah menjadi variable
penentu hasil performa model SSL.

1.  **Normalisasi Data *Ground Truth***

Normalisasi audio dilakukan melalui pipeline pemrosesan bertahap untuk
memastikan memastikan semua data memenuhi syarat untuk tiap model SSL.
Dataset Quran-MD disimpan dalam format *Parquet* dengan audio
terkompresi MP3 (bitrate 64kbps) sebagai raw bytes. Proses ekstraksi dan
konversi mengikuti tahapan berikut

1.  **Ekstraksi Audio dari *Parquet***

Dalam proses ini digunakan library datasets dan soundfile. Audio dalam
parquet disimpan sebagai struktur {bytes, path} dengan format MP3.
Library datasets secara default melakukan decoding otomatis menggunakan
*soundfile,* yang mana tidak mendukung format MP3. Oleh karena itu,
decoding otomatis dinonaktifkan *cast_column("audio",
Audio(decode=False))* untuk mengakses raw byte secara langsung.

2.  **Decoding MP3 dengan *librosa***

Raw bytes MP3 didecode menggunakan *librosa.load()* dengan parameter
sr=None untuk mempertahankan sampling rate asli tanpa resampling paksa.
Pendekatan ini mencegah degradasi kualitas akibat double resampling.
Fungsi ini mengembalikan numpy array dengan shape (channels, samples)
dan sampling rate asli.

3.  **Resampling menjadi 16kHz**

Audio dengan sampling rate tidak sama dengan 16kHz dilakukan resampling
menggunakan *librosa.resample()* dengan algoritma polyphase filtering
default. Resampling ini menggunakan anti-aliasing filter untuk mencegah
aliasing artifacts yang dapat mengganggu ekstraksi fitur audio
downstream.

4.  **Konversi ke Mono**

Audio stereo (jika ada) dikonversi ke mono menggunakan librosa.to_mono()
dengan melakukan averaging pada channel. Meskipun dataset Quran-MD
secara default sudah mono, langkah ini ditambahkan sebagai safeguard
untuk menangani kasus edge.

5.  **Penyimpanan WAV PCM 16-bit**

Audio yang telah dinormalisasi disimpan dalam format WAV dengan encoding
PCM 16-bit menggunakan *soundfile.write()*. Format ini dipilih karena:

- Lossless compression (tidak ada degradasi kualitas)

- Kompatibilitas universal dengan library audio processing

- Sesuai dengan requirement input Wav2Vec2 dan Data2Vec2

  1.  **Normalisasi data query (testing)**

Tahapan data preparation pada data query yaitu mentransformasi rekaman
video setoran hafalan mahasiswa menjadi unit-unit audio per ayat yang
diperuntukkan untuk proses berikutnya. Proses ini melibatkan ekstraksi
audio, penyelarasan teks ke suara (force alignment) dan pemotongan audio
secara otomatis.

1.  **Ekstraksi Audio**

Seluruh berkas video dalam format .mp4, .mov, .avi, .mkv yang diunggah
oleh mahasiswa diekstraksi komponen audionya saja. Proses ini
menggunakan library *moviepy* untuk aliran audio dari video kedalam
format MP3.

2.  **Normalisasi Nama surah dan Nama berkas**

Mengingat variasi penamaan berkas oleh mahasiswa (contoh: \"78-annaba\",
\"an-nas.mp4\", \"114. an nas\"), dilakukan proses normalisasi teks
untuk memetakan nama berkas ke ID surah yang valid pada dataset Quran.
Proses ini meliputi

1.  *Character Cleaning:* menghapus angka diawal nama, mengubah
    separator (dash/underscore) menjadi spasi, dan menghapus karaktek
    non-alfanumerik.

2.  *String Mapping:* memastikan file dengan termapping dengan benar,
    contohnya: surah ke 78 yaitu An-Naba jika hanya diberikan nama
    "naba" maka akan menjadi "an-naba" . yang mana "an-naba" merupakan
    value yang telah disetarakan dan menjadi master nama surah ke 78.

<!-- -->

3.  **Transkripsi dan penyelarasan waktu (*Force Alignment*)**

Untuk membagi satu audio utuh surah menjadi potongan per ayat, digunakan
model WhisperX (berbasis-model large-v2). Prosesnya sebagai berikut

- Transkripsi Bahasa Arab: Model melakukan inferensi awal untuk
  mendeteksi kata-kata dalam bahasa Arab.

- Phoneme-level Alignment: Menggunakan model penyelarasan khusus untuk
  mendapatkan stempel waktu (*timestamp*) yang presisi di tingkat kata.

- Identifikasi Awal Surah: Sistem secara otomatis mendeteksi kata
  pertama dari surah (seperti *\'amma* pada An-Naba) untuk melewati
  bagian *ta\'awwudh* atau perkenalan (*intro*) yang dilakukan oleh
  mahasiswa

4.  **Segmentasi Ayat berbasis Rasio Kata**

Setelah stempel waktu kata didapatkan, dilakukan pembagian audio
berdasarkan jumlah kata tiap ayat yang bersumber dari API Al-Quran
Cloud. Audio dipotong menggunakan *pydub* berdasarkan rentang waktu
mulai dari kata pertama ayat hingga kata terakhir ayat tersebut. Hasil
akhirnya adalah berkas audio individu (ayah_01.mp3, ayah_02.mp3, dst)
dengan format PCM yang terorganisir dalam direktori per mahasiswa dan
per surah.

### Pembagian set *dev* dan set *test* 

Untuk menjamin pemilihan lapisan optimal tidak bias terhadapt data yang
sama yang digunakan untuk pelaporan akhir, himpunan query dibagi menjadi
dua bagian set pengembangan 70%, dan set uji 30%. Pembagian dilakukan
secara terstratifikasi berdasarkan pasangan (nomor surah, dan nomor
ayat), sehingga distribusi ayat pada kedua set tetap seimbang dan tidak
ada ayat yang sepenuhnya hilang dari salah satu set. Set pengembangan
digunakan utnuk menyapu seluruh lapisan (layer sweep) dan memilih
lapisan dengna MAP tertinggi bagi masing-masing model, sedangkan set uji
digunakan semata-mata untuk melaporkan kinerja akhir pada lapisan
terpilih. Protocol dua-set ini mengikuti praktik standar evaluasi
*query-by-example* pada tolak ukur SUPERB \[3\], yaitu memilih lapisan
terbaik pada set pengembangan dan melaporkannya pada set uji, guna
menghindari kebocoran informasi yang dapat menghasilkan estimasi kinerja
yang terlalu optimis.

##  Modeling

Fase Modeling dilakukan untuk mengekstrak frozen embedding dari kedua
model Self-Supervised Learning tanpa proses fine-tuning, kemudian
menghitung skor kemiripan untuk menghasilkan retrieval.

### Ekstraksi Frozen Embedding per-lapisan 

Kedua model, Wav2vec2 dan Data2vec, digunakan dalam kondisi *frozen*
yaitu seluruh bobot model dipertahankan sebagaimana haisl pretraining,
tanpa pembaruan parameter apa pun. Dari setiap klip audio diekstraksi
representasi pada 13 lapisan (satu lapisan proyeksi fitur dan dua belas
lapisan Transformer), masing-masing berdimensi 768. Untuk memperoleh
satu vector tetap per klip, representasi tingkat frame pada tiap lapisan
diagregasi melalui mean pooling sepanjang dimensi waktu, sehingga setiap
klip direpresentasikan oleh satu vector berdimensi 768 pada tiap
lapisan.

Ekstraksi dilakukan secara terpisah untuk himpunan query (7.119 Klip)
dan *database reference* (17.130 klip) pada kedua model. Urutan untuk
ekstraksi harus dijaga, dikarenakan vector pada indeks yang sama baik
pada Wav2vec2 maupun Data2vec selalu merujuk pada klip audio yang sama.
Konsistensi ini merupakan prasyarat agar perbandingan kinerja kedua
model bersifat adil dan bebas dari ketidakselarasan data. Verifikasi
menunjukkan seluruh keluaran bebas dari nilai NaN, dengan hanya klip
gagal yang telah ditangani pada fase sebelumnya.

### Data Cleaning 

Berbeda dengan normalisasi audio yang menyeragamkan format masukan,
tahap data cleaning bertujuan menyingkirkan klip yang gagal menghasilkan
embedding valid agar tidak mencemari perhitungan kemiripan. *Cleaning*
dilakukan secara bertahap mengikuti alur ekstraksi embedding.

1.  **Deteksi klip gagal saat ekstraksi**

Pada saat frozen embedding, setiap klip diproses satu per satu oleh
masing-masing model. Sejumlah kecil klip gagal diproses karena durasinya
terlalu pendek, sehingga jumlah frame yang tersisa setelah melewat
encoder konvolusi pada lapisan pertama tidak mencukupi untuk membentuk
representasi. Kegagalan ini ditangkap sebagai pengecualian pada saat
pemanggilan model. Klip yang gagal akan tetap disimpan pada buffer
sebagai vector nol dan ditandai melalui kolom failed pada manifest
output. Kegagalan teridentifikasi pada 1 klip dihimpunan query (Surah
Al-Fil ayat pertama) dan 3 klip pada database referensi.

2.  **Vektor Nol**

Vektor nol berbahaya bagi evaluasi berbasis cosine similarity karena
menghasilkan skor kemiripan nol (atau tidak terdefinisi/NaN) terhadap
seluruh kandidat. Akibatnya urutan peringkat klip terbuset menjadi acak
dan dapat merusak perhitungan MAP maupun MRR secara tidak disadari.

3.  **Penjaminan Keselarasan Baris**

Mekanisme union menjaga keselarasan baris (row alignment) antar kedua
model. Setiap indeks query dan referensi merujuk pada klip yang identik,
sehingga perbandingan kinerja kedua model itu selaras dan tidak ada
perbedaan data. Verifikasi terakhir memastikan himpunan data bersih
tidak lagi memuat vektor bernilai nol maupun NaN.

### Similarity Scoring dan Ranking 

Peringkat retrieval dibentuk dengan menghitung cosine similarity antara
vector embedding setiap query dan seluruh vector embedding dalam basis
data referensi pada lapisan yang sama. Untuk setiap query, seluruh klip
referensi diurutkan menurun berdasarkan skor peringkat teratas.

Perlu ditegaskan bahwa cosine similarity di sini berperan sebagai
scoring function yang menentukan urutan peringkat bukan sebagai metrik
evaluasi. Pemilihan cosine similarity sesuai dengan sifat retrieval
berbasis frozen embedding, yang mengukur kedekatan arah antar vector
pada ruang representasi tanpa bergantung pada magnitudonya. Kualitas
peringkat yang dihasilkan selanjutnya dievaluasi menggunakan metrik MAP,
Top-K Accuracy, dan MRR pada fase *Evaluation*

##  Evaluation

Pada penelitian ini hasil akan dievaluasi per rumusan masalah, setiap
temuan dipaparkan lalu langsung dibahas sebelum berpindah pada persoalan
yang lainnya. Untuk menjaga objektivitas, himpunan query dibagi menjadi
dua bagian yang tidak terikat melalui pemisahan terstratifikasi
berdasasrkan pasangan (surah, ayat) dev set (4.886 query, 70%) dan test
set (2.232 query, 30%). Stratifikasi memastikan setiap ayat terwakili
secara proporsiaonal pada kedua himpunan, sehingga distribusi ayat akan
merata.

Dev set digunakan sebagai tahap penalaan, sleuruh 13 lapisan
representasi tiap model diuji, dan lapisan dengan MAP tertinggi dipilih
sebagai konfigurasi terbaik (Wav2vec2 lapisan 7 dan Data2vec lapisan 5).
Test set yang tidak pernah disentuh selama pemilihan lapisan-baru
digunakan pada tahap akhir untuk melaporkan metrik final. Pemisahan ini
krusial untuk menghindari bias seleksi, apabila lapisan terbaik dipilih
dan dilaporkan pada data yang sama, skor yang dihasilkan akan cenderung
tinggi. Dengan memisahkan tahap pemilihan (dev set) dari tahap pelaporan
(test set), skor akhir yang diperoleh benar-benar mencerminkan kemampuan
generalisasi model secara adil.

Proses retrieval itu sendiri dilakukan sebagai berikut, setiap vector
embedding query dibandingkan terhadap seluruh 17.130 vektor pada
database referensi menggunakan cosine similarity sebagai *scoring
function*. Hasilnya akan dilakukan perankingan berdasarkan skor. Dan
signifikansi bootstrap (B = 10.000) pada selisih MAP per-query
menentukan apakah perbedaan antar model bermakna secara statistic atau
sekadar fluktuasi acak.

### Kinerja Retrieval Frozen Embedding 

Pada test set, konfiugarsi terbaik masing-masing model menghasilkan
kinerja sebagai berikut

  ------------------------------------------------------------------------
  Model          Lapisan       MAP       MRR     Top-1     Top-5    Top-10
                 Terbaik                                         
  ----------- ---------- --------- --------- --------- --------- ---------
  Wav2vec2             7    0,0178    0,1146    0,0730    0,1496    0,1904

  Data2vec             5    0,0188    0,1124    0,0699    0,1465    0,1935
  ------------------------------------------------------------------------

Kedua Model menunjukkan kinerja retrieval yang rendah secara absolut
(MAP di bawah 0,02 dan Top-1 di bawah 8%). Analisis per-query pada test
set menegaskan pola ini, dari 2.232 query, tidak ada satu pun yang
mencapai Average Precision sempurna, sekaligus tidak ada query yang
bernilai nol, artinya setiap query berhasil menarik setidaknya satu klip
relevan ke dalam peringkat. Namun, tidak ada yang diperingkatkan secara
bersih di posisi teratas. Temuan ini mengindikasikan bahwa frozen
embedding dari model SSL, dipadukan dengan scoring function cosine
similarity, belum memadai sebagai basis retrieval ayat yang andal tanpa
adaptasi lebih lanjut.

### Perbandingan Wav2vec2 dan Data2vec 

Selisih MAP antar model pada test set adalah +0,0009 (data2vec di atas
Wav2vec2). Uji signifikansi bootstrap (B = 10.000) menghasilkan selang
kepercayaan 95% -0,00005, +0,0024 yang memnuat nol, sehingga perbedaan
tidak signifikasn secara statistic. Secara per kueri, Data2vec unggul
pada 1.164 query dan Wav2vec2 1.068 query, perbandingan ini nyaris
seimbang.

Berdasarkan analisis signifikansi menggunakan bootstrap confidence
interval, perbedaan kinerja antara Wav2Vec2 dan Data2Vec tidak
menunjukkan dukungan statistik yang kuat. Dengan demikian, tidak
terdapat bukti yang cukup untuk menyatakan bahwa salah satu model lebih
unggul secara meyakinkan pada tugas retrieval ayat berbasis cosine
similarity dalam skenario penelitian ini.

### Lapisan paling informatif 

Lapisan terbaik terletak di pertengahan jaringan wav2vec2, lebih
tepatnya pada lapisan 7 dan Data2vec pada lapisan 5, lapisan terbaik
pada kedua model untuk tugas retrieval ternyata bukan pada lapisan
terakhir. Kinerja menaik dari lapisan awal, memuncak di pertengahan,
lalu menurun pada lapisan akhir (Wav2vec2 L12 MAP=0,0054, Data2vec
L12=0,0062).

Pola *inverted-U* ini konsisten dengen literatur analisis layer-wise
pada model SSL audio \[27\], yang menunjukkan bahwa lapisan akhir
cenderung terspesialisasi bagi tujuan pretraining dan kehilangan
Sebagian informasi fonetik yang justru berguna bagi tugas retrieval.
Temuan ini menvalidasi keputusan metodologis menyapu seluruh 13 lapisan
alih-alih mengasumsikan lapisan akhir sebagai representasi terbaik.

![[]{#_Toc236811332 .anchor}Gambar 4.1 Hasil Evaluasi Tiap-tiap
Layer](media/image9.png){width="5.508333333333334in"
height="1.439636920384952in"}

##  Deployment (Pertimbangan Penerapan)

Fase Deployment pada penelitian ini tidak berupa penerapan sistem pada
lingkungan produksi. Melainkan pertimbangan penerapan dari temuan yang
diperoleh, mengingat penelitian yang dilakukan adalah komparatif dan
menghasilkan prototype. Bukan siap pakai.

### Skema Penerapan Sistem Retrieval 

Secara arsitektural, sistem retrieval berbasis frozen embedding dapat
diterapkan melalui tiga tahap yang pertama yaitu proses ekstrasi untuk
data ground truth pada database referensi menjadi vector embedding,
kemudian mengektrasi data audio query dari lapisan model Wav2vec2 dan
Data2vec yang terbaik lalu dibandingkan dengan database referensi
menggunakan scoring function cosine similarity, lalu proses terakhir
dibuatkan perankingan hasil mana yang paling cocok dengan data yang ada
di database referensi berdasarkan kemiripannya. Skema ini efisien karena
ekstraksi referensi hanya perlu dilakukan sekali, sedangkan untuk proses
pencarian cukup menghitung cosine similarity pada vektor yang sudah
tersimpan.

### Kelayakan penerapan berdasarkan temuan 

Namun demikian, kinerja yang diperoleh belum memenuhi kelayakan untuk
penerapan operasional. Dengan MAP di bawah 0,02 dan Top-1 di bawah 8%,
sistem dalam kondisinya saat ini belum dapat diandalkan sebagai alat
pencarian ayat yang akurat bagi pengguna akhir. Oleh karena itu,
penerapan nyata mensyaratkan peningkatan terlebih dahulu, misalnya
melalui fine-tuning pada domain, pembelajaran metrik (metric learning),
atau strategi agregasi embedding yang lebih canggih daripada mean
pooling.

# 

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

\[12\] Z. Li, L. Wu, T. Li, and Y. Yan, "Improves Neural Acoustic Word
Embeddings Query by Example Spoken Term Detection with Wav2vec
Pretraining and Circle Loss," in *2021 12th International Symposium on
Chinese Spoken Language Processing (ISCSLP)*, Hong Kong: IEEE, Jan.
2021, pp. 1--5. doi: 10.1109/ISCSLP49672.2021.9362065.

\[13\] Y. Xie, Z. Zhang, and Y. Yang, "Siamese Network with wav2vec
Feature for Spoofing Speech Detection," in *Interspeech 2021*, ISCA,
Aug. 2021, pp. 4269--4273. doi: 10.21437/Interspeech.2021-847.

\[14\] C. Tang, Y. Wang, X. Chen, and W.-Q. Zhang, "Exploring Effective
Fusion Algorithms for Speech Based Self-Supervised Learning Models,"
Dec. 20, 2022, *arXiv*: arXiv:2212.10092. doi:
10.48550/arXiv.2212.10092.

\[15\] J. W. Yoon, S. M. Kim, and N. S. Kim, "MCR-Data2vec 2.0:
Improving Self-supervised Speech Pre-training via Model-level
Consistency Regularization," Jun. 14, 2023, *arXiv*: arXiv:2306.08463.
doi: 10.48550/arXiv.2306.08463.

\[16\] S. Yang *et al.*, "A Large-Scale Evaluation of Speech Foundation
Models," May 29, 2024, *arXiv*: arXiv:2404.09385. doi:
10.48550/arXiv.2404.09385.

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

\[30\] S. Zaiem, Y. Kemiche, T. Parcollet, S. Essid, and M. Ravanelli,
"Speech Self-Supervised Representations Benchmarking: a Case for Larger
Probing Heads," Feb. 21, 2024, *arXiv*: arXiv:2308.14456. doi:
10.48550/arXiv.2308.14456.

# Draft Jawaban Munaqosah: Pertanyaan 19-37

Dokumen ini disusun sebagai jawaban lisan. Setiap bagian diawali jawaban langsung, lalu penjelasan jika penguji meminta rincian. Path bukti ditulis relatif terhadap root repositori.

## Q19 - Di mana proses normalisasi dilakukan?

**Pertanyaan:** Di mana proses normalisasi dilakukan? Tunjukkan skrip Python-nya.

**Jawaban langsung:** Normalisasi audio referensi dilakukan dalam `backend-skripsi/pre-processing/normalisasi-data.py`. Audio dibaca pada sampling rate asli, diubah menjadi 16 kHz bila perlu, diubah menjadi mono bila memiliki lebih dari satu kanal, lalu disimpan sebagai WAV PCM 16-bit. Untuk rekaman mahasiswa, `prepare-audio.py` mengekstrak audio menjadi FLAC mono 16 kHz melalui FFmpeg. Saat inferensi, `backends/audio_loader.py` juga memastikan masukan model dimuat sebagai mono 16 kHz.

**Penjelasan lebih dalam:** Jadi normalisasi diterapkan pada jalur persiapan data dan dijaga lagi pada jalur pemuatan model. Tujuannya bukan memperbaiki isi bacaan, melainkan menyeragamkan format teknis agar Wav2Vec2 dan Data2Vec menerima masukan yang setara.

**Cuplikan kode:**

```python
# backend-skripsi/pre-processing/normalisasi-data.py:48-69
audio_array, original_sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
if original_sr != 16000:
    audio_array = librosa.resample(
        audio_array, orig_sr=original_sr, target_sr=16000
    )
if len(audio_array.shape) > 1:
    audio_array = librosa.to_mono(audio_array.T)
sf.write(str(output_path), audio_array, 16000, subtype="PCM_16")
```

```python
# backend-skripsi/pre-processing/prepare-audio.py:43-49
result = subprocess.run([
    'ffmpeg', '-y', '-loglevel', 'error', '-i', str(video_path),
    '-vn', '-ac', '1', '-ar', str(SAMPLE_RATE),
    '-c:a', 'flac', '-compression_level', '8', str(audio_path),
])
```

**Bukti:** `normalisasi-data.py:94-130` juga memvalidasi bahwa hasil memiliki `samplerate == 16000` dan `channels == 1`.

## Q20 - Apa itu kanal mono dan mengapa 16 kHz?

**Jawaban langsung:** Mono berarti audio hanya memiliki satu kanal, bukan kanal kiri dan kanan yang terpisah seperti stereo. Audio dibuat mono agar satu sinyal ucapan tidak digandakan atau dipengaruhi perbedaan kanal. Sampling rate 16 kHz dipakai karena kedua model dalam penelitian menerima audio ucapan pada 16 kHz, sehingga satu detik audio direpresentasikan oleh 16.000 sampel.

**Penjelasan lebih dalam:** Menurut teorema Nyquist, 16 kHz dapat merepresentasikan frekuensi sampai sekitar 8 kHz, yang mencakup bagian utama informasi ucapan. Menyeragamkan semua berkas ke mono 16 kHz juga mencegah format rekaman asal menjadi faktor pembaur dalam perbandingan model. Bukti implementasinya ada pada `prepare-audio.py:27,47`, `backends/wav2vec2.py:25-30`, dan `backends/data2vec.py:24-29`.

## Q21 - Contoh keterlacakan rekaman induk dan metode segmentasi

**Jawaban langsung:** Satu klip mahasiswa dapat dilacak melalui identitas klip, path hasil segmentasi, NIM, surah, ayat, serta status dan metode segmentasinya. Contoh nyata pada manifes final adalah klip `1207050036_097_003`, milik Farzan Argani, surah 97 ayat 3. Klip itu berada pada `1207050036_Farzan Argani/al-qadr/ayah_03.mp3` dan diberi provenance `audited_success_fallback`, metode `time_proportional`, dengan alasan bahwa hanya tiga kata bacaan terdeteksi untuk lima ayat.

**Penjelasan lebih dalam:** Nama folder dan `relpath` menghubungkan klip ayat ke rekaman mahasiswa atau kelompok rekaman asal, sedangkan kolom audit menjelaskan bagaimana segmen dihasilkan. Manifes final tidak menyimpan nama berkas video induk sebagai kolom tersendiri, tetapi manifes sebelum segmentasi menyimpannya pada field `source` di `prepare-audio.py:95-104`. Hubungan tersebut dijaga lewat NIM, folder, surah, dan slug.

**Bukti:** Contoh terdapat pada `backend-skripsi/embeddings/query-layerwise-20260803-v1/query/wav2vec2/manifest.csv:8234`. Struktur pencatatannya dibentuk oleh `pre-processing/segment_validation.py:536-558`, dengan kolom `relpath`, `nim`, `surah_id`, `ayah_id`, `audit_provenance`, `audit_fallback`, dan `audit_detail`.

## Q22 - Contoh disjoint pada Skenario B

**Jawaban langsung:** Disjoint berarti seorang qari hanya boleh berada pada satu sisi, sebagai pemilik kueri atau pemilik basis data, tidak pada keduanya. Contoh rasio B-70:30 membagi 30 qari Quran-MD menjadi 21 qari basis data dan 9 qari kueri. Semua klip milik satu qari mengikuti qari tersebut, sehingga model diuji lintas qari.

**Penjelasan lebih dalam:** Kode membuat dua himpunan pemilik, membentuk mask klip berdasarkan `reciter_id`, lalu memeriksa irisan kedua himpunan. Jika ada satu qari saja pada kedua sisi, proses dihentikan dengan `ValueError`. Dengan demikian, yang dipisah lebih dulu adalah identitas qari, bukan klip secara acak.

```python
# backend-skripsi/experiments/scenarios.py:338-350
query_subset = manifest.loc[q_mask, :]
database_subset = manifest.loc[db_mask, :]
overlap = set(q_owners) & set(db_owners)
if overlap:
    raise ValueError("Scenario B: owner overlap detected")
```

**Bukti empiris:** `Skripsi.md:1950-1956` mencatat B-60:40 sampai B-90:10 sebagai 18/12, 21/9, 24/6, dan 27/3 qari basis data/kueri.

## Q23 - Bagaimana data dibagi pada rasio 60:40, 70:30, dan seterusnya?

**Jawaban langsung:** Angka pertama adalah proporsi pemilik pada sisi basis data dan angka kedua adalah proporsi pemilik pada sisi kueri. Daftar pemilik diurutkan agar stabil, lalu dipermutasi dengan generator acak ber-seed 42. Jumlah pemilik basis data dihitung dengan `round(rasio × jumlah pemilik)`; sisanya menjadi pemilik kueri. Setelah itu, seluruh klip seorang pemilik mengikuti sisinya.

**Penjelasan lebih dalam:** Pada 60 mahasiswa, rasio 70:30 menghasilkan 42 mahasiswa basis data dan 18 mahasiswa kueri. Pada 30 qari, rasio yang sama menghasilkan 21 qari basis data dan 9 qari kueri. Pembagian dilakukan pada tingkat pemilik untuk menguji generalisasi ke pembaca yang tidak ada di basis data dan mencegah kebocoran identitas. Ini bukan pembagian jumlah klip secara tepat, sehingga persentase klip dapat berbeda karena tiap pemilik mempunyai jumlah klip berbeda.

```python
# backend-skripsi/experiments/splits.py:270-281
owners = sorted({_stable_value(value) for value in manifest[owner_column]})
rng = np.random.default_rng(seed)
permutation = rng.permutation(len(owners))
database_count = int(round(float(owner_ratio) * len(owners)))
database_owners = sorted(owners[int(index)] for index in permutation[:database_count])
query_owners = sorted(set(owners) - set(database_owners))
db_mask = manifest[owner_column].map(_stable_value).isin(list(database_owners)).to_numpy(dtype=bool)
q_mask = ~db_mask
```

**Bukti:** `results/scenario-matrix-60-students-v1/owner_splits.json` merekam seed, daftar pemilik, mask, jumlah baris, dan checksum untuk setiap rasio.

## Q24 - Mengapa rasio tidak berlaku pada Skenario A?

**Jawaban langsung:** Skenario A tidak membagi satu korpus berdasarkan pemilik. Semua 25.829 klip mahasiswa menjadi kueri dan semua 17.127 klip Quran-MD menjadi basis data. Kedua sisi sudah terpisah oleh sumber data dan jenis pemilik, jadi rasio pemilik tidak memiliki objek yang perlu dibagi.

**Penjelasan lebih dalam:** Memaksakan rasio pada A akan mengubah pertanyaan eksperimennya dan membuang sebagian data tanpa alasan. Karena itu konfigurasi menyimpan `owner_ratio="not_applicable"`, bukan 60:40 sampai 90:10. Implementasinya ada pada `experiments/scenarios.py:270-310`, sedangkan aturan matriksnya ada pada `experiments/config.py:22-30`.

## Q25 - Apakah pembagian dilakukan secara stratified?

**Jawaban langsung:** Ya untuk pembagian development dan test, tetapi tidak untuk pembagian owner ratio. Development dan test distratifikasi per pasangan `(surah_id, ayah_id)`. Owner ratio membagi daftar identitas pemilik secara deterministik dengan seed 42, bukan melakukan stratifikasi per ayat.

**Penjelasan lebih dalam:** Pada split development/test, klip yang memiliki label ayat sama dikelompokkan. Indeks dalam tiap kelompok diacak, sekitar 70% masuk development dan sisanya test. Cara ini menjaga tiap ayat tetap terwakili sebanyak mungkin. Pada owner split, semua klip pemilik harus tetap bersama, sehingga unit pembagiannya adalah NIM untuk mahasiswa atau `reciter_id` untuk qari.

```python
# backend-skripsi/experiments/runner.py:82-94
rng = np.random.default_rng(seed)
grouped = manifest.iloc[covered_indices].groupby(
    ["surah_id", "ayah_id"], sort=False
)
for _, local_indices in grouped.indices.items():
    indices = covered_indices[np.asarray(local_indices, dtype=np.int64)]
    rng.shuffle(indices)
    count = max(1, int(len(indices) * dev_ratio))
    dev.extend(indices[:count])
    test.extend(indices[count:])
```

## Q26 - Apa itu AP nol dan bagaimana dapat terjadi?

**Jawaban langsung:** AP, atau Average Precision, dihitung untuk satu kueri dari presisi pada setiap peringkat yang berisi dokumen relevan. Dalam konvensi evaluasi yang menetapkan AP nol untuk kueri tanpa hit relevan, nilai nol terjadi bila tidak ada dokumen relevan dalam daftar yang dievaluasi. Pada full ranking, jika basis data memang memiliki sedikitnya satu dokumen relevan, AP akan lebih besar dari nol karena dokumen itu pasti mempunyai suatu peringkat.

**Penjelasan lebih dalam:** Dalam evaluasi utama penelitian ini, pencarian dilakukan terhadap seluruh basis data dan kueri tanpa pasangan relevan disaring sebelum split. Karena setiap kueri evaluasi memiliki sedikitnya satu dokumen dengan `(surah, ayat)` yang sama, AP nol akibat ketiadaan pasangan relevan tidak muncul pada hasil final. Contoh hipotetis: kueri Al-Ikhlas ayat 1 dibandingkan dengan basis data yang hanya memuat ayat 2 sampai 4. Secara konvensi kueri itu dapat diberi AP nol, tetapi implementasi penelitian ini justru melewatinya melalui `if n_rel == 0: continue`. `runner.py:48-58,386-387` lebih dahulu membentuk dan memakai mask cakupan agar kueri seperti itu tidak masuk split evaluasi.

## Q27 - Apakah evaluasi hanya memakai 30% dari seluruh data?

**Jawaban langsung:** Evaluasi final memakai test set, tetapi angka 30% berlaku pada kueri yang tercakup dalam setiap sel, bukan 30% dari gabungan seluruh korpus. Basis data tidak dipotong 30%; seluruh basis data sesuai skenario tetap dipakai untuk meranking setiap kueri test.

**Penjelasan lebih dalam:** Development dipakai untuk memilih layer, sedangkan test dipakai sekali untuk melaporkan MAP, MRR, Top-1, Top-5, dan Top-10. Proporsi aktual test adalah komplemen dari development aktual, jadi sekitar 30,9% sampai 42,1% karena pembulatan per kelompok ayat. Contohnya, Skenario A memiliki 17.839 kueri development dan 7.990 kueri test, sementara basis datanya tetap 17.127 referensi. Jadi bukan berarti hanya 30% seluruh data penelitian yang dievaluasi.

## Q28 - Apa perbedaan development set, test set, dan owner ratio?

**Jawaban langsung:** Ketiganya berbeda. Owner ratio menentukan siapa yang menjadi pemilik basis data dan siapa yang menjadi pemilik kueri. Setelah sisi kueri terbentuk dan cakupannya diperiksa, klip kueri itu dibagi lagi menjadi development dan test. Development memilih layer, sedangkan test menilai konfigurasi final.

**Penjelasan lebih dalam:** Urutannya adalah: bentuk skenario dan owner split, tentukan kueri yang memiliki pasangan relevan, bagi kueri menjadi development/test secara terstratifikasi, uji 13 titik pada development, lalu uji satu titik terpilih pada test. Owner ratio berlaku pada B, C, dan D. Split development/test berlaku pada semua 13 sel, termasuk A. Bukti alur terdapat pada `experiments/runner.py:369-408`.

## Q29 - Bagaimana pengacakan deterministik dilakukan?

**Jawaban langsung:** Program memakai `np.random.default_rng(42)`. Untuk owner split, generator menghasilkan permutasi indeks pemilik. Untuk development/test, generator yang sama mengacak indeks klip di dalam setiap kelompok ayat. Input dan seed yang sama menghasilkan pembagian yang sama.

**Penjelasan lebih dalam:** Deterministik bukan berarti tidak diacak, tetapi hasil acaknya dapat direproduksi. Daftar pemilik diurutkan sebelum dipermutasi, pembagian disimpan dalam `owner_splits.json`, dan mask development/test disimpan dalam `split.json`. Seed 42 juga tercatat pada konfigurasi, sehingga eksperimen dapat diaudit. Cuplikan kodenya ada pada Q23 dan Q25; konstanta kanoniknya adalah `SEED = 42` pada `experiments/config.py:44`.

## Q30 - Mengapa proporsi development dapat turun sampai 57,9%?

**Jawaban langsung:** Karena 70% dihitung dan dibulatkan ke bawah secara terpisah pada setiap kelompok ayat, bukan sekali dari total kueri. Pada kelompok kecil, kehilangan proporsinya besar. Misalnya, jika satu ayat hanya memiliki 2 klip, `int(2 × 0,7) = 1`, sehingga development hanya 50%. Jika ada 3 klip, hasilnya 2 atau 66,7%. Banyak kelompok kecil membuat proporsi gabungan turun.

**Penjelasan lebih dalam:** Kode memakai `count = max(1, int(len(indices) * 0.7))`. Aturan minimal satu melindungi representasi development, tetapi tidak memaksa test selalu ada pada kelompok berukuran satu. Komposisi kelompok berubah pada setiap owner ratio karena pemilik kueri berbeda dan jumlah klip tiap pemilik tidak seragam. Akumulasi efek pembulatan itulah yang menghasilkan rentang empiris 57,9% sampai 69,1%, bukan kesalahan seed atau kehilangan data.

**Bukti:** Rumus ada pada `experiments/runner.py:89-94`; rentang empiris dan alasannya dicatat pada `Skripsi.md:2054-2066`.

## Q31 - Mengapa development dan test dipilih 70% dan 30%?

**Jawaban langsung:** Rasio 70:30 dipilih sebagai kompromi: development perlu cukup besar untuk membandingkan 13 titik representasi secara stabil, sedangkan test harus tetap cukup besar untuk evaluasi final yang independen. Rasio ini adalah keputusan desain eksperimen, bukan hukum bahwa 70:30 selalu paling baik.

**Penjelasan lebih dalam:** Jika development terlalu kecil, pemilihan layer lebih peka terhadap variasi sampel. Jika test terlalu kecil, estimasi hasil final dan bootstrap menjadi kurang stabil. Pemisahan fungsi kedua himpunan lebih penting daripada angka persisnya: test tidak boleh ikut menentukan layer. Konfigurasi merekam `DEV_RATIO = 0.7` pada `experiments/config.py:51`, sementara pembulatan per ayat menghasilkan proporsi aktual yang sudah dilaporkan secara transparan.

## Q32 - Apa yang dimaksud artefak akhir pada hasil modeling?

**Jawaban langsung:** Artefak akhir adalah berkas keluaran komputasi yang disimpan dan divalidasi, bukan hanya angka yang tampil di tabel. Untuk embedding, artefaknya meliputi manifes dan `layer_00.npy` sampai `layer_12.npy`. Setiap berkas layer berisi matriks `N × 768` bertipe `float32`, dan baris ke-i harus sesuai dengan baris ke-i pada manifes.

**Penjelasan lebih dalam:** Setiap klip mula-mula menghasilkan 13 vektor berdimensi 768. Vektor dengan nomor titik yang sama dirakit menjadi satu matriks per layer. Artefak evaluasi kemudian mencakup `config.json`, `manifest.json`, `split.json`, `dev_sweep.csv`, `test.csv`, `metrics.json`, dan `audit.json`. Validasi memastikan jumlah layer, bentuk, tipe data, nilai finite, keselarasan model, dan checksum. Bukti format embedding terdapat pada `Skripsi.md:2082-2097`; kontrak artefak evaluasi ada pada `experiments/config.py:79-100`.

## Q33 - Mengapa pemilihan titik menggunakan MAP?

**Jawaban langsung:** MAP dipakai karena satu ayat dapat memiliki banyak dokumen relevan dari pembaca yang berbeda. MAP menilai kualitas urutan seluruh dokumen relevan, sehingga paling sesuai untuk memilih representasi bagi retrieval multi-relevan. MRR hanya menekankan dokumen relevan pertama, sedangkan Top-K hanya menilai ada atau tidaknya hit dalam batas K.

**Penjelasan lebih dalam:** Jika layer dipilih dengan MRR, layer dapat terlihat baik hanya karena satu hasil relevan muncul sangat awal walaupun hasil relevan lain tersebar buruk. Top-K juga mengabaikan urutan setelah batas K. MAP memberi sinyal pemilihan yang lebih lengkap untuk tujuan penelitian. Setelah layer dipilih pada development dengan MAP, test tetap melaporkan MAP, MRR, dan Top-1/5/10 agar aspek ranking lain terlihat. Cosine similarity hanya fungsi pemberi skor untuk menyusun ranking, bukan metrik evaluasi.

**Bukti kode:** `experiments/evaluator.py:298-315` memilih nilai terbesar dengan kriteria default `MAP`, lalu `run_cell_evaluation()` pada baris 379-389 hanya mengevaluasi layer terpilih pada test.

## Q34 - Mengapa layer terbaik bukan layer terakhir?

**Jawaban langsung:** Karena kedalaman Transformer mengubah jenis informasi yang direpresentasikan. Layer tengah dapat mempertahankan pola akustik dan fonetik yang lebih berguna untuk mencocokkan ayat, sedangkan layer akhir makin dipengaruhi tujuan pralatih model. Jadi layer terakhir tidak otomatis paling cocok untuk cosine retrieval.

**Penjelasan lebih dalam:** Itu adalah interpretasi berdasarkan hasil retrieval per layer, bukan klaim bahwa isi fonetik setiap layer diukur secara langsung dalam penelitian ini. Secara empiris, MAP development naik menuju layer tengah lalu turun pada layer akhir. Contohnya, Skenario A memilih Wav2Vec2 titik 7 dan Data2Vec titik 5; banyak sel lain memilih Wav2Vec2 titik 7 atau 8 dan Data2Vec titik 5 atau 6. Kesimpulan yang aman adalah bahwa titik optimum bersifat task-dependent dan ditentukan oleh performa retrieval development, bukan oleh posisi layer semata.

## Q35 - Bagaimana evaluasi tiap layer dilakukan?

**Jawaban langsung:** Program memuat stack embedding berbentuk `(13, N, 768)`. Untuk setiap indeks titik 0 sampai 12, embedding kueri dan basis data pada titik yang sama dinormalisasi, cosine similarity dipakai untuk memberi skor, hasil diurutkan, lalu MAP, MRR, dan Top-K dihitung pada development. Titik dengan MAP tertinggi dipilih dan hanya titik itu yang dipanggil pada test.

**Cuplikan pemanggilan data dan evaluasi:**

```python
# backend-skripsi/experiments/runner.py:118-123
arrays = [load_layer_embeddings(Path(split_dir), model, layer, manifest)
          for layer in selected]
stack = np.stack(arrays, axis=0).astype(np.float32, copy=False)
```

```python
# backend-skripsi/experiments/evaluator.py:290-315
for layer in range(n_layers):
    results[layer] = evaluate_layer(
        q[layer], d[layer], qkeys, dkeys, dev_indices, top_k=top_k
    )
return max(dev_results,
           key=lambda layer: dev_results[layer]["MAP"])
```

```python
# backend-skripsi/experiments/evaluator.py:167-185
for row, query_idx in enumerate(blk):
    rel = ((qkeys[query_idx]["s"] == dkeys["s"])
           & (qkeys[query_idx]["a"] == dkeys["a"]))
    n_rel = int(rel.sum())
    if n_rel == 0:
        continue
    scores = sims[row]
    order = np.argsort(-scores, kind="stable")
    ranked = rel[order]
    hits = np.flatnonzero(ranked)
    ap = float(np.mean((np.arange(len(hits)) + 1) / (hits + 1)))
```

**Penjelasan titik:** Titik 0 bukan keluaran Transformer pertama. Titik 0 adalah hasil proyeksi fitur sebelum blok Transformer pertama. Titik 1 sampai 12 masing-masing adalah keluaran blok Transformer 1 sampai 12. Karena itu ada 13 titik, tetapi hanya 12 blok Transformer.

## Q36 - Apa yang dimaksud keluaran blok Transformer?

**Jawaban langsung:** Keluaran blok Transformer adalah urutan representasi kontekstual setelah sinyal melewati satu blok Transformer. Untuk satu klip pada titik tertentu, bentuknya sekitar `(T, 768)`: `T` adalah jumlah langkah waktu dan 768 adalah dimensi fitur. Penelitian merata-ratakan dimensi waktu dengan mean pooling sehingga setiap titik menghasilkan satu vektor 768 dimensi.

**Penjelasan lebih dalam:** Hubungannya dengan penelitian adalah setiap keluaran blok menjadi kandidat representasi untuk similarity search. `output_hidden_states=True` meminta model mengembalikan semua titik, bukan hanya keluaran terakhir. Kode lalu melakukan `hidden.mean(dim=1)`. Titik 0 adalah proyeksi fitur sebelum blok pertama, sedangkan titik 1 sampai 12 adalah keluaran 12 blok. Penelitian mengukur performa retrieval dari vektor-vektor tersebut, tetapi tidak mengukur kandungan linguistik atau fonetik layer secara langsung.

```python
# backend-skripsi/backends/wav2vec2.py:65-72
outputs = self.model(**inputs, output_hidden_states=True)
# tuple 13 elemen; tiap elemen berbentuk (1, T, 768)
for idx, hidden in enumerate(outputs.hidden_states):
    pooled = hidden.mean(dim=1).squeeze().cpu().numpy()
    layer_embeddings[idx] = pooled
```

Implementasi Data2Vec yang setara terdapat pada `backend-skripsi/backends/data2vec.py:64-71`.

## Q37 - Bagaimana evaluasi owner ratio dan skenario dilakukan?

**Jawaban langsung:** Evaluasi membentuk 13 sel: A satu kali tanpa owner ratio, lalu B, C, dan D masing-masing pada 60:40, 70:30, 80:20, dan 90:10. Untuk setiap sel dan model, program membentuk keanggotaan kueri dan basis data, memeriksa kebocoran pemilik, membagi kueri menjadi development/test, memilih layer dengan MAP development, lalu menghitung MAP, MRR, Top-1, Top-5, dan Top-10 pada test.

**Penjelasan skenario:** A adalah mahasiswa ke seluruh Quran-MD. B adalah Quran-MD ke Quran-MD lintas qari. C adalah mahasiswa ke mahasiswa lintas NIM. D adalah mahasiswa kueri ke basis data gabungan mahasiswa lain dan seluruh Quran-MD. Pada D, audit juga memeriksa tidak ada overlap path audio. `owner_ratio` selalu berarti proporsi pemilik sisi basis data; sisanya berada pada sisi kueri.

```python
# backend-skripsi/experiments/runner.py:369-408
if cell.scenario != "A":
    corpus = "reference" if cell.scenario == "B" else "query"
    split = owner_splits[corpus][float(cell.owner_ratio)]
membership = build_scenario_membership(
    cell, qman, rman, split_artifact=split.to_dict() if split else None,
    seed=config.seed,
)
coverage = compute_coverage_mask(membership_qkeys, membership_dkeys)
dev, test = stratified_dev_test_split(
    membership.query_rows, config.dev_ratio, config.seed, coverage
)
result = run_cell_evaluation(
    qstack, dstack, qkeys, dkeys, dev, test,
    config.top_k, config.bootstrap_b, config.seed
)
```

**Contoh audit nyata:** Pada D-90:10, `cells/D_90-10/data2vec/audit.json` mencatat 6 pemilik kueri, 54 pemilik basis data mahasiswa, 1.833 klip kueri, dan 41.123 item basis data, terdiri dari 23.996 klip mahasiswa lain ditambah 17.127 Quran-MD. `owner_overlap_count` dan `path_overlap_count` keduanya nol. Hasil setiap sel tersimpan pada `dev_sweep.csv` dan `test.csv`; ringkasannya terdapat pada `results/scenario-matrix-60-students-v1/summary_test.csv`.

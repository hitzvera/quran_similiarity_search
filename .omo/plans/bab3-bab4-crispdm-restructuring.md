# BAB III / BAB IV CRISP-DM Restructuring Plan

> Status: **DRAFT PLAN — REVISED** (not yet executed). `Skripsi.md` is untouched.
> Last updated: 2026-08-05 (revision 2).
> Source of locked facts: `.omo/notes/definitive-corpus-state.md`.
>
> **Revision 2 corrections**: (1) Added Section 2 pre-edit blocker for missing image assets `media/image5.png`–`image8.png`; all four are broken references. (2) Rewrote Section 6 migration map to treat Gambar 3.1–3.4 as broken source references with inspect/reuse/regenerate/BLOCKED protocol. (3) Replaced checklist-only verification gates (Section 9) with executable QA table containing exact tool, scope/action, and expected result. (4) Marked 17,130 Quran-MD bacaan as "must be verified" — not locked. (5) Described 4.2.x subsections as conceptual destinations pending user numbering approval. (6) Preserved Tabel 3.1's 81-stage statistics as collection-stage observations separate from the 81→60 funnel — both concepts retained.

---

## 1. Scope and Non-Goals

### In scope

1. Separate **methodology** (what was planned/done) from **results** (what was observed) across BAB III and BAB IV, following CRISP-DM phase semantics.
2. Migrate every empirical observation currently inside BAB III (tables, figures, distribution findings, naming-convention findings, outlier findings) into BAB IV `Hasil Data Understanding`.
3. Replace every stale numeric claim in BAB IV that was derived from the obsolete 7,119-clip / 42-student / 396-pair / 20-surah / 7,118-query regime with either (a) a locked-fact value from the definitive corpus audit or (b) an explicit **BLOCKED pending Vast.ai rerun** marker.
4. Make the 81 (collection-stage) to 60 (validated-student) funnel explicit and safe.
5. Preserve CRISP-DM procedure descriptions in BAB III.
6. Preserve Markdown/Pandoc grid-table syntax and figure cross-references. Do not reflow tables.

### Non-goals (explicitly out of scope)

- Do NOT edit `Skripsi.md` in this plan document. This plan only prescribes edits.
- Do NOT invent causes for the 81 to 60 reduction.
- Do NOT fabricate final MAP / MRR / Top-K values, selected layers, bootstrap intervals, scenario sizes, or deployment conclusions.
- Do NOT introduce new citations.
- Do NOT commit to git.
- Do NOT silently renumber sections, figures, or tables. Any renumbering requires explicit user confirmation.
- Do NOT delete grid-table syntax (`+---+---+`) or Pandoc figure markup.

---

## 2. Pre-Edit Blocker: Missing Image Assets

**CRITICAL**: The following image files referenced in `Skripsi.md` do NOT exist in `laporan/Skripsi/media/`:

| Missing reference | Skripsi.md line | Referenced caption |
|---|---|---|
| `media/image5.png` | 1218 | Gambar 3.1 Ringkasan Dataset Quran-MD |
| `media/image6.png` | 1238 | Gambar 3.2 Karakteristik Audio |
| `media/image7.png` | 1281 | Gambar 3.3 Total Distribusi Video |
| `media/image8.png` | 1303 | Gambar 3.4 Box Plot Showing Outlier |

**Existing media files** in `laporan/Skripsi/media/`:
- `bab4_du_surah_coverage.png`
- `bab4_du_scenario_sizes.png`
- `bab4_du_duration_dist.png`
- `bab4_du_audit_status.png`
- `data_understanding_flow.{png,pdf}`

**Definitive EDA figures** in `backend-skripsi/eda_output/definitive_corpus/`:
- `fig01_distribusi_klip_per_mahasiswa.png`
- `fig02_distribusi_surah_per_mahasiswa.png`
- `fig03_klip_per_surah.png`
- `fig04_mahasiswa_per_surah.png`
- `fig05_scatter_klip_vs_surah.png`
- `fig06_provenans_klip_tervalidasi.png`
- `fig07_kelengkapan_surah.png`
- `fig08_ayat_per_surah_referensi.png`

### Image asset resolution protocol

For each broken Gambar 3.x reference, apply this protocol in order:

1. **Inspect candidate existing outputs** for semantic fit:
   - Compare the caption concept (e.g., "Ringkasan Dataset Quran-MD") against existing `bab4_du_*.png` and `fig0_*.png` files.
   - Use `look_at` to visually inspect candidate images and confirm they depict the claimed concept on the **validated 60-student corpus** (not the obsolete 81-student collection-stage data).

2. **Reuse only after visual/content verification**:
   - If a candidate image semantically matches AND is computed from the validated 60-student / 25,829-clip corpus, it may be reused.
   - Copy the verified image to `laporan/Skripsi/media/` with a descriptive filename (e.g., `bab4_du_dataset_summary.png`).
   - Update the `Skripsi.md` reference to point to the new filename.

3. **Regenerate from definitive audited data** if no suitable candidate exists:
   - Write a new plotting script that reads from `backend-skripsi/eda_output/definitive_corpus/` CSVs/JSONs.
   - Generate the figure on the validated corpus.
   - Save to `laporan/Skripsi/media/` with a descriptive filename.

4. **Mark the figure asset BLOCKED** if neither reuse nor regeneration is possible:
   - Insert `[BLOCKED: asset gambar tidak tersedia — menunggu regenerasi]` in place of the figure reference.
   - Migrate only the surrounding empirical prose and caption concept to BAB IV.
   - The figure itself remains absent until regenerated.

**Never reuse an image based only on filename.** Always verify visual content and data provenance.

**Do not infer that old collection-stage plots (computed on 81 students / 7,119 clips) represent the validated 60-student corpus.** Any plot derived from the obsolete regime must be regenerated.

---

## 3. Locked Corpus Facts (source of truth)

These values are fixed and may be used in any rewritten prose without rerun.

| Quantity | Value |
|---|---|
| Collection-stage students (raw upload roster) | 81 |
| Validated students (after audit + quality filtering) | 60 |
| Raw MP3 files in `backend-skripsi/data/segmented/` | 25,945 |
| Zero-byte files excluded | 116 |
| Valid audio clips | 25,829 |
| Surahs represented (Al-Fatihah + Juz Amma) | 38 |
| Valid (student, surah) recording pairs | 1,795 |

### Provenance breakdown of 25,829 clips

| Category | Clips | Source recordings |
|---|---|---|
| Ordinary audited-success clips | 24,872 | recordings with matching audit rows marked success |
| Fallback proportional-time clips | 710 | 91 recordings (time-proportional allocation) |
| Clips from recordings without matching audit rows | 247 | 19 recordings |
| **Total** | **25,829** | |

### 81 to 60 funnel (safe, non-inventive)

| Stage | Count | Notes |
|---|---|---|
| Collection-stage students (raw upload roster) | 81 | Initial tahfidz submission roster. |
| Validated students after audit + quality filtering | 60 | Students whose submissions survived audit and quality filtering. |
| Reduction (81 to 60) | 21 | **Cause not to be invented.** Only documented, verified causes may be stated. |

Do NOT equate 81 and 60. Do NOT attribute the reduction to specific reasons unless verified.

### Scoring vs. evaluation metrics (locked distinction)

- **Cosine similarity** = scoring function (similarity function), NOT an evaluation metric. Cosine distance (1 - sim) violates the triangle inequality (Schubert 2021, arXiv:2107.04071).
- **MAP, MRR, Top-K** = evaluation metrics for the retrieval task (Manning IIR 2008).

---

## 3. Editorial Rule: Methodology vs. Results

Apply this rule consistently when deciding where content belongs.

| Content type | Goes in BAB III (Metodologi) | Goes in BAB IV (Hasil dan Pembahasan) |
|---|---|---|
| CRISP-DM phase description (what the phase means, what sub-processes were planned) | YES | no |
| Algorithm / model / pipeline specification | YES | no |
| Parameter choices and their justification | YES | no |
| Observed counts, distributions, completeness stats, outlier findings | no | YES |
| Tables/figures that report empirical observations | no | YES |
| Naming-convention findings, missingness patterns, zero-byte exclusions | no | YES |
| Final metric values (MAP/MRR/Top-K), selected layers, bootstrap intervals | no | YES |
| Deployment conclusions based on observed metrics | no | YES |

Rule of thumb: BAB III = "what we planned to do and why." BAB IV = "what we actually observed."

---

## 4. Target Outline for BAB III (after restructuring)

BAB III keeps the CRISP-DM procedure narrative. Empirical observations move out.

### 3.1 Business Understanding (lines 1136-1192)
- **Keep as-is.** Pure methodology: problem translation, objectives, evaluation criteria, layer-wise rationale.
- No edits needed beyond typo-level cleanup if user authorizes.

### 3.2 Data Understanding (lines 1194-1314) -> METHODOLOGY ONLY
Target content after migration:

1. **Paragraph 1-2 (1196-1215):** Keep. Describes the Data Understanding phase intent and the Quran-MD source at the plan level.
2. **Gambar 3.1 (1217-1219):** MIGRATE to BAB IV. Replace with a single sentence noting that dataset structure visualization is presented in BAB IV.
3. **Paragraph on audio characteristics plan (1221-1235):** Keep the methodological intent (what will be examined: format, sampling rate, channels, distributions). Remove the sentence that presents Gambar 3.2 as if it already exists here.
4. **Gambar 3.2 (1237-1239):** MIGRATE to BAB IV. Replace with forward reference.
5. **Transition paragraph (1241-1243):** Keep.
6. **Paragraphs on 81 students, Tabel 3.1, Gambar 3.3, naming-convention findings, Gambar 3.4 (1245-1314):** ALL MIGRATE to BAB IV. Replace with a short methodological note that the query dataset was collected from students as part of tahfidz coursework and that detailed completeness/distribution findings are reported in BAB IV. Mention the 81-student collection roster fact here (methodology: "data were collected from 81 students") but do NOT present the 25/56 split, means, or distributions here.

### 3.3 Data Preparation (lines 1316-1372)
- **Keep as-is.** Pure methodology: selection, normalization, WhisperX forced alignment, proportional word-ratio segmentation.

### 3.4 Modeling (lines 1374-1435)
- **Keep as-is.** Pure methodology: frozen embedding, layer-wise extraction, mean pooling, cosine similarity as scoring function.

### 3.5 Evaluation (lines 1437-1527)
- **Keep as-is.** Pure methodology: MAP/MRR/Top-K definitions, layer-wise evaluation protocol, dev/test split rationale, three scenarios (A/B/C), bootstrap significance rule, qualitative analysis plan.

### 3.6 Deployment (lines 1529-1566)
- **Keep as-is.** Pure methodology: synthesis, recommendation, future work direction.

---

## 6. Passage-Level Migration Map (BAB III -> BAB IV)

Every empirical item currently in BAB III Data Understanding (lines 1194-1314) must move to BAB IV `Hasil Data Understanding`. **All four figure references (Gambar 3.1-3.4) are broken source references** — the underlying image files do not exist. Apply the image asset resolution protocol from Section 2 before finalizing any figure placement.

| BAB III location | Item | Conceptual destination in BAB IV (pending numbering confirmation) | Image asset status | Notes |
|---|---|---|---|---|
| 1217-1219 | Gambar 3.1 (Ringkasan Dataset Quran-MD, `media/image5.png`) | Conceptual: subsection on struktur dataset Quran-MD | **BROKEN** — `image5.png` missing. Apply Section 2 protocol. Candidate: inspect `fig08_ayat_per_surah_referensi.png` or regenerate. | Do not renumber until user confirms subsection structure. |
| 1237-1239 | Gambar 3.2 (Karakteristik Audio, `media/image6.png`) | Conceptual: subsection on karakteristik audio | **BROKEN** — `image6.png` missing. Apply Section 2 protocol. Candidate: inspect `bab4_du_duration_dist.png` or regenerate. | Do not renumber until user confirms. |
| 1255-1271 | Tabel 3.1 (Summary Statistic Dataset Query: 81/25/56/22.2/28/0/38) | Conceptual: subsection on kelengkapan pengumpulan | N/A (table, not image) | **Retain as collection-stage observation** if values are verified against the raw upload roster. **Also add** the 81→60 funnel table as a separate concept. Do NOT substitute one for the other. |
| 1273-1278 | Prose interpreting Tabel 3.1 ("25 mahasiswa (30.9%)...") | Same conceptual subsection | N/A | **BLOCKED**: 25/56 split is collection-stage only. May be retained as collection-stage observation if verified. **Also** add prose on the 60 validated students. Keep both concepts separate. |
| 1280-1287 | Gambar 3.3 (Total Distribusi Video, `media/image7.png`) | Conceptual: same subsection as kelengkapan | **BROKEN** — `image7.png` missing. Apply Section 2 protocol. Candidate: inspect `fig01_distribusi_klip_per_mahasiswa.png` or regenerate on validated 60-student set. | Old collection-stage plot must NOT be reused for validated corpus. |
| 1284-1287 | Prose on "15 surah paling sering missing" | Same conceptual subsection | N/A | **BLOCKED**: derived from 81-student regime. Must be recomputed on 60 validated students. Candidate data: `fig07_kelengkapan_surah.png` or `per_surah_coverage.csv`. |
| 1289-1300 | Naming-convention findings (NIM_NAMA, separators, prefixes, capitalization) | Conceptual: subsection on normalisasi penamaan berkas | N/A | Pure observation, can be kept as-is. No image dependency. |
| 1302-1304 | Gambar 3.4 (Box Plot Showing Outlier, `media/image8.png`) | Conceptual: subsection on distribusi dan outlier | **BROKEN** — `image8.png` missing. Apply Section 2 protocol. Candidate: inspect `fig01_distribusi_klip_per_mahasiswa.png` or regenerate on validated 60-student set. | Old collection-stage plot must NOT be reused for validated corpus. |
| 1306-1314 | Prose on outlier and heterogeneity | Same conceptual subsection | N/A | Keep, but tie to validated 60-student set. |

### Tables to place in BAB IV (both concepts preserved separately)

**Table A: Collection-stage summary statistic (Tabel 3.1 concept, retained if verified)**

```
+-----------------------------------------+-------+
| Metric                                  | Value |
+=========================================+=======+
| Total mahasiswa (collection-stage)      | 81    |
+-----------------------------------------+-------+
| Lengkap (38 surah)                      | 25    |
+-----------------------------------------+-------+
| Tidak Lengkap                           | 56    |
+-----------------------------------------+-------+
| Rata-rata video                         | 22.2  |
+-----------------------------------------+-------+
| Median video                            | 28.0  |
+-----------------------------------------+-------+
| Min Video                               | 0     |
+-----------------------------------------+-------+
| Max Video                               | 38    |
+-----------------------------------------+-------+
```

*Status: Retain only if values are verified against the raw upload roster. This table describes the 81-student collection stage, NOT the validated 60-student corpus.*

**Table B: 81→60 validation funnel (new, locked facts)**

```
+-----------------------------------------+-------+--------------------------------------+
| Stage                                   | Count | Notes                                |
+=========================================+=======+======================================+
| Collection-stage students (raw roster)  | 81    | Initial tahfidz submission roster.   |
+-----------------------------------------+-------+--------------------------------------+
| Validated students after audit + QC     | 60    | Survived audit and quality filtering.|
+-----------------------------------------+-------+--------------------------------------+
| Reduction                               | 21    | Cause not invented; only documented  |
|                                         |       | causes may be stated.                |
+-----------------------------------------+-------+--------------------------------------+
```

*Status: Locked facts from definitive corpus audit. Both Table A and Table B must appear — they describe different stages.*

---

## 7. BAB IV Destination Map and Stale-Claim Map

### 7.1 Conceptual BAB IV structure (pending user approval of subsection numbering)

**IMPORTANT**: The 4.2.x subsection numbers below are **conceptual destinations** only. Do not create or renumber subsections without explicit user approval. Final numbering must be confirmed by the user before any edits are applied.

- **4.1 Hasil Business Understanding** (lines 1572-1732): keep as-is.
- **4.2 Hasil Data Understanding** (lines 1734-1863): **HEAVY EDIT REQUIRED.**
  - [CONCEPTUAL] Karakteristik Himpunan Data Mahasiswa -> **REWRITE** with locked facts (60 students, 38 surahs, 1,795 pairs, 25,829 clips, provenance 24,872+710+247).
  - [CONCEPTUAL] Karakteristik Himpunan Data Quran-MD -> keep/verify. **Note**: The "17,130 bacaan" figure is NOT locked — it must be verified against the definitive corpus audit before use.
  - [CONCEPTUAL] Struktur Dataset Quran-MD (receives Gambar 3.1 concept — image asset BROKEN, apply Section 2 protocol).
  - [CONCEPTUAL] Karakteristik Audio (receives Gambar 3.2 concept — image asset BROKEN, apply Section 2 protocol).
  - [CONCEPTUAL] Kelengkapan Pengumpulan Mahasiswa (receives Tabel 3.1 concept as collection-stage observation + 81→60 funnel table as separate concept; receives Gambar 3.3 and 3.4 concepts — both BROKEN, apply Section 2 protocol).
  - [CONCEPTUAL] Normalisasi Penamaan Berkas (receives naming-convention findings).
  - [CONCEPTUAL] Distribusi dan Outlier (may merge with Kelengkapan subsection — pending user decision).
  - Cakupan Relevansi dan Ukuran Skenario -> **BLOCKED pending rerun.** Current numbers (7,118 / 5,139 / 2,452 / 17,127 / 11,988 / 4,666) are obsolete.
  - Kualitas Data -> **BLOCKED pending rerun.** Current 353/43 pair counts are obsolete.
- **4.3 Hasil Data Preparation** (lines 1865-2002): keep methodology; remove any empirical claims.
- **4.4 Hasil Modeling** (lines 2004-2078):
  - Line 2022: "7.119 Klip" -> **BLOCKED pending rerun.** Must be replaced with validated corpus size.
  - Data Cleaning subsection: 1-clip / 3-clip failure counts -> **BLOCKED pending rerun.**
- **4.5 Hasil Evaluation** (lines 2080-2169): **ALL BLOCKED pending rerun.**
  - Line 2086-2089: dev/test split sizes (4,886 / 2,232) -> **BLOCKED.**
  - Line 2093: selected layers (Wav2vec2 L7, Data2vec L5) -> **BLOCKED.**
  - Lines 2115-2122: performance table (MAP 0.0178 / 0.0188 etc.) -> **BLOCKED.**
  - Lines 2137-2142: bootstrap CI, per-query win counts -> **BLOCKED.**
  - Lines 2151-2165: layer analysis, Gambar 4.1 -> **BLOCKED.**
- **4.6 Deployment** (lines 2171-2201):
  - Lines 2194-2201: "MAP di bawah 0,02 dan Top-1 di bawah 8%" -> **BLOCKED.** Deployment conclusions must wait for rerun.

### 7.2 Stale-claim map (every 7,119-derived value)

| Location (line) | Stale claim | Status |
|---|---|---|
| 1784 | "42 mahasiswa" | WRONG. Replace with 60 validated students. |
| 1785 | "396 pasangan (mahasiswa, surah)" | WRONG. Replace with 1,795 pairs. |
| 1785 | "20 surah" | WRONG. Replace with 38 surahs. |
| 1829 | Skenario A: 7,118 query / 17,127 database | BLOCKED pending rerun. |
| 1831 | Skenario B: 5,139 / 11,988 | BLOCKED pending rerun. |
| 1833 | Skenario C: 2,452 / 4,666 / 99.31% | BLOCKED pending rerun. |
| 1843-1845 | "17 query tidak memiliki dokumen relevan" | BLOCKED pending rerun. |
| 1851-1857 | "353 lengkap / 43 tidak lengkap" | BLOCKED pending rerun. |
| 1860-1862 | "1 klip query / 3 klip database gagal" | BLOCKED pending rerun. |
| 2022 | "7.119 Klip" | BLOCKED pending rerun. |
| 2047-2048 | "1 klip (Al-Fil ayat 1) / 3 klip database" | BLOCKED pending rerun. |
| 2086-2087 | "dev set (4.886 query, 70%) / test set (2.232 query, 30%)" | BLOCKED pending rerun. |
| 2093 | "Wav2vec2 lapisan 7 dan Data2vec lapisan 5" | BLOCKED pending rerun. |
| 2103 | "17.130 vektor pada database referensi" | Verify against rerun. |
| 2115-2122 | Performance table (MAP/MRR/Top-K values) | BLOCKED pending rerun. |
| 2125-2133 | "2.232 query... tidak ada AP sempurna... tidak ada nol" | BLOCKED pending rerun. |
| 2137-2142 | Bootstrap CI "-0,00005, +0,0024", per-query wins 1,164 / 1,068 | BLOCKED pending rerun. |
| 2154-2158 | "Lapisan 7 / lapisan 5", "L12 MAP=0,0054 / L12=0,0062" | BLOCKED pending rerun. |
| 2194-2201 | "MAP di bawah 0,02 dan Top-1 di bawah 8%" | BLOCKED pending rerun. |

---

## 8. Pre-Rerun vs. Post-Rerun Editing Phases

### Phase A: Pre-rerun edits (safe to execute NOW)

These edits do not depend on Vast.ai rerun output.

1. **Resolve image assets** per Section 2 protocol for all four broken Gambar 3.x references.
2. **Migrate empirical prose from BAB III to BAB IV** per Section 6 migration map. Do NOT move image files — they are broken references. Migrate only prose/caption concepts.
3. **Place both tables in BAB IV**: Table A (collection-stage 81-student statistics, if verified) and Table B (81→60 funnel, locked facts). Keep as separate concepts.
4. **Replace stale 42/396/20 claims** in BAB IV with locked facts (60 students, 38 surahs, 1,795 pairs, 25,829 clips, provenance breakdown). Mark 17,130 as "must be verified."
5. **Strip methodology-violating prose** from BAB III 3.2 (replace migrated figures/tables with forward references to BAB IV).
6. **Fix scoring/metric distinction** wherever cosine similarity is mislabeled as a metric (e.g., line 757 "metrik utama" and line 233 list).
7. **Mark every BLOCKED cell** in BAB IV with an explicit inline comment or placeholder text such as `[BLOCKED: menunggu hasil rerun Vast.ai]` so no stale value is read as final.
8. **Renumbering freeze:** do NOT renumber sections, figures, or tables until user confirms. Use temporary labels like `Gambar 4.x (penomoran menunggu konfirmasi)` in the draft.

### Phase B: Post-rerun edits (execute ONLY after Vast.ai results are verified)

1. Fill every BLOCKED cell with the verified rerun value:
   - Scenario sizes (A/B/C query and database clip counts).
   - Cakupan relevansi per skenario.
   - Data quality counts (segmentasi lengkap/tidak lengkap, klip gagal).
   - Dev/test split sizes.
   - Selected layers per model.
   - MAP / MRR / Top-K tables.
   - Bootstrap confidence intervals and per-query win counts.
   - Layer-wise curve data and Gambar 4.x.
   - Deployment conclusions.
2. Remove all `[BLOCKED]` markers.
3. Finalize figure/table numbering.
4. Re-verify scoring/metric distinction in all new prose.

---

## 9. Executable QA Verification Gates

Each gate must pass before proceeding to the next phase. All checks use exact tools, scopes, and expected results.

### Gate G1 (pre-edit)

| # | Tool | Scope / Action | Expected Result |
|---|---|---|---|
| G1.1 | Read | `backend-skripsi/eda_output/definitive_corpus/corpus_stats.json` | File exists and contains locked facts (60 students, 25,829 clips, 38 surahs, 1,795 pairs). |
| G1.2 | Grep | `Skripsi.md` for regex `media/image[5-8]\.png` | 4 matches found (lines 1218, 1238, 1281, 1303) — confirms broken references exist before edit. |
| G1.3 | Bash | `Test-Path "laporan/Skripsi/media/image5.png"` (repeat for image6/7/8) | All return `False` — confirms assets missing. |
| G1.4 | Read | This plan file | User has reviewed and authorized Phase A edits. |

### Gate G2 (after Phase A edits)

| # | Tool | Scope / Action | Expected Result |
|---|---|---|---|
| G2.1 | Grep | `Skripsi.md` lines 1115-1568 (BAB III) for regex `Tabel 3\.1\|Gambar 3\.[1-4]` | 0 matches — all empirical artifacts migrated out of BAB III. |
| G2.2 | Grep | `Skripsi.md` lines 1734-1864 (BAB IV 4.2) for regex `\b42\b.*mahasiswa\|396.*pasangan\|20 surah` | 0 matches — stale 42/396/20 claims removed. |
| G2.3 | Grep | `Skripsi.md` lines 1734-1864 for regex `\b60\b.*mahasiswa\|1.795\|25.829\|38 surah` | ≥3 matches — locked facts present in BAB IV. |
| G2.4 | Grep | `Skripsi.md` for regex `7[.,]119\|7[.,]118\|4\.886\|2\.232` | All matches must be adjacent (within 2 lines) to `[BLOCKED` marker. |
| G2.5 | Grep | `Skripsi.md` for regex `MAP.*0,0178\|MAP.*0,0188\|MRR.*0,1146\|Top-1.*0,0730` | All matches must be adjacent to `[BLOCKED` marker. |
| G2.6 | Grep | `Skripsi.md` lines 1115-2202 for regex `cosine.*metrik\|metrik.*cosine` (case-insensitive) | 0 matches — cosine never called an evaluation metric. |
| G2.7 | Grep | `Skripsi.md` for regex `^\+[-=]+\+` (grid-table borders) | Count unchanged from pre-edit baseline — grid-table syntax preserved. |
| G2.8 | Grep | `Skripsi.md` for regex `media/image[5-8]\.png` | 0 matches — broken references removed or replaced with valid paths. |
| G2.9 | Bash | `Test-Path` on every `media/*.png` referenced in `Skripsi.md` | All return `True` — all image targets exist. |
| G2.10 | Read | `Skripsi.md` section/figure/table numbering | No silent renumbering performed (or user has explicitly approved new numbering). |
| G2.11 | Bash | `pandoc --version` | If exit code 0, proceed to G2.12. If not installed, skip G2.12. |
| G2.12 | Bash | `pandoc --from markdown --to html laporan/Skripsi/Skripsi.md --output NUL` | Exit code 0 — Pandoc parse succeeds (only if G2.11 passed). |

### Gate G3 (after Vast.ai rerun, before Phase B)

| # | Tool | Scope / Action | Expected Result |
|---|---|---|---|
| G3.1 | Read | `backend-skripsi/eda_output/` rerun outputs | New scenario sizes, selected layers, and metric tables present. |
| G3.2 | Read | Vast.ai logs / extraction notes | Rerun completion confirmed. |
| G3.3 | Manual | User review of rerun results | User has reviewed and authorized Phase B edits. |

### Gate G4 (after Phase B edits)

| # | Tool | Scope / Action | Expected Result |
|---|---|---|---|
| G4.1 | Grep | `Skripsi.md` for regex `\[BLOCKED` | 0 matches — all BLOCKED markers removed. |
| G4.2 | Grep | `Skripsi.md` lines 2080-2169 (BAB IV 4.5 Evaluation) for regex `MAP.*\d+,\d+\|MRR.*\d+,\d+\|Top-\d+.*\d+,\d+` | ≥3 matches — metric tables populated. |
| G4.3 | Grep | `Skripsi.md` lines 2171-2202 (BAB IV 4.6 Deployment) for regex `MAP di bawah\|Top-1 di bawah` | 0 matches — deployment conclusions no longer reference obsolete metrics. |
| G4.4 | Read | `Skripsi.md` section/figure/table numbering | Final numbering confirmed by user. |
| G4.5 | Bash | `pandoc --version` | If exit code 0, proceed to G4.6. If not installed, skip G4.6. |
| G4.6 | Bash | `pandoc --from markdown --to html laporan/Skripsi/Skripsi.md --output NUL` | Exit code 0 — full-document Pandoc build succeeds (only if G4.5 passed). |

---

## 10. Exact Execution Order

Execute in this order. Do not skip steps.

1. **User reviews this plan.** (Current step.)
2. **User authorizes Phase A edits.**
3. **Resolve image assets** per Section 2 protocol for Gambar 3.1-3.4 concepts. For each: inspect candidates, reuse if verified, regenerate if needed, or mark BLOCKED.
4. **Migrate BAB III empirical prose to BAB IV** per Section 6 migration map. Do NOT move image files — they are broken references. Migrate only the prose/caption concepts.
5. **Place both tables in BAB IV**: Table A (collection-stage 81-student statistics, if verified) and Table B (81→60 funnel, locked facts). Keep as separate concepts.
6. **Rewrite BAB III 3.2** to contain only methodology, with forward references to BAB IV.
7. **Rewrite BAB IV 4.2** opening with locked facts (60 / 38 / 1,795 / 25,829 / provenance). Mark 17,130 as "must be verified."
8. **Mark every BLOCKED cell** in BAB IV with `[BLOCKED: menunggu hasil rerun Vast.ai]`.
9. **Fix scoring/metric distinction** across edited sections.
10. **Run Gate G2 checks** (Section 9 table).
11. **Wait for Vast.ai rerun to complete.** (Do not proceed until results are verified.)
12. **User reviews rerun results.**
13. **Execute Phase B edits**: fill BLOCKED cells, finalize numbering (with user approval), rewrite deployment conclusions.
14. **Run Gate G4 checks** (Section 9 table).
15. **Final Pandoc build verification** (only if Pandoc available).

---

## 11. Constraints and Reminders

- `Skripsi.md` was NOT edited in the creation of this plan.
- No corpus files were altered.
- No git commit was made.
- No papers are cited beyond what is already in the thesis or in `definitive-corpus-state.md`.
- Section/figure/table renumbering requires explicit user confirmation. Do not do it silently.
- The Vast.ai rerun is NOT complete. Do not claim completion until results are verified.
- The 81-to-60 reduction cause must not be invented.

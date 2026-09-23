# Definitive Corpus State

> Source-of-truth note for the Qur'an recitation corpus used in this thesis.
> Last updated: 2026-08-05.
> This note records validated facts only. Do not invent exclusion causes or cite unverified papers here.

---

## 1. Source Directory

- **Definitive source**: `backend-skripsi/data/segmented/`
- `segmented_1/` is **obsolete**. Do not use it for any computation or reporting.

---

## 2. Corpus Counts (Locked)

| Quantity | Value |
|---|---|
| Raw MP3 files in `segmented/` | 25,945 |
| Zero-byte files excluded | 116 |
| Valid audio clips | 25,829 |
| Validated students | 60 |
| Surahs represented | 38 |
| Valid (student, surah) recording pairs | 1,795 |

---

## 3. Provenance Breakdown (25,829 clips)

| Category | Clips | Source recordings |
|---|---|---|
| Ordinary audited-success clips | 24,872 | recordings with matching audit rows marked success |
| Fallback proportional-time clips | 710 | 91 recordings (time-proportional allocation) |
| Clips from recordings without matching audit rows | 247 | 19 recordings |
| **Total** | **25,829** | |

---

## 4. Student Count Clarification

- **81** refers to the number of students at the **collection stage** (raw upload roster).
- **60** is the count of **validated students** after audit and quality filtering.
- These two numbers must **never** be equated or conflated.
- The specific reasons for the 81-to-60 reduction are **not to be invented**. Only documented, verified causes may be stated.

---

## 5. Obsolete Experimental Artifacts

- Existing query embeddings and A/B/C experiment results were computed over **7,119 clips**.
- These results are **obsolete** for the final BAB IV (Results chapter) metrics.
- Final retrieval metrics must be recomputed over the full validated corpus (25,829 clips / 1,795 recordings).

---

## 6. Scoring vs. Evaluation Metrics

- **Cosine similarity** is a **scoring function** (similarity function), not an evaluation metric.
  - Cosine distance (1 - sim) is not a true mathematical metric; it violates the triangle inequality (Schubert 2021, arXiv:2107.04071).
- **MAP (Mean Average Precision)**, **MRR (Mean Reciprocal Rank)**, and **Top-K** are the **evaluation metrics** for the retrieval task.
- This distinction must be maintained in all thesis text and code comments.

---

## 7. Key File Paths

| Purpose | Path |
|---|---|
| Definitive segmented audio | `backend-skripsi/data/segmented/` |
| Upload manifest | `backend-skripsi/data/upload/manifest.json` |
| Segmentation reconciliation | `backend-skripsi/data/validation/segmentation_reconciliation.json` |
| Validated query manifest (CSV) | `backend-skripsi/data/validation/query_manifest_validated.csv` |
| Validated query manifest (JSON) | `backend-skripsi/data/validation/query_manifest_validated.json` |
| Segmentation audit CSV | `backend-skripsi/data/segmentation_audit.csv` |
| Definitive corpus audit script | `backend-skripsi/scripts/audit_definitive_corpus.py` |
| Query set audit script | `backend-skripsi/audit_query_set.py` |
| Query audit summary output | `backend-skripsi/eda_output/query_audit_summary.txt` |
| Definitive generated statistics, tables, validation checks, and figures | `backend-skripsi/eda_output/definitive_corpus/` |
| Vast.ai bundle archive | `backend-skripsi/vast_bundle.tar.gz` |
| Vast.ai run script | `backend-skripsi/vast_bundle/run_vast.sh` |
| Vast.ai extraction notes | `backend-skripsi/extraction/VASTAI_EXTRACTION.md` |

---

## 8. Pending Work

- [ ] Rerun full retrieval evaluation (Wav2Vec2 and Data2vec) over the validated 25,829-clip corpus on Vast.ai.
- [ ] Compute final BAB IV metrics (MAP, MRR, Top-K) from the rerun results.
- [ ] The Vast.ai rerun is **not yet complete**. Do not claim completion until results are verified.

---

## 9. Constraints on This Note

- `Skripsi.md` was **not edited** in the creation of this note.
- No corpus files were altered.
- No git commit was made.
- No papers are cited beyond what is needed for the scoring/metric distinction.

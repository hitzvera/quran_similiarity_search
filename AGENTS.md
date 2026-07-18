# AGENTS.md

Instruction file for AI coding agents working in this repository.
Read this fully before acting. It encodes hard-won facts that are expensive to rediscover.

## What this repo IS

This is a **thesis-writing repository (documents), NOT a code project.**
There is no build, no test suite, no application to run. The deliverable is a
written undergraduate thesis (skripsi S1 Teknik Informatika).

- Student: Mujahid Ansori Majid (NIM 1197050093), UIN Sunan Gunung Djati Bandung, 2026.
- Topic: Comparing **Wav2Vec2 vs Data2vec** as *frozen* speech embeddings for
  **retrieval / similarity search of Qur'an verses** — transcription-free, no
  fine-tuning. Data: Al-Fatihah + Juz Amma. Scoring by cosine similarity;
  evaluation by MAP / Top-K / MRR.

## Primary file — handle with care

`laporan/Skripsi/Skripsi.md`

- Format: **Markdown/Pandoc**, converted from Word. Tables are **grid tables**
  (`+---+---+`). LaTeX math `$...$` is used and rendered via Pandoc. **Preserve
  both.** Do not reflow or "clean up" tables.
- **Do NOT modify `Skripsi.md` unless the user explicitly asks.** Default output
  mode is ready-to-paste text in chat, not file edits.
- Section numbering is currently inconsistent (e.g. `7.1/7.2/7.3` where `2.x`
  is expected). Do not silently renumber; confirm with user first.

Companion: `laporan/Skripsi/SKILL.md` (title page + DAFTAR ISI),
`laporan/Skripsi/Kerangka Pemikiran.png` (framework diagram).

## Hard rules

1. **Never `Read` image files directly** — the model cannot read them as text.
   Use `look_at` (or a multimodal/librarian tool) to extract content from
   `Kerangka Pemikiran.png` and any other image/PDF.
2. **Cosine similarity is a SCORING/similarity FUNCTION, not an evaluation
   metric.** (Manning IIR 2008 §6.3 = scoring; §8 = MAP/MRR/Top-K = evaluation
   metrics.) Cosine *distance* (1−sim) is not a true mathematical metric — it
   violates the triangle inequality (Schubert 2021, arXiv:2107.04071).
   `Skripsi.md` currently mislabels cosine as "metrik utama" (~line 757) and
   lists it alongside MAP/Top-K as a metric (~line 233). Keep this distinction
   correct in any new text.
3. **Verify every citation before using it.** This project has a history of
   hallucinated / unverifiable papers. Do not cite a paper unless it is
   confirmed against a real source (PDF in `ref/`, DOI, arXiv ID).
4. **H1 vs H2 hypotheses are the novelty shield.** They must appear in **two
   places with identical wording**: (a) end of Latar Belakang (motivation),
   (b) sub-section 2.11 of Tinjauan Pustaka (synthesis).
   - H1: Data2vec wins (full contextual-latent target → richer phonetic features).
   - H2: Wav2Vec2 wins (contrastive loss → more clusterable embeddings for
     cosine retrieval).
5. Research gap = direction **A+B**: (A) acknowledge Aswat already compared
   wav2vec vs data2vec for Arabic *ASR*, then show the gap; (B) contrastive
   hypothesis — test whether the ASR advantage transfers to *retrieval*.

## Known bugs to fix in the SotA table (when editing is authorized)

- [14] Aswat year `(2021)` → **2023** (Alkanhal et al., ArabicNLP/ACL 2023,
  DOI 10.18653/v1/2023.arabicnlp-1.10).
- [7] Xie year `(2022)` → **2021** (Interspeech 2021,
  DOI 10.21437/Interspeech.2021-847).
- Row [a] wrong authors "R Prajana / Kavitha S N" → should be **Baevski et al. 2020**.
- Row [f] "Tujuan" column was copy-pasted from row [e]; needs its own content.

## Repository layout

- `laporan/` — the thesis itself (main work is here).
- `ref/` — **43 reference PDFs** (Aswat, Data2vec/Wav2Vec2 SOTA, HuBERT,
  Large-Scale Eval, Query-by-Example, etc.). Read PDFs via `look_at`/multimodal,
  not `Read`. This is the source of truth for citations.
- `refMan/`, `revisi/`, `templates/`, `guide/`, `daily-logs/`,
  `kebutuhan-sidang/` — supporting material.
- `.omo/` — agent plans/drafts (e.g. `.omo/plans/abstract-plan.md`).
- `glosarium-2.txt` — Indonesian glossary of technical terms (empiris, latent
  representation, vector embedding, fonetik, testbed, artikulatoris).

## Git

- Branch: `main`. Remotes include `dev`, `draft_proposal-v.1`, `v.0.1-init`.
- Never commit unless explicitly requested.

## Verified paper facts (safe to cite)

- **Aswat** — Alkanhal et al., ArabicNLP/ACL 2023. data2vec > wav2vec for Arabic
  ASR; WER on MGB-2 = 10.3%.
- **Baevski et al. 2020** (wav2vec 2.0) and **Baevski et al. 2022** (data2vec).
- **Nay San 2021** (arXiv:2103.14583); **Hu/Settle/Livescu 2021**
  (arXiv:2011.11807, AP 0.57 > DTW 0.49); **Zhaoqi Li 2021** (ISCSLP);
  **Xie et al. 2021** (Siamese + wav2vec spoofing, ASVspoof2019 LA,
  EER 4.07%→1.15%).
- **Yoon MCR-Data2vec 2.0** (Interspeech 2023, arXiv:2306.08463);
  **Yang, Large-Scale Eval** (TASLP 2024, arXiv:2404.09385 — Data2vec best for
  VC/PhoneRec, weaker for SID/OOD-ASR); **Sinha 2025** (Interspeech — optimal
  layers Wav2Vec2=22 / Data2vec=22 / HuBERT=24); **Pasad 2021/2023**
  (layer-wise; note: does NOT cover Data2vec); **Schubert 2021**
  (arXiv:2107.04071, cosine-distance triangle-inequality); **Manning IIR 2008**.
- **Tang et al. NCMMSC 2022** (arXiv:2212.10092) — HuBERT vs Data2vec fusion is
  task-dependent; **do NOT quote its WER numbers**.

## Do-NOT-cite (unverified / likely hallucinated)

Until independently confirmed against a real source, do not cite:
CBVeRse, Riwaya-ID, Moustafa & Aly (arXiv:2111.06331), a supposed "Quranic ASR"
arXiv:2606.19747 (hallucinated ID), and any Qur'an-specific paper claimed but
not yet shown to/confirmed by the user. Note: `ref/` does contain
`Gemini Embedding 2` and `WavRag` PDFs, but their relevance to the framework was
rejected — do not reintroduce them into the Kerangka Pemikiran without asking.

## Working style expected here

- The user drives; do not start editing `Skripsi.md` on your own initiative.
- Prefer ready-to-paste Indonesian academic prose over file edits.
- Match the document's existing register (formal Indonesian, IEEE-style citations).

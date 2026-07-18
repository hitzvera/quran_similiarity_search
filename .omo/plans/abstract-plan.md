# Abstract Writing Plan: Quran Audio Similarity Search Research

## TL;DR

> **Quick Summary**: Create a 150-250 word abstract (both English and Bahasa Indonesia) for skripsi research comparing Wav2Vec 2.0 and Data2Vec embedding models for Quran recitation verse/passage matching using cosine similarity search.
>
> **Deliverables**:
> - Abstract structure outline with section-by-section guidance
> - English abstract template with placeholders
> - Bahasa Indonesia abstract template with placeholders
> - Writing tips and best practices for academic abstracts
>
> **Estimated Effort**: Quick (writing task, no code implementation)
> **Parallel Execution**: NO (sequential writing process)
> **Critical Path**: Research summary → Structure outline → English draft → Indonesian translation

---

## Context

### Original Request
User is conducting research on audio similarity search for Quran recitation, specifically:
- Converting audio to embeddings using speech models
- Finding similar verses/passages using cosine similarity
- Comparing Wav2Vec 2.0 vs Data2Vec (multimodal) performance
- Evaluating with multiple metrics (Top-K accuracy, similarity scores, precision/recall/F1, MAP, speed)
- Needs abstract draft for skripsi (undergraduate thesis) in both languages

### Interview Summary
**Key Discussions**:
- **Research Focus**: Model comparison between Wav2Vec 2.0 and Data2Vec
- **Application**: Verse/passage matching from Quran recitation audio
- **Dataset**: Quran recitation dataset (specific source TBD)
- **Status**: Experiments in progress, needs abstract draft
- **Format**: Standard skripsi abstract (150-250 words), bilingual (EN/ID)

**Research Components**:
- Input: Audio query (Quran recitation)
- Process: Audio → Embedding → Cosine similarity search
- Models: Wav2Vec 2.0 (speech-specific) vs Data2Vec (multimodal)
- Output: Ranked list of similar verses/passages
- Evaluation: Top-K accuracy, similarity scores, precision/recall/F1, MAP, speed

---

## Work Objectives

### Core Objective
Provide a clear, structured abstract template that the user can complete with their actual experimental results, following academic standards for skripsi abstracts.

### Concrete Deliverables
1. Abstract structure outline with 5 sections (Background, Objective, Method, Results, Conclusion)
2. English abstract template (150-250 words) with [PLACEHOLDER] markers
3. Bahasa Indonesia abstract template (150-250 words) with [PLACEHOLDER] markers
4. Section-by-section writing guide
5. Common mistakes to avoid

### Definition of Done
- [ ] User can fill in placeholders with their actual data
- [ ] Abstract follows standard academic structure
- [ ] Both language versions are consistent
- [ ] Word count is within 150-250 words per version

### Must Have
- Clear problem statement (why verse matching matters)
- Specific methodology (models, dataset, similarity method)
- Quantitative results placeholders (with units/metrics)
- Clear conclusion (which model performed better, by how much)
- Keywords (5-7 terms for indexing)

### Must NOT Have (Guardrails)
- Vague statements without specifics ("good results", "performed well")
- Technical jargon without explanation (assume general CS audience)
- Citations or references (abstracts don't include these)
- Figures, tables, or equations
- Excessive background (keep to 1-2 sentences)
- Future work mentions (save for conclusion chapter)

---

## Abstract Structure Outline

### Standard 5-Section Structure (150-250 words total)

#### 1. Background/Context (1-2 sentences, ~30 words)
**Purpose**: Establish the research problem and its importance

**What to include**:
- The domain: Quran recitation audio processing
- The problem: Need for efficient verse/passage matching
- Why it matters: Applications in education, recitation analysis, etc.

**Example starter**:
> "Quran recitation audio similarity search enables automated verse matching for educational and analytical applications."

**Tips**:
- Start broad, then narrow to your specific problem
- Avoid generic statements like "In today's digital age..."
- Focus on the technical problem, not religious context

---

#### 2. Objective (1 sentence, ~20 words)
**Purpose**: State what this research aims to achieve

**What to include**:
- The goal: Compare embedding models for audio similarity
- The specific task: Verse/passage matching
- The approach: Cosine similarity on embeddings

**Example**:
> "This research compares Wav2Vec 2.0 and Data2Vec embedding models for Quran verse matching using cosine similarity search."

**Tips**:
- Use active voice ("This research compares..." not "A comparison is made...")
- Be specific about what you're comparing
- Mention the end task (verse matching)

---

#### 3. Methodology (2-3 sentences, ~60-80 words)
**Purpose**: Explain HOW you did the research

**What to include**:
- Dataset: Quran recitation dataset (size, source, preprocessing)
- Models: Wav2Vec 2.0 and Data2Vec (brief description)
- Process: Audio → Embedding → Cosine similarity → Ranked results
- Evaluation: Metrics used (Top-K accuracy, precision/recall/F1, MAP, speed)

**Example**:
> "We processed [X] Quran recitation audio samples from [dataset source], extracting embeddings using Wav2Vec 2.0 and Data2Vec models. Cosine similarity was computed between query audio and database embeddings to retrieve top-K matching verses. Performance was evaluated using Top-1/3/5 accuracy, precision, recall, F1-score, Mean Average Precision (MAP), and processing time."

**Tips**:
- Include dataset size (number of samples/hours)
- Mention preprocessing steps if critical (sampling rate, normalization)
- List all evaluation metrics
- Keep it concise - save details for methodology chapter

---

#### 4. Results (2-3 sentences, ~60-80 words)
**Purpose**: Present key findings with specific numbers

**What to include**:
- Best performing model (with specific metrics)
- Comparison between models (difference in performance)
- Notable trade-offs (accuracy vs speed, etc.)
- Statistical significance (if applicable)

**Example**:
> "Data2Vec achieved [X]% Top-1 accuracy, outperforming Wav2Vec 2.0 by [Y]% (p<0.05). Data2Vec also showed higher MAP scores ([A] vs [B]) and better F1 ([C] vs [D]). However, Wav2Vec 2.0 processed queries [Z]x faster ([time1] vs [time2] seconds per query)."

**Tips**:
- Use specific numbers, not vague terms
- Include units (% accuracy, seconds, etc.)
- Highlight the most important 2-3 metrics
- Mention trade-offs if they exist
- Use comparative language ("outperformed", "improved by", "reduced")

---

#### 5. Conclusion (1-2 sentences, ~30 words)
**Purpose**: State the main takeaway and implications

**What to include**:
- Which model is recommended (and why)
- The implication (what this means for the field)
- Optional: Limitation or future direction (keep brief)

**Example**:
> "Data2Vec is recommended for high-accuracy Quran verse matching, while Wav2Vec 2.0 suits real-time applications requiring faster processing. These findings guide model selection for audio-based Quranic applications."

**Tips**:
- Don't introduce new information
- Connect back to the objective
- Keep it forward-looking but grounded in your results
- Avoid overclaiming ("solves the problem" → "contributes to")

---

#### 6. Keywords (5-7 terms)
**Purpose**: Help indexing and searchability

**Format**: Comma-separated, lowercase (except proper nouns)

**Suggested keywords for your research**:
- Quran recitation
- Audio similarity search
- Speech embeddings
- Wav2Vec 2.0
- Data2Vec
- Cosine similarity
- Verse matching

---

## English Abstract Template

```
[BACKGROUND - 1-2 sentences]
Quran recitation audio similarity search enables automated verse matching for educational and analytical applications. [Optional: Add specific problem or gap your research addresses]

[OBJECTIVE - 1 sentence]
This research compares Wav2Vec 2.0 and Data2Vec embedding models for Quran verse matching using cosine similarity search.

[METHODOLOGY - 2-3 sentences]
We processed [NUMBER] Quran recitation audio samples from [DATASET SOURCE], extracting embeddings using Wav2Vec 2.0 and Data2Vec models. Cosine similarity was computed between query audio and database embeddings to retrieve top-K matching verses. Performance was evaluated using Top-1/3/5 accuracy, precision, recall, F1-score, Mean Average Precision (MAP), and processing time.

[RESULTS - 2-3 sentences]
Data2Vec achieved [X]% Top-1 accuracy, outperforming Wav2Vec 2.0 by [Y]% (p<0.05). Data2Vec also showed higher MAP scores ([A] vs [B]) and better F1 ([C] vs [D]). However, Wav2Vec 2.0 processed queries [Z]x faster ([TIME1] vs [TIME2] seconds per query).

[CONCLUSION - 1-2 sentences]
Data2Vec is recommended for high-accuracy Quran verse matching, while Wav2Vec 2.0 suits real-time applications requiring faster processing. These findings guide model selection for audio-based Quranic applications.

Keywords: Quran recitation, audio similarity search, speech embeddings, Wav2Vec 2.0, Data2Vec, cosine similarity, verse matching
```

**Word count target**: 150-250 words (excluding keywords)

---

## Bahasa Indonesia Abstract Template (Abstrak)

```
[LATAR BELAKANG - 1-2 kalimat]
Pencarian kemiripan audio tilawah Al-Quran memungkinkan pencocokan ayat otomatis untuk aplikasi pendidikan dan analitis. [Opsional: Tambahkan masalah spesifik atau celah yang diteliti]

[TUJUAN - 1 kalimat]
Penelitian ini membandingkan model embedding Wav2Vec 2.0 dan Data2Vec untuk pencocokan ayat Al-Quran menggunakan pencarian kemiripan cosine.

[METODOLOGI - 2-3 kalimat]
Kami memproses [JUMLAH] sampel audio tilawah Al-Quran dari [SUMBER DATASET], mengekstrak embedding menggunakan model Wav2Vec 2.0 dan Data2Vec. Kemiripan cosine dihitung antara audio query dan embedding database untuk mengambil top-K ayat yang cocok. Kinerja dievaluasi menggunakan akurasi Top-1/3/5, presisi, recall, F1-score, Mean Average Precision (MAP), dan waktu pemrosesan.

[HASIL - 2-3 kalimat]
Data2Vec mencapai akurasi Top-1 [X]%, mengungguli Wav2Vec 2.0 sebesar [Y]% (p<0.05). Data2Vec juga menunjukkan skor MAP yang lebih tinggi ([A] vs [B]) dan F1 yang lebih baik ([C] vs [D]). Namun, Wav2Vec 2.0 memproses query [Z]x lebih cepat ([WAKTU1] vs [WAKTU2] detik per query).

[KESIMPULAN - 1-2 kalimat]
Data2Vec direkomendasikan untuk pencocokan ayat Al-Quran dengan akurasi tinggi, sementara Wav2Vec 2.0 cocok untuk aplikasi real-time yang membutuhkan pemrosesan lebih cepat. Temuan ini memandu pemilihan model untuk aplikasi Al-Quran berbasis audio.

Kata kunci: tilawah Al-Quran, pencarian kemiripan audio, embedding ucapan, Wav2Vec 2.0, Data2Vec, kemiripan cosine, pencocokan ayat
```

**Target jumlah kata**: 150-250 kata (tidak termasuk kata kunci)

---

## TODOs

- [ ] 1. Fill in English Abstract Template

  **What to do**:
  - Replace all [PLACEHOLDER] markers with your actual data
  - Ensure word count is 150-250 words
  - Check that all 5 sections are present and balanced
  - Verify specific numbers are included (not vague terms)

  **Must NOT do**:
  - Add citations or references
  - Include figures or tables
  - Use vague language ("good results", "performed well")
  - Exceed 250 words

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Reason**: Academic writing task requiring clarity and precision

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Step 1)
  - **Blocks**: Step 2 (Indonesian translation)
  - **Blocked By**: None (can start immediately)

  **References**:
  - Use the English template above
  - Fill in with your experimental results
  - Ensure consistency between sections

  **Acceptance Criteria**:
  - [ ] All placeholders replaced with actual data
  - [ ] Word count: 150-250 words
  - [ ] All 5 sections present (Background, Objective, Method, Results, Conclusion)
  - [ ] Specific numbers included (accuracy %, time, etc.)
  - [ ] Keywords listed (5-7 terms)

  **Evidence to Capture**:
  - [ ] Final English abstract text
  - [ ] Word count verification

  **Commit**: NO (this is a writing task, not code)

---

- [ ] 2. Fill in Bahasa Indonesia Abstract Template

  **What to do**:
  - Translate your English abstract to Bahasa Indonesia
  - Use the Indonesian template provided
  - Ensure technical terms are correctly translated
  - Maintain 150-250 word count
  - Verify consistency with English version

  **Must NOT do**:
  - Change the meaning or add new information
  - Use informal language
  - Exceed 250 words

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Reason**: Translation and academic writing in Bahasa Indonesia

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Step 2)
  - **Blocks**: Step 3 (Final review)
  - **Blocked By**: Step 1 (English version must be complete)

  **References**:
  - Use the Indonesian template above
  - Reference your completed English abstract
  - Technical terms: embedding → embedding, cosine similarity → kemiripan cosine

  **Acceptance Criteria**:
  - [ ] All placeholders replaced with actual data
  - [ ] Word count: 150-250 words
  - [ ] Consistent with English version (same meaning, structure)
  - [ ] Proper Bahasa Indonesia academic language
  - [ ] Keywords translated appropriately

  **Evidence to Capture**:
  - [ ] Final Indonesian abstract text
  - [ ] Word count verification

  **Commit**: NO (this is a writing task, not code)

---

- [ ] 3. Final Review and Integration

  **What to do**:
  - Review both abstracts side-by-side
  - Check for consistency in numbers and claims
  - Verify word counts are within limits
  - Ensure keywords match between languages
  - Get feedback from advisor if needed

  **Must NOT do**:
  - Make major changes at this stage
  - Add new information not in the templates
  - Deviate from the 5-section structure

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Reason**: Review and quality assurance

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Step 3)
  - **Blocks**: None (final step)
  - **Blocked By**: Steps 1 and 2

  **References**:
  - Both completed abstracts
  - Original templates for structure verification

  **Acceptance Criteria**:
  - [ ] Both abstracts are consistent
  - [ ] Word counts are 150-250 each
  - [ ] All sections are balanced and complete
  - [ ] Keywords match between languages
  - [ ] Ready for submission to skripsi document

  **Evidence to Capture**:
  - [ ] Final abstracts in both languages
  - [ ] Word count verification for both
  - [ ] Consistency check notes

  **Commit**: NO (this is a writing task, not code)

---

## Writing Tips and Best Practices

### DO:
1. **Be specific**: Use exact numbers ("87.3% accuracy" not "high accuracy")
2. **Use active voice**: "We compared..." not "A comparison was made..."
3. **Focus on your contribution**: What did YOU do, not what others did
4. **Keep it self-contained**: Reader shouldn't need to read the full paper to understand
5. **Match tense correctly**:
   - Past tense for what you did: "We processed...", "Data2Vec achieved..."
   - Present tense for general facts: "Cosine similarity measures..."
   - Present tense for conclusions: "Data2Vec is recommended..."
6. **Use comparative language**: "outperformed", "improved by", "reduced", "faster than"
7. **Include trade-offs**: If one model is better in accuracy but slower, mention both

### DON'T:
1. **Don't cite references**: Abstracts don't include citations
2. **Don't use jargon without context**: Assume general CS audience
3. **Don't include figures/tables**: Save for the main document
4. **Don't exceed word limit**: 150-250 words is standard for skripsi
5. **Don't be vague**: Avoid "good results", "performed well", "significant improvement"
6. **Don't introduce new information**: Everything in abstract should be in the paper
7. **Don't use first person singular**: Use "We" (even if solo author, it's academic convention)
8. **Don't mention future work**: Save for conclusion chapter

### Common Mistakes to Avoid:

❌ **Too much background**:
> "The Quran is the holy book of Islam, revealed to Prophet Muhammad peace be upon him. It consists of 114 surahs and is recited in Arabic. Many people want to learn Quran recitation..."

✅ **Better**:
> "Quran recitation audio similarity search enables automated verse matching for educational applications."

---

❌ **Vague results**:
> "The results showed that Data2Vec performed well and was better than Wav2Vec 2.0."

✅ **Better**:
> "Data2Vec achieved 87.3% Top-1 accuracy, outperforming Wav2Vec 2.0 by 12.5% (p<0.05)."

---

❌ **Missing methodology details**:
> "We used some audio data and compared two models."

✅ **Better**:
> "We processed 1,200 Quran recitation audio samples from the Quran.com dataset, extracting embeddings using Wav2Vec 2.0 and Data2Vec models."

---

❌ **Overclaiming**:
> "This research solves the problem of Quran verse matching."

✅ **Better**:
> "This research contributes to Quran verse matching by comparing embedding models and identifying optimal approaches."

---

## Success Criteria

### Verification Checklist
- [ ] English abstract: 150-250 words
- [ ] Indonesian abstract: 150-250 words
- [ ] Both abstracts have 5 sections (Background, Objective, Method, Results, Conclusion)
- [ ] Specific numbers included (no vague terms)
- [ ] Keywords listed (5-7 terms) in both languages
- [ ] Consistent between languages (same meaning, structure, numbers)
- [ ] No citations, figures, or tables
- [ ] Academic language (formal, third person, past tense for methods)
- [ ] Ready to paste into Skripsi.docx

### Final Output Location
Once completed, paste both abstracts into:
- `laporan/Skripsi/Skripsi.docx` (Indonesian abstract first, then English)
- Or follow your university's format requirements

---

## Next Steps

1. **Complete your experiments** if not yet done
2. **Fill in the English template** with your actual results
3. **Translate to Indonesian** using the provided template
4. **Review both versions** for consistency
5. **Get advisor feedback** before finalizing
6. **Paste into Skripsi.docx**

Good luck with your skripsi! 🎓

Looking at your existing setup — frozen Wav2Vec2/Data2Vec embeddings evaluated with MAP and Top-K accuracy — let me check what's actually available for Arabic/Quranic audio before I suggest a direction, since picking the wrong starting checkpoint or the wrong fine-tuning objective would cost you a lot of GPU time for nothing.

Orchestrated practical implementation strategy for transfer learning setup
Orchestrated practical implementation strategy for transfer learning setup

The Arabic XLSR checkpoints and the speaker-recognition fine-tuning precedent both point to a clear answer. Let me show you the core idea first, then walk through the practical setup.

Your current setup — frozen Wav2Vec2/Data2Vec embeddings scored with MAP and Top-K — already is transfer learning. That's the "feature extraction" flavor: you take representations learned from unlabeled speech and use them as-is on a new task, zero additional training. If you need the term for your metodologi section, that's it, and it's already correctly applied.

If what you actually want is the other flavor — updating the model's weights — there's one thing to get right before writing any code. Nearly every "wav2vec2 Quran" model on Hugging Face fine-tunes with a CTC loss for phonetic or word-level transcription of recitation. That's the standard recipe, and it's the wrong objective for you: it optimizes for transcribing correctly, not for pulling same-verse embeddings together and pushing different-verse embeddings apart — which is exactly what MAP and Top-K measure. Fine-tune that way and you might get a model that transcribes well while its embedding space is no more retrieval-friendly than your frozen baseline.
TBOGamer22/wav2vec2-quran-phonetics · Hugging Face +2

The closest real precedent for what you want is speaker recognition fine-tuning of Wav2Vec2, where the model processes pairs of audio segments and is trained on batches built from several speakers with multiple utterances each, using same-speaker and different-speaker pairs to shape the embedding space. Swap "speaker identity" for "ayah identity" and that's your fine-tuning objective.
arXiv

This lines up almost exactly with data you already have. Your 50-reciter segmentation pipeline gives you several different recordings of the same ayah — that's a ready-made positive-pair signal, no extra labeling needed.

Practically:

Starting checkpoint: skip English wav2vec2-base. Use an Arabic XLSR checkpoint — elgeish/wav2vec2-large-xlsr-53-arabic or jonatasgrosman/wav2vec2-large-xlsr-53-arabic are both established, built by fine-tuning facebook/wav2vec2-large-xlsr-53 on Arabic speech corpora, so the phonetic inventory starts much closer to Quranic Arabic than a generic multilingual base. There's precedent for exactly your use case, too — a Quran verse-identification project fine-tuned elgeish's Arabic XLSR model using data from the Quran ASR Challenge dataset on Kaggle. Also worth trying as an experiment: starting from an existing Quran-specific checkpoint (there are a few phonetic and syllable-level ASR ones on Hugging Face) since they're already adapted to tajweed and recitation-specific acoustics — you'd just load the encoder backbone and drop their CTC head. I'd treat that as a secondary experiment, not the safe default, since those checkpoints are far less battle-tested than the big Arabic XLSR ones.
Hugging Face
PromptLayer
What to freeze: freeze the CNN feature encoder — low-level acoustic features don't need retraining — and the lower half of the transformer layers. Fine-tune only the upper layers plus a small projection head on top. On a limited Quranic-audio dataset, full fine-tuning overfits fast.
Loss: InfoNCE / NT-Xent with in-batch negatives. For a batch containing several ayahs with 2+ reciters each, every same-ayah pair is a free positive and everything else in the batch is a free negative — batch construction matters more than the exact loss formula here.
Split by reciter, not just by ayah: hold out entire reciters for validation and test, not just held-out ayahs from reciters already seen in training. That mirrors the real use case — recognizing an ayah from a voice the model has never heard.
Reuse your existing eval code exactly — the same MAP/Top-K computation from your frozen baseline, just pointed at the fine-tuned embeddings. That's what makes "frozen vs. fine-tuned" a legitimate comparison instead of two different measurements.

One asymmetry worth knowing before you commit scope: Arabic-specific community checkpoints are common for Wav2Vec2, much rarer for Data2Vec-audio. A clean frozen-vs-fine-tuned comparison is easiest to run on the Wav2Vec2 side only, unless you want to first adapt a Data2Vec-audio base to Arabic yourself — more work, probably not worth it this close to submission.

Given you're finishing, not restarting, I'd frame this as an added ablation rather than a replacement: "we evaluate both frozen SSL embeddings and contrastively fine-tuned embeddings, and observe..." That's a stronger contribution than either result alone, and it builds on the comparison you've already written instead of requiring you to redo it.

Here's a working skeleton — a dataset that samples same-ayah positive pairs from your segmented clips, the model wrapper with the freezing logic, and the InfoNCE loss:

Orchestrated Python implementation with refined encoder freezing methodology
Orchestrated Python implementation with refined encoder freezing methodology

The dataset class samples a same-ayah, different-reciter pair on every call; the model wrapper handles the freezing and pooling; train() is written to accept your existing MAP/Top-K function directly via eval_fn, so plugging it in is a one-line change rather than a rewrite. Swap in your manifest format and checkpoint of choice.

Want help wiring the dataset class to your actual segmented-clips folder structure, or adapting your existing MAP/Top-K eval function to score these fine-tuned embeddings instead of the frozen ones?

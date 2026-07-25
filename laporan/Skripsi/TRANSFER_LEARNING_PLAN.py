"""
Contrastive fine-tuning of Wav2Vec2 for Quranic ayah retrieval.

This is a skeleton, not a drop-in script. It assumes you already have a
manifest of segmented ayah-level audio clips (path, ayah_id, reciter) from
your existing segmentation pipeline. Wire the manifest loading to your own
data before running.

Objective: instead of CTC fine-tuning (which optimizes for transcription),
this trains the model so that different reciters reciting the SAME ayah
produce embeddings close together, and different ayahs produce embeddings
far apart -- directly aligned with MAP / Top-K retrieval evaluation.

pip install torch torchaudio transformers --break-system-packages
"""

import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# ---------------------------------------------------------------------------
# 1. Dataset: samples an (anchor, positive) pair per __getitem__, where the
#    positive is a DIFFERENT reciter reciting the SAME ayah as the anchor.
# ---------------------------------------------------------------------------


class AyahRecitationDataset(Dataset):
    """
    manifest: list of dicts, e.g.
        {"path": "/data/clips/husary_002255.wav", "ayah_id": "2_255", "reciter": "husary"}
    ayah_id must uniquely identify (surah, ayah), e.g. f"{surah}_{ayah}".
    """

    def __init__(self, manifest, sample_rate=16000, max_duration=10.0):
        self.manifest = manifest
        self.sr = sample_rate
        self.max_len = int(max_duration * sample_rate)

        self.by_ayah = defaultdict(list)
        for i, item in enumerate(manifest):
            self.by_ayah[item["ayah_id"]].append(i)

        # ayahs with only one reciter can't form a positive pair -- surface
        # this instead of silently falling back to anchor==positive
        singletons = [a for a, idxs in self.by_ayah.items() if len(idxs) < 2]
        if singletons:
            print(
                f"warning: {len(singletons)} ayahs have only one reciter clip "
                f"and will reuse the anchor as its own positive (weak signal)"
            )

    def __len__(self):
        return len(self.manifest)

    def _load(self, path):
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav.mean(dim=0)  # mono
        if wav.shape[0] > self.max_len:
            start = random.randint(0, wav.shape[0] - self.max_len)
            wav = wav[start : start + self.max_len]
        return wav

    def __getitem__(self, idx):
        anchor_item = self.manifest[idx]
        ayah_id = anchor_item["ayah_id"]

        candidates = [i for i in self.by_ayah[ayah_id] if i != idx]
        pos_idx = random.choice(candidates) if candidates else idx
        pos_item = self.manifest[pos_idx]

        return {
            "anchor": self._load(anchor_item["path"]),
            "positive": self._load(pos_item["path"]),
            "ayah_id": ayah_id,
        }


def make_collate_fn(feature_extractor, sample_rate=16000):
    def collate_fn(batch):
        anchors = [b["anchor"].numpy() for b in batch]
        positives = [b["positive"].numpy() for b in batch]
        ayah_ids = [b["ayah_id"] for b in batch]

        anchor_inputs = feature_extractor(
            anchors, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        positive_inputs = feature_extractor(
            positives, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        return anchor_inputs, positive_inputs, ayah_ids

    return collate_fn


# ---------------------------------------------------------------------------
# 2. Model: Wav2Vec2 backbone (frozen CNN + partially frozen transformer)
#    + mean pooling + small projection head, L2-normalized output.
# ---------------------------------------------------------------------------


class Wav2Vec2Embedder(nn.Module):
    def __init__(
        self,
        model_name="elgeish/wav2vec2-large-xlsr-53-arabic",
        freeze_feature_encoder=True,
        freeze_transformer_layers=6,
        proj_dim=256,
    ):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_feature_encoder:
            self.backbone.freeze_feature_encoder()  # freezes the CNN front-end

        if freeze_transformer_layers > 0:
            for layer in self.backbone.encoder.layers[:freeze_transformer_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_dim),
        )

    def forward(self, input_values, attention_mask=None):
        out = self.backbone(input_values, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, T, H)

        if attention_mask is not None:
            feat_mask = self.backbone._get_feature_vector_attention_mask(
                hidden.shape[1], attention_mask
            ).unsqueeze(-1)
            hidden = hidden * feat_mask
            pooled = hidden.sum(dim=1) / feat_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)

        emb = self.proj(pooled)
        return F.normalize(emb, dim=-1)


# ---------------------------------------------------------------------------
# 3. Loss: symmetric InfoNCE / NT-Xent with in-batch negatives.
#    anchor_i and positive_i are a positive pair; every other positive_j
#    in the batch is a free negative for anchor_i (and vice versa).
# ---------------------------------------------------------------------------


def info_nce_loss(anchor_emb, positive_emb, temperature=0.07):
    logits = anchor_emb @ positive_emb.t() / temperature  # (B, B)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return (loss_a + loss_b) / 2


# ---------------------------------------------------------------------------
# 4. Training loop skeleton. Plug in your own manifest + eval function.
# ---------------------------------------------------------------------------


def train(
    manifest_train,
    manifest_val,
    model_name="elgeish/wav2vec2-large-xlsr-53-arabic",
    epochs=10,
    batch_size=8,
    lr=1e-5,
    device="cuda",
    eval_fn=None,  # e.g. your existing MAP / Top-K function, signature: eval_fn(model, manifest_val) -> dict
):
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    collate_fn = make_collate_fn(fe)

    train_ds = AyahRecitationDataset(manifest_train)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    model = Wav2Vec2Embedder(model_name).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=lr)

    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"training {n_trainable:,} / {n_total:,} parameters")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for anchor_inputs, positive_inputs, _ in train_loader:
            anchor_inputs = {k: v.to(device) for k, v in anchor_inputs.items()}
            positive_inputs = {k: v.to(device) for k, v in positive_inputs.items()}

            anchor_emb = model(**anchor_inputs)
            positive_emb = model(**positive_inputs)

            loss = info_nce_loss(anchor_emb, positive_emb)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"epoch {epoch}: train_loss={avg_loss:.4f}")

        if eval_fn is not None:
            model.eval()
            with torch.no_grad():
                metrics = eval_fn(model, manifest_val)
            print(f"epoch {epoch}: val_metrics={metrics}")

    return model


if __name__ == "__main__":
    # Example manifest format -- replace with your actual segmented clips.
    # manifest_train = [
    #     {"path": "...", "ayah_id": "1_1", "reciter": "husary"},
    #     {"path": "...", "ayah_id": "1_1", "reciter": "sudais"},
    #     ...
    # ]
    # model = train(manifest_train, manifest_val, eval_fn=your_map_topk_eval_fn)
    pass
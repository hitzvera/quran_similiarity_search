"""
Generator diagram alur fase Data Understanding (CRISP-DM) untuk BAB III.

Menghasilkan diagram alur empat sub-tugas Data Understanding
(Collect -> Describe -> Explore -> Verify) beserta cabang keputusan
iteratif kembali ke Business Understanding.

Output: media/data_understanding_flow.png (dan .pdf opsional)

Cara pakai:
    pip install matplotlib
    python diagram_data_understanding.py
"""

import os
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Konfigurasi tampilan
# ---------------------------------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Sans"

COL_PHASE = "#DCE6F5"      # biru muda: kotak fase utama
COL_PHASE_EDGE = "#2F5496"
COL_TERMINAL = "#D7E8D2"   # hijau muda: mulai / selesai
COL_TERMINAL_EDGE = "#548235"
COL_DECISION = "#FCE4C4"   # oranye muda: keputusan
COL_DECISION_EDGE = "#C55A11"
COL_LOOP = "#F4CCCC"       # merah muda: loop iteratif
COL_LOOP_EDGE = "#A61C00"
COL_TEXT = "#1A1A1A"

FIG_W, FIG_H = 9.5, 13.5   # inci


def add_box(ax, x, y, w, h, title, body=None, face=COL_PHASE,
            edge=COL_PHASE_EDGE, style="round", title_size=11, body_size=9):
    """Gambar satu kotak dengan judul (dan opsional daftar isi di bawahnya)."""
    boxstyle = "round,pad=0.02,rounding_size=0.08" if style == "round" else "square,pad=0.02"
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=boxstyle, linewidth=1.6,
        facecolor=face, edgecolor=edge, zorder=2,
    )
    ax.add_patch(patch)

    if body:
        ax.text(x, y + h / 2 - 0.18, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold", color=COL_TEXT, zorder=3)
        ax.text(x, y + h / 2 - 0.52, body, ha="center", va="top",
                fontsize=body_size, color=COL_TEXT, zorder=3, linespacing=1.35)
    else:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=COL_TEXT, zorder=3)


def add_diamond(ax, x, y, w, h, text, face=COL_DECISION, edge=COL_DECISION_EDGE):
    """Gambar node keputusan berbentuk belah ketupat."""
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    poly = plt.Polygon(pts, closed=True, linewidth=1.6,
                       facecolor=face, edgecolor=edge, zorder=2)
    ax.add_patch(poly)
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=COL_TEXT, zorder=3)


def arrow(ax, x1, y1, x2, y2, color="#404040", label=None,
          lx=None, ly=None, connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
        linewidth=1.5, color=color, zorder=1,
        connectionstyle=connectionstyle,
    ))
    if label:
        ax.text(lx if lx is not None else (x1 + x2) / 2,
                ly if ly is not None else (y1 + y2) / 2,
                label, ha="center", va="center", fontsize=9,
                fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"),
                zorder=4)


def bullet(items, width=46):
    """Format daftar poin dengan bullet dan wrapping rapi."""
    lines = []
    for it in items:
        wrapped = textwrap.wrap(it, width=width)
        lines.append("\u2022 " + wrapped[0])
        for cont in wrapped[1:]:
            lines.append("   " + cont)
    return "\n".join(lines)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")

    cx = 5.0  # sumbu tengah vertikal

    # --- Terminal: Mulai ---
    add_box(ax, cx, 19.2, 4.2, 0.9, "Mulai: Fase Data Understanding",
            face=COL_TERMINAL, edge=COL_TERMINAL_EDGE, title_size=11)

    # --- 1. Collect Initial Data ---
    add_box(ax, cx, 17.4, 6.6, 1.9,
            "1. Collect Initial Data",
            bullet([
                "Mengakses dataset Quran-MD (HuggingFace), sub-dataset tingkat ayat",
                "Mendokumentasikan struktur: audio, teks Arab, metadata qori",
                "Menyeleksi cakupan sesuai batasan: Surah Al-Fatihah + Juz Amma",
            ]))

    # --- 2. Describe Data ---
    add_box(ax, cx, 14.9, 6.6, 2.1,
            "2. Describe Data (Sifat Permukaan)",
            bullet([
                "Format berkas audio (mp3)",
                "Sampling rate asli & jumlah kanal (mono/stereo)",
                "Cakupan: jumlah surah, ayat unik, dan jumlah qori",
            ]))

    # --- 3. Explore Data (EDA) ---
    add_box(ax, cx, 12.2, 6.6, 2.3,
            "3. Explore Data (EDA)",
            bullet([
                "Distribusi jumlah qori per ayat (menentukan |R| untuk MAP/MRR)",
                "Distribusi durasi rekaman antar-ayat (dampak ke temporal pooling)",
                "Distribusi jumlah ayat per surah pada subset",
            ]))

    # --- 4. Verify Data Quality ---
    add_box(ax, cx, 9.4, 6.6, 2.3,
            "4. Verify Data Quality",
            bullet([
                "Ayat kekurangan variasi qori (tak punya pasangan relevan)",
                "Rekaman rusak / terpotong / derau berlebih",
                "Konsistensi anotasi batas ayat (satu berkas = satu ayat utuh)",
            ]))

    # --- Keputusan ---
    add_diamond(ax, cx, 6.4, 4.6, 2.0,
                "Data layak &\nmendukung tujuan?")

    # --- Loop iteratif (kiri) ---
    add_box(ax, 1.7, 6.4, 2.7, 1.7,
            "Revisi Business\nUnderstanding",
            bullet(["Sesuaikan tujuan", "atau batasan (iteratif)"], width=22),
            face=COL_LOOP, edge=COL_LOOP_EDGE, title_size=10, body_size=8.5)

    # --- Terminal: lanjut ---
    add_box(ax, cx, 3.9, 5.0, 0.9,
            "Lanjut ke Fase Data Preparation",
            face=COL_TERMINAL, edge=COL_TERMINAL_EDGE, title_size=11)

    # -----------------------------------------------------------------------
    # Panah alur
    # -----------------------------------------------------------------------
    arrow(ax, cx, 18.75, cx, 18.35)   # mulai -> collect
    arrow(ax, cx, 16.45, cx, 15.95)   # collect -> describe
    arrow(ax, cx, 13.85, cx, 13.35)   # describe -> explore
    arrow(ax, cx, 11.05, cx, 10.55)   # explore -> verify
    arrow(ax, cx, 8.25, cx, 7.4)      # verify -> keputusan
    arrow(ax, cx, 5.4, cx, 4.35, label="Ya", lx=cx + 0.35, ly=4.9)

    # keputusan -> loop (Tidak)
    arrow(ax, cx - 2.3, 6.4, 3.05, 6.4, color=COL_LOOP_EDGE,
          label="Tidak", lx=3.75, ly=6.7)
    # loop -> collect (naik kembali, melengkung)
    arrow(ax, 1.7, 7.25, 1.7, 17.4, color=COL_LOOP_EDGE,
          connectionstyle="arc3,rad=0.0")
    arrow(ax, 1.7, 17.4, cx - 3.3, 17.4, color=COL_LOOP_EDGE)

    # -----------------------------------------------------------------------
    # Judul
    # -----------------------------------------------------------------------
    ax.text(cx, 19.85,
            "Alur Fase Data Understanding pada Kerangka CRISP-DM",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=COL_TEXT)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "data_understanding_flow.png")
    pdf_path = os.path.join(out_dir, "data_understanding_flow.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")  # vektor, kualitas cetak
    print(f"Tersimpan:\n  {png_path}\n  {pdf_path}")


if __name__ == "__main__":
    main()

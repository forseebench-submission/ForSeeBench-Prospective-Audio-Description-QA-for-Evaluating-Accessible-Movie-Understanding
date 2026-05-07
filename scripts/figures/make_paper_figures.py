"""Build NeurIPS-style figures for the ForSeeBench paper from the curated
examples in `figure_examples/`.

Outputs (PDF + PNG) → `figures/paper_figures/`:
  - fig_teaser.{pdf,png}     — main teaser (good_examples/example_03)
  - fig_filter.{pdf,png}     — nontriviality filtering 2x2 grid
  - fig_ad_source.{pdf,png}  — AD-source substitution (baseline_comparison/example_01)

Each panel is its own ``fig.add_axes`` rectangle so panels never overlap. Text
inside a panel is in panel-local coordinates [0..1].
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT = Path("/jumbo/jinlab/Sally/ForSeeBench")
FE = ROOT / "figure_examples"
OUT = ROOT / "figures/paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

# Palette
INK = "#1F2933"
MUTED = "#6B7280"
SUBTLE = "#9CA3AF"
BLUE = "#3D6FB6"
DARK_BLUE = "#1E3A8A"
GREEN = "#3F8F5C"
DARK_GREEN = "#1E5631"
RED = "#C0413E"
DARK_RED = "#7B1F1C"
ORANGE = "#D97A2D"
GOLD = "#E2A83C"
PURPLE = "#7E5BB0"

PRIOR_HI = "#FFF3D2"
GOLD_HI = "#DDEFD2"
RED_HI = "#FBE3E1"
BLUE_HI = "#E8F0FB"
PANEL_FACE = "#FFFFFF"
PANEL_TINT = "#F8FAFC"


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def setup_panel(fig, rect, *, ec=BLUE, fc=PANEL_FACE, lw=1.4, header=None,
                header_color=None):
    """Create a sub-axes for one panel, draw the bounding box + optional
    header bar. Returns the axes (already configured to use 0..1 coords).
    """
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Outer rounded rectangle
    ax.add_patch(FancyBboxPatch(
        (0.005, 0.005), 0.99, 0.99,
        boxstyle="round,pad=0.0,rounding_size=0.020",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    ))
    if header:
        bar_h = 0.13
        bar_color = header_color or ec
        ax.add_patch(FancyBboxPatch(
            (0.005, 1 - bar_h - 0.005), 0.99, bar_h,
            boxstyle="round,pad=0.0,rounding_size=0.020",
            linewidth=0, edgecolor=bar_color, facecolor=bar_color,
            zorder=2,
        ))
        # Cover bottom corners of the bar so the rounding only happens at the
        # top of the panel.
        ax.add_patch(patches.Rectangle((0.005, 1 - bar_h - 0.005),
                                       0.99, bar_h * 0.30,
                                       fc=bar_color, ec=bar_color, lw=0,
                                       zorder=2))
        ax.text(0.025, 1 - bar_h / 2 - 0.005, header,
                fontsize=9.0, color="white", fontweight="bold",
                va="center", zorder=3)
    return ax


def section_marker(ax, x, y, n, title, color=BLUE):
    """Big numbered circle marker like the reference figure."""
    ax.add_patch(plt.Circle((x, y), 0.034, color=color, zorder=10))
    ax.text(x, y, str(n), color="white", ha="center", va="center",
            fontsize=11, fontweight="bold", zorder=11)
    ax.text(x + 0.055, y, title, color=color, fontsize=10,
            fontweight="bold", va="center", zorder=11)


# ---------------------------------------------------------------------------
# Figure 1 — main teaser
# ---------------------------------------------------------------------------

def make_teaser():
    """Teaser using baseline_comparison_examples/example_01 (Tracy / The
    Roommate). The figure makes the text-only answerer visible by splitting
    the canvas into two regions:

      TOP  ── AD-generation lane: dimmed video frames → AD model → AD text.
      DIVIDER ── a horizontal "firewall" labelled "the answerer reads only
                 what is below this line."
      BOTTOM ── answerer's view: prior AD timeline, hidden target card,
                 question with four plausible options.

    The example: prior AD says ``In her room, Tracy sits up in bed and rubs
    her brow.''  Target: ``Tracy groggily gets up and crosses the dim
    room.''  All four options are plausible morning behaviours; the bridge
    is ``sits up + rubs brow → about to get out of bed.''
    """
    ex = FE / "baseline_comparison_examples/example_01"
    ctx_frames = sorted((ex / "frames").glob("context_*.jpg"))
    # Pick three representative frames: t0, tmid, tend.
    chosen = []
    for tag in ["t0", "tmid", "tend"]:
        for fp in ctx_frames:
            if fp.name.endswith(f"__{tag}.jpg"):
                chosen.append(fp)
                break
    chosen = chosen[:3]

    target_sentence = "Tracy groggily gets up and crosses the dim room."
    options = [
        "Tracy gets out of bed and walks across the room.",
        "Tracy sits back down on the bed and continues to rub her brow.",
        "Tracy picks up a book and starts reading.",
        "Tracy talks to someone off-screen.",
    ]
    correct_idx = 0

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    # Two-region layout: AD-generation lane on top, answerer's view below.
    fig = plt.figure(figsize=(9.5, 5.4))
    fig.patch.set_facecolor("white")

    # Single axes, fig-relative 0..1 coords.
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # ===== Title =====
    ax.text(0.5, 0.965,
            "ForSeeBench: Prospective Audio-Description QA",
            ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(0.5, 0.928,
            "The answerer is text-only — it reads prior AD lines, never the video.",
            ha="center", va="center", fontsize=9.0, color=MUTED, style="italic")

    # ===== TOP REGION: AD-generation lane =====
    # Bounded box covering the full top region with a faint grey background to
    # signal "this entire region is the AD-model's input — not the answerer's."
    top_y0, top_y1 = 0.700, 0.895
    ax.add_patch(FancyBboxPatch(
        (0.018, top_y0), 0.964, top_y1 - top_y0,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        lw=0.8, ec=SUBTLE, fc="#F5F5F7", zorder=1,
    ))

    # Region label (top-left of the lane)
    ax.text(0.030, top_y1 - 0.013,
            "AD-generation pipeline  ·  the answerer never sees this",
            fontsize=8.4, color=MUTED, fontweight="bold", style="italic",
            va="top", zorder=3)

    # Layout inside the top region: 3 frames → arrow → AD model → arrow → AD text
    fr_y0 = 0.715
    fr_h = 0.140
    fr_w = 0.085
    fr_gap = 0.010
    frame_x_start = 0.030
    for i, fp in enumerate(chosen):
        x = frame_x_start + i * (fr_w + fr_gap)
        # Render frames at full intensity so the bedroom visual is legible.
        # The "AD-model only" framing is carried by the panel-level grey
        # background and the explicit caption above the strip.
        ax.imshow(mpimg.imread(fp), extent=(x, x + fr_w, fr_y0, fr_y0 + fr_h),
                  aspect="auto", zorder=2)
        ax.add_patch(patches.Rectangle((x, fr_y0), fr_w, fr_h,
                                       fill=False, lw=0.7, ec="#111", zorder=3))
    # Frames label
    ax.text(frame_x_start + ((fr_w + fr_gap) * len(chosen) - fr_gap) / 2,
            top_y0 + 0.018,
            "video frames",
            ha="center", va="center", fontsize=7.0, color=MUTED, style="italic")

    # Arrow → AD model
    arrow1_x = frame_x_start + len(chosen) * (fr_w + fr_gap) + 0.005
    ax.add_patch(FancyArrowPatch(
        (arrow1_x, fr_y0 + fr_h / 2),
        (arrow1_x + 0.040, fr_y0 + fr_h / 2),
        arrowstyle="-|>", mutation_scale=11, lw=1.4, color=BLUE, zorder=4,
    ))

    # AD model box
    model_x = arrow1_x + 0.045
    model_w = 0.150
    ax.add_patch(FancyBboxPatch(
        (model_x, fr_y0), model_w, fr_h,
        boxstyle="round,pad=0.005,rounding_size=0.020",
        lw=1.2, ec=BLUE, fc=BLUE_HI, zorder=2,
    ))
    ax.text(model_x + model_w / 2, fr_y0 + fr_h / 2,
            "AD model\n(MAD-eval, NarrAD,\nAutoAD-Zero)",
            ha="center", va="center", fontsize=7.4, color=DARK_BLUE,
            fontweight="bold", linespacing=1.30, zorder=3)

    # Arrow → AD text
    ax.add_patch(FancyArrowPatch(
        (model_x + model_w + 0.005, fr_y0 + fr_h / 2),
        (model_x + model_w + 0.045, fr_y0 + fr_h / 2),
        arrowstyle="-|>", mutation_scale=11, lw=1.4, color=BLUE, zorder=4,
    ))

    # AD text snippet box (preview of what feeds into the answerer)
    text_x = model_x + model_w + 0.050
    text_w = 0.290
    ax.add_patch(FancyBboxPatch(
        (text_x, fr_y0), text_w, fr_h,
        boxstyle="round,pad=0.005,rounding_size=0.020",
        lw=1.0, ec=ORANGE, fc="#FFFFFF", zorder=2,
    ))
    ax.text(text_x + 0.010, fr_y0 + fr_h - 0.018,
            "AD text",
            fontsize=7.4, color=ORANGE, fontweight="bold", va="top", zorder=3)
    ax.text(text_x + 0.010, fr_y0 + fr_h - 0.053,
            "“In her room, Tracy sits up\n  in bed and rubs her brow.”",
            fontsize=7.4, color=INK, va="top", linespacing=1.30, zorder=3)

    # "Drops down to answerer" arrow & label
    answerer_arrow_x = text_x + text_w + 0.018
    ax.text(answerer_arrow_x, fr_y0 + fr_h + 0.012,
            "only the AD text\ncrosses to the answerer",
            ha="left", va="bottom", fontsize=7.0, color=GREEN,
            fontweight="bold", style="italic")
    ax.add_patch(FancyArrowPatch(
        (answerer_arrow_x + 0.018, fr_y0 + fr_h * 0.55),
        (answerer_arrow_x + 0.018, top_y0 - 0.018),
        arrowstyle="-|>", mutation_scale=11, lw=1.6, color=GREEN, zorder=4,
    ))

    # ===== FIREWALL DIVIDER =====
    fw_y = 0.685
    ax.plot([0.018, 0.982], [fw_y, fw_y], color=GREEN,
            lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax.text(0.500, fw_y - 0.018,
            "▼  the answerer reads only what is below this line  ▼",
            ha="center", va="center", fontsize=8.4, color=GREEN,
            fontweight="bold", style="italic", zorder=4)

    # ===== BOTTOM REGION: answerer's view =====
    # Three columns: prior AD (left+center), hidden ??? (top-right), QA (bottom-right).
    # Layout
    B = (0.018, 0.605)   # prior AD column (left, right)
    C = (0.625, 0.982)   # right stack

    # ---- Prior AD context (left) ----
    b_x0, b_x1 = B
    bw = b_x1 - b_x0
    body_y0 = 0.040
    body_y1 = 0.638
    ax.add_patch(FancyBboxPatch(
        (b_x0, body_y0), bw, body_y1 - body_y0,
        boxstyle="round,pad=0.0,rounding_size=0.014",
        lw=1.2, ec=BLUE, fc=PANEL_FACE, zorder=1,
    ))
    # Header strip
    ax.add_patch(FancyBboxPatch(
        (b_x0, body_y1 - 0.060), bw, 0.060,
        boxstyle="round,pad=0.0,rounding_size=0.014",
        lw=0, fc=BLUE, zorder=2,
    ))
    ax.add_patch(patches.Rectangle(
        (b_x0, body_y1 - 0.060), bw, 0.020, fc=BLUE, ec=BLUE, lw=0, zorder=2,
    ))
    ax.text(b_x0 + 0.014, body_y1 - 0.030,
            "Prior AD context  ·  visible to the answerer",
            fontsize=10, color="white", fontweight="bold",
            va="center", zorder=3)

    # The bridge AD line, with a soft tint
    ad_y_top = body_y1 - 0.110
    ad_h = 0.220
    ax.add_patch(FancyBboxPatch(
        (b_x0 + 0.014, ad_y_top - ad_h + 0.025), bw - 0.028, ad_h,
        boxstyle="round,pad=0.0,rounding_size=0.014",
        lw=0, fc=PRIOR_HI, zorder=2,
    ))
    ax.text(b_x0 + 0.024, ad_y_top, "AD₁",
            fontsize=10, fontweight="bold", color=ORANGE,
            family="monospace", va="top", zorder=4)
    ax.text(b_x0 + 0.064, ad_y_top, "−00:08",
            fontsize=9.5, color=ORANGE, fontweight="bold",
            family="monospace", va="top", zorder=4)
    ax.text(b_x0 + 0.140, ad_y_top,
            wrap("In her room, Tracy sits up in bed and rubs her brow.", 38),
            fontsize=10.5, color=INK, va="top",
            linespacing=1.30, zorder=4)

    # Bridge keyword callout under the AD line
    ax.text(b_x0 + 0.024, ad_y_top - 0.155,
            "▸ bridge cue",
            fontsize=8.4, color=DARK_RED, fontweight="bold", style="italic",
            va="top", zorder=4)
    ax.text(b_x0 + 0.135, ad_y_top - 0.155,
            "“sits up in bed”  +  “rubs her brow”",
            fontsize=9.5, color=ORANGE, fontweight="bold",
            va="top", zorder=4)
    ax.text(b_x0 + 0.024, ad_y_top - 0.193,
            "      →  she is waking up; about to leave the bed.",
            fontsize=9, color=DARK_RED, style="italic", va="top", zorder=4)

    # Bottom note
    ax.text((b_x0 + b_x1) / 2, 0.073,
            "→ infer the next narrated visual update from this AD line alone.",
            ha="center", va="center", fontsize=8.4, color=MUTED, style="italic")

    # ---- Hidden future AD card (top-right) ----
    c_x0, c_x1 = C
    cw = c_x1 - c_x0
    c_top_y0, c_top_y1 = 0.380, 0.638
    ax.add_patch(FancyBboxPatch(
        (c_x0, c_top_y0), cw, c_top_y1 - c_top_y0,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        lw=1.2, ec=RED, fc=RED_HI, zorder=1,
    ))
    ax.text((c_x0 + c_x1) / 2, c_top_y1 - 0.025,
            "Future AD  ·  hidden",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=DARK_RED, zorder=2)
    ax.text((c_x0 + c_x1) / 2, (c_top_y0 + c_top_y1) / 2,
            "?  ?  ?",
            ha="center", va="center", fontsize=38, fontweight="bold",
            color=RED, zorder=2)
    ax.text((c_x0 + c_x1) / 2, c_top_y0 + 0.020,
            f"target:  “{target_sentence}”",
            ha="center", va="center", fontsize=6.8, color=MUTED,
            style="italic", zorder=2)

    # ---- QA card (bottom-right) ----
    c_bot_y0, c_bot_y1 = 0.040, 0.355
    bh = c_bot_y1 - c_bot_y0
    ax.add_patch(FancyBboxPatch(
        (c_x0, c_bot_y0), cw, bh,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        lw=1.2, ec=GREEN, fc=PANEL_FACE, zorder=1,
    ))
    ax.text(c_x0 + 0.014, c_bot_y1 - 0.025,
            "Q:  What happens next?",
            fontsize=10.0, fontweight="bold", color=INK,
            va="center", zorder=2)

    # Render four options
    opt_y_top = c_bot_y1 - 0.060
    opt_h = 0.060
    opt_gap = 0.005
    for j in range(4):
        oy_top = opt_y_top - j * (opt_h + opt_gap)
        oy_bot = oy_top - opt_h
        is_gold = (j == correct_idx)
        if is_gold:
            ax.add_patch(FancyBboxPatch(
                (c_x0 + 0.008, oy_bot), cw - 0.016, opt_h,
                boxstyle="round,pad=0.005,rounding_size=0.014",
                lw=0, fc=GOLD_HI, zorder=2,
            ))
        letter = chr(ord("A") + j)
        ax.text(c_x0 + 0.018, oy_top - opt_h / 2, f"({letter})",
                fontsize=9.6, color=GREEN if is_gold else INK,
                fontweight="bold", va="center", zorder=3)
        ax.text(c_x0 + 0.054, oy_top - opt_h / 2, options[j],
                fontsize=8.6, color=INK, va="center", zorder=3)
        if is_gold:
            ax.text(c_x1 - 0.014, oy_top - opt_h / 2, "✓",
                    fontsize=12, color=GREEN, fontweight="bold",
                    ha="right", va="center", zorder=3)

    # ===== Footer caption =====
    ax.text(0.5, 0.012,
            "All four options are independently plausible bedroom actions.  "
            "Only the bridge cue (sits up + rubs brow → waking up) makes "
            "“gets out of bed and walks across the room” the natural pick.",
            ha="center", va="center", fontsize=7.6, color=INK, style="italic")

    pdf = OUT / "fig_teaser.pdf"
    png = OUT / "fig_teaser.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}  +  {png}")


def _legacy_make_teaser_unused():
    """Old multi-block layout left in place for reference; not called."""
    chosen = []
    by_tag = {"t0": [], "tmid": [], "tend": []}
    for fp in []:
        for tag in by_tag:
            if fp.name.endswith(f"__{tag}.jpg"):
                by_tag[tag].append(fp)
                break
    if by_tag["t0"]:
        chosen.append(by_tag["t0"][0])
    if len(by_tag["tmid"]) > 1:
        chosen.append(by_tag["tmid"][-1])
    elif by_tag["tmid"]:
        chosen.append(by_tag["tmid"][0])
    if by_tag["tend"]:
        chosen.append(by_tag["tend"][-1])
    chosen = chosen[:3]

    frame_top = 0.785
    frame_h = 0.18
    frame_gap = 0.020
    margin = 0.013
    fx0, fx1 = a_x0 + margin, a_x1 - margin
    for i, fp in enumerate(chosen):
        y_top = frame_top - i * (frame_h + frame_gap)
        ax.imshow(mpimg.imread(fp),
                  extent=(fx0, fx1, y_top - frame_h, y_top),
                  aspect="auto", zorder=2)
        ax.add_patch(patches.Rectangle(
            (fx0, y_top - frame_h), fx1 - fx0, frame_h,
            fill=False, lw=0.7, ec="#111", zorder=3,
        ))
    ax.text((a_x0 + a_x1) / 2, 0.115,
            "movie clip\nframes",
            fontsize=6.6, color=MUTED, ha="center",
            va="center", style="italic")

    # ===== Column B: prior AD context =====
    b_x0, b_x1 = B
    bw = b_x1 - b_x0
    # Outer panel
    ax.add_patch(FancyBboxPatch(
        (b_x0, 0.080), bw, 0.785,
        boxstyle="round,pad=0.0,rounding_size=0.014",
        lw=1.2, ec=BLUE, fc=PANEL_FACE, zorder=1,
    ))
    # Header bar
    ax.add_patch(FancyBboxPatch(
        (b_x0, 0.808), bw, 0.057,
        boxstyle="round,pad=0.0,rounding_size=0.014",
        lw=0, fc=BLUE, zorder=2,
    ))
    ax.add_patch(patches.Rectangle(
        (b_x0, 0.808), bw, 0.018, fc=BLUE, ec=BLUE, lw=0, zorder=2,
    ))
    ax.text(b_x0 + 0.014, 0.838,
            "Prior AD context  ·  visible to the answerer",
            fontsize=10, color="white", fontweight="bold", va="center", zorder=3)

    # AD1
    ad1_y_top = 0.770
    ax.text(b_x0 + 0.020, ad1_y_top, "AD₁",
            fontsize=10, fontweight="bold", color=MUTED,
            family="monospace", va="top", zorder=3)
    ax.text(b_x0 + 0.060, ad1_y_top, "−00:18",
            fontsize=9.5, color=MUTED, family="monospace", va="top", zorder=3)
    ax.text(b_x0 + 0.135, ad1_y_top,
            "Charlie strides out of the restaurant,\nhis eyes cast downward.",
            fontsize=9.5, color=INK, va="top", linespacing=1.30, zorder=3)

    # AD2 — bridge row, soft tint behind
    ad2_y_top = 0.610
    ad2_h = 0.245
    ax.add_patch(FancyBboxPatch(
        (b_x0 + 0.012, ad2_y_top - ad2_h + 0.020), bw - 0.024, ad2_h,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        lw=0, fc=PRIOR_HI, zorder=2,
    ))
    ax.text(b_x0 + 0.020, ad2_y_top, "AD₂",
            fontsize=10, fontweight="bold", color=ORANGE,
            family="monospace", va="top", zorder=4)
    ax.text(b_x0 + 0.060, ad2_y_top, "−00:14",
            fontsize=9.5, color=ORANGE, fontweight="bold",
            family="monospace", va="top", zorder=4)

    # Render AD2 body as plain INK text wrapped to fit. Bridge keywords are
    # surfaced separately below as a callout — that avoids inline-color
    # alignment issues entirely.
    ad2_body = ("As Charlie steps into the street, a fisherman in waterproof "
                "bib pants passes him carrying a coil of rope slung over his shoulder.")
    ax.text(b_x0 + 0.135, ad2_y_top, wrap(ad2_body, 38),
            fontsize=9.5, color=INK, va="top", linespacing=1.30, zorder=4)

    # Bridge keyword callout (orange bold, single line)
    ax.text(b_x0 + 0.020, ad2_y_top - 0.180,
            "▸ bridge keywords",
            fontsize=8.4, color=DARK_RED, fontweight="bold",
            style="italic", va="top", zorder=4)
    ax.text(b_x0 + 0.155, ad2_y_top - 0.180,
            "fisherman  ·  waterproofs  ·  rope",
            fontsize=9.0, color=ORANGE, fontweight="bold",
            va="top", zorder=4)
    ax.text(b_x0 + 0.020, ad2_y_top - 0.220,
            "      →  implies a waterfront setting",
            fontsize=8.4, color=DARK_RED, style="italic",
            va="top", zorder=4)

    # Bottom note inside column B
    ax.text((b_x0 + b_x1) / 2, 0.115,
            "→ infer the next narrated visual update from these AD lines alone.",
            ha="center", va="center", fontsize=8.2, color=MUTED, style="italic")

    # ===== Column C top: hidden future AD =====
    c_x0, c_x1 = C
    cw = c_x1 - c_x0
    # Top card y range
    c_top_y0, c_top_y1 = 0.520, 0.865
    ax.add_patch(FancyBboxPatch(
        (c_x0, c_top_y0), cw, c_top_y1 - c_top_y0,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        lw=1.2, ec=RED, fc=RED_HI, zorder=1,
    ))
    ax.text((c_x0 + c_x1) / 2, c_top_y1 - 0.040,
            "Future AD  ·  hidden from the answerer",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=DARK_RED, zorder=2)
    ax.text((c_x0 + c_x1) / 2, (c_top_y0 + c_top_y1) / 2,
            "?  ?  ?",
            ha="center", va="center", fontsize=44, fontweight="bold",
            color=RED, zorder=2)
    ax.text((c_x0 + c_x1) / 2, c_top_y0 + 0.030,
            f"hidden target: “{target_sentence}”",
            ha="center", va="center", fontsize=7.4, color=MUTED,
            style="italic", zorder=2)

    # ===== Column C bottom: question + options =====
    c_bot_y0, c_bot_y1 = 0.080, 0.490
    bh = c_bot_y1 - c_bot_y0
    ax.add_patch(FancyBboxPatch(
        (c_x0, c_bot_y0), cw, bh,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        lw=1.2, ec=GREEN, fc=PANEL_FACE, zorder=1,
    ))
    ax.text(c_x0 + 0.012, c_bot_y1 - 0.030,
            "Q:  What happens next?",
            fontsize=10.5, fontweight="bold", color=INK, va="center", zorder=2)

    # Render four options. The correct one (A) gets a soft-gold tint, a green
    # check, and the keyword "water" highlighted in orange to mirror the bridge.
    opt_y_top = c_bot_y1 - 0.080
    opt_h = 0.080
    opt_gap = 0.005
    for j in range(4):
        oy_top = opt_y_top - j * (opt_h + opt_gap)
        oy_bot = oy_top - opt_h
        is_gold = (j == correct_idx)
        if is_gold:
            ax.add_patch(FancyBboxPatch(
                (c_x0 + 0.008, oy_bot), cw - 0.016, opt_h,
                boxstyle="round,pad=0.005,rounding_size=0.014",
                lw=0, fc=GOLD_HI, zorder=2,
            ))
        letter = chr(ord("A") + j)
        ax.text(c_x0 + 0.018, oy_top - opt_h / 2, f"({letter})",
                fontsize=10, color=GREEN if is_gold else INK,
                fontweight="bold", va="center", zorder=3)

        if is_gold:
            # Inline highlight of "water"
            ax.text(c_x0 + 0.055, oy_top - opt_h / 2,
                    "Charlie gazes intently at the ",
                    fontsize=9, color=INK, va="center", zorder=3)
            ax.text(c_x0 + 0.276, oy_top - opt_h / 2,
                    "water",
                    fontsize=9, color=ORANGE, fontweight="bold",
                    va="center", zorder=3)
            ax.text(c_x0 + 0.318, oy_top - opt_h / 2,
                    ".", fontsize=9, color=INK, va="center", zorder=3)
            ax.text(c_x1 - 0.020, oy_top - opt_h / 2, "✓",
                    fontsize=12, color=GREEN, fontweight="bold",
                    ha="right", va="center", zorder=3)
        else:
            ax.text(c_x0 + 0.055, oy_top - opt_h / 2,
                    options[j], fontsize=9, color=INK,
                    va="center", zorder=3)

    # ===== Footer caption =====
    ax.text(0.5, 0.025,
            "All four options are independently plausible street actions.  Only the bridge clue in AD₂ "
            "(a fisherman with waterproofs and rope → waterfront setting) makes “gazes at the water” the natural pick.",
            ha="center", va="center", fontsize=7.6, color=INK, style="italic")

    pdf = OUT / "fig_teaser.pdf"
    png = OUT / "fig_teaser.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}  +  {png}")


def WHITE_HEX() -> str:
    return "#FFFFFF"


# ---------------------------------------------------------------------------
# Figure 2 — filtering: rejected vs. retained
# ---------------------------------------------------------------------------

def _qa_panel(fig, rect, *, header, header_color, badge, badge_color,
              ctx_lines, target_text, options, correct_idx, distractors,
              frame_path, takeaway, takeaway_color):
    ax = setup_panel(fig, rect, ec=header_color, fc=PANEL_FACE,
                     header=header, header_color=header_color)
    # Badge in header
    ax.text(0.97, 0.94, badge, fontsize=8.6, color="white",
            fontweight="bold", ha="right", va="center", zorder=4)

    # Frame thumbnail (top-left of body)
    fx, fy, fw, fh = 0.03, 0.55, 0.30, 0.30
    if frame_path and frame_path.exists():
        ax.imshow(mpimg.imread(frame_path), extent=(fx, fx + fw, fy, fy + fh),
                  aspect="auto", zorder=2)
        ax.add_patch(patches.Rectangle((fx, fy), fw, fh,
                                       fill=False, lw=0.7, ec="#111", zorder=3))
    else:
        ax.add_patch(patches.Rectangle((fx, fy), fw, fh,
                                       fc="#E5E7EB", ec="#9CA3AF",
                                       lw=0.6, zorder=2))
        ax.text(fx + fw / 2, fy + fh / 2,
                "(raw clip not\nredistributed)",
                ha="center", va="center", fontsize=7.0, color="#6B7280", zorder=3)
    ax.text(fx + fw / 2, fy - 0.020, "context frame",
            ha="center", fontsize=6.6, color=MUTED, va="top")

    # Prior AD on the right of the frame
    tx = fx + fw + 0.025
    tw = 1 - tx - 0.03
    ax.text(tx, 0.84, "Prior AD context",
            fontsize=8.0, color=MUTED, fontweight="bold", va="top")
    yy = 0.80
    for line in ctx_lines:
        ax.text(tx, yy, "•", fontsize=8.4, color=INK, fontweight="bold", va="top")
        ax.text(tx + 0.022, yy, wrap(line, 38),
                fontsize=7.6, color=INK, va="top")
        # rough line height per wrap row
        rows = max(1, (len(line) + 37) // 38)
        yy -= 0.045 + 0.040 * (rows - 1)

    # Hidden target row (full width)
    ax.text(0.03, 0.50,
            "Hidden target AD:",
            fontsize=7.8, fontweight="bold", color=DARK_RED, va="top")
    ax.text(0.225, 0.50, wrap(target_text, 76),
            fontsize=8.0, color=INK, va="top", style="italic")

    # Options (lower half)
    yy = 0.43
    for j, opt in enumerate(options):
        letter = chr(ord("A") + j)
        is_gold = (j == correct_idx)
        if is_gold:
            ax.add_patch(FancyBboxPatch(
                (0.025, yy - 0.045), 0.95, 0.055,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                lw=0, fc=GOLD_HI, zorder=2))
        ax.text(0.040, yy, f"({letter})",
                fontsize=8.4, color=GREEN if is_gold else INK,
                fontweight="bold", va="top", zorder=3)
        ax.text(0.085, yy, wrap(opt, 56),
                fontsize=7.8, color=INK, va="top", zorder=3)
        ax.text(0.965, yy - 0.024, distractors[j],
                fontsize=6.0, color=MUTED, family="monospace",
                ha="right", va="top", zorder=3)
        if is_gold:
            ax.text(0.945, yy, "✓", fontsize=11, color=GREEN,
                    fontweight="bold", ha="right", va="top", zorder=3)
        yy -= 0.075

    # Take-away tag (bottom)
    ax.add_patch(FancyBboxPatch(
        (0.025, 0.030), 0.95, 0.075,
        boxstyle="round,pad=0.005,rounding_size=0.014",
        lw=0, fc=takeaway_color + "26",  # 15% alpha hex (24)
        zorder=2,
    ))
    ax.text(0.5, 0.067, wrap(takeaway, 95),
            ha="center", va="center", fontsize=7.4,
            color=takeaway_color, fontweight="bold", zorder=3, style="italic")


def make_filter_figure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(13.0, 8.5))
    fig.patch.set_facecolor("white")

    # Title
    t_ax = fig.add_axes([0, 0.94, 1, 0.06]); t_ax.axis("off")
    t_ax.text(0.5, 0.75,
              "Why nontriviality filtering matters",
              ha="center", va="center", fontsize=15.5, fontweight="bold", color=INK)
    t_ax.text(0.5, 0.18,
              "Construction-time filters drop items answerable without prior-AD evidence. "
              "Retained items keep the typed-distractor structure that requires forward-relevant context.",
              ha="center", va="center", fontsize=9.0, color=MUTED, style="italic")

    # Column headers
    h_ax = fig.add_axes([0, 0.88, 1, 0.04]); h_ax.axis("off")
    h_ax.text(0.252, 0.6,
              "REJECTED  (validation: distractor_quality = 0.0)",
              ha="center", va="center", fontsize=11.0, fontweight="bold", color=RED)
    h_ax.text(0.752, 0.6,
              "RETAINED  (787-item benchmark)",
              ha="center", va="center", fontsize=11.0, fontweight="bold", color=GREEN)

    # ----------------- bad_04 (top-left) -----------------
    bad4 = FE / "bad_examples/example_04"
    bad4_meta = json.loads((bad4 / "metadata.json").read_text())
    bad4_ctx = [
        "VALJEAN gets up and hauls MARIUS up onto his shoulder.",
        "With the unconscious MARIUS over his shoulder, VALJEAN wades up through the knee-deep sewerage.",
    ]
    bad4_target = "VALJEAN holds MARIUS up next to him."
    bad4_options = [
        "VALJEAN supports MARIUS while navigating the sewer.",
        "MARIUS regains consciousness and starts walking.",
        "VALJEAN drops MARIUS and runs away.",
        "VALJEAN throws MARIUS into the sewer.",
    ]
    bad4_frame = next(iter(sorted((bad4 / "frames").glob("context_02*tmid*"))), None)
    _qa_panel(fig, [0.025, 0.46, 0.450, 0.42],
              header="bad_04 · 1027_Les_Miserables · participant_update",
              header_color=RED, badge="REJECTED", badge_color=RED,
              ctx_lines=bad4_ctx, target_text=bad4_target,
              options=bad4_options, correct_idx=0,
              distractors=bad4_meta["distractor_metadata"],
              frame_path=bad4_frame,
              takeaway="(A) is a near-paraphrase of both the target AND the prior AD → answerable "
                       "from wording alone, no forward evidence required.",
              takeaway_color=RED)

    # ----------------- bad_03 (bottom-left) -----------------
    bad3 = FE / "bad_examples/example_03"
    bad3_meta = json.loads((bad3 / "metadata.json").read_text())
    bad3_ctx = [
        "MICHAEL takes a nozzle from a fuel pump, sprays gas over the back of the van, and lights it.",
    ]
    bad3_target = "The possessed flee as the van is engulfed in flames."
    bad3_options = [
        "The van bursts into flames.",
        "The possessed chase Michael away.",
        "The van explodes.",
        "Michael runs away from the van.",
    ]
    _qa_panel(fig, [0.025, 0.025, 0.450, 0.42],
              header="bad_03 · 1026_Legion · spatial_consequence",
              header_color=RED, badge="REJECTED", badge_color=RED,
              ctx_lines=bad3_ctx, target_text=bad3_target,
              options=bad3_options, correct_idx=0,
              distractors=bad3_meta["distractor_metadata"],
              frame_path=None,
              takeaway="Target is the immediate physical consequence of the prior AD (lights it → bursts into flames). "
                       "Options (A) and (C) are near-synonyms.",
              takeaway_color=RED)

    # ----------------- good_03 (top-right) -----------------
    g3 = FE / "good_examples/example_03"
    g3_meta = json.loads((g3 / "metadata.json").read_text())
    g3_ctx = ["VOLDEMORT forces HARRY to bow."]
    g3_target = "HARRY writhes in agony."
    g3_options = [
        "HARRY experiences intense pain.",
        "HARRY starts to laugh uncontrollably.",
        "HARRY suddenly gains superhuman strength.",
        "HARRY begins to float upwards.",
    ]
    g3_frame = next(iter(sorted((g3 / "frames").glob("context_01*tmid*"))), None)
    _qa_panel(fig, [0.525, 0.46, 0.450, 0.42],
              header="good_03 · 1051_Harry_Potter… · state_change",
              header_color=GREEN, badge="KEPT", badge_color=GREEN,
              ctx_lines=g3_ctx, target_text=g3_target,
              options=g3_options, correct_idx=0,
              distractors=g3_meta["distractor_metadata"],
              frame_path=g3_frame,
              takeaway="Prior AD provides a causal precondition (forced to bow) that supports the future state. "
                       "Distractors are concrete and unambiguously wrong.",
              takeaway_color=GREEN)

    # ----------------- good_02 (bottom-right) -----------------
    g2 = FE / "good_examples/example_02"
    g2_meta = json.loads((g2 / "metadata.json").read_text())
    g2_ctx = ["Mina moves towards Hansel.",
              "Mina kisses him."]
    g2_target = "They embrace passionately."
    g2_options = [
        "Mina and Hansel kiss passionately.",
        "Hansel and Gretel kiss passionately.",
        "The witch appears and chases them away.",
        "The forest becomes covered in snow.",
    ]
    g2_frame = next(iter(sorted((g2 / "frames").glob("context_02*tmid*"))), None)
    _qa_panel(fig, [0.525, 0.025, 0.450, 0.42],
              header="good_02 · 3031_HANSEL_GRETEL… · participant_update",
              header_color=GREEN, badge="KEPT", badge_color=GREEN,
              ctx_lines=g2_ctx, target_text=g2_target,
              options=g2_options, correct_idx=0,
              distractors=g2_meta["distractor_metadata"],
              frame_path=g2_frame,
              takeaway="Named participants in prior AD make (B) a clean entity_swapped distractor. "
                       "(C) and (D) probe movie-prior shortcuts.",
              takeaway_color=GREEN)

    pdf = OUT / "fig_filter.pdf"
    png = OUT / "fig_filter.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}  +  {png}")


# ---------------------------------------------------------------------------
# Figure 3 — AD-source substitution (Tracy / The Roommate, bc_01)
# ---------------------------------------------------------------------------

def make_ad_source_figure():
    bc1 = FE / "baseline_comparison_examples/example_01"
    options = [
        "Tracy gets out of bed and walks across the room.",
        "Tracy sits back down on the bed and continues to rub her brow.",
        "Tracy picks up a book and starts reading.",
        "Tracy talks to someone off-screen.",
    ]
    correct_idx = 0
    distractor_meta = ["correct", "already_happened", "plausible_unsupported", "contradicts_context"]
    target_sentence = "Tracy groggily gets up and crosses the dim room."
    sources = [
        ("Human MAD-eval AD",
         "In her room, Tracy sits up in bed and rubs her brow.",
         True,
         "Tracy gets out of bed and walks across the room.",
         GREEN, GOLD_HI,
         "preserves the entity Tracy and the location bed / room",
         "kept_keywords:  Tracy · bed · room"),
        ("NarrAD",
         "Rebecca watches Sarah sleep, then wipes her face thoughtfully.",
         False,
         "Tracy sits back down on the bed and continues to rub her brow.",
         RED, RED_HI,
         "drops Tracy → wrong protagonist (Rebecca / Sarah)",
         "dropped_keywords:  Tracy"),
        ("AutoAD-Zero",
         "Tracy covers her face with her hands in front of the mirror.",
         False,
         "Tracy sits back down on the bed and continues to rub her brow.",
         RED, RED_HI,
         "drops bed → wrong location (mirror)",
         "dropped_keywords:  bed · room"),
    ]
    ctx_frames = sorted((bc1 / "frames").glob("context_01__*"))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(13.5, 8.0))
    fig.patch.set_facecolor("white")

    # Title
    t_ax = fig.add_axes([0, 0.94, 1, 0.06]); t_ax.axis("off")
    t_ax.text(0.5, 0.78,
              "AD-source substitution: same item, three AD sources, three predictions",
              ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
    t_ax.text(0.5, 0.18,
              "Question, options, and selected context positions are fixed — only the AD source filling those positions changes "
              "(item: 3074_THE_ROOMMATE / participant_update).",
              ha="center", va="center", fontsize=9.0, color=MUTED, style="italic")

    # ===== Top: question + frames =====
    p_top = setup_panel(fig, [0.020, 0.66, 0.96, 0.275], ec=BLUE,
                        header="Shared item   ·   selected adaptive context: 1 clip",
                        header_color=BLUE)

    # Frame strip (left half)
    for i, fp in enumerate(ctx_frames[:3]):
        x0 = 0.030 + i * 0.115
        p_top.imshow(mpimg.imread(fp),
                     extent=(x0, x0 + 0.105, 0.18, 0.66),
                     aspect="auto", zorder=2)
        p_top.add_patch(patches.Rectangle((x0, 0.18), 0.105, 0.48,
                                          fill=False, lw=0.7, ec="#111", zorder=3))
        p_top.text(x0 + 0.052, 0.13, ["t=0", "t=mid", "t=end"][i],
                   ha="center", fontsize=7.5, color=MUTED, va="top")
    p_top.text(0.030 + 0.155, 0.075,
               "context frames (visualization only — answerer is text-only)",
               fontsize=7.5, color=MUTED, va="top", style="italic", ha="center")

    # Question + options (right half, top)
    qx = 0.40
    p_top.text(qx, 0.74, "Q:  What happens next?",
               fontsize=11.5, fontweight="bold", color=INK, va="top")
    yy = 0.62
    for j, opt in enumerate(options):
        letter = chr(ord("A") + j)
        is_gold = (j == correct_idx)
        if is_gold:
            p_top.add_patch(FancyBboxPatch(
                (qx - 0.005, yy - 0.085), 0.395, 0.095,
                boxstyle="round,pad=0.005,rounding_size=0.018",
                lw=0, fc=GOLD_HI, zorder=2))
        p_top.text(qx, yy, f"({letter})",
                   fontsize=9.0, color=GREEN if is_gold else INK,
                   fontweight="bold", va="top", zorder=3)
        p_top.text(qx + 0.030, yy, wrap(opt, 50),
                   fontsize=8.5, color=INK, va="top", zorder=3)
        if is_gold:
            p_top.text(qx + 0.385, yy - 0.005, "✓",
                       fontsize=14, color=GREEN, fontweight="bold",
                       ha="right", va="top", zorder=3)
        p_top.text(qx + 0.385, yy - 0.060, distractor_meta[j],
                   fontsize=6.6, color=MUTED, family="monospace",
                   ha="right", va="top", zorder=3)
        yy -= 0.115

    # Hidden target line
    p_top.text(0.815, 0.74, "Hidden target AD",
               fontsize=8.6, color=DARK_RED, fontweight="bold", va="top")
    p_top.add_patch(FancyBboxPatch(
        (0.812, 0.20), 0.180, 0.50,
        boxstyle="round,pad=0.008,rounding_size=0.020",
        lw=1.0, ec=RED, fc=RED_HI, zorder=2))
    p_top.text(0.902, 0.50, "?  ?", ha="center", va="center",
               fontsize=22, color=RED, fontweight="bold", zorder=3)
    p_top.text(0.902, 0.34,
               wrap(f"“{target_sentence}”", 22),
               ha="center", fontsize=7.0, color=DARK_RED, va="top",
               style="italic", zorder=3)
    p_top.text(0.902, 0.10,
               "withheld during evaluation",
               ha="center", fontsize=7.2, color=MUTED, va="top", style="italic")

    # ===== Bottom: three AD-source rows =====
    bottom_top = 0.62
    bottom_bot = 0.060
    row_h = (bottom_top - bottom_bot - 2 * 0.020) / 3
    for i, (label, ad_text, is_correct, pred, color, hi_color, drop_note, kw_note) in enumerate(sources):
        ry = bottom_top - row_h - i * (row_h + 0.020)
        # Panel for source row
        ax_row = setup_panel(fig, [0.020, ry, 0.96, row_h], ec=color, fc=hi_color,
                             header=None)

        # Source label badge (left)
        ax_row.add_patch(FancyBboxPatch(
            (0.012, 0.30), 0.16, 0.40,
            boxstyle="round,pad=0.0,rounding_size=0.018",
            lw=0, fc=color, zorder=2))
        ax_row.text(0.092, 0.50, label, fontsize=10, color="white",
                    fontweight="bold", ha="center", va="center", zorder=3)

        # Middle: AD text + drop note
        midx = 0.190
        ax_row.text(midx, 0.84, "AD filling the selected context position:",
                    fontsize=7.6, color=MUTED, fontweight="bold", va="top")
        ax_row.text(midx, 0.65, wrap(ad_text, 70),
                    fontsize=9.5, color=INK, va="top")
        ax_row.text(midx, 0.30, drop_note,
                    fontsize=8.2, color=color, fontweight="bold",
                    style="italic", va="top")
        ax_row.text(midx, 0.13, kw_note,
                    fontsize=7.0, color=MUTED, family="monospace", va="top")

        # Right: prediction box
        px = 0.690
        pw = 0.295
        ax_row.add_patch(FancyBboxPatch(
            (px, 0.10), pw, 0.80,
            boxstyle="round,pad=0.005,rounding_size=0.018",
            lw=1.2, ec=color, fc="white", zorder=2))
        ax_row.text(px + 0.012, 0.84, "Prospective QA prediction:",
                    fontsize=7.6, color=MUTED, fontweight="bold", va="top")
        verdict = "✓  correct" if is_correct else "✗  incorrect"
        ax_row.text(px + 0.012, 0.66, verdict,
                    fontsize=11, color=color, fontweight="bold", va="top")
        ax_row.text(px + 0.012, 0.43, wrap(pred, 38),
                    fontsize=8.6, color=INK, va="top")

    # Footer
    foot = fig.add_axes([0.0, 0.0, 1.0, 0.045]); foot.axis("off")
    foot.text(0.5, 0.55,
              "Same question, options, answerer (Qwen2.5-VL).  "
              "Only the AD source filling the selected context changes.  "
              "Human MAD-eval AD preserves the entity (Tracy) and location (bed / room) needed for the correct answer.",
              ha="center", va="center", fontsize=8.4, color=INK, style="italic")

    pdf = OUT / "fig_ad_source.pdf"
    png = OUT / "fig_ad_source.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}  +  {png}")


def main() -> None:
    print("Building paper figures →", OUT)
    make_teaser()
    make_filter_figure()
    make_ad_source_figure()
    print("Done.")


if __name__ == "__main__":
    main()

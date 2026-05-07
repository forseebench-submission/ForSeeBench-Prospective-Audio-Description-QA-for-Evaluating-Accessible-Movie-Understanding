"""Generate additional NeurIPS-style figures for ForSeeBench using the
curated examples in `figure_examples/`.

Outputs (PDF + PNG) → `figures/paper_figures/`:
  - fig_good_gallery.{pdf,png}    — six diverse good examples in a 2x3 grid
  - fig_ad_source_grid.{pdf,png}  — three baseline_comparison items, each
                                    with all three AD sources
  - fig_target_types.{pdf,png}    — one example per target type (5 panels)
  - fig_evidence_chain.{pdf,png}  — keyword-bridge chains for four examples
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path("/jumbo/jinlab/Sally/ForSeeBench")
FE = ROOT / "figure_examples"
OUT = ROOT / "figures/paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

# Palette (kept identical to make_paper_figures.py for visual consistency)
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
TEAL = "#1B7F8E"
PRIOR_HI = "#FFF3D2"
GOLD_HI = "#DDEFD2"
RED_HI = "#FBE3E1"
BLUE_HI = "#E8F0FB"
PURPLE_HI = "#EFE7F4"


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def setup_panel(fig, rect, *, ec=BLUE, fc="#FFFFFF", lw=1.4, header=None,
                header_color=None, header_fontsize=9.0):
    ax = fig.add_axes(rect); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
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
            linewidth=0, edgecolor=bar_color, facecolor=bar_color, zorder=2,
        ))
        ax.add_patch(patches.Rectangle((0.005, 1 - bar_h - 0.005),
                                       0.99, bar_h * 0.30,
                                       fc=bar_color, ec=bar_color, lw=0, zorder=2))
        ax.text(0.025, 1 - bar_h / 2 - 0.005, header,
                fontsize=header_fontsize, color="white",
                fontweight="bold", va="center", zorder=3)
    return ax


def first_frame(folder: Path, pattern: str = "context_01*tmid*") -> Path | None:
    matches = sorted(folder.glob(f"frames/{pattern}.jpg"))
    return matches[0] if matches else None


def all_context_frames(folder: Path) -> list[Path]:
    return sorted(folder.glob("frames/context_*.jpg"))


def normalize_words(text: str) -> set[str]:
    stop = {"the", "a", "an", "and", "of", "in", "on", "at", "to", "is", "are",
            "with", "for", "into", "from", "by", "as", "his", "her", "their",
            "they", "he", "she", "it", "its", "this", "that", "these", "those",
            "be", "been", "but", "off", "out", "up", "over", "while", "then",
            "next", "now", "him", "them", "she's", "he's"}
    return {w for w in re.findall(r"[a-zA-Z']+", text.lower())
            if w not in stop and len(w) > 2}


def load_meta(folder: Path) -> dict:
    return json.loads((folder / "metadata.json").read_text())


# ---------------------------------------------------------------------------
# Figure: good gallery (2 rows × 3 cols)
# ---------------------------------------------------------------------------

GOOD_GALLERY_SPECS = [
    # (folder, prior_lines, target, options, correct_idx, distractors,
    #  bridge_keyword, target_type_label)
    ("good_examples/example_03",
     ["VOLDEMORT forces HARRY to bow."],
     "HARRY writhes in agony.",
     ["HARRY experiences intense pain.",
      "HARRY starts to laugh uncontrollably.",
      "HARRY suddenly gains superhuman strength.",
      "HARRY begins to float upwards."],
     0, ["correct", "unrelated", "plausible_unsupported", "contradicts_context"],
     "forces … to bow", "state_change"),

    ("good_examples/example_02",
     ["Mina moves towards Hansel.", "Mina kisses him."],
     "They embrace passionately.",
     ["Mina and Hansel kiss passionately.",
      "Hansel and Gretel kiss passionately.",
      "The witch appears and chases them away.",
      "The forest becomes covered in snow."],
     0, ["correct", "entity_swapped", "plausible_unsupported", "contradicts_context"],
     "Mina … Hansel", "participant_update"),

    ("good_examples/example_04",
     ["From a watch tower, Nantz and Imlay look over the extent of the destruction."],
     "Nantz peers through binoculars.",
     ["Nantz uses binoculars to observe the damage.",
      "Imlay takes over observing the damage.",
      "The destruction stops spreading.",
      "A new character appears on the scene."],
     0, ["correct", "entity_swapped", "plausible_unsupported", "contradicts_context"],
     "watch tower / destruction", "state_change"),

    ("good_examples/example_06",
     ["… which leads to a massive walk-in closet."],
     "Matty enters.",
     ["Matty walks into the walk-in closet.",
      "The walk-in closet suddenly disappears.",
      "Matty finds a secret room behind a bookshelf.",
      "A ghost appears in the walk-in closet."],
     0, ["correct", "unrelated", "plausible_unsupported", "contradicts_context"],
     "walk-in closet", "participant_update"),

    ("good_examples/example_07",
     ["As Stephen goes, Sarah smiles after him.",
      "Later, Sarah steps off the elevator on her floor."],
     "Crossing toward her bed.",
     ["Sarah walks towards her bed.",
      "Stephen enters the bedroom.",
      "Sarah sits on the couch.",
      "The doorbell rings."],
     0, ["correct", "already_happened", "plausible_unsupported", "contradicts_context"],
     "Sarah · elevator → floor", "spatial_consequence"),

    ("good_examples/example_10",
     ["MOODY places the spider in front of HERMIONE.",
      "HERMIONE shakes her head."],
     "The spider dies.",
     ["Hermione observes the dead spider.",
      "The spider moves.",
      "Hermione talks to the spider.",
      "The spider is alive."],
     0, ["correct", "already_happened", "plausible_unsupported", "contradicts_context"],
     "spider · Hermione", "object_reveal"),
]


def _short_movie(movie: str, limit: int = 24) -> str:
    """Shorten LSMDC movie names for figure headers."""
    if len(movie) <= limit:
        return movie
    # Drop the leading 4-digit LSMDC prefix and trim words.
    body = movie
    if re.match(r"^\d{4}_", body):
        body = body[5:]
    # Replace underscores with spaces for readability and truncate.
    pretty = body.replace("_", " ")
    if len(pretty) <= limit:
        return pretty
    return pretty[:limit - 1].rstrip() + "…"


def _gallery_panel(fig, rect, *, folder, prior_lines, target, options,
                   correct_idx, distractors, bridge, target_type):
    folder_p = FE / folder
    name_id = folder.split("/")[-1]                 # e.g., example_03
    movie = _short_movie(load_meta(folder_p)["movie"])
    ax = setup_panel(fig, rect, ec=GREEN, fc="#FFFFFF", lw=1.2,
                     header=f"{name_id}  ·  {movie}  ·  {target_type}",
                     header_color=GREEN, header_fontsize=8.0)

    # Frame thumbnail (top of body)
    fp = first_frame(folder_p)
    fx, fy, fw, fh = 0.030, 0.640, 0.230, 0.215
    if fp:
        ax.imshow(mpimg.imread(fp), extent=(fx, fx + fw, fy, fy + fh),
                  aspect="auto", zorder=2)
        ax.add_patch(patches.Rectangle((fx, fy), fw, fh,
                                       fill=False, lw=0.7, ec="#111", zorder=3))
    else:
        ax.add_patch(patches.Rectangle((fx, fy), fw, fh, fc="#E5E7EB",
                                       ec="#9CA3AF", lw=0.6, zorder=2))
    ax.text(fx + fw / 2, fy - 0.018, "context frame", ha="center",
            fontsize=6.6, color=MUTED, va="top")

    # Prior AD on the right of the frame
    tx = fx + fw + 0.025
    ax.text(tx, 0.85, "Prior AD context",
            fontsize=7.4, color=MUTED, fontweight="bold", va="top")
    yy = 0.81
    for line in prior_lines:
        ax.text(tx, yy, "•", fontsize=7.6, color=INK, fontweight="bold", va="top")
        ax.text(tx + 0.022, yy, wrap(line, 50),
                fontsize=7.4, color=INK, va="top")
        rows = max(1, (len(line) + 49) // 50)
        yy -= 0.045 + 0.038 * (rows - 1)

    # Bridge keyword note (orange chip)
    ax.add_patch(FancyBboxPatch(
        (tx - 0.005, 0.575), 0.755, 0.045,
        boxstyle="round,pad=0.004,rounding_size=0.014",
        lw=0, fc=PRIOR_HI, zorder=2,
    ))
    ax.text(tx + 0.005, 0.605, "bridge: ", fontsize=7.0,
            color=ORANGE, fontweight="bold", va="top", zorder=3)
    ax.text(tx + 0.075, 0.605, bridge, fontsize=7.6,
            color=DARK_RED, fontweight="bold", va="top", style="italic", zorder=3)

    # Hidden target line
    ax.text(0.030, 0.515, "Hidden target AD:", fontsize=7.4,
            color=DARK_RED, fontweight="bold", va="top")
    ax.text(0.235, 0.515, wrap(target, 75), fontsize=7.6,
            color=INK, va="top", style="italic")

    # Question + options
    ax.text(0.030, 0.435, "Q: What happens next?", fontsize=8.4,
            fontweight="bold", color=INK, va="top")
    yy = 0.380
    for j, opt in enumerate(options):
        letter = chr(ord("A") + j)
        is_gold = (j == correct_idx)
        if is_gold:
            ax.add_patch(FancyBboxPatch(
                (0.020, yy - 0.058), 0.96, 0.068,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                lw=0, fc=GOLD_HI, zorder=2,
            ))
        ax.text(0.035, yy, f"({letter})",
                fontsize=8.0, color=GREEN if is_gold else INK,
                fontweight="bold", va="top", zorder=3)
        ax.text(0.080, yy, wrap(opt, 60), fontsize=7.5,
                color=INK, va="top", zorder=3)
        if is_gold:
            ax.text(0.965, yy, "✓", fontsize=11, color=GREEN,
                    fontweight="bold", ha="right", va="top", zorder=3)
        ax.text(0.965, yy - 0.030, distractors[j],
                fontsize=5.8, color=MUTED, family="monospace",
                ha="right", va="top", zorder=3)
        yy -= 0.075


def make_good_gallery():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(15.0, 10.0))
    fig.patch.set_facecolor("white")
    t_ax = fig.add_axes([0, 0.94, 1, 0.06]); t_ax.axis("off")
    t_ax.text(0.5, 0.78,
              "ForSeeBench gallery: forward-relevant evidence across target types",
              ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
    t_ax.text(0.5, 0.22,
              "Each panel shows a retained ForSeeBench item: prior AD context, hidden target AD, "
              "the four-option question, and the bridging keyword that ties prior context to the correct answer.",
              ha="center", va="center", fontsize=9.0, color=MUTED, style="italic")

    rows, cols = 2, 3
    margin_x, margin_y = 0.018, 0.030
    grid_top, grid_bot = 0.93, 0.030
    grid_h = grid_top - grid_bot
    panel_w = (1 - (cols + 1) * margin_x) / cols
    panel_h = (grid_h - (rows + 1) * margin_y) / rows

    for idx, spec in enumerate(GOOD_GALLERY_SPECS[:rows * cols]):
        r = idx // cols
        c = idx % cols
        x0 = margin_x + c * (panel_w + margin_x)
        y0 = grid_bot + margin_y + (rows - 1 - r) * (panel_h + margin_y)
        _gallery_panel(
            fig, [x0, y0, panel_w, panel_h],
            folder=spec[0], prior_lines=spec[1], target=spec[2],
            options=spec[3], correct_idx=spec[4], distractors=spec[5],
            bridge=spec[6], target_type=spec[7],
        )

    foot = fig.add_axes([0, 0.0, 1, 0.025]); foot.axis("off")
    foot.text(0.5, 0.5,
              "All six items are sourced verbatim from the released benchmark "
              "(`data/processed/all_movies/eval_all10.jsonl`).  "
              "Distractor labels follow the typed taxonomy: already_happened, entity_swapped, "
              "plausible_unsupported, contradicts_context, unrelated.",
              ha="center", va="center", fontsize=7.6, color=INK, style="italic")

    pdf = OUT / "fig_good_gallery.pdf"
    png = OUT / "fig_good_gallery.png"
    fig.savefig(pdf, bbox_inches="tight"); fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}  +  {png}")


# ---------------------------------------------------------------------------
# Figure: AD-source grid across multiple items
# ---------------------------------------------------------------------------

AD_GRID_SPECS = [
    # (folder_name, target, options, correct_idx,
    #  human_ad, narrad_ad, autoad_ad,
    #  human_pred, narrad_pred, autoad_pred,
    #  drop_summary)
    ("baseline_comparison_examples/example_01",
     "Tracy groggily gets up and crosses the dim room.",
     ["Tracy gets out of bed and walks across the room.",
      "Tracy sits back down on the bed and continues to rub her brow.",
      "Tracy picks up a book and starts reading.",
      "Tracy talks to someone off-screen."], 0,
     "In her room, Tracy sits up in bed and rubs her brow.",
     "Rebecca watches Sarah sleep, then wipes her face thoughtfully.",
     "Tracy covers her face with her hands in front of the mirror.",
     "Tracy gets out of bed and walks across the room.",
     "Tracy sits back down on the bed and continues to rub her brow.",
     "Tracy sits back down on the bed and continues to rub her brow.",
     "NarrAD swaps the protagonist; AutoAD-Zero changes the location."),

    ("baseline_comparison_examples/example_03",
     "Hansel then wraps a chain around her neck and hoists her up using the oven door as a pulley.",
     ["Hansel uses the oven door as a pulley to lift the witch.",
      "The witch escapes by using the oven door as a ladder.",
      "Hansel throws the witch into the fire.",
      "The witch transforms into a bird."], 0,
     "Hansel heaves a hanging cauldron into the witches face knocking her down.",
     "The gingerbread house is in a tumultuous state.",
     "Hansel runs through the maze, looking around frantically.",
     "Hansel uses the oven door as a pulley to lift the witch.",
     "The witch transforms into a bird.",
     "The witch escapes by using the oven door as a ladder.",
     "Generated AD strips out both the actor (Hansel) and the object of action (witch / cauldron)."),

    ("baseline_comparison_examples/example_05",
     "And takes the receiver.",
     ["MERRILL takes the receiver.",
      "The car roof collapses.",
      "GRAHAM takes the receiver.",
      "The receiver vibrates."], 0,
     "MERRILL reaches out his hand. · MERRILL and GRAHAM both hold the receiver across the car roof.",
     "Merrill leans towards the rear, focusing intently on something.",
     "Graham reaches out and pulls Sara closer to him.",
     "MERRILL takes the receiver.",
     "GRAHAM takes the receiver.",
     "GRAHAM takes the receiver.",
     "Without a named co-actor (MERRILL holding the receiver), the answerer can't pick the right entity."),
]


def _ad_grid_row(fig, rect, *, folder, target, options, correct_idx,
                 human_ad, narrad_ad, autoad_ad,
                 human_pred, narrad_pred, autoad_pred,
                 drop_summary):
    folder_p = FE / folder
    movie = _short_movie(load_meta(folder_p)["movie"])
    ax = setup_panel(fig, rect, ec=BLUE, fc="#FFFFFF", lw=1.2,
                     header=f"{folder.split('/')[-1]} · {movie}",
                     header_color=BLUE, header_fontsize=8.6)

    # Question + correct answer (top-left)
    ax.text(0.025, 0.85, "Q: What happens next?",
            fontsize=8.4, fontweight="bold", color=INK, va="top")
    # Hidden target
    ax.text(0.025, 0.78, "Hidden target AD:", fontsize=7.0,
            color=DARK_RED, fontweight="bold", va="top")
    ax.text(0.025, 0.745, wrap(target, 78),
            fontsize=7.4, color=INK, va="top", style="italic")

    # Correct option (full width banner)
    correct = options[correct_idx]
    ax.add_patch(FancyBboxPatch(
        (0.025, 0.53), 0.95, 0.078,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        lw=0, fc=GOLD_HI, zorder=2,
    ))
    ax.text(0.040, 0.585, "✓ correct option:",
            fontsize=7.4, color=GREEN, fontweight="bold", va="top", zorder=3)
    ax.text(0.180, 0.585, correct,
            fontsize=8.0, color=INK, va="top", fontweight="bold", zorder=3)
    # Drop summary
    ax.add_patch(FancyBboxPatch(
        (0.025, 0.448), 0.95, 0.060,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        lw=0, fc=PRIOR_HI, zorder=2,
    ))
    ax.text(0.040, 0.490, "evidence gap:",
            fontsize=7.0, color=ORANGE, fontweight="bold", va="top", zorder=3)
    ax.text(0.150, 0.490, drop_summary,
            fontsize=7.6, color=DARK_RED, va="top", style="italic", zorder=3)

    # Three columns for the three sources
    col_w = (0.95 - 2 * 0.013) / 3
    sources = [
        ("Human MAD-eval AD", human_ad, human_pred,
         human_pred == correct, GREEN),
        ("NarrAD",            narrad_ad, narrad_pred,
         narrad_pred == correct, RED),
        ("AutoAD-Zero",       autoad_ad, autoad_pred,
         autoad_pred == correct, RED),
    ]
    for j, (label, ad_text, pred, is_correct, color) in enumerate(sources):
        x0 = 0.025 + j * (col_w + 0.013)
        # Source header strip
        ax.add_patch(FancyBboxPatch(
            (x0, 0.36), col_w, 0.058,
            boxstyle="round,pad=0.0,rounding_size=0.014",
            lw=0, fc=color, zorder=2,
        ))
        ax.text(x0 + col_w / 2, 0.389, label,
                ha="center", va="center", fontsize=8.2,
                color="white", fontweight="bold", zorder=3)
        # AD text below
        ax.add_patch(FancyBboxPatch(
            (x0, 0.180), col_w, 0.175,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            lw=0.7, ec=color, fc="white", zorder=2,
        ))
        ax.text(x0 + 0.010, 0.340, "AD filling selected position:",
                fontsize=6.4, color=MUTED, fontweight="bold", va="top")
        ax.text(x0 + 0.010, 0.310, wrap(ad_text, 38),
                fontsize=7.0, color=INK, va="top")

        # Prediction box
        verdict = "✓ correct" if is_correct else "✗ incorrect"
        ax.add_patch(FancyBboxPatch(
            (x0, 0.030), col_w, 0.140,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            lw=1.0, ec=color,
            fc=GOLD_HI if is_correct else RED_HI, zorder=2,
        ))
        ax.text(x0 + 0.010, 0.155, "prospective QA prediction:",
                fontsize=6.4, color=MUTED, fontweight="bold", va="top")
        ax.text(x0 + 0.010, 0.125, verdict,
                fontsize=8.2, color=color, fontweight="bold", va="top")
        ax.text(x0 + 0.010, 0.095, wrap(pred, 32),
                fontsize=7.0, color=INK, va="top")


def make_ad_source_grid():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(15.0, 11.5))
    fig.patch.set_facecolor("white")
    t_ax = fig.add_axes([0, 0.95, 1, 0.05]); t_ax.axis("off")
    t_ax.text(0.5, 0.7,
              "AD-source substitution across three items",
              ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
    t_ax.text(0.5, 0.20,
              "Same questions, options, and selected positions — only the AD source filling those positions changes. "
              "Human MAD-eval AD predicts correctly; NarrAD and AutoAD-Zero mispredict because the bridging entity or location "
              "is dropped from the generated context.",
              ha="center", va="center", fontsize=9.0, color=MUTED, style="italic")

    rows = len(AD_GRID_SPECS)
    margin = 0.020
    top, bot = 0.94, 0.030
    h_total = top - bot
    panel_h = (h_total - (rows + 1) * margin) / rows
    for i, spec in enumerate(AD_GRID_SPECS):
        y0 = top - margin - (i + 1) * panel_h - i * margin
        _ad_grid_row(fig, [margin, y0, 1 - 2 * margin, panel_h],
                     folder=spec[0], target=spec[1], options=spec[2],
                     correct_idx=spec[3],
                     human_ad=spec[4], narrad_ad=spec[5], autoad_ad=spec[6],
                     human_pred=spec[7], narrad_pred=spec[8], autoad_pred=spec[9],
                     drop_summary=spec[10])

    foot = fig.add_axes([0, 0.0, 1, 0.025]); foot.axis("off")
    foot.text(0.5, 0.5,
              "Predictions are read directly from the released QA outputs in "
              "`outputs/evaluation/{ground_truth,narrad,autoad_zero}/all_10_movies_adaptive_source_neutral/predictions.jsonl`.",
              ha="center", va="center", fontsize=7.6, color=INK, style="italic")

    pdf = OUT / "fig_ad_source_grid.pdf"
    png = OUT / "fig_ad_source_grid.png"
    fig.savefig(pdf, bbox_inches="tight"); fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}  +  {png}")


# ---------------------------------------------------------------------------
# Figure: target-type diversity — one panel per type
# ---------------------------------------------------------------------------

TARGET_TYPE_SPECS = [
    # (folder, target_type, color, hi_color, prior, target, correct)
    ("good_examples/example_02", "participant_update", BLUE, BLUE_HI,
     "Mina moves towards Hansel.\nMina kisses him.",
     "They embrace passionately.",
     "Mina and Hansel kiss passionately."),
    ("good_examples/example_09", "action_transition", PURPLE, PURPLE_HI,
     "GRAHAM pushes his hand against the door lit by MORGAN torch.",
     "GRAHAM then turns to face the others.",
     "GRAHAM prepares to enter the room."),
    ("good_examples/example_03", "state_change", GREEN, GOLD_HI,
     "VOLDEMORT forces HARRY to bow.",
     "HARRY writhes in agony.",
     "HARRY experiences intense pain."),
    ("good_examples/example_07", "spatial_consequence", TEAL, "#DCEEEC",
     "As Stephen goes, Sarah smiles after him.\nLater, Sarah steps off the elevator on her floor.",
     "Crossing toward her bed.",
     "Sarah walks towards her bed."),
    ("good_examples/example_10", "object_reveal", ORANGE, "#FCEEDB",
     "MOODY places the spider in front of HERMIONE.\nHERMIONE shakes her head.",
     "The spider dies.",
     "Hermione observes the dead spider."),
]


def _type_panel(fig, rect, *, folder, target_type, color, hi_color,
                prior, target, correct):
    folder_p = FE / folder
    movie = _short_movie(load_meta(folder_p)["movie"], limit=32)
    ax = setup_panel(fig, rect, ec=color, fc="#FFFFFF", lw=1.3,
                     header=f"target type:  {target_type}",
                     header_color=color, header_fontsize=9.2)
    # Movie / id sub-label
    ax.text(0.025, 0.825, f"{folder.split('/')[-1]} · {movie}",
            fontsize=7.4, color=MUTED, va="top", style="italic")
    # Frame
    fp = first_frame(folder_p)
    fx, fy, fw, fh = 0.025, 0.555, 0.30, 0.22
    if fp:
        ax.imshow(mpimg.imread(fp), extent=(fx, fx + fw, fy, fy + fh),
                  aspect="auto", zorder=2)
        ax.add_patch(patches.Rectangle((fx, fy), fw, fh,
                                       fill=False, lw=0.7, ec="#111", zorder=3))
    # Prior AD
    tx = fx + fw + 0.030
    ax.text(tx, 0.78, "Prior AD context", fontsize=7.6,
            color=MUTED, fontweight="bold", va="top")
    ax.add_patch(FancyBboxPatch(
        (tx - 0.005, 0.500), 0.62, 0.265,
        boxstyle="round,pad=0.005,rounding_size=0.015",
        lw=0, fc=hi_color, zorder=1,
    ))
    ax.text(tx + 0.010, 0.745,
            "\n".join(["• " + l for l in prior.split("\n")]),
            fontsize=7.8, color=INK, va="top", linespacing=1.30)
    # Hidden target row
    ax.text(0.025, 0.475, "Hidden target AD:",
            fontsize=7.6, color=DARK_RED, fontweight="bold", va="top")
    ax.text(0.025, 0.420, wrap(f"“{target}”", 78),
            fontsize=8.0, color=INK, va="top", style="italic")
    # Correct answer banner
    ax.add_patch(FancyBboxPatch(
        (0.025, 0.220), 0.95, 0.115,
        boxstyle="round,pad=0.005,rounding_size=0.014",
        lw=1.2, ec=color, fc=hi_color, zorder=2,
    ))
    ax.text(0.040, 0.310, "✓ correct option:",
            fontsize=8.0, color=color, fontweight="bold", va="top", zorder=3)
    ax.text(0.040, 0.265, wrap(correct, 80),
            fontsize=8.4, color=INK, va="top", fontweight="bold", zorder=3)
    # Footer interpretation
    interpretation = {
        "participant_update":   "A new participant enters or becomes the focus of the scene.",
        "action_transition":    "A character transitions from one action to the next.",
        "state_change":         "A visible state of a character or object changes.",
        "spatial_consequence":  "A spatial movement / location change implied by the prior AD.",
        "object_reveal":        "An object becomes narratively important / changes status.",
    }
    ax.text(0.025, 0.155, interpretation.get(target_type, ""),
            fontsize=7.4, color=MUTED, va="top", style="italic")


def make_target_types():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(15.0, 9.5))
    fig.patch.set_facecolor("white")
    t_ax = fig.add_axes([0, 0.94, 1, 0.06]); t_ax.axis("off")
    t_ax.text(0.5, 0.78,
              "Target-type taxonomy: ForSeeBench is broader than action prediction",
              ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
    t_ax.text(0.5, 0.22,
              "Five of the six retained target types, one example each. "
              "(visible_text_update, n=7, omitted for compact layout.)",
              ha="center", va="center", fontsize=9.0, color=MUTED, style="italic")

    # 5 panels in a 2x3 grid; bottom-right is the legend / counts panel
    rows, cols = 2, 3
    margin_x, margin_y = 0.020, 0.030
    grid_top, grid_bot = 0.93, 0.030
    grid_h = grid_top - grid_bot
    panel_w = (1 - (cols + 1) * margin_x) / cols
    panel_h = (grid_h - (rows + 1) * margin_y) / rows

    for idx, spec in enumerate(TARGET_TYPE_SPECS):
        r = idx // cols
        c = idx % cols
        x0 = margin_x + c * (panel_w + margin_x)
        y0 = grid_bot + margin_y + (rows - 1 - r) * (panel_h + margin_y)
        _type_panel(fig, [x0, y0, panel_w, panel_h],
                    folder=spec[0], target_type=spec[1],
                    color=spec[2], hi_color=spec[3],
                    prior=spec[4], target=spec[5], correct=spec[6])

    # 6th cell: counts table
    last_idx = len(TARGET_TYPE_SPECS)
    r = last_idx // cols; c = last_idx % cols
    x0 = margin_x + c * (panel_w + margin_x)
    y0 = grid_bot + margin_y + (rows - 1 - r) * (panel_h + margin_y)
    legend = setup_panel(fig, [x0, y0, panel_w, panel_h],
                         ec=MUTED, fc="#FBFBFC", lw=1.0,
                         header="benchmark composition (787 items)",
                         header_color=MUTED, header_fontsize=8.6)
    counts = [
        ("participant_update",  "379"),
        ("action_transition",   "196"),
        ("state_change",        "147"),
        ("object_reveal",       " 30"),
        ("spatial_consequence", " 28"),
        ("visible_text_update", "  7"),
    ]
    legend.text(0.07, 0.825, "Target type",
                fontsize=8.2, fontweight="bold", color=INK, va="top")
    legend.text(0.78, 0.825, "Count",
                fontsize=8.2, fontweight="bold", color=INK, ha="center", va="top")
    yy = 0.745
    for tt, n in counts:
        legend.text(0.07, yy, tt, fontsize=8.2, color=INK,
                    family="monospace", va="top")
        legend.text(0.78, yy, n, fontsize=8.2, color=INK,
                    family="monospace", ha="center", va="top")
        yy -= 0.090

    legend.text(0.07, 0.15, "Adaptive context length",
                fontsize=8.2, fontweight="bold", color=INK, va="top")
    legend.text(0.07, 0.105,
                "median = 1 clip · mean = 2.01 · max = 9",
                fontsize=8.0, color=MUTED, va="top", style="italic")

    pdf = OUT / "fig_target_types.pdf"
    png = OUT / "fig_target_types.png"
    fig.savefig(pdf, bbox_inches="tight"); fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}  +  {png}")


# ---------------------------------------------------------------------------
# Figure: evidence-chain (keyword bridge)
# ---------------------------------------------------------------------------

EVIDENCE_CHAIN_SPECS = [
    # (folder, prior_lines (list), bridge_phrase, target, correct,
    #  arrow_color, panel_color, panel_hi)
    ("good_examples/example_03",
     ["VOLDEMORT forces HARRY to bow."],
     "forces … to bow",
     "HARRY writhes in agony.",
     "HARRY experiences intense pain.",
     ORANGE, GREEN, GOLD_HI),
    ("good_examples/example_06",
     ["… which leads to a massive walk-in closet."],
     "walk-in closet",
     "Matty enters.",
     "Matty walks into the walk-in closet.",
     ORANGE, BLUE, BLUE_HI),
    ("good_examples/example_04",
     ["From a watch tower, Nantz and Imlay look over the extent of the destruction."],
     "watch tower · destruction",
     "Nantz peers through binoculars.",
     "Nantz uses binoculars to observe the damage.",
     ORANGE, TEAL, "#DCEEEC"),
    ("good_examples/example_10",
     ["MOODY places the spider in front of HERMIONE.",
      "HERMIONE shakes her head."],
     "spider · Hermione",
     "The spider dies.",
     "Hermione observes the dead spider.",
     ORANGE, PURPLE, PURPLE_HI),
]


def _chain_row(fig, rect, *, folder, prior_lines, bridge, target, correct,
               arrow_color, panel_color, panel_hi):
    folder_p = FE / folder
    movie = _short_movie(load_meta(folder_p)["movie"], limit=32)
    ax = setup_panel(fig, rect, ec=panel_color, fc="#FFFFFF", lw=1.2,
                     header=f"{folder.split('/')[-1]} · {movie}",
                     header_color=panel_color, header_fontsize=8.6)

    # Three-step pill chain across the panel: PRIOR → BRIDGE → ANSWER
    pad = 0.020
    band_y, band_h = 0.30, 0.45
    pill_w = 0.30
    # Pill 1: Prior AD
    ax.add_patch(FancyBboxPatch(
        (pad, band_y), pill_w, band_h,
        boxstyle="round,pad=0.005,rounding_size=0.020",
        lw=1.0, ec=panel_color, fc=panel_hi, zorder=2,
    ))
    ax.text(pad + 0.012, band_y + band_h - 0.040,
            "Prior AD context",
            fontsize=7.8, color=panel_color,
            fontweight="bold", va="top")
    ax.text(pad + 0.012, band_y + band_h - 0.080,
            wrap("\n".join("• " + l for l in prior_lines), 32),
            fontsize=7.6, color=INK, va="top", linespacing=1.30)

    # Pill 2: Bridge keyword
    pill2_x = pad + pill_w + 0.045
    pill2_w = 0.22
    ax.add_patch(FancyBboxPatch(
        (pill2_x, band_y + band_h * 0.20), pill2_w, band_h * 0.60,
        boxstyle="round,pad=0.005,rounding_size=0.022",
        lw=1.4, ec=arrow_color, fc=PRIOR_HI, zorder=2,
    ))
    ax.text(pill2_x + pill2_w / 2, band_y + band_h * 0.65,
            "bridge keyword",
            ha="center", va="top", fontsize=7.4, color=arrow_color,
            fontweight="bold")
    ax.text(pill2_x + pill2_w / 2, band_y + band_h * 0.42,
            bridge, ha="center", va="top",
            fontsize=8.6, color=DARK_RED, fontweight="bold", style="italic")

    # Pill 3: Correct answer (right)
    pill3_x = pill2_x + pill2_w + 0.045
    pill3_w = 1 - pill3_x - pad
    ax.add_patch(FancyBboxPatch(
        (pill3_x, band_y), pill3_w, band_h,
        boxstyle="round,pad=0.005,rounding_size=0.020",
        lw=1.4, ec=panel_color, fc=GOLD_HI, zorder=2,
    ))
    ax.text(pill3_x + 0.012, band_y + band_h - 0.040,
            "✓  correct option (paraphrases hidden target)",
            fontsize=7.8, color=panel_color, fontweight="bold", va="top")
    ax.text(pill3_x + 0.012, band_y + band_h - 0.080,
            wrap(correct, 38),
            fontsize=8.0, color=INK, va="top", fontweight="bold")

    # Hidden target line below pill 3
    ax.text(pill3_x + 0.012, band_y - 0.005,
            "hidden target AD:",
            fontsize=6.8, color=DARK_RED, fontweight="bold", va="top")
    ax.text(pill3_x + 0.012, band_y - 0.045,
            wrap(f"“{target}”", 40),
            fontsize=7.4, color=INK, va="top", style="italic")

    # Arrows between pills
    ax.add_patch(FancyArrowPatch(
        (pad + pill_w + 0.005, band_y + band_h / 2),
        (pill2_x - 0.005, band_y + band_h / 2),
        arrowstyle="-|>", mutation_scale=14, lw=1.6, color=arrow_color, zorder=4,
    ))
    ax.add_patch(FancyArrowPatch(
        (pill2_x + pill2_w + 0.005, band_y + band_h / 2),
        (pill3_x - 0.005, band_y + band_h / 2),
        arrowstyle="-|>", mutation_scale=14, lw=1.6, color=arrow_color, zorder=4,
    ))

    # Frame thumbnail (top-right corner of panel)
    fp = first_frame(folder_p)
    if fp:
        fx, fy, fw, fh = 0.78, 0.83, 0.20, 0.040  # tiny thumbnail near header? skip
        # Better: top-left thumbnail above the chain
        fx, fy, fw, fh = pad, 0.79, 0.20, 0.075
        ax.imshow(mpimg.imread(fp), extent=(fx, fx + fw, fy, fy + fh),
                  aspect="auto", zorder=2)
        ax.add_patch(patches.Rectangle((fx, fy), fw, fh,
                                       fill=False, lw=0.6, ec="#111", zorder=3))
        ax.text(fx + fw + 0.012, fy + fh / 2,
                "context frame (visualization only — answerer is text-only)",
                fontsize=7.0, color=MUTED, va="center", style="italic")


def make_evidence_chain():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(13.0, 11.0))
    fig.patch.set_facecolor("white")
    t_ax = fig.add_axes([0, 0.94, 1, 0.06]); t_ax.axis("off")
    t_ax.text(0.5, 0.78,
              "Forward-relevant evidence: a single keyword bridges prior AD to the correct option",
              ha="center", va="center", fontsize=14.5, fontweight="bold", color=INK)
    t_ax.text(0.5, 0.22,
              "Each row reads left-to-right: prior AD context  →  bridging keyword  →  correct option (paraphrases the hidden target). "
              "The bridge keyword is a single object, participant, or location.",
              ha="center", va="center", fontsize=9.0, color=MUTED, style="italic")

    rows = len(EVIDENCE_CHAIN_SPECS)
    margin = 0.022
    top, bot = 0.93, 0.030
    h_total = top - bot
    panel_h = (h_total - (rows + 1) * margin) / rows
    for i, spec in enumerate(EVIDENCE_CHAIN_SPECS):
        y0 = top - margin - (i + 1) * panel_h - i * margin
        _chain_row(fig, [margin, y0, 1 - 2 * margin, panel_h],
                   folder=spec[0], prior_lines=spec[1], bridge=spec[2],
                   target=spec[3], correct=spec[4],
                   arrow_color=spec[5], panel_color=spec[6], panel_hi=spec[7])

    foot = fig.add_axes([0, 0, 1, 0.025]); foot.axis("off")
    foot.text(0.5, 0.5,
              "These chains illustrate the construction-time guarantee: every retained item carries an evidence span "
              "(`evidence[].span`) that overlaps a content keyword in the correct option.",
              ha="center", va="center", fontsize=7.6, color=INK, style="italic")

    pdf = OUT / "fig_evidence_chain.pdf"
    png = OUT / "fig_evidence_chain.png"
    fig.savefig(pdf, bbox_inches="tight"); fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}  +  {png}")


def main() -> None:
    print("Building extra paper figures →", OUT)
    make_good_gallery()
    make_ad_source_grid()
    make_target_types()
    make_evidence_chain()
    print("Done.")


if __name__ == "__main__":
    main()

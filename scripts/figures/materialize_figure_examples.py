"""Materialize per-example folders for `figure_examples/` from the manifest.

Reads `figure_examples/_selection_manifest.json` and writes:
  figure_examples/good_examples/example_XX/...
  figure_examples/bad_examples/example_XX/...
  figure_examples/baseline_comparison_examples/example_XX/...

Each folder contains the human-readable text files and a `metadata.json`
referencing the original benchmark item id, paths to source clips, and the
relevant prediction outcomes. Frame extraction is done by a separate script.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path("/jumbo/jinlab/Sally/ForSeeBench")
EVAL_JSONL = ROOT / "data/processed/all_movies/eval_all10.jsonl"
RESULTS_CSV = ROOT / "Results.csv"
PRED_DIR = ROOT / "outputs/evaluation"
ADAPTIVE = {
    "Human MAD-eval AD": PRED_DIR / "ground_truth/all_10_movies_adaptive_source_neutral/predictions.jsonl",
    "NarrAD": PRED_DIR / "narrad/all_10_movies_adaptive_source_neutral/predictions.jsonl",
    "AutoAD-Zero": PRED_DIR / "autoad_zero/all_10_movies_adaptive_source_neutral/predictions.jsonl",
}
NOCTX = PRED_DIR / "predicc/all_10_movies_run_fixed_k0/no_context/k_0/predictions.jsonl"
FIXED = {
    ("Human MAD-eval AD", 8): PRED_DIR / "predicc/all_10_movies_run_fixed_k0/ground_truth/k_8/predictions.jsonl",
    ("NarrAD", 8): PRED_DIR / "predicc/all_10_movies_run_fixed_k0/narrad/k_8/predictions.jsonl",
    ("AutoAD-Zero", 8): PRED_DIR / "predicc/all_10_movies_run_fixed_k0/autoad_zero/k_8/predictions.jsonl",
}

OUT = ROOT / "figure_examples"
GOOD_DIR = OUT / "good_examples"
BAD_DIR = OUT / "bad_examples"
BC_DIR = OUT / "baseline_comparison_examples"
for d in (GOOD_DIR, BAD_DIR, BC_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as h:
        return [json.loads(line) for line in h if line.strip()]


def index_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with open(path) as h:
        for line in h:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def csv_index() -> dict:
    idx = {}
    with open(RESULTS_CSV) as h:
        for row in csv.DictReader(h):
            idx[(row["movie"], row["start"], row["end"])] = row
    return idx


def clip_id_to_ts(clip_id: str) -> tuple[str, str]:
    parts = clip_id.split("__")
    if len(parts) < 2:
        return ("", "")
    rest = parts[1]
    bits = rest.split("_")
    pat = re.compile(r"^\d+p\d+$")
    ts = [b for b in bits if pat.match(b)]
    if len(ts) >= 2:
        return ts[0].replace("p", "."), ts[-1].replace("p", ".")
    return ("", "")


def _ts_variants(t: str) -> list[str]:
    """Generate timestamp string variants (handle trailing-zero stripping)."""
    out = {t}
    if t.endswith(".0"):
        out.add(t[:-2])
    if "." in t:
        try:
            f = float(t)
            if f == int(f):
                out.add(str(int(f)))
            out.add(str(f))
        except ValueError:
            pass
    return list(out)


def csv_lookup(csv_idx, movie_folder: str, clip_id: str) -> dict | None:
    s, e = clip_id_to_ts(clip_id)
    if not s:
        return None
    movie_candidates = [movie_folder]
    if re.match(r"^\d{4}_", movie_folder):
        movie_candidates.append(movie_folder[5:])
        movie_candidates.append(movie_folder[5:].upper())
    for k in movie_candidates:
        for sv in _ts_variants(s):
            for ev in _ts_variants(e):
                row = csv_idx.get((k, sv, ev))
                if row is not None:
                    return row
    # Fuzzy fallback: tolerate small (≤0.3s) timestamp drift between MAD-eval
    # and Results.csv. Only used when the exact key misses, and we still
    # require both endpoints to be within tolerance.
    try:
        s_f, e_f = float(s), float(e)
    except ValueError:
        return None
    best = None
    best_err = 0.4
    for (m_k, sv, ev), row in csv_idx.items():
        if m_k not in movie_candidates:
            continue
        try:
            sv_f, ev_f = float(sv), float(ev)
        except ValueError:
            continue
        err = max(abs(sv_f - s_f), abs(ev_f - e_f))
        if err < best_err:
            best_err = err
            best = row
    return best


def fmt_options(options: list[str], answer_idx: int, distractor_meta: list[str] | None = None) -> str:
    lines = []
    for i, opt in enumerate(options):
        marker = "*" if i == answer_idx else " "
        meta = ""
        if distractor_meta and i < len(distractor_meta):
            meta = f"  [{distractor_meta[i]}]"
        lines.append(f"  ({chr(65+i)}){marker} {opt}{meta}")
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.write_text(content)


def materialize_good(items_by_id, manifest, preds, csv_idx) -> list[dict]:
    summaries = []
    for i, g in enumerate(manifest["good"]):
        idx = i + 1
        it = items_by_id[g["id"]]
        edir = GOOD_DIR / f"example_{idx:02d}"
        edir.mkdir(exist_ok=True)
        (edir / "frames").mkdir(exist_ok=True)

        ctx_lines = []
        for j, (cid, sent) in enumerate(zip(it.get("context_clip_ids") or [], it.get("context_sentences") or [])):
            ctx_lines.append(f"[{j+1}] (clip {cid}) {sent}")
        write_text(edir / "context.txt", "\n".join(ctx_lines))

        write_text(edir / "question.txt",
                   f"{it['question']}\n\n{fmt_options(it['options'], it['answer_idx'], it.get('distractor_metadata'))}\n")
        write_text(edir / "target_ad.txt", f"[hidden] {it['target_sentence']}\n")

        # Evidence chain
        ev_lines = ["EVIDENCE CHAIN", "=" * 14, "", "Prior clue (from selected adaptive context):"]
        ev_spans = it.get("evidence") or []
        for ev in ev_spans:
            ev_lines.append(f"  - {ev.get('span','')} (clip {ev.get('clip_id','')}, type={ev.get('evidence_type','')})")
        ev_lines.append("")
        ev_lines.append(f"Hidden target event:\n  - {it['target_sentence']}")
        ev_lines.append("")
        ev_lines.append(f"Question: {it['question']}")
        ev_lines.append(f"Correct answer: {it['answer_text']}")
        ev_lines.append("")
        ev_lines.append("Why each distractor is wrong:")
        for j, opt in enumerate(it["options"]):
            if j == it["answer_idx"]:
                continue
            label = (it.get("distractor_metadata") or [None]*len(it["options"]))[j]
            ev_lines.append(f"  - ({chr(65+j)}) {opt}  [{label}]")
        write_text(edir / "evidence_chain.txt", "\n".join(ev_lines) + "\n")

        why = [
            f"Why example_{idx:02d} is a good teaser candidate:",
            f"- Movie: {it['movie']}",
            f"- Target type: {it.get('target_type')}",
            f"- Selected adaptive context length: {len(it.get('context_clip_ids') or [])} clip(s)",
            f"- Predictability: {it.get('predictability')}, expectedness: {it.get('expectedness')}",
            f"- Heuristic score: {g['score']}",
            "- Heuristic notes: " + ", ".join(g.get("notes") or []),
            "",
            "Outcomes (per AD source):",
        ]
        for src, p in preds.items():
            row = p.get(g["id"])
            if row is None:
                why.append(f"  - {src} adaptive: (no prediction)")
            else:
                why.append(f"  - {src} adaptive: {'correct' if row.get('correct') else 'incorrect'}; predicted: {row.get('prediction_text')}")
        no_p = preds.get("__noctx__", {}).get(g["id"])
        if no_p is not None:
            why.append(f"  - No-context (k=0): {'correct' if no_p.get('correct') else 'incorrect'}; predicted: {no_p.get('prediction_text')}")
        write_text(edir / "why_good.txt", "\n".join(why) + "\n")

        # Figure panel
        panel = [
            f"# Figure panel — `good_examples/example_{idx:02d}`",
            "",
            f"**Movie ID (LSMDC):** `{it['movie']}` ({it.get('target_type')})",
            "",
            "**Prior AD context (visible to model):**",
            "",
        ]
        for j, sent in enumerate(it.get("context_sentences") or []):
            panel.append(f"> [{j+1}] {sent}")
        panel += [
            "",
            f"**Question:** {it['question']}",
            "",
            "**Options:**",
            "",
        ]
        for j, opt in enumerate(it["options"]):
            star = " ★ correct" if j == it["answer_idx"] else ""
            panel.append(f"- ({chr(65+j)}) {opt}{star}")
        panel += [
            "",
            f"**Hidden target AD (kept off-screen in the panel):** _{it['target_sentence']}_",
            "",
            "**Suggested keyword highlights in the prior AD:**",
        ]
        for ev in ev_spans:
            panel.append(f"- `{ev.get('span','')}`")
        panel.append("")
        panel.append("**Frame strip:** show 2–3 frames from each context clip (left), the question + options (center). Hidden-target frames go behind a `?` token.")
        write_text(edir / "figure_panel.md", "\n".join(panel) + "\n")

        # Metadata
        ctx_clip_paths = [c.get("video_path") for c in (it.get("context") or [])]
        target_path = (it.get("target") or {}).get("video_path")
        meta = {
            "id": it["id"],
            "movie": it["movie"],
            "target_clip_id": it["target_clip_id"],
            "target_type": it.get("target_type"),
            "reasoning_type": it.get("reasoning_type"),
            "continuity_type": it.get("continuity_type"),
            "predictability": it.get("predictability"),
            "expectedness": it.get("expectedness"),
            "context_clip_ids": it.get("context_clip_ids"),
            "context_video_paths": ctx_clip_paths,
            "target_video_path": target_path,
            "evidence": it.get("evidence"),
            "distractor_metadata": it.get("distractor_metadata"),
            "selection_score": g["score"],
            "selection_notes": g.get("notes"),
            "outcomes": {
                src: ({"correct": p.get(g["id"], {}).get("correct"),
                       "prediction_text": p.get(g["id"], {}).get("prediction_text")} if p.get(g["id"]) else None)
                for src, p in preds.items() if src != "__noctx__"
            },
            "no_context_outcome": (
                {"correct": preds.get("__noctx__", {}).get(g["id"], {}).get("correct"),
                 "prediction_text": preds.get("__noctx__", {}).get(g["id"], {}).get("prediction_text")}
                if preds.get("__noctx__", {}).get(g["id"]) else None
            ),
        }
        (edir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        summaries.append({
            "example": f"example_{idx:02d}",
            "id": it["id"],
            "movie": it["movie"],
            "target_type": it.get("target_type"),
            "score": g["score"],
            "ctx_paths": ctx_clip_paths,
            "target_path": target_path,
        })
    return summaries


def materialize_bad(items_by_id, manifest, preds) -> list[dict]:
    summaries = []
    for i, b in enumerate(manifest["bad"]):
        idx = i + 1
        edir = BAD_DIR / f"example_{idx:02d}"
        edir.mkdir(exist_ok=True)
        (edir / "frames").mkdir(exist_ok=True)

        ctx = b.get("context") or []
        ctx_paths = [c.get("video_path") for c in ctx if isinstance(c, dict)]
        target_path = (b.get("target") or {}).get("video_path") if isinstance(b.get("target"), dict) else None

        # Context
        ctx_sentences = b.get("context_sentences") or []
        ctx_clip_ids = [c.get("clip_id") for c in ctx if isinstance(c, dict)]
        # If there are no sentences but ctx contains them, fall back
        if not ctx_sentences and ctx:
            ctx_sentences = [c.get("audio_description", "") for c in ctx if isinstance(c, dict)]
        ctx_lines = []
        for j, (cid, sent) in enumerate(zip(ctx_clip_ids or ["?"]*len(ctx_sentences), ctx_sentences)):
            ctx_lines.append(f"[{j+1}] (clip {cid}) {sent}")
        write_text(edir / "context.txt", "\n".join(ctx_lines) or "(no prior context)\n")

        if b.get("options"):
            write_text(edir / "question.txt",
                       f"What happens next?\n\n{fmt_options(b['options'], b['answer_idx'], b.get('distractor_metadata'))}\n")
        else:
            write_text(edir / "question.txt", "(question not generated for this rejected block)\n")

        target_sentence = b.get("target_sentence") or ""
        write_text(edir / "target_ad.txt", f"[hidden target] {target_sentence}\n")

        why = [f"Why example_{idx:02d} is a bad / filtered example:", ""]
        why.append(f"Source: {b.get('source')}")
        why.append(f"Movie: {b.get('movie')}")
        why.append(f"Item id: {b.get('id')}")
        why.append("")
        vmeta = b.get("validation_metadata") or {}
        if vmeta:
            why.append("Validation metadata:")
            for k, v in vmeta.items():
                why.append(f"  - {k}: {v}")
            why.append("")
        # Reasoned summary
        reasons = []
        if (vmeta or {}).get("distractor_quality") == 0.0:
            reasons.append("- Distractor quality scored 0.0 in Qwen validation: distractors are weak, redundant, or stylistically distinguishable from the correct answer.")
        # Detect option-0 paraphrase
        opts = b.get("options") or []
        if opts and target_sentence:
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, opts[b.get("answer_idx", 0)].lower(), target_sentence.lower()).ratio()
            if ratio > 0.45:
                reasons.append(f"- Option (A) is a near-paraphrase of the hidden target (ratio≈{ratio:.2f}); a model can match wording rather than reason from prior AD.")
        # Trivial pattern: distractors all entity_swapped or all-already_happened
        dmeta = b.get("distractor_metadata") or []
        non_correct = [d for j, d in enumerate(dmeta) if j != b.get("answer_idx", 0)]
        if non_correct and len(set(non_correct)) <= 2:
            reasons.append(f"- Distractors are concentrated in only {len(set(non_correct))} type(s): {sorted(set(non_correct))} — limited diagnostic coverage.")
        # Already-happened pattern
        if "already_happened" in dmeta and target_sentence and ctx_sentences:
            for sent in ctx_sentences:
                if sent.lower().split()[:3] == target_sentence.lower().split()[:3]:
                    reasons.append("- Target essentially restates the most recent prior AD line (redundant continuation).")
                    break
        if not reasons:
            reasons.append("- Filtered for failing one or more validation gates (see metadata).")
        why.extend(reasons)
        write_text(edir / "why_bad.txt", "\n".join(why) + "\n")

        panel = [
            f"# Figure panel — `bad_examples/example_{idx:02d}`",
            "",
            f"**Movie ID:** `{b.get('movie')}` (target type: {b.get('target_type')})",
            "",
            "**Prior AD context:**",
            "",
        ]
        for j, sent in enumerate(ctx_sentences):
            panel.append(f"> [{j+1}] {sent}")
        if not ctx_sentences:
            panel.append("> (none)")
        panel += ["", f"**Hidden target AD:** _{target_sentence}_", ""]
        if opts:
            panel.append("**Options as generated by the pipeline:**")
            panel.append("")
            for j, opt in enumerate(opts):
                star = " ★ correct" if j == b.get("answer_idx", 0) else ""
                panel.append(f"- ({chr(65+j)}) {opt}{star}")
            panel.append("")
        panel.append("**Why filtered:** " + "; ".join(r.lstrip("- ") for r in reasons))
        write_text(edir / "figure_panel.md", "\n".join(panel) + "\n")

        meta = {
            "id": b.get("id"),
            "movie": b.get("movie"),
            "source": b.get("source"),
            "target_type": b.get("target_type"),
            "expectedness": b.get("expectedness"),
            "predictability": b.get("predictability"),
            "context_clip_ids": ctx_clip_ids,
            "context_video_paths": ctx_paths,
            "target_video_path": target_path,
            "validation_metadata": vmeta,
            "distractor_metadata": b.get("distractor_metadata"),
        }
        (edir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        if not ctx_paths and not target_path:
            (edir / "frames" / "frames_unavailable.txt").write_text("Source clips not present for this rejected item.\n")
        summaries.append({
            "example": f"example_{idx:02d}",
            "id": b.get("id"),
            "movie": b.get("movie"),
            "ctx_paths": ctx_paths,
            "target_path": target_path,
        })
    return summaries


def materialize_baseline(items_by_id, manifest, preds, csv_idx) -> list[dict]:
    summaries = []
    for i, b in enumerate(manifest["baseline_comparison"]):
        idx = i + 1
        it = items_by_id[b["id"]]
        edir = BC_DIR / f"example_{idx:02d}"
        edir.mkdir(exist_ok=True)
        (edir / "frames").mkdir(exist_ok=True)

        # Build per-source context text from Results.csv at the same context clip ids.
        per_source_ctx = {"reference": [], "narrad": [], "autoad": []}
        for cid in it.get("context_clip_ids") or []:
            row = csv_lookup(csv_idx, it["movie"], cid)
            if row is None:
                per_source_ctx["reference"].append(f"[clip {cid}] (not in Results.csv)")
                per_source_ctx["narrad"].append(f"[clip {cid}] (not in Results.csv)")
                per_source_ctx["autoad"].append(f"[clip {cid}] (not in Results.csv)")
                continue
            per_source_ctx["reference"].append(f"[clip {cid}] {row.get('ground_truth','')}")
            per_source_ctx["narrad"].append(f"[clip {cid}] {row.get('NarrAD','')}")
            per_source_ctx["autoad"].append(f"[clip {cid}] {row.get('AutoAD-Zero','')}")
        write_text(edir / "reference_context.txt", "\n".join(per_source_ctx["reference"]) + "\n")
        write_text(edir / "narrad_context.txt", "\n".join(per_source_ctx["narrad"]) + "\n")
        write_text(edir / "autoad_context.txt", "\n".join(per_source_ctx["autoad"]) + "\n")

        write_text(edir / "question.txt",
                   f"{it['question']}\n\n{fmt_options(it['options'], it['answer_idx'], it.get('distractor_metadata'))}\n")
        write_text(edir / "target_ad.txt", f"[hidden] {it['target_sentence']}\n")

        # Outcomes
        outcomes = {}
        for src, p in preds.items():
            if src == "__noctx__":
                continue
            r = p.get(b["id"])
            if r is None:
                outcomes[src] = None
            else:
                outcomes[src] = {
                    "correct": r.get("correct"),
                    "prediction_text": r.get("prediction_text"),
                }
        no_p = preds.get("__noctx__", {}).get(b["id"])
        outcomes["No-context (k=0)"] = (
            {"correct": no_p.get("correct"), "prediction_text": no_p.get("prediction_text")} if no_p else None
        )
        (edir / "model_results.json").write_text(json.dumps(outcomes, indent=2, default=str))

        # Why comparison is good
        why = [
            f"Why example_{idx:02d} is a strong baseline-comparison example:",
            "",
            f"- Movie: {it['movie']}",
            f"- Target type: {it.get('target_type')}",
            f"- Adaptive context length: {len(it.get('context_clip_ids') or [])} clip(s)",
            f"- Heuristic score: {b['score']:.2f}",
            "- Heuristic notes: " + ", ".join(b.get("notes") or []),
            "",
            "Outcomes:",
        ]
        for src, val in outcomes.items():
            if val is None:
                why.append(f"  - {src}: (no prediction)")
            else:
                why.append(f"  - {src}: {'correct' if val['correct'] else 'incorrect'} -> {val['prediction_text']}")
        why += ["", "Why human MAD-eval AD wins:"]
        # Identify keywords in human AD that are absent from generated context.
        ans_words = set(re.findall(r"[A-Za-z']+", it["answer_text"].lower()))
        for cid in it.get("context_clip_ids") or []:
            row = csv_lookup(csv_idx, it["movie"], cid)
            if row is None:
                continue
            gt = row.get("ground_truth", "")
            narrad = row.get("NarrAD", "")
            autoad = row.get("AutoAD-Zero", "")
            gt_words = set(re.findall(r"[A-Za-z']+", gt.lower()))
            narrad_words = set(re.findall(r"[A-Za-z']+", narrad.lower()))
            autoad_words = set(re.findall(r"[A-Za-z']+", autoad.lower()))
            bridge = (gt_words & ans_words) - {"the", "a", "an", "and", "of", "to", "her", "his", "their"}
            narrad_drops = bridge - narrad_words
            autoad_drops = bridge - autoad_words
            why.append(f"  - clip {cid} bridge keywords (in human AD ∩ correct answer): {sorted(bridge) or '(none)'}")
            if narrad_drops:
                why.append(f"    NarrAD drops: {sorted(narrad_drops)}")
            if autoad_drops:
                why.append(f"    AutoAD-Zero drops: {sorted(autoad_drops)}")
        write_text(edir / "why_comparison_is_good.txt", "\n".join(why) + "\n")

        panel = [
            f"# Figure panel — `baseline_comparison_examples/example_{idx:02d}`",
            "",
            f"**Movie:** `{it['movie']}` (target: {it.get('target_type')})",
            "",
            f"**Question:** {it['question']}",
            "",
            "**Options:**",
            "",
        ]
        for j, opt in enumerate(it["options"]):
            star = " ★ correct" if j == it["answer_idx"] else ""
            panel.append(f"- ({chr(65+j)}) {opt}{star}")
        panel += ["", f"**Hidden target AD:** _{it['target_sentence']}_", ""]
        panel.append("**Three AD sources fill the same selected context positions:**")
        panel.append("")
        panel.append("| Source | Context filled into selected position(s) | Adaptive prediction |")
        panel.append("|---|---|---|")
        for src_label, key in [("Human MAD-eval AD", "reference"), ("NarrAD", "narrad"), ("AutoAD-Zero", "autoad")]:
            ctx_text = " ".join(per_source_ctx[key])
            pred = outcomes.get(src_label) or {}
            pred_str = pred.get("prediction_text") or "(no prediction)"
            verdict = "✓" if pred.get("correct") else "✗"
            panel.append(f"| {src_label} | {ctx_text} | {verdict} {pred_str} |")
        panel += [
            "",
            "**Suggested keyword highlights:** entities/objects that appear in the human AD but disappear from NarrAD / AutoAD-Zero on the same clip.",
            "",
            "**Frame strip:** 2–3 frames from each context clip on the left; question + per-source predictions on the right; target frames behind a `?` token.",
        ]
        write_text(edir / "figure_panel.md", "\n".join(panel) + "\n")

        ctx_clip_paths = [c.get("video_path") for c in (it.get("context") or [])]
        target_path = (it.get("target") or {}).get("video_path")
        meta = {
            "id": it["id"],
            "movie": it["movie"],
            "target_type": it.get("target_type"),
            "context_clip_ids": it.get("context_clip_ids"),
            "context_video_paths": ctx_clip_paths,
            "target_video_path": target_path,
            "evidence": it.get("evidence"),
            "selection_score": b["score"],
            "selection_notes": b.get("notes"),
            "per_source_context": per_source_ctx,
            "outcomes": outcomes,
        }
        (edir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        summaries.append({
            "example": f"example_{idx:02d}",
            "id": it["id"],
            "movie": it["movie"],
            "ctx_paths": ctx_clip_paths,
            "target_path": target_path,
        })
    return summaries


def main() -> int:
    items_by_id = {it["id"]: it for it in load_jsonl(EVAL_JSONL)}
    manifest = json.loads((OUT / "_selection_manifest.json").read_text())

    preds = {src: index_predictions(p) for src, p in ADAPTIVE.items()}
    preds["__noctx__"] = index_predictions(NOCTX)

    csv_idx = csv_index()

    good = materialize_good(items_by_id, manifest, preds, csv_idx)
    bad = materialize_bad(items_by_id, manifest, preds)
    bc = materialize_baseline(items_by_id, manifest, preds, csv_idx)

    summary = {"good": good, "bad": bad, "baseline_comparison": bc}
    (OUT / "_materialize_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: len(v) for k, v in summary.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

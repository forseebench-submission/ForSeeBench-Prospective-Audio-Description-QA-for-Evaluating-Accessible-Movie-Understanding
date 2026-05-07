"""Select curated figure examples for the ForSeeBench paper.

Runs the heuristic selection described in `figure_examples_plan.md`:
  - 10 good teaser examples
  - 5 bad / filtered examples (from interim/per_movie/*/rejected_examples.jsonl)
  - 6 baseline-comparison examples (Human-AD adaptive correct,
    NarrAD/AutoAD-Zero adaptive incorrect)

Writes a single manifest JSON describing the selection. A separate script
(`materialize_figure_examples.py`) consumes that manifest and writes the
per-example folders + frames.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/jumbo/jinlab/Sally/ForSeeBench")
EVAL_JSONL = ROOT / "data/processed/all_movies/eval_all10.jsonl"
RESULTS_CSV = ROOT / "Results.csv"
PRED_DIR = ROOT / "outputs/evaluation"
ADAPTIVE_GT = PRED_DIR / "ground_truth/all_10_movies_adaptive_source_neutral/predictions.jsonl"
ADAPTIVE_NARRAD = PRED_DIR / "narrad/all_10_movies_adaptive_source_neutral/predictions.jsonl"
ADAPTIVE_AUTOAD = PRED_DIR / "autoad_zero/all_10_movies_adaptive_source_neutral/predictions.jsonl"
NOCTX = PRED_DIR / "predicc/all_10_movies_run_fixed_k0/no_context/k_0/predictions.jsonl"
FIXED_GT8 = PRED_DIR / "predicc/all_10_movies_run_fixed_k0/ground_truth/k_8/predictions.jsonl"
FIXED_NARRAD8 = PRED_DIR / "predicc/all_10_movies_run_fixed_k0/narrad/k_8/predictions.jsonl"
FIXED_AUTOAD8 = PRED_DIR / "predicc/all_10_movies_run_fixed_k0/autoad_zero/k_8/predictions.jsonl"

INTERIM_PER_MOVIE = ROOT / "data/interim/per_movie"
OUT_DIR = ROOT / "figure_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "_selection_manifest.json"

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "is", "are",
    "with", "for", "into", "from", "by", "as", "his", "her", "their", "they",
    "he", "she", "it", "its", "this", "that", "these", "those", "be", "been",
    "but", "off", "out", "up", "over", "while", "then", "next", "now",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_predictions_indexed(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with open(path) as handle:
        for line in handle:
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def normalize_words(text: str) -> set[str]:
    if not text:
        return set()
    return {
        tok
        for tok in re.findall(r"[a-zA-Z']+", text.lower())
        if tok not in STOPWORDS and len(tok) > 2
    }


def token_overlap(a: str, b: str) -> float:
    A = normalize_words(a)
    B = normalize_words(b)
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, min(len(A), len(B)))


def evidence_supports_answer(evidence_list: list[dict], answer_text: str) -> bool:
    """True if any evidence span shares meaningful tokens with the correct answer."""
    answer_tokens = normalize_words(answer_text)
    if not answer_tokens:
        return False
    for ev in evidence_list or []:
        span = ev.get("span") or ""
        span_tokens = normalize_words(span)
        if span_tokens & answer_tokens:
            return True
    return False


def load_results_csv() -> dict:
    """Index Results.csv by (movie, start_str, end_str) → row dict.

    The clip_id format used elsewhere is `<movie>__<startp>_<endp>` (dots replaced
    with `p`). We index by a normalized key derived from the timestamp strings.
    """
    by_movie_ts = {}
    with open(RESULTS_CSV) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["movie"], row["start"], row["end"])
            by_movie_ts[key] = row
    return by_movie_ts


def clip_id_to_ts(clip_id: str) -> tuple[str, str]:
    # `1005_Signs__225p1_227p561` -> ('225.1', '227.561')
    parts = clip_id.split("__")
    if len(parts) < 2:
        return ("", "")
    rest = parts[1]
    bits = rest.split("_")
    # bits could be ['225p1', '227p561'] for two halves
    if len(bits) == 2:
        s, e = bits
    else:
        # Sometimes there are >2 underscore separated tokens, but ts uses 'p' for dot.
        # Re-merge: timestamps always look like 123p456
        ts_pattern = re.compile(r"^\d+p\d+$")
        ts_tokens = [b for b in bits if ts_pattern.match(b)]
        if len(ts_tokens) >= 2:
            s, e = ts_tokens[0], ts_tokens[-1]
        else:
            return ("", "")
    return (s.replace("p", "."), e.replace("p", "."))


def normalize_movie_for_csv(movie_folder: str) -> str:
    """Map mad-eval-videos folder names to the `movie` column in Results.csv."""
    # Results.csv movie field examples: 'HARRY_POTTER_AND_THE_GOBLET_OF_FIRE',
    # '1005_Signs', etc. Drop the 4-digit LSMDC prefix when the folder has it.
    if re.match(r"^\d{4}_", movie_folder):
        return movie_folder[5:]
    return movie_folder


def csv_lookup(csv_idx: dict, movie_folder: str, clip_id: str) -> dict | None:
    s, e = clip_id_to_ts(clip_id)
    if not s or not e:
        return None
    csv_movie = normalize_movie_for_csv(movie_folder)
    # Try exact then with movie folder string and original.
    for movie_key in (csv_movie, movie_folder, movie_folder.upper(),
                      movie_folder.upper().replace("_", "_")):
        row = csv_idx.get((movie_key, s, e))
        if row is not None:
            return row
    return None


def main() -> int:
    print("Loading dataset and predictions...", file=sys.stderr)
    items = load_jsonl(EVAL_JSONL)
    items_by_id = {it["id"]: it for it in items}
    print(f"  {len(items)} eval items", file=sys.stderr)

    pred_gt = load_predictions_indexed(ADAPTIVE_GT)
    pred_narrad = load_predictions_indexed(ADAPTIVE_NARRAD)
    pred_autoad = load_predictions_indexed(ADAPTIVE_AUTOAD)
    pred_noctx = load_predictions_indexed(NOCTX)
    pred_fixed_gt8 = load_predictions_indexed(FIXED_GT8)
    pred_fixed_narrad8 = load_predictions_indexed(FIXED_NARRAD8)
    pred_fixed_autoad8 = load_predictions_indexed(FIXED_AUTOAD8)
    print(
        f"  preds: gt={len(pred_gt)} narrad={len(pred_narrad)} "
        f"autoad={len(pred_autoad)} k0={len(pred_noctx)}",
        file=sys.stderr,
    )

    csv_idx = load_results_csv()

    # ------------------------------------------------------------------
    # 1) Bad / filtered examples — read interim/per_movie/*/rejected_examples.jsonl
    # ------------------------------------------------------------------
    bad_records = []
    for movie_folder in sorted(os.listdir(INTERIM_PER_MOVIE)):
        path = INTERIM_PER_MOVIE / movie_folder / "rejected_examples.jsonl"
        if not path.exists():
            continue
        with open(path) as handle:
            for line in handle:
                rec = json.loads(line)
                rec["_source"] = "rejected_examples"
                rec["_movie_folder"] = movie_folder
                bad_records.append(rec)
    print(f"  bad: {len(bad_records)} rejected examples", file=sys.stderr)

    # If we have fewer than 5, supplement with retained items where the no-context
    # baseline already gets it correct AND answer paraphrases the target.
    if len(bad_records) < 5:
        supplemental = []
        for it in items:
            noctx = pred_noctx.get(it["id"])
            if not noctx or not noctx.get("correct"):
                continue
            overlap = token_overlap(it["answer_text"], it["target_sentence"])
            if overlap >= 0.7:
                rec = dict(it)
                rec["_source"] = "near_paraphrase"
                rec["_movie_folder"] = it["movie"]
                supplemental.append((overlap, rec))
        supplemental.sort(reverse=True, key=lambda x: x[0])
        for _, r in supplemental[: 5 - len(bad_records)]:
            bad_records.append(r)
    print(f"  bad after supplement: {len(bad_records)}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 2) Good teaser examples
    # ------------------------------------------------------------------
    good_scored = []
    for it in items:
        score = 0.0
        notes = []

        gt_p = pred_gt.get(it["id"])
        if gt_p and gt_p.get("correct"):
            score += 3
            notes.append("human_adaptive_correct")
        narrad_p = pred_narrad.get(it["id"])
        if narrad_p and narrad_p.get("correct"):
            score += 1
            notes.append("narrad_adaptive_correct")
        # No bonus from autoad correctness
        autoad_p = pred_autoad.get(it["id"])

        ctx_len = len(it.get("context_clip_ids") or [])
        if ctx_len <= 2:
            score += 2
            notes.append(f"compact_ctx_{ctx_len}")

        if (it.get("expectedness") or 0) >= 0.7:
            score += 2
            notes.append("expectedness_ge_0.7")

        if evidence_supports_answer(it.get("evidence") or [], it["answer_text"]):
            score += 2
            notes.append("evidence_supports_answer")

        if it.get("target_type") in {
            "object_reveal",
            "spatial_consequence",
            "state_change",
            "participant_update",
        }:
            score += 1
            notes.append(f"concrete_target_{it['target_type']}")

        if any(not s for s in (it.get("context_sentences") or [])):
            score -= 2
            notes.append("empty_context_sentence")

        noctx_p = pred_noctx.get(it["id"])
        if noctx_p and noctx_p.get("correct"):
            score -= 2
            notes.append("noctx_correct")

        overlap_with_target = token_overlap(it["answer_text"], it["target_sentence"])
        overlap_with_context = max(
            (token_overlap(it["answer_text"], s) for s in (it.get("context_sentences") or [""])),
            default=0.0,
        )
        if overlap_with_target >= 0.6:
            score -= 2
            notes.append(f"answer_paraphrases_target_{overlap_with_target:.2f}")
        if overlap_with_context > overlap_with_target + 0.1 and overlap_with_context >= 0.3:
            # Answer text is closer to prior context than to the target — looks like
            # the question got reversed (correct option paraphrases the context).
            score -= 5
            notes.append(
                f"answer_resembles_context_more_{overlap_with_context:.2f}_vs_target_{overlap_with_target:.2f}"
            )

        if it.get("question_type") != "what_happens_next":
            score -= 1
            notes.append(f"q_type_{it.get('question_type')}")

        # Slight bonus when at least one context clip has a video file (figure-friendly).
        ctx_paths = [c.get("video_path") for c in (it.get("context") or [])]
        if any(p for p in ctx_paths):
            score += 1
            notes.append("video_available")

        good_scored.append((score, it, notes))

    good_scored.sort(reverse=True, key=lambda x: (x[0], len(x[1].get("evidence") or [])))

    # Diversify: prefer one example per movie initially, then fill.
    seen_movies = defaultdict(int)
    seen_targets = defaultdict(int)
    good_top = []
    for score, it, notes in good_scored:
        if score < 6:
            continue
        movie = it["movie"]
        ttype = it.get("target_type", "")
        if seen_movies[movie] >= 2:
            continue
        if seen_targets[ttype] >= 4:
            continue
        good_top.append({"score": score, "id": it["id"], "movie": movie,
                         "target_type": ttype, "notes": notes})
        seen_movies[movie] += 1
        seen_targets[ttype] += 1
        if len(good_top) >= 14:
            break

    # Trim to 10
    good_top = good_top[:10]
    print(f"  good: {len(good_top)} top examples", file=sys.stderr)

    # ------------------------------------------------------------------
    # 3) Baseline comparison
    # ------------------------------------------------------------------
    baseline_scored = []
    for it in items:
        gt_p = pred_gt.get(it["id"])
        narrad_p = pred_narrad.get(it["id"])
        autoad_p = pred_autoad.get(it["id"])
        if not gt_p or not narrad_p or not autoad_p:
            continue
        if not gt_p.get("correct"):
            continue
        narrad_correct = bool(narrad_p.get("correct"))
        autoad_correct = bool(autoad_p.get("correct"))
        if narrad_correct and autoad_correct:
            continue  # need at least one wrong

        # Require non-trivial: no-context baseline must be wrong, otherwise the
        # context wasn't really doing the work for human AD either.
        noctx_p = pred_noctx.get(it["id"])
        if noctx_p and noctx_p.get("correct"):
            continue

        # Skip items where the answer text is closer to the prior context than to
        # the target — those are question-generation quirks, not legitimate
        # forward-evidence cases.
        ans = it["answer_text"]
        target_overlap = token_overlap(ans, it["target_sentence"])
        ctx_overlap = max(
            (token_overlap(ans, s) for s in (it.get("context_sentences") or [""])),
            default=0.0,
        )
        if ctx_overlap > target_overlap + 0.1 and ctx_overlap >= 0.3:
            continue
        # Skip items where the correct option does not visibly track the target
        # sentence — usually a question-generation quirk.
        if target_overlap < 0.15:
            continue

        score = 0.0
        notes = ["human_adaptive_correct"]
        if not narrad_correct:
            score += 1.5
            notes.append("narrad_adaptive_wrong")
        if not autoad_correct:
            score += 1.5
            notes.append("autoad_adaptive_wrong")
        if not narrad_correct and not autoad_correct:
            score += 2  # both wrong
            notes.append("both_generated_wrong")

        # noctx is already required to be wrong above.
        score += 2
        notes.append("noctx_wrong_required")

        # Inspect Results.csv: does the gt evidence text differ from generated AD on
        # the selected context clip(s)?
        bridge_words = set()
        for ev in it.get("evidence") or []:
            bridge_words |= normalize_words(ev.get("span") or "")
        if not bridge_words:
            for s in it.get("context_sentences") or []:
                bridge_words |= normalize_words(s)
        ans_words = normalize_words(it["answer_text"])
        bridge_ans = bridge_words & ans_words
        if bridge_ans:
            score += 1
            notes.append(f"bridge_keywords:{','.join(sorted(bridge_ans))[:60]}")

        # For each context clip, compare gt and narrad/autoad text from Results.csv.
        narrad_misses = 0
        autoad_misses = 0
        for cid in it.get("context_clip_ids") or []:
            csv_row = csv_lookup(csv_idx, it["movie"], cid)
            if csv_row is None:
                continue
            gt_text = csv_row.get("ground_truth", "")
            narrad_text = csv_row.get("NarrAD", "")
            autoad_text = csv_row.get("AutoAD-Zero", "")
            gt_tokens = normalize_words(gt_text)
            narrad_tokens = normalize_words(narrad_text)
            autoad_tokens = normalize_words(autoad_text)
            for w in bridge_ans:
                if w in gt_tokens and w not in narrad_tokens:
                    narrad_misses += 1
                if w in gt_tokens and w not in autoad_tokens:
                    autoad_misses += 1
        if narrad_misses:
            score += 1.5
            notes.append(f"narrad_drops_{narrad_misses}_keywords")
        if autoad_misses:
            score += 1.5
            notes.append(f"autoad_drops_{autoad_misses}_keywords")

        if len(it.get("context_clip_ids") or []) <= 2:
            score += 2
            notes.append(f"compact_ctx_{len(it['context_clip_ids'])}")

        baseline_scored.append((score, it, notes,
                                {"narrad_correct": narrad_correct,
                                 "autoad_correct": autoad_correct,
                                 "noctx_correct": bool(noctx_p and noctx_p.get("correct")),
                                 "narrad_dropped_kw": narrad_misses,
                                 "autoad_dropped_kw": autoad_misses}))

    baseline_scored.sort(reverse=True, key=lambda x: x[0])
    seen_movies_bc = defaultdict(int)
    baseline_top = []
    for score, it, notes, extras in baseline_scored:
        if seen_movies_bc[it["movie"]] >= 2:
            continue
        baseline_top.append({"score": score, "id": it["id"], "movie": it["movie"],
                             "target_type": it.get("target_type"),
                             "notes": notes, "outcomes": extras})
        seen_movies_bc[it["movie"]] += 1
        if len(baseline_top) >= 8:
            break
    baseline_top = baseline_top[:6]
    print(f"  baseline_comparison: {len(baseline_top)} examples", file=sys.stderr)

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    manifest = {
        "good": good_top,
        "bad": [
            {
                "source": r.get("_source"),
                "movie": r.get("_movie_folder"),
                "id": r.get("id"),
                "target_sentence": r.get("target_sentence"),
                "context_sentences": r.get("context_sentences"),
                "options": r.get("options"),
                "answer_idx": r.get("answer_idx"),
                "validation_metadata": r.get("validation_metadata"),
                "distractor_metadata": r.get("distractor_metadata"),
                "target_type": r.get("target_type"),
                "expectedness": r.get("expectedness"),
                "predictability": r.get("predictability"),
                "context": r.get("context"),
                "target": r.get("target"),
            }
            for r in bad_records[:5]
        ],
        "baseline_comparison": baseline_top,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Wrote {MANIFEST_PATH}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

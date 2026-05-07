#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_movie_pipeline.sh --movie MOVIE_ID [options]

Options:
  --movie MOVIE_ID          Movie id to process, e.g. 1005_Signs. Required.
  --gpu GPU_ID              CUDA device id. Default: current CUDA_VISIBLE_DEVICES or 0.
  --input PATH              Raw LSMDC/MAD CSV/TSV/JSONL or directory.
                            Default: /jumbo/jinlab/datasets/MAD/MAD-EVAL_named_combined.csv
  --video-root DIR          MAD clip video root. Default: data/raw/lsmdc/mad-eval-videos.
  --out-root DIR            Root for per-movie outputs. Default: data/interim/per_movie
  --processed-root DIR      Root for per-movie processed exports. Default: data/processed/per_movie
  --window-size N           Ordered block size for stage 02. Default: 10.
  --qwen-config PATH        Base Qwen config. Default: configs/qwen.yaml.
  --context-config PATH     Base context config. Default: configs/context_selection.yaml.
  --dataset-config PATH     Base dataset config. Default: configs/dataset_generation.yaml.
  --limit N                 Optional limit for stage 01 parse rows after movie filtering.
  --max-blocks N            Optional max blocks for stage 02.
  --extraction-limit N      Optional limit for stage 03.
  --candidate-limit N       Optional limit for stage 04.
  --rerun-contexts          Recompute stage 02b context selection outputs for this movie.
  --resume                  Resume stages when supported.
EOF
}

MOVIE=""
GPU="${CUDA_VISIBLE_DEVICES:-0}"
INPUT="/jumbo/jinlab/datasets/MAD/MAD-EVAL_named_combined.csv"
VIDEO_ROOT="data/raw/lsmdc/mad-eval-videos"
OUT_ROOT="data/interim/per_movie"
PROCESSED_ROOT="data/processed/per_movie"
WINDOW_SIZE="10"
QWEN_CONFIG="configs/qwen.yaml"
CONTEXT_CONFIG="configs/context_selection.yaml"
DATASET_CONFIG="configs/dataset_generation.yaml"
LIMIT=""
MAX_BLOCKS=""
EXTRACTION_LIMIT=""
CANDIDATE_LIMIT=""
RERUN_CONTEXTS=0
RESUME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --movie) MOVIE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --input) INPUT="$2"; shift 2 ;;
    --video-root) VIDEO_ROOT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --processed-root) PROCESSED_ROOT="$2"; shift 2 ;;
    --window-size) WINDOW_SIZE="$2"; shift 2 ;;
    --qwen-config) QWEN_CONFIG="$2"; shift 2 ;;
    --context-config) CONTEXT_CONFIG="$2"; shift 2 ;;
    --dataset-config) DATASET_CONFIG="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --max-blocks) MAX_BLOCKS="$2"; shift 2 ;;
    --extraction-limit) EXTRACTION_LIMIT="$2"; shift 2 ;;
    --candidate-limit) CANDIDATE_LIMIT="$2"; shift 2 ;;
    --rerun-contexts) RERUN_CONTEXTS=1; shift ;;
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MOVIE" ]]; then
  echo "--movie is required" >&2
  usage
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU"

MOVIE_DIR="${OUT_ROOT}/${MOVIE}"
PROCESSED_DIR="${PROCESSED_ROOT}/${MOVIE}"
CONFIG_DIR="${MOVIE_DIR}/configs"
LOG_DIR="${MOVIE_DIR}/logs"
mkdir -p "$MOVIE_DIR" "$PROCESSED_DIR" "$CONFIG_DIR" "$LOG_DIR"

CONTEXT_CONFIG_RUNTIME="${CONFIG_DIR}/context_selection.yaml"
DATASET_CONFIG_RUNTIME="${CONFIG_DIR}/dataset_generation.yaml"

python - "$CONTEXT_CONFIG" "$CONTEXT_CONFIG_RUNTIME" "$MOVIE_DIR" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, movie_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
with open(src, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
cfg["output_selected"] = str(movie_dir / "selected_contexts.jsonl")
cfg["output_rejected"] = str(movie_dir / "rejected_contexts.jsonl")
cfg["output_challenge"] = str(movie_dir / "challenge_contexts.jsonl")
cfg["progress_output"] = str(movie_dir / "context_selection_progress.json")
with open(dst, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

python - "$DATASET_CONFIG" "$DATASET_CONFIG_RUNTIME" "$MOVIE_DIR" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, movie_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
with open(src, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
cfg["progress_output_extract_actions"] = str(movie_dir / "progress_extract_actions.json")
cfg["progress_output_generate_examples"] = str(movie_dir / "progress_generate_examples.json")
cfg["progress_output_validate_filter"] = str(movie_dir / "progress_validate_filter.json")
with open(dst, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

RESUME_FLAG=()
if [[ "$RESUME" -eq 1 ]]; then
  RESUME_FLAG=(--resume)
fi

PARSE_LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  PARSE_LIMIT_ARGS=(--limit "$LIMIT")
fi

MAX_BLOCKS_ARGS=()
if [[ -n "$MAX_BLOCKS" ]]; then
  MAX_BLOCKS_ARGS=(--max_blocks "$MAX_BLOCKS")
fi

EXTRACTION_LIMIT_ARGS=()
if [[ -n "$EXTRACTION_LIMIT" ]]; then
  EXTRACTION_LIMIT_ARGS=(--limit "$EXTRACTION_LIMIT")
fi

CANDIDATE_LIMIT_ARGS=()
if [[ -n "$CANDIDATE_LIMIT" ]]; then
  CANDIDATE_LIMIT_ARGS=(--limit "$CANDIDATE_LIMIT")
fi

jsonl_count() {
  local path="$1"
  if [[ -f "$path" ]]; then
    wc -l < "$path" | tr -d ' '
  else
    echo 0
  fi
}

stage_summary() {
  local stage="$1"
  shift
  echo "[$(date -Is)] ${stage} summary: $*"
}

echo "[$(date -Is)] movie=${MOVIE} gpu=${CUDA_VISIBLE_DEVICES} out=${MOVIE_DIR}"

if [[ "$RERUN_CONTEXTS" -eq 1 ]]; then
  rm -f \
    "${MOVIE_DIR}/selected_contexts.jsonl" \
    "${MOVIE_DIR}/rejected_contexts.jsonl" \
    "${MOVIE_DIR}/challenge_contexts.jsonl" \
    "${MOVIE_DIR}/context_selection_progress.json"
fi

python scripts/01_parse_lsmdc.py \
  --input "$INPUT" \
  --output "${MOVIE_DIR}/parsed_sequences.jsonl" \
  --movie "$MOVIE" \
  --video_root "$VIDEO_ROOT" \
  "${PARSE_LIMIT_ARGS[@]}" \
  "${RESUME_FLAG[@]}" \
  2>&1 | tee "${LOG_DIR}/01_parse.log"
stage_summary "01_parse" "parsed=$(jsonl_count "${MOVIE_DIR}/parsed_sequences.jsonl") output=${MOVIE_DIR}/parsed_sequences.jsonl"

python scripts/02_build_windows.py \
  --input "${MOVIE_DIR}/parsed_sequences.jsonl" \
  --output "${MOVIE_DIR}/search_blocks_w${WINDOW_SIZE}.jsonl" \
  --max_window_clips "$WINDOW_SIZE" \
  --movie "$MOVIE" \
  "${MAX_BLOCKS_ARGS[@]}" \
  "${RESUME_FLAG[@]}" \
  2>&1 | tee "${LOG_DIR}/02_build_windows.log"
stage_summary "02_build_windows" "input=$(jsonl_count "${MOVIE_DIR}/parsed_sequences.jsonl") blocks=$(jsonl_count "${MOVIE_DIR}/search_blocks_w${WINDOW_SIZE}.jsonl") output=${MOVIE_DIR}/search_blocks_w${WINDOW_SIZE}.jsonl"

python scripts/02b_select_valid_contexts.py \
  --input "${MOVIE_DIR}/search_blocks_w${WINDOW_SIZE}.jsonl" \
  --qwen_config "$QWEN_CONFIG" \
  --context_config "$CONTEXT_CONFIG_RUNTIME" \
  --selected_output "${MOVIE_DIR}/selected_contexts.jsonl" \
  --rejected_output "${MOVIE_DIR}/rejected_contexts.jsonl" \
  --challenge_output "${MOVIE_DIR}/challenge_contexts.jsonl" \
  --movie "$MOVIE" \
  "${RESUME_FLAG[@]}" \
  2>&1 | tee "${LOG_DIR}/02b_select_contexts.log"
stage_summary "02b_select_contexts" "blocks=$(jsonl_count "${MOVIE_DIR}/search_blocks_w${WINDOW_SIZE}.jsonl") selected=$(jsonl_count "${MOVIE_DIR}/selected_contexts.jsonl") rejected=$(jsonl_count "${MOVIE_DIR}/rejected_contexts.jsonl") challenge=$(jsonl_count "${MOVIE_DIR}/challenge_contexts.jsonl")"
python scripts/summarize_pipeline_outputs.py --movie_dir "$MOVIE_DIR" --processed_dir "$PROCESSED_DIR" --stage "02b_contexts"

python scripts/03_qwen_extract_actions.py \
  --input "${MOVIE_DIR}/selected_contexts.jsonl" \
  --output "${MOVIE_DIR}/qwen_action_extractions.jsonl" \
  --config "$QWEN_CONFIG" \
  --movie "$MOVIE" \
  "${EXTRACTION_LIMIT_ARGS[@]}" \
  "${RESUME_FLAG[@]}" \
  2>&1 | tee "${LOG_DIR}/03_extract_actions.log"
stage_summary "03_extract_actions" "input_selected=$(jsonl_count "${MOVIE_DIR}/selected_contexts.jsonl") processed=$(jsonl_count "${MOVIE_DIR}/qwen_action_extractions.jsonl") output=${MOVIE_DIR}/qwen_action_extractions.jsonl"

python scripts/04_generate_benchmark_examples.py \
  --input "${MOVIE_DIR}/qwen_action_extractions.jsonl" \
  --output "${MOVIE_DIR}/candidate_examples.jsonl" \
  --config "$QWEN_CONFIG" \
  --movie "$MOVIE" \
  "${CANDIDATE_LIMIT_ARGS[@]}" \
  "${RESUME_FLAG[@]}" \
  2>&1 | tee "${LOG_DIR}/04_generate_examples.log"
stage_summary "04_generate_examples" "input_extractions=$(jsonl_count "${MOVIE_DIR}/qwen_action_extractions.jsonl") processed=$(jsonl_count "${MOVIE_DIR}/candidate_examples.jsonl") output=${MOVIE_DIR}/candidate_examples.jsonl"
python scripts/summarize_pipeline_outputs.py --movie_dir "$MOVIE_DIR" --processed_dir "$PROCESSED_DIR" --stage "04_candidates"

python scripts/05_validate_and_filter.py \
  --input "${MOVIE_DIR}/candidate_examples.jsonl" \
  --kept "${MOVIE_DIR}/kept_examples.jsonl" \
  --rejected "${MOVIE_DIR}/rejected_examples.jsonl" \
  --challenge "${MOVIE_DIR}/challenge_examples.jsonl" \
  --config "$QWEN_CONFIG" \
  --dataset_config "$DATASET_CONFIG_RUNTIME" \
  "${RESUME_FLAG[@]}" \
  2>&1 | tee "${LOG_DIR}/05_validate_filter.log"
stage_summary "05_validate_filter" "input_candidates=$(jsonl_count "${MOVIE_DIR}/candidate_examples.jsonl") kept=$(jsonl_count "${MOVIE_DIR}/kept_examples.jsonl") rejected=$(jsonl_count "${MOVIE_DIR}/rejected_examples.jsonl") challenge=$(jsonl_count "${MOVIE_DIR}/challenge_examples.jsonl")"
python scripts/summarize_pipeline_outputs.py --movie_dir "$MOVIE_DIR" --processed_dir "$PROCESSED_DIR" --stage "05_validate"

python scripts/06_split_dataset.py \
  --input "${MOVIE_DIR}/kept_examples.jsonl" \
  --challenge_input "${MOVIE_DIR}/challenge_examples.jsonl" \
  --out_dir "$PROCESSED_DIR" \
  --split_by movie \
  "${RESUME_FLAG[@]}" \
  2>&1 | tee "${LOG_DIR}/06_split_dataset.log"
stage_summary "06_split_dataset" "train=$(jsonl_count "${PROCESSED_DIR}/train.jsonl") val=$(jsonl_count "${PROCESSED_DIR}/val.jsonl") test=$(jsonl_count "${PROCESSED_DIR}/test.jsonl") challenge=$(jsonl_count "${PROCESSED_DIR}/challenge_unpredictable.jsonl") processed_dir=${PROCESSED_DIR}"
python scripts/summarize_pipeline_outputs.py --movie_dir "$MOVIE_DIR" --processed_dir "$PROCESSED_DIR" --stage "06_split"

python scripts/07_export_dataset_card.py \
  --train "${PROCESSED_DIR}/train.jsonl" \
  --val "${PROCESSED_DIR}/val.jsonl" \
  --test "${PROCESSED_DIR}/test.jsonl" \
  --challenge "${PROCESSED_DIR}/challenge_unpredictable.jsonl" \
  --output "${PROCESSED_DIR}/dataset_card.md" \
  "${RESUME_FLAG[@]}" \
  2>&1 | tee "${LOG_DIR}/07_dataset_card.log"
stage_summary "07_dataset_card" "output=${PROCESSED_DIR}/dataset_card.md"

echo "[$(date -Is)] done movie=${MOVIE} gpu=${CUDA_VISIBLE_DEVICES}"

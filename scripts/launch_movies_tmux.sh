#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/launch_movies_tmux.sh --movies MOVIE1,MOVIE2,... [options]
  scripts/launch_movies_tmux.sh --movies-file movies.txt [options]

Options:
  --movies CSV             Comma-separated movie ids.
  --movies-file PATH       One movie id per line. Blank lines and # comments are ignored.
  --gpus CSV               GPU ids to use. Default: 0,1,2,3,4,5,6,7.
  --session NAME           tmux session name. Default: forseebench.
  --window-size N          Ordered block size passed to run_movie_pipeline.sh. Default: 10.
  --out-root DIR           Root for per-movie interim outputs. Default: data/interim/per_movie.
  --processed-root DIR     Root for per-movie processed outputs. Default: data/processed/per_movie.
  --input PATH             Raw LSMDC/MAD input path.
  --video-root DIR         MAD clip video root. Default: data/raw/lsmdc/mad-eval-videos.
  --qwen-config PATH       Qwen config. Default: configs/qwen.yaml.
  --context-config PATH    Context config. Default: configs/context_selection.yaml.
  --dataset-config PATH    Dataset config. Default: configs/dataset_generation.yaml.
  --rerun-contexts         Recompute stage 02b context selection outputs for each movie.
  --resume                 Resume stages when supported.

This creates one tmux window per GPU. Each window runs its assigned movies
sequentially; the next movie on that GPU starts after the previous movie finishes.
EOF
}

MOVIES_CSV=""
MOVIES_FILE=""
GPUS_CSV="0,1,2,3,4,5,6,7"
SESSION="forseebench"
WINDOW_SIZE="10"
OUT_ROOT="data/interim/per_movie"
PROCESSED_ROOT="data/processed/per_movie"
INPUT="/jumbo/jinlab/datasets/MAD/MAD-EVAL_named_combined.csv"
VIDEO_ROOT="data/raw/lsmdc/mad-eval-videos"
QWEN_CONFIG="configs/qwen.yaml"
CONTEXT_CONFIG="configs/context_selection.yaml"
DATASET_CONFIG="configs/dataset_generation.yaml"
RESUME=0
RERUN_CONTEXTS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --movies) [[ $# -ge 2 ]] || { echo "--movies requires a value" >&2; exit 2; }; MOVIES_CSV="$2"; shift 2 ;;
    --movies-file) [[ $# -ge 2 ]] || { echo "--movies-file requires a value" >&2; exit 2; }; MOVIES_FILE="$2"; shift 2 ;;
    --gpus) [[ $# -ge 2 ]] || { echo "--gpus requires a value" >&2; exit 2; }; GPUS_CSV="$2"; shift 2 ;;
    --session) [[ $# -ge 2 ]] || { echo "--session requires a value" >&2; exit 2; }; SESSION="$2"; shift 2 ;;
    --window-size) [[ $# -ge 2 ]] || { echo "--window-size requires a value" >&2; exit 2; }; WINDOW_SIZE="$2"; shift 2 ;;
    --out-root) [[ $# -ge 2 ]] || { echo "--out-root requires a value" >&2; exit 2; }; OUT_ROOT="$2"; shift 2 ;;
    --processed-root) [[ $# -ge 2 ]] || { echo "--processed-root requires a value" >&2; exit 2; }; PROCESSED_ROOT="$2"; shift 2 ;;
    --input) [[ $# -ge 2 ]] || { echo "--input requires a value" >&2; exit 2; }; INPUT="$2"; shift 2 ;;
    --video-root) [[ $# -ge 2 ]] || { echo "--video-root requires a value" >&2; exit 2; }; VIDEO_ROOT="$2"; shift 2 ;;
    --qwen-config) [[ $# -ge 2 ]] || { echo "--qwen-config requires a value" >&2; exit 2; }; QWEN_CONFIG="$2"; shift 2 ;;
    --context-config) [[ $# -ge 2 ]] || { echo "--context-config requires a value" >&2; exit 2; }; CONTEXT_CONFIG="$2"; shift 2 ;;
    --dataset-config) [[ $# -ge 2 ]] || { echo "--dataset-config requires a value" >&2; exit 2; }; DATASET_CONFIG="$2"; shift 2 ;;
    --rerun-contexts) RERUN_CONTEXTS=1; shift ;;
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MOVIES_CSV" && -z "$MOVIES_FILE" ]]; then
  echo "Provide --movies or --movies-file" >&2
  usage
  exit 2
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found on PATH" >&2
  exit 1
fi

MOVIES=()
if [[ -n "$MOVIES_CSV" ]]; then
  IFS=',' read -r -a MOVIES <<< "$MOVIES_CSV"
else
  while IFS= read -r line; do
    line="${line%%#*}"
    line="${line//[[:space:]]/}"
    [[ -n "$line" ]] && MOVIES+=("$line")
  done < "$MOVIES_FILE"
fi

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ "${#MOVIES[@]}" -eq 0 ]]; then
  echo "No movies provided" >&2
  exit 2
fi
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPUs provided" >&2
  exit 2
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

COMMON_ARGS=(
  --window-size "$WINDOW_SIZE"
  --out-root "$OUT_ROOT"
  --processed-root "$PROCESSED_ROOT"
  --input "$INPUT"
  --video-root "$VIDEO_ROOT"
  --qwen-config "$QWEN_CONFIG"
  --context-config "$CONTEXT_CONFIG"
  --dataset-config "$DATASET_CONFIG"
)
if [[ "$RESUME" -eq 1 ]]; then
  COMMON_ARGS+=(--resume)
fi
if [[ "$RERUN_CONTEXTS" -eq 1 ]]; then
  COMMON_ARGS+=(--rerun-contexts)
fi

tmux new-session -d -s "$SESSION" -n "gpu${GPUS[0]}"

for gpu_index in "${!GPUS[@]}"; do
  gpu="${GPUS[$gpu_index]}"
  window="gpu${gpu}"
  if [[ "$gpu_index" -ne 0 ]]; then
    tmux new-window -t "$SESSION" -n "$window"
  fi

  commands=()
  for movie_index in "${!MOVIES[@]}"; do
    if (( movie_index % ${#GPUS[@]} == gpu_index )); then
      movie="${MOVIES[$movie_index]}"
      commands+=("scripts/run_movie_pipeline.sh --movie '$movie' --gpu '$gpu' ${COMMON_ARGS[*]}")
    fi
  done

  if [[ "${#commands[@]}" -eq 0 ]]; then
    tmux send-keys -t "${SESSION}:${window}" "echo 'No movies assigned to GPU ${gpu}'" C-m
    continue
  fi

  joined=""
  for cmd in "${commands[@]}"; do
    if [[ -z "$joined" ]]; then
      joined="$cmd"
    else
      joined="${joined} && ${cmd}"
    fi
  done
  tmux send-keys -t "${SESSION}:${window}" "cd '$PWD' && ${joined}" C-m
done

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"

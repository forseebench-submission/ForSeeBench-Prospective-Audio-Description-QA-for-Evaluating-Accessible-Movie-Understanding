# ForSeeBench

ForSeeBench is an evaluation benchmark for prospective audio-description question answering. A model receives prior movie audio-description (AD) context and a multiple-choice question about a withheld future AD target; scoring checks whether the model selects the answer grounded in the hidden target.

This repository is an anonymous review artifact for a double-blind conference submission. It contains code, scripts, configs, tests, documentation, and tiny sample Q/A files. It is not a training-data release and should not contain author identity, private paths, raw media, or restricted source assets.

## Artifact Targets

- GitHub code artifact: `https://github.com/forseebench-submission/ForSeeBench-Prospective-Audio-Description-QA-for-Evaluating-Accessible-Movie-Understanding.git`
- Hugging Face Dataset artifact: `https://huggingface.co/datasets/forseebench/forseebench`
- Optional Hugging Face Space artifact: `forseebench/forseebench-reviewer-demo`

## Release Representation

ForSeeBench is released as an evaluation benchmark rather than a training corpus. The intended Hugging Face layout is:

```text
data/forseebench_public.jsonl          # no answers, for model prediction
data/forseebench_with_answers.jsonl    # answer-bearing scoring file
sample_data/sample_public.jsonl
sample_data/sample_with_answers.jsonl
sample_data/sample_predictions.jsonl
schema.md
```

The current repository includes sample files and metadata. Full benchmark files should be added only after final field-level redistribution review.

## Source Data Boundary

ForSeeBench is derived from MAD/MAD-eval audio-description data. Raw movie videos, movie clips, audio tracks, subtitles, dialogue files, full MAD/MAD-eval source assets, and other restricted source assets are not redistributed. Users who need source assets must obtain them from the original providers under their own terms.

## Repository Contents

- `src/forseebench/`: parsing, construction, schema validation, and evaluation helpers.
- `scripts/`: construction, evaluation, validation, anonymization, release-prep, and HF upload utilities.
- `hf_dataset/`: Hugging Face dataset card, sample Q/A files, schema, Croissant/RAI notes, and metadata draft.
- `docs/`: reviewer-facing documentation and release policy notes.
- `tests/`: lightweight unit and smoke tests.
- `paper/`: active anonymized paper source.
- `agent/`: audit and readiness reports.

## Installation

Sample validation and generic MCQ scoring use the Python standard library plus this repository's `src` package.

```bash
python -m venv .venv
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
```

For tests:

```bash
python -m pip install pytest
pytest
```

TODO(author): add a complete dependency file for full construction and Qwen-based evaluation.

## Reviewer Quickstart

```bash
python scripts/check_anonymization.py
python scripts/check_submission_ready.py
python scripts/validate_dataset.py --input hf_dataset/sample_data/sample_public.jsonl --schema public
python scripts/validate_dataset.py --input hf_dataset/sample_data/sample_with_answers.jsonl --schema with_answers
python scripts/evaluate_mcq.py \
  --input hf_dataset/sample_data/sample_with_answers.jsonl \
  --predictions hf_dataset/sample_data/sample_predictions.jsonl
```

These commands use derived Q/A JSONL only. They do not require raw videos, clips, audio, subtitles, restricted MAD/MAD-eval assets, GPUs, or Qwen.

## Evaluation

Prediction rows are JSONL objects:

```json
{"id": "example-id", "prediction": 0}
```

Scoring command:

```bash
python scripts/evaluate_mcq.py \
  --input <forseebench_with_answers.jsonl> \
  --predictions <prediction_rows.jsonl>
```

Generated-AD and PrediCC-style analysis scripts are also present, but require full artifacts and Qwen setup:

```bash
python scripts/evaluate_autoad_mcq.py --help
python scripts/evaluate_predicc.py --help
```

## Citation

```bibtex
@misc{forseebench2026,
  title = {ForSeeBench: Prospective Audio-Description QA for Evaluating Accessible Movie Understanding},
  author = {Anonymous Author(s)},
  year = {2026},
  note = {Anonymous benchmark submission artifact}
}
```

## License

TODO(author): add final code license and derived benchmark license or access terms. Original source assets remain governed by their original providers' terms.

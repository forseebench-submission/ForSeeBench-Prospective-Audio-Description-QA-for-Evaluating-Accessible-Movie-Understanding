# Dataset Construction

This document summarizes the implemented construction pipeline. ForSeeBench is represented for release as an evaluation benchmark, not a training dataset.

## Source Boundary

The implementation consumes MAD/LSMDC-style audio-description rows. The paper instantiates ForSeeBench on 10 MAD-eval movies. Raw movie videos, clips, audio, subtitles/dialogue, full MAD/MAD-eval assets, private source paths, and credentials are cited/referenced but not redistributed.

## Pipeline

1. Parse source AD rows into normalized clip records.
2. Build bounded 10-clip search regions.
3. Use Qwen-assisted target/context selection to find nontrivial future targets with strictly prior evidence.
4. Extract target/action and expectedness metadata.
5. Generate four-option prospective QA examples.
6. Validate schema, target answerability, selected-context inferability, leakage risks, grounding, and distractor quality.
7. Export release-facing benchmark rows.

Helpful commands:

```bash
python scripts/01_parse_lsmdc.py --help
python scripts/02_build_windows.py --help
python scripts/02b_select_valid_contexts.py --help
python scripts/03_qwen_extract_actions.py --help
python scripts/04_generate_benchmark_examples.py --help
python scripts/05_validate_and_filter.py --help
python scripts/06_split_dataset.py --help
python scripts/07_export_dataset_card.py --help
```

## Release Rows

The paper-facing release avoids train/validation/test terminology unless a future paper revision defines an official split policy. The current release-facing Q/A files are:

- `hf_dataset/data/qna_test.jsonl`: public no-answer examples for model prediction;
- `hf_dataset/data/qna_with_answers.jsonl`: answer-bearing examples for scoring and reproducibility.

They are exported from `data/processed/all_movies/eval_all10.jsonl` with:

```bash
python scripts/export_release_qna.py
```

Public rows include prior context, question, options, question type, target type, context length, and release-safe source identifiers. Answer-bearing rows add answer keys and clean audit fields such as hidden target AD and evidence metadata when legally safe.

## Filtering And Nontriviality

The implemented filtering keeps examples that are nontrivial, predictable, evidence-bearing, and high quality under configured thresholds. It rejects generic, static, redundant, low-information, unsupported, or leakage-prone targets.

TODO(author): finalize which source-derived text fields and evidence spans may be hosted in the full answer-bearing file.

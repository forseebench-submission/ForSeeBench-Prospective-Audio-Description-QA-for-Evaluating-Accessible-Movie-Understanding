# Evaluation Protocol

## Task

ForSeeBench evaluates prospective audio-description question answering. A system receives prior AD context and answer options, but not the hidden future AD target. It predicts the option index that best matches the withheld target.

## Files

- Public/no-answer file: `forseebench_public.jsonl`
- Answer-bearing/scoring file: `forseebench_with_answers.jsonl`
- Prediction file: JSONL rows with `id` and `prediction`

The benchmark is not framed as train/validation/test splits.

## Public Inputs

Public rows contain:

- `id`
- `source_id`
- `prior_context`
- `question`
- `options`
- `question_type`
- `target_type`
- `context_length`
- optional clean diagnostics such as `expectedness`

Public rows exclude `answer_idx`, `answer_text`, hidden target AD, raw paths, and raw source assets.

## Scoring Inputs

Answer-bearing rows add:

- `answer_idx`
- `answer_text`
- optional `hidden_target_ad` if legally safe;
- optional clean evidence metadata if legally safe.

## Metric

`scripts/evaluate_mcq.py` computes exact multiple-choice accuracy. Missing predictions count as incorrect.

Sample command:

```bash
python scripts/evaluate_mcq.py \
  --input hf_dataset/sample_data/sample_with_answers.jsonl \
  --predictions hf_dataset/sample_data/sample_predictions.jsonl
```

## PrediCC And Context Analyses

The paper also evaluates no-context, adaptive selected-context, and fixed-window PrediCC settings. Those full analyses require full artifacts and Qwen setup:

```bash
python scripts/evaluate_autoad_mcq.py --help
python scripts/evaluate_predicc.py --help
```

TODO(author): provide final command manifests, model revision, hardware/runtime, and checksums for all reported tables.

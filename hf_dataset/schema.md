# ForSeeBench Release Schema

ForSeeBench is released as an evaluation benchmark rather than a training corpus. The full release has one public no-answer Q/A file for prediction and one answer-bearing Q/A file for scoring/reproducibility.

## Public / No-Answer Rows

File:

```text
data/qna_test.jsonl
```

Required fields:

- `id`: stable example id.
- `source_id`: movie/source identifier, included only if release-safe.
- `prior_context`: list of prior context objects. Each object has `clip_id` and `text`.
- `question`: multiple-choice question about the hidden future AD target.
- `options`: list of exactly four answer options.
- `question_type`: controlled question type.
- `target_type`: controlled target type.
- `context_length`: number of prior context clips.

Optional public diagnostic fields:

- `expectedness`: numeric score in `[0, 1]` or `null`.

Forbidden in public/no-answer rows:

- `answer_idx`
- `answer_text`
- `hidden_target_ad`
- raw video paths or local paths
- raw prompt/response logs

## Answer-Bearing / Scoring Rows

File:

```text
data/qna_with_answers.jsonl
```

Required fields:

- all public/no-answer fields;
- `answer_idx`: correct option index in `[0, 3]`;
- `answer_text`: text matching `options[answer_idx]`.

Optional scoring/audit fields:

- `hidden_target_ad`: withheld target AD sentence, if legally safe for release;
- `evidence`: list of clean evidence objects with `clip_id` and `span`, if legally safe.
- `distractor_metadata`: optional distractor type labels, if useful for audit.

## Source Assets

The schema does not include raw movie videos, clips, audio tracks, subtitles, dialogue files, full MAD/MAD-eval assets, local paths, or credentials.

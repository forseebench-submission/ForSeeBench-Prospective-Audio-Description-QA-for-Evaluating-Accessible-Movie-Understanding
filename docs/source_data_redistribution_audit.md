# Source Data Redistribution Audit

ForSeeBench is derived from MAD/MAD-eval audio-description data. The release cites source data but does not redistribute raw source assets. Raw videos are not redistributed, and raw source assets must not be uploaded.

## Raw Source Assets: Do Not Release

- raw movie videos;
- movie clips;
- raw audio tracks;
- subtitles/dialogue files if restricted;
- full MAD/MAD-eval source assets;
- local filesystem paths;
- credentials, API keys, or access tokens.

## Derived Benchmark Artifacts: Candidate Release

- public no-answer Q/A JSONL;
- answer-bearing scoring JSONL;
- sample Q/A files;
- sample predictions;
- schema;
- target type and question type;
- context length and clean context metadata;
- evidence metadata if legally safe;
- hidden target AD only in the answer-bearing file if legally safe.

## Public No-Answer File

The public file should exclude `answer_idx`, `answer_text`, hidden target AD, raw paths, and source assets. It is for model prediction only.

## Answer-Bearing File

The scoring file may include answer keys, hidden target AD, and evidence metadata only after the author confirms redistribution rights.

## Uncertain Fields

TODO(author): confirm release status for source/movie identifiers, clip ids, prior AD text, hidden target AD, evidence spans, generated AD outputs, and validation metadata.

## GitHub Policy

GitHub may include code, configs safe for release, docs, tests, schema examples, and tiny sample Q/A files. GitHub must exclude raw assets, local paths, `.env`/secrets, old git history, author identity, caches, build artifacts, `node_modules`, and private outputs.

For this artifact, GitHub and Hugging Face may also include the full derived Q/A benchmark files after field-level review:

- `hf_dataset/data/qna_test.jsonl`
- `hf_dataset/data/qna_with_answers.jsonl`

These files are derived benchmark artifacts, not raw MAD/MAD-eval source assets. They must still be reviewed for source-field license terms before any public non-review release.

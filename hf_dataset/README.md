---
language:
- en
license: other
task_categories:
- question-answering
tags:
- benchmark
- evaluation
- audio-description
- multiple-choice
- prospective-reasoning
pretty_name: ForSeeBench
---

# ForSeeBench

ForSeeBench is an evaluation benchmark for prospective audio-description question answering. A system receives prior audio-description (AD) context and a multiple-choice question about a withheld future AD target. The benchmark measures whether prior AD context supports the next salient visual update that a human AD writer chose to narrate.

This dataset repository is an anonymous review artifact for a double-blind conference submission. It is not a training corpus.

## Task

Given:

- prior AD context;
- a multiple-choice question;
- four answer options;
- target/question metadata;

predict the option index that best answers the question about the hidden future AD target.

## File Layout

Intended full release:

```text
data/forseebench_public.jsonl
data/forseebench_with_answers.jsonl
sample_data/sample_public.jsonl
sample_data/sample_with_answers.jsonl
sample_data/sample_predictions.jsonl
schema.md
README.md
croissant_rai_notes.md
croissant_metadata_draft.json
```

Current staging upload includes sample files and metadata only. Full benchmark files will be added after final field-level redistribution review.

## Files

- `data/forseebench_public.jsonl`: public no-answer benchmark file for model prediction. TODO(author): add after final release review.
- `data/forseebench_with_answers.jsonl`: answer-bearing scoring file for evaluation and reproducibility. TODO(author): add after final release review.
- `sample_data/sample_public.jsonl`: small no-answer sample.
- `sample_data/sample_with_answers.jsonl`: small answer-bearing sample.
- `sample_data/sample_predictions.jsonl`: tiny prediction file for evaluator smoke tests.
- `schema.md`: release schema.

The benchmark is intentionally not presented as train/validation/test splits because the paper defines an evaluation set rather than a supervised training protocol.

## Source Data And Redistribution

ForSeeBench is derived from MAD/MAD-eval audio-description data. This release contains derived Q/A benchmark artifacts and metadata only.

This release does not redistribute raw movie videos, movie clips, audio tracks, subtitles, dialogue files, full MAD/MAD-eval source assets, or other restricted source assets. Users who need source assets must obtain them from the original providers under their own terms.

## Source Citation

```bibtex
@inproceedings{soldan2022mad,
  title = {MAD: A Scalable Dataset for Language Grounding in Videos From Movie Audio Descriptions},
  author = {Soldan, Mattia and Pardo, Alejandro and Alc{\'a}zar, Juan Le{\'o}n and Caba, Fabian and Zhao, Chen and Giancola, Silvio and Ghanem, Bernard},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2022}
}
```

TODO(author): confirm final source-data URL, version, access terms, and required attribution language.

## Evaluation

Sample validation:

```bash
python scripts/validate_dataset.py --input hf_dataset/sample_data/sample_public.jsonl --schema public
python scripts/validate_dataset.py --input hf_dataset/sample_data/sample_with_answers.jsonl --schema with_answers
```

Sample scoring:

```bash
python scripts/evaluate_mcq.py \
  --input hf_dataset/sample_data/sample_with_answers.jsonl \
  --predictions hf_dataset/sample_data/sample_predictions.jsonl
```

The evaluator reports multiple-choice accuracy, missing predictions, number correct, and label distributions.

## Intended Use

- Evaluate prospective QA from prior AD context.
- Compare context policies or AD sources under fixed questions and options.
- Audit forward-evidence preservation using target type, question type, context length, and clean evidence metadata where legally safe.

## Out-of-Scope Use

- Training models on restricted source assets.
- Redistributing raw movie/video/audio/subtitle/source assets.
- Certifying complete accessibility quality or blind/low-vision viewer comprehension.
- High-stakes deployment or unrestricted future prediction.

## Licensing

TODO(author): finalize the derived benchmark license or access terms.

The derived benchmark license may differ from the licenses or terms governing MAD/MAD-eval and underlying movie assets. Original source assets remain governed by their original providers' terms.

## Citation

```bibtex
@misc{forseebench2026,
  title = {ForSeeBench: Prospective Audio-Description QA for Evaluating Accessible Movie Understanding},
  author = {Anonymous Author(s)},
  year = {2026},
  note = {Anonymous benchmark submission artifact}
}
```

---
pretty_name: ForSeeBench
license: other
task_categories:
- question-answering
language:
- en
tags:
- audio-description
- accessibility
- benchmark
- multiple-choice-qa
- video-understanding
- prospective-qa
---

# ForSeeBench

## Dataset Summary

ForSeeBench is an evaluation benchmark for prospective audio-description question answering. Sighted viewers receive dense visual cues continuously: actions, bodies, expressions, scene changes, objects, and spatial updates all help them build a mental model of a movie. Blind and low-vision viewers often depend on audio description (AD) for that information, because the soundtrack alone does not carry it.

Each ForSeeBench item gives prior AD context and a multiple-choice question about a withheld future human-written AD target. The task is to choose the answer option that best matches the upcoming target, measuring whether the prior AD stream preserves evidence for the next salient visual update. This complements captioning, video QA, and general video-understanding evaluations, which often ask systems to describe or answer questions about visible content without directly testing whether AD supports the unfolding viewing experience.

This is an anonymous review artifact for a double-blind benchmark submission. It is not a training corpus.

| Property | Value |
| --- | --- |
| Task | Multiple-choice prospective audio-description QA |
| Size | 787 Q/A items |
| Language | English |
| Modality | Text derived from movie audio descriptions |
| Source | MAD/MAD-eval audio-description data |
| Raw media redistributed | No |

## Dataset Files

- `data/qna_test.jsonl`: public no-answer Q/A file for model prediction.
- `data/qna_with_answers.jsonl`: answer-bearing scoring file for evaluation and reproducibility.
- `sample_data/sample_public.jsonl`: two-row no-answer sample for smoke tests.
- `sample_data/sample_with_answers.jsonl`: two-row answer-bearing sample for smoke tests.
- `sample_data/sample_predictions.jsonl`: sample predictions for evaluator smoke tests.
- `schema.md`: field definitions.

The full Q/A files contain 787 derived benchmark items. The sample files are included only to make validation and scoring checks fast.

These files are benchmark artifacts, not training or model-development splits. `qna_test.jsonl` is the prediction input file, and `qna_with_answers.jsonl` is the scoring/reproducibility file.

## Source Data

ForSeeBench is derived from MAD/MAD-eval audio-description data. This release contains derived Q/A benchmark artifacts and metadata only.

Raw movie videos, movie clips, audio tracks, subtitles, dialogue files, full MAD/MAD-eval source assets, and other restricted source assets are not redistributed. Users who need source assets must obtain them from the original providers under their own terms.

MAD / MAD-eval citation:

```bibtex
@inproceedings{soldan2022mad,
  author = {Soldan, Mattia and Pardo, Alejandro and Alc{\'a}zar, Juan Le{\'o}n and Caba Heilbron, Fabian and Zhao, Chen and Giancola, Silvio and Ghanem, Bernard},
  title = {{MAD}: A Scalable Dataset for Language Grounding in Videos from Movie Audio Descriptions},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = {5026--5035},
  year = {2022}
}
```

If using the underlying Movie Description/LSMDC lineage directly, cite Rohrbach et al., *Movie Description*, IJCV 2017.

## Task And Evaluation

Systems receive prior AD context, a question, and four answer options from `data/qna_test.jsonl`. They output one JSONL prediction per example:

```json
{"id": "example-id", "prediction": 0}
```

Scoring uses `data/qna_with_answers.jsonl`:

```bash
python scripts/evaluate_mcq.py \
  --input hf_dataset/data/qna_with_answers.jsonl \
  --predictions path/to/predictions.jsonl
```

The evaluator reports multiple-choice accuracy, missing predictions, number correct, and label distributions.

## Intended Use And Limits

ForSeeBench is intended for evaluating prospective QA from prior AD context and comparing context policies or AD sources under fixed questions and options. It should not be used as a training corpus unless a future release explicitly defines such a protocol.

The benchmark does not redistribute raw media or source datasets, does not certify complete accessibility quality, and should not be used for high-stakes deployment claims or unrestricted future-event prediction.

## Citation

ForSeeBench:

```bibtex
@misc{forseebench2026,
  title = {ForSeeBench: Prospective Audio-Description QA for Evaluating Accessible Movie Understanding},
  author = {Anonymous Author(s)},
  year = {2026},
  note = {Anonymous benchmark submission artifact}
}
```

MAD / MAD-eval:

```bibtex
@inproceedings{soldan2022mad,
  author = {Soldan, Mattia and Pardo, Alejandro and Alc{\'a}zar, Juan Le{\'o}n and Caba Heilbron, Fabian and Zhao, Chen and Giancola, Silvio and Ghanem, Bernard},
  title = {{MAD}: A Scalable Dataset for Language Grounding in Videos from Movie Audio Descriptions},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = {5026--5035},
  year = {2022}
}
```

## License

The dataset license is `TODO(author)`. Original source assets remain governed by their original providers' terms.

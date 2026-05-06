# Source Dataset, Model, And Asset Citations

This document records source datasets, models, and generated-output artifacts that ForSeeBench references. Unknown URLs/licenses remain `TODO(author)`.

## MAD / MAD-eval

Primary source for movie audio-description examples.

```bibtex
@inproceedings{soldan2022mad,
  title = {MAD: A Scalable Dataset for Language Grounding in Videos From Movie Audio Descriptions},
  author = {Soldan, Mattia and Pardo, Alejandro and Alc{\'a}zar, Juan Le{\'o}n and Caba, Fabian and Zhao, Chen and Giancola, Silvio and Ghanem, Bernard},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2022}
}
```

- Paper citation present: yes, as `soldan2022mad`.
- URL/license/access terms: TODO(author).
- Redistribution: raw/full MAD/MAD-eval assets must not be redistributed. Derived AD text fields require source-field confirmation.

## LSMDC / Movie Description

- Role: source lineage and loader naming.
- Paper citation present: yes, as `rohrbach2017movie`.
- URL/license/access terms: TODO(author).
- Redistribution: do not redistribute raw LSMDC/Movie Description assets unless permitted.

## AutoAD-Zero / AutoAD

- Role: generated AD baseline/source in evaluation scripts and paper analysis.
- Paper citation present: yes, `han2023autoad`; AutoAD II also appears in references.
- URL/license/generated-output terms: TODO(author).
- Redistribution: do not upload generated outputs until provenance and redistribution rights are confirmed.

## NarrAD

- Role: generated AD source in evaluation artifacts/paper context.
- Complete citation/URL/license: TODO(author).
- Redistribution: do not upload NarrAD outputs until citation, provenance, and generated-output terms are confirmed.

## Qwen / Qwen2.5-VL-7B-Instruct

- Role: automated curation, validation, and answerer for full experiments.
- Model card URL/license/revision/citation: TODO(author).
- Redistribution: model weights are not redistributed.

## Movie Videos, Audio, Subtitles, Dialogue

- Role: underlying source assets behind the AD stream.
- Terms: TODO(author), via original providers/source datasets.
- Redistribution: do not redistribute raw videos, clips, audio, subtitles, dialogue files, or restricted source assets.

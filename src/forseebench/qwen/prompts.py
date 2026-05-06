"""Prompt templates for Qwen-assisted benchmark construction."""

from __future__ import annotations

import json
from typing import Any


EXTRACT_TARGET_ACTION_PROMPT = """You are an automated benchmark curator for ForSeeBench, a benchmark for evidence-grounded prospective audio-description question answering.

This stage runs after target/context selection. Given selected prior context clips and the selected hidden target clip:
- normalize the target sentence into a compact target development for QA generation.

The target sentence and target type are already known from upstream curation.
Do not copy or reclassify them in your JSON. The pipeline will attach those fields deterministically from the selected target clip and upstream labels.

The target development must be:
- concrete and informative enough to be the gold answer in a multiple-choice question
- grounded in the target clip sentence
- useful as an imminent development to anticipate from prior AD context
- not merely unmotivated generic motion, static description, or redundant continuation
- not invented from the context or movie knowledge
- not copied from an earlier context clip unless the target adds new action, state, spatial, participant, object/reveal, social/emotional, risk, or narrative-relevant information

Return strict JSON only.

Context clips:
{context_clips}

Actual next clip:
{target_clip}

Return only these fields. The JSON below is a schema template; replace every placeholder with values for this target.
{{
  "normalized_target": "one concise target development suitable as the gold answer"
}}"""


SCORE_EXPECTEDNESS_PROMPT = """You are an automated benchmark curator scoring how expected a target development is from selected prior AD/video-description context in ForSeeBench.

`expectedness` is a dataset difficulty / obviousness score.
It is not model confidence.

Definition:
- 1.0: the target development is nearly entailed by the prior context and may be too easy
- 0.0: the target development is unsupported, surprising, or likely follows a scene cut

Guidance:
- 0.9-1.0: almost directly implied / trivial
- 0.7-0.9: strongly expected
- 0.4-0.7: moderately expected; useful benchmark range
- 0.2-0.4: weakly expected or ambiguous
- 0.0-0.2: unsupported, surprising, or likely scene cut

Use only evidence in the provided context, not movie priors, genre expectations, or commonsense alone.

Return strict JSON only.

Context clips:
{context_clips}

Target development:
{target_action}

Return only this field. The JSON below is a schema template; replace the placeholder score with the actual score from the rubric.
{{
  "expectedness": "<float in [0,1]>"
}}"""


IDENTIFY_EVIDENCE_PROMPT = """You are an automated benchmark curator for ForSeeBench, a benchmark for evidence-grounded prospective ADQA.

Given selected prior context clips and the actual target development, determine whether the target is predictable from concrete evidence in the prior context.

A target is predictable only if the previous AD/video-description context contains evidence that points toward the actual target development.

Do not mark an example predictable merely because the action is common, plausible, genre-typical, or known from movie priors.
Dialogue cues can support an answer only when the cue is present in the provided context and the target remains grounded in the target AD sentence.

Evidence types:
- physical_precondition
- motion_trajectory
- object_affordance
- social_cue
- dialogue_cue
- emotional_cue
- scene_script
- hazard_cue
- other

Evidence span rules:
- Use exact short substrings from the provided context clips.
- Every evidence item must name one prior clip_id and one concrete span.
- Do not cite the target clip as evidence.
- If there is no exact span, return an empty evidence list and do not mark the item predictable.

Continuity types:
- continuous_physical
- continuous_social
- dialogue_continuous
- scene_cut_semantically_linked
- discontinuous_unpredictable

Predictability labels:
- predictable
- underdetermined
- unpredictable

Decision rules:
- predictable: concrete prior evidence supports the target development, even if it is not certain
- underdetermined: the target is meaningful but the context does not narrow the future enough
- unpredictable: context is unrelated, too weak, or likely separated by a hard scene change
- should_keep_for_main_benchmark must be true only for predictable examples with at least one precise evidence span
- Use only the predictability, continuity_type, evidence_type, and reasoning_type labels listed in this prompt; these labels are downstreamed into QA generation and validation.

Return strict JSON only.

Context clips:
{context_clips}

Actual target development:
{target_action}

Return only these fields. The JSON below is a schema template; replace every label, score, boolean, id, and span with values for this item.
{{
  "predictability": "predictable|underdetermined|unpredictable",
  "continuity_type": "continuous_physical|continuous_social|dialogue_continuous|scene_cut_semantically_linked|discontinuous_unpredictable",
  "evidence": [
    {{
      "clip_id": "prior_clip_id",
      "span": "exact supporting words from that prior clip",
      "evidence_type": "physical_precondition|motion_trajectory|object_affordance|social_cue|dialogue_cue|emotional_cue|scene_script|hazard_cue|other"
    }}
  ],
  "reasoning_type": ["physical_precondition|motion_trajectory|object_affordance|social_interaction|dialogue_grounded|intent_driven|scene_script|hazard_prediction|other"],
  "evidence_sufficiency": "<float in [0,1]>",
  "should_keep_for_main_benchmark": "<true|false>"
}}"""


GENERATE_QUESTION_PROMPT = """You are an automated benchmark curator for ForSeeBench. This stage generates a multiple-choice QA item after target/context curation.

Given the selected prior context, hidden target sentence/development, target type, evidence, and reasoning type, write one question and four answer options.

Research objective:
- The item should test whether prior AD/video-description context helps anticipate the hidden target development.
- The gold answer is grounded in the target sentence, but the selected prior context should make that answer more inferable than no context.
- Do not write generic recognition, current-scene, movie-trivia, or commonsense-only questions.

Category consistency:
- Use the supplied target_type to choose question_type; do not invent new question_type labels.
- question_type must be one of: what_happens_next, what_changes_next, what_is_revealed_next, what_text_appears_next.
- Do not use target_type labels such as action_transition, state_change, participant_update, spatial_consequence, object_reveal, or visible_text_update as question_type.
- Use the supplied evidence and reasoning_type as constraints for distractor design.
- Use only these distractor labels: correct, already_happened, entity_swapped, plausible_unsupported, contradicts_context, unrelated.

Requirements:
- Exactly one option must be correct.
- The correct option must be the actual target development grounded in the hidden target sentence.
- The QA must be answerable from the target sentence/clip.
- Distractors must be hard but wrong.
- Distractors should be plausible from the scene but unsupported or contradicted by the context.
- Do not make the correct answer longer, more detailed, or stylistically different in an obvious way.
- Do not include "all of the above", "none of the above", impossible meta-options, or options that differ only by tiny wording.
- Do not reveal the answer through target-clip wording in the question.
- The question should ask about the target's concrete update/development, not current-scene description or generic movie plot.
- The correct answer must not be obvious from generic priors, dialogue alone, movie knowledge, or option style.
- Include at least three of these distractor types when possible:
  - already_happened
  - entity_swapped
  - plausible_unsupported
  - contradicts_context
  - unrelated

Answer-option rules:
- Keep all four options similar in length and specificity.
- Each option should be a complete target development, not a label or fragment.
- Put the correct answer at any index; do not always use index 0.
- `distractor_metadata` must have exactly four labels and the correct answer's label must be `correct`.

Question guidance:
- `action_transition` or `spatial_consequence`: ask what development happens next
- `state_change`: ask what visibly changes next
- `participant_update`: ask what development involving the participant happens next
- `object_reveal`: ask what is revealed next
- `visible_text_update`: ask what text becomes visible next

Return strict JSON only.

Context:
{context_clips}

Actual target development:
{target_action}

Evidence:
{evidence}

Return only these fields. The JSON below is a schema template; replace the question, options, answer_idx, and labels with values for this item. Do not copy the sample answer index.
Use a JSON integer, not a string, for answer_idx.
{{
  "question_type": "what_happens_next|what_changes_next|what_is_revealed_next|what_text_appears_next",
  "question": "ask about the next target development",
  "options": ["gold target development", "hard wrong option", "hard wrong option", "hard wrong option"],
  "answer_idx": 0,
  "distractor_metadata": ["correct", "entity_swapped", "plausible_unsupported", "contradicts_context"]
}}"""


VALIDATE_EXAMPLE_PROMPT = """You are an automated benchmark curator and strict dataset quality validator for ForSeeBench, a NeurIPS benchmark.

You must reject examples that are noisy, ambiguous, trivial, leaked, or not evidence-grounded.

Research objective:
- Keep examples only when they support the benchmark claim: selected prior AD/video-description context improves anticipation of a hidden target sentence.
- Reject examples that can be solved by shortcuts instead of context-grounded anticipation.

Category consistency:
- Check that target_type, predictability, continuity_type, reasoning_type, evidence_type, question_type, and distractor_metadata use the allowed labels already established by upstream stages.
- Reject or list a failure reason if a downstream category contradicts upstream curation, such as a predictable label with no evidence spans or a correct answer not labeled correct.

Check these curator gates:
- oracle_target_answerability: the target sentence alone supports the gold answer.
- no_context_leakage: the answer is not obvious from the question/options alone.
- movie_prior_leakage: the answer is not obvious from movie title, genre, character priors, or external movie knowledge.
- dialogue_only_leakage: the answer cannot be solved from dialogue/audio alone without the AD context.
- wrong_context_leakage: shuffled, mismatched, or wrong-time context would not support the gold answer.
- selected_context_inferability: selected prior AD context makes the target more inferable than no context.
- grounding: the gold answer and rationale are grounded in the target sentence and do not cite outside knowledge.
- distractor_quality: exactly one answer is correct; distractors are plausible, distinct, and not absurd.

Reject if any required gate fails, if the target is trivial, or if the example is just caption continuation without context-grounded anticipation.

Return strict JSON only.

Example:
{full_example}

Return only these fields. The JSON below is a schema template; replace every score, boolean, and failure reason with values for this example.
Use an empty failure_reasons list only when all gates pass; otherwise list the failed gate names.
Use JSON booleans true/false, not strings, for should_keep.
{{
  "should_keep": true,
  "qwen_confidence": "<float in [0,1]>",
  "evidence_sufficiency": "<float in [0,1]>",
  "distractor_quality": "<float in [0,1]>",
  "failure_reasons": [],
  "recommended_fix": "short actionable fix or empty string"
}}"""


CONTINUITY_CLASSIFIER_PROMPT = """You are an automated benchmark curator selecting relevant temporal context for ForSeeBench, an evidence-grounded prospective ADQA benchmark.

Treat the provided context as a candidate context for the target clip, not as automatically valid context.

Use:
- candidate context clips
- target clip
- timestamp gap
- semantic similarity scores
- entity, action, and location overlap

Classify:
- continuity_type:
  - continuous_physical
  - continuous_social
  - dialogue_continuous
  - scene_cut_semantically_linked
  - discontinuous_unpredictable
- predictability:
  - predictable
  - underdetermined
  - unpredictable

Decision rules:
- `continuous_physical`: same local event continues with shared people, objects, space, state, or motion.
- `continuous_social`: same interaction continues with shared participants, intentions, reactions, or emotional cues.
- `dialogue_continuous`: continuity comes mainly from quoted or implied dialogue, not enough AD/context evidence by itself.
- `scene_cut_semantically_linked`: a new shot/scene is related but not directly continuous.
- `discontinuous_unpredictable`: target cannot be inferred from the candidate context.
- `should_keep` is true only for predictable items with continuous_physical or continuous_social continuity and at least one exact evidence span.
- Use only the continuity_type, predictability, evidence_type, and reasoning_type labels listed in this prompt; these labels are downstreamed into QA generation and validation.

Return strict JSON only.

Candidate context clips:
{context_clips}

Target clip:
{target_clip}

Signals:
{signals}

Return only these fields. The JSON below is a schema template; replace every label, score, boolean, id, and span with values for this item.
{{
  "continuity_type": "continuous_physical|continuous_social|dialogue_continuous|scene_cut_semantically_linked|discontinuous_unpredictable",
  "predictability": "predictable|underdetermined|unpredictable",
  "evidence": [
    {{
      "clip_id": "prior_clip_id",
      "span": "exact supporting words from that prior clip",
      "evidence_type": "physical_precondition|motion_trajectory|object_affordance|social_cue|dialogue_cue|emotional_cue|scene_script|hazard_cue|other"
    }}
  ],
  "evidence_sufficiency": "<float in [0,1]>",
  "reasoning_type": ["physical_precondition|motion_trajectory|object_affordance|social_interaction|dialogue_grounded|intent_driven|scene_script|hazard_prediction|other"],
  "should_keep": "<true|false>",
  "rejection_reason": null
}}"""


BLOCK_SELECTION_PROMPT = """You are an automated benchmark curator for ForSeeBench.

For each ordered block, Qwen receives ordered video/audio-description clip information and decides:
1. whether a benchmarkable target exists,
2. which clip is the target,
3. which earlier clips are the minimal supporting context,
4. whether the selected target is nontrivial and predictable from prior context.

You are given:
- an ordered block of clips from one movie, oldest to newest
- the block is only a bounded search region, not a pre-decided context/target split

Your job:
- decide whether any clip in the ordered block is a valid benchmark target using the rubric below
- choose exactly one target clip if one exists
- if multiple valid targets exist, select the earliest valid nontrivial target in block order
- prefer selecting a target with some earlier supporting context over rejecting the block when the block contains coherent local buildup
- do not force a target when the prior clips are only atmospheric, biographical, static, or unrelated to the target update
- if a concrete target exists but the prior context is too weak for the main benchmark, select it and label target_triviality="meaningful_but_noninferable" and predictability="underdetermined" instead of returning no target
- do not assume the final clip in the block is the target
- do not assume the immediate next clip is always the target
- reject the block only if no clip has any meaningful earlier buildup inside the block
- choose only earlier clips in the same block as supporting context; never choose context after or at the target
- select minimal earlier context clips only, strictly before the target
- reject trivial, generic, redundant, or unsupported targets
- classify target_type, target_triviality, predictability, and meaningful_but_noninferable cases
- decide whether the selected target is nontrivial and predictable from prior context

Good benchmark target sentence:
- concrete and informative enough to support a multiple-choice QA item
- grounded in the selected target AD sentence/clip
- more inferable from selected prior context than from no context
- useful as a short-horizon anticipatory target, not just a current-scene description
- may involve action, state, spatial relation, participant update, object/reveal, social/emotional reaction, risk, visible text, or narrative-relevant development
- not obvious from generic commonsense, movie knowledge, dialogue alone, or option style
- does not need to be dramatic, surprising, or uniquely predictable; ordinary actions are valid when earlier clips make that specific action more likely
- should contain a concrete target update that a QA question can ask about without relying on hidden movie knowledge

Target/context criteria:
- Nontrivial targets are target sentences that can support QA and become more inferable from selected prior context.
- A simple surface action can be nontrivial if prior context meaningfully builds toward that exact moment.
- Triviality is not decided by action difficulty alone; it is decided by whether the target would test context-grounded anticipation.
- Trivial targets are generic, redundant, low-information, obvious without context, prior-solvable, or too vague for robust QA.
- Meaningful-but-non-inferable targets are target sentences that are good target content but lack sufficient prior buildup.
- Use meaningful_but_noninferable for audit/challenge rows when the target itself is concrete and QA-worthy but the prior context does not support fair anticipation.
- Do not require the prior context to determine the target with certainty. For ForSeeBench, useful anticipation means the context makes the target a better forecast than no-context guessing.
- A valid target/context pair must preserve the same local thread: the same participant, object, place, motion path, social exchange, danger, goal, or interrupted action should continue from context into target.
- Reject target/context pairs where the context only gives identity/background/mood and does not help forecast the target's concrete update.
- Fragmentary target sentences are allowed only when selected context includes the immediate antecedent needed to understand the fragment. For example, a target like "Knocking him and his torch to the floor" must include the prior clip that names the force or actor doing the knocking.
- If the target uses pronouns or omitted subjects ("he", "she", "it", "they", "knocking...", "then..."), selected context must include the nearest earlier clip that resolves the participant/action thread.

Context selection rules:
- Context clips must occur strictly before the chosen target.
- Context should not be fixed to one sentence or one clip.
- Select the minimal sufficient suffix or subset of earlier clips in the ordered block, usually 1-4 clips unless more are clearly needed.
- Include every clip cited in selected_context_spans.
- Evidence spans must be exact short substrings from earlier AD/video-description clips.
- Prior context can support the target through entity continuity, spatial setup, object continuity, action precondition, social/emotional cue, dialogue cue, risk cue, scene goal, or narrative setup.
- Do not use movie knowledge, genre priors, or generic commonsense as evidence.
- If no earlier context meaningfully supports the target, label it meaningful_but_noninferable or reject the block.
- Only `nontrivial` and predictable targets are eligible for the main benchmark.
- If a prior clip sets up a person, object, place, motion path, social situation, danger, goal, or interrupted action, and a later target develops that setup, select that later target and cite the setup span.
- Evidence must explain one concrete detail in the target sentence. If the cited span cannot answer "why was this target more likely?", it is not valid evidence.
- Evidence must include the most direct antecedent clip when one exists, not only earlier background setup. Do not skip the immediately preceding setup if it directly causes or explains the target.
- Do not cite atmospheric setup alone, such as clouds, lighting, music-like mood, flags, static rooms, or character identity, unless that span directly sets up the target action or state change.
- Do not cite a previous violent, magical, or surprising event as support for an unrelated next shot; that is a scene transition, not anticipatory evidence.

Allowed target types:
- action_transition
- state_change
- participant_update
- spatial_consequence
- object_reveal
- visible_text_update

Triviality labels:
- nontrivial
- trivial_generic_motion
- trivial_static_state
- trivial_redundant_continuation
- trivial_low_information
- meaningful_but_noninferable

Use trivial labels as follows:
- nontrivial: target sentence has enough detail for MCQ options and selected prior context makes it more inferable, even if the surface action is simple
- trivial_generic_motion: low-information movement such as looking, walking, turning, standing, smiling, entering, or leaving only when prior context does not make that exact moment informative or anticipatable
- trivial_static_state: mostly describes someone or something remaining in place or a static scene
- trivial_redundant_continuation: repeats a prior clip with no new benchmarkable information
- trivial_low_information: too vague, underspecified, pronoun-heavy, or generic to support a robust MCQ
- meaningful_but_noninferable: informative target, but the prior clips do not provide enough evidence to prefer it over other plausible futures

Examples of trivial targets:
- "He looks around." when no prior context makes the look meaningful.
- "She walks away." when no prior context builds toward why leaving matters.
- "They stand in the room."
- "He turns." when it is only generic motion with no context-supported implication.

Examples of nontrivial targets:
- "She walks away." if prior context establishes a confrontation, decision, threat, or departure setup.
- "He turns." if prior context establishes he heard or noticed something important.
- "They stand in the room." if prior context builds toward a tense pause, reveal, or reaction.
- "Water starts running into the sink basin." if prior context establishes the person moved to the sink or prepared to use it.

Predictability labels:
- predictable
- underdetermined
- unpredictable

Use predictability labels as follows:
- predictable: selected context makes the target more likely than no-context guessing and provides concrete evidence spans; certainty is not required
- underdetermined: target is meaningful and has some local continuity, but the selected context is too weak to support a fair anticipatory QA item
- unpredictable: prior clips are unrelated, give no usable buildup, or the target depends mainly on new information in the target clip

Reasoning types:
- physical_precondition
- motion_trajectory
- object_affordance
- social_interaction
- dialogue_grounded
- intent_driven
- scene_script
- hazard_prediction
- other

Continuity types:
- continuous_physical
- continuous_social
- dialogue_continuous
- scene_cut_semantically_linked
- discontinuous_unpredictable

Category consistency:
- The selected target_type, target_triviality, predictability, continuity_type, evidence_type, and reasoning_type labels are downstreamed into target extraction, QA generation, and validation.
- Use only labels listed in this prompt.
- Do not invent alternate labels such as eventive_action, temporally_reversed, or target_action_hint.
- Do not output target_sentence, context_sentences, or audio-description text except exact evidence spans; the pipeline derives those source fields from clip IDs.
- Use evidence_type only from the evidence-type labels listed in this prompt; do not use target-type labels as evidence_type.
- reasoning_type must be an array, even if there is only one reason.
- target_validity_reason must refer to the selected target sentence, not a different possible target in the block.
- evidence_sufficiency must reflect actual support strength, not a default value:
  - 0.0-0.2: no useful support
  - 0.3-0.5: weak or mostly generic continuity
  - 0.6-0.75: usable support but not strong
  - 0.76-0.9: clear direct setup for the target update
  - above 0.9: nearly explicit setup without revealing the answer

Reject these false-positive patterns:
- Identity/background only -> action: "the dad wears a clerical collar" does not support "he sits up in bed."
- Atmosphere only -> action: "dark clouds cover the moon" does not support a specific body movement unless the target directly continues that visual event.
- Prior shock -> unrelated insert: a death, flash, explosion, or threat does not support an unrelated cutaway such as a kettle boiling.
- Object/place mention -> arbitrary action: a tent, flag, house, or room only supports target details that develop that same object/place thread.
- Context that already gives away the answer is leakage, not anticipation.

If a block contains a coherent setup followed by a concrete development, select the earliest such development as the target and return its minimal supporting prior context.
If a concrete QA-worthy target exists but the context is too weak, still return found_valid_target=true with that target, target_triviality="meaningful_but_noninferable", predictability="underdetermined", and the best weak prior context/evidence you can cite.
Return found_valid_target=false only when no concrete QA-worthy target exists in the block or all possible targets are generic/static/redundant/low-information.

Return no prose, markdown, or comments outside the JSON object.

Return strict JSON only.
Use JSON booleans true/false and JSON null, not strings, for boolean/null fields.

Block clips:
{block_clips}

Return only these fields. The JSON below is a schema template; replace every target id, position, label, score, boolean, context id, and evidence span with values for this ordered block.
{{
  "found_valid_target": true,
  "target_clip_id": "clip_id_from_block_or_null",
  "target_position_in_block": "<1-based integer position in block or null>",
  "target_type": "action_transition|state_change|participant_update|spatial_consequence|object_reveal|visible_text_update",
  "target_triviality": "nontrivial|trivial_generic_motion|trivial_static_state|trivial_redundant_continuation|trivial_low_information|meaningful_but_noninferable",
  "target_validity_reason": "short reason for the target/context decision",
  "selected_context_clip_ids": ["prior_clip_id_or_empty_if_rejected"],
  "selected_context_spans": [
    {{
      "clip_id": "prior_clip_id_or_empty_if_rejected",
      "span": "exact supporting words from prior clip, or empty if rejected",
      "evidence_type": "physical_precondition|motion_trajectory|object_affordance|social_cue|dialogue_cue|emotional_cue|scene_script|hazard_cue|other"
    }}
  ],
  "predictability": "predictable|underdetermined|unpredictable",
  "continuity_type": "continuous_physical|continuous_social|dialogue_continuous|scene_cut_semantically_linked|discontinuous_unpredictable",
  "evidence_sufficiency": "<float in [0,1]>",
  "reasoning_type": ["physical_precondition"]
}}"""


def format_context_clips(context: list[dict[str, Any]]) -> str:
    """Render context clips into a prompt-friendly block."""

    parts: list[str] = []
    for row in context:
        parts.append(
            f"- {row['clip_id']} [{row.get('timestamp_start')}, {row.get('timestamp_end')}]: "
            f"{row.get('audio_description', '')}"
        )
    return "\n".join(parts)


def collect_video_paths(rows: list[dict[str, Any]]) -> list[str]:
    """Return existing video paths from prompt rows in order."""

    paths: list[str] = []
    for row in rows:
        path = row.get("video_path")
        if isinstance(path, str) and path:
            paths.append(path)
    return paths


def make_extract_action_prompt(context: list[dict[str, Any]], target: dict[str, Any]) -> str:
    return EXTRACT_TARGET_ACTION_PROMPT.format(
        context_clips=format_context_clips(context),
        target_clip=json.dumps(target, ensure_ascii=False, indent=2),
    )


def make_score_expectedness_prompt(context: list[dict[str, Any]], target_action: dict[str, Any]) -> str:
    return SCORE_EXPECTEDNESS_PROMPT.format(
        context_clips=format_context_clips(context),
        target_action=json.dumps(target_action, ensure_ascii=False, indent=2),
    )


def make_identify_evidence_prompt(context: list[dict[str, Any]], target_action: dict[str, Any]) -> str:
    return IDENTIFY_EVIDENCE_PROMPT.format(
        context_clips=format_context_clips(context),
        target_action=json.dumps(target_action, ensure_ascii=False, indent=2),
    )


def make_question_prompt(
    context: list[dict[str, Any]],
    target_action: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> str:
    return GENERATE_QUESTION_PROMPT.format(
        context_clips=format_context_clips(context),
        target_action=json.dumps(target_action, ensure_ascii=False, indent=2),
        evidence=json.dumps(evidence, ensure_ascii=False, indent=2),
    )


def make_validate_prompt(example: dict[str, Any]) -> str:
    return VALIDATE_EXAMPLE_PROMPT.format(full_example=json.dumps(example, ensure_ascii=False, indent=2))


def make_continuity_prompt(candidate: dict[str, Any]) -> str:
    signals = {
        "timestamp_gap": candidate.get("timestamp_gap"),
        "semantic_similarity_last": candidate.get("semantic_similarity_last"),
        "semantic_similarity_mean": candidate.get("semantic_similarity_mean"),
        "entity_overlap": candidate.get("entity_overlap"),
        "action_overlap": candidate.get("action_overlap"),
        "location_overlap": candidate.get("location_overlap"),
        "context_entities": candidate.get("context_entities"),
        "target_entities": candidate.get("target_entities"),
    }
    return CONTINUITY_CLASSIFIER_PROMPT.format(
        context_clips=format_context_clips(candidate["context"]),
        target_clip=json.dumps(candidate["target"], ensure_ascii=False, indent=2),
        signals=json.dumps(signals, ensure_ascii=False, indent=2),
    )


def make_block_selection_prompt(block: dict[str, Any]) -> str:
    return BLOCK_SELECTION_PROMPT.format(block_clips=format_context_clips(block["clips"]))


def make_block_selection_video_prompt(block: dict[str, Any]) -> tuple[str, list[str]]:
    """Return the block-selection prompt plus ordered video paths for Qwen-VL."""

    clips = block["clips"]
    prompt = make_block_selection_prompt(block)
    return prompt, collect_video_paths(clips)

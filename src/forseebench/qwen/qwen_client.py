"""Qwen local Hugging Face backend for Qwen2.5-VL."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from forseebench.qwen.parse_outputs import parse_json_object


@dataclass(slots=True)
class QwenCallResult:
    """Structured result from one Qwen call."""

    prompt_name: str
    raw_text: str
    parsed: dict[str, Any]


@dataclass(slots=True)
class QwenClient:
    """Support local Hugging Face execution for Qwen2.5-VL."""

    backend: str = "local_hf"
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    timeout_seconds: int = 120
    temperature: float = 0.1
    top_p: float = 0.8
    max_tokens: int = 2048
    cache_dir: str | None = None
    device_map: str = "auto"
    torch_dtype: str = "auto"
    attn_implementation: str | None = None
    max_calls: int | None = None
    video_fps: float | None = 1.0
    video_max_pixels: int | None = 150528
    video_min_pixels: int | None = None
    calls_made: int = field(init=False, default=0)
    failure_log_path: Path = field(init=False)
    _hf_model: Any | None = field(init=False, default=None)
    _hf_processor: Any | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.calls_made = 0
        self.failure_log_path = Path("data/interim/qwen_failures.jsonl")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QwenClient":
        return cls(**config)

    def is_enabled(self) -> bool:
        return self.backend != "disabled"

    def generate_json(self, prompt_name: str, prompt: str, *, temperature: float | None = None) -> QwenCallResult:
        """Generate and parse a strict JSON response."""

        return self.generate_json_with_videos(prompt_name, prompt, video_paths=None, temperature=temperature)

    def generate_json_with_videos(
        self,
        prompt_name: str,
        prompt: str,
        *,
        video_paths: list[str] | None = None,
        temperature: float | None = None,
    ) -> QwenCallResult:
        """Generate and parse a strict JSON response with optional video inputs."""

        if not self.is_enabled():
            raise RuntimeError("Qwen backend is disabled")
        if self.max_calls is not None and self.calls_made >= self.max_calls:
            raise RuntimeError("Qwen max_calls limit reached")
        self.calls_made += 1

        if self.backend == "local_hf":
            raw_text = self._generate_via_local_hf(prompt, video_paths=video_paths, temperature=temperature)
        else:
            raise ValueError(f"unsupported Qwen backend: {self.backend}")

        try:
            parsed = parse_json_object(raw_text)
        except Exception as exc:
            self._record_failure(prompt_name, prompt, raw_text, str(exc))
            raise
        return QwenCallResult(prompt_name=prompt_name, raw_text=raw_text, parsed=parsed)

    def _generate_via_local_hf(
        self,
        prompt: str,
        *,
        video_paths: list[str] | None,
        temperature: float | None,
    ) -> str:
        model, processor = self._load_local_hf()
        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in video_paths or []:
            video_content: dict[str, Any] = {"type": "video", "video": path}
            if self.video_fps is not None:
                video_content["fps"] = self.video_fps
            if self.video_max_pixels is not None:
                video_content["max_pixels"] = self.video_max_pixels
            if self.video_min_pixels is not None:
                video_content["min_pixels"] = self.video_min_pixels
            user_content.append(video_content)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Return only strict JSON."}]},
            {"role": "user", "content": user_content},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processor_kwargs: dict[str, Any] = {"text": [text], "padding": True, "return_tensors": "pt"}
        if video_paths:
            image_inputs, video_inputs = self._process_vision_info(messages)
            if image_inputs is not None:
                processor_kwargs["images"] = image_inputs
            if video_inputs is not None:
                processor_kwargs["videos"] = video_inputs
        inputs = processor(**processor_kwargs)
        inputs = inputs.to(model.device)
        effective_temperature = self.temperature if temperature is None else temperature
        output_ids = model.generate(
            **inputs,
            do_sample=effective_temperature > 0,
            temperature=effective_temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_tokens,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        decoded = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""

    def _process_vision_info(self, messages: list[dict[str, Any]]) -> tuple[Any | None, Any | None]:
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as exc:
            raise RuntimeError(
                "Video inputs require qwen-vl-utils. Install with: pip install qwen-vl-utils"
            ) from exc
        return process_vision_info(messages)

    def _load_local_hf(self) -> tuple[Any, Any]:
        if self._hf_model is not None and self._hf_processor is not None:
            return self._hf_model, self._hf_processor
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(
                "Qwen2.5-VL local backend requires the latest Hugging Face transformers. "
                "Install with: pip install git+https://github.com/huggingface/transformers accelerate"
            ) from exc
        model_kwargs: dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "device_map": self.device_map,
            "torch_dtype": self._resolve_torch_dtype(self.torch_dtype),
        }
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        self._hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model,
            **model_kwargs,
        )
        self._hf_processor = AutoProcessor.from_pretrained(
            self.model,
            cache_dir=self.cache_dir,
        )
        return self._hf_model, self._hf_processor

    def _resolve_torch_dtype(self, value: str | None) -> Any:
        if value in {None, "", "auto"}:
            return value or "auto"
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch is required to resolve a non-auto torch_dtype") from exc
        try:
            return getattr(torch, value)
        except AttributeError as exc:
            raise ValueError(f"unsupported torch_dtype setting: {value}") from exc

    def _record_failure(self, prompt_name: str, prompt: str, output: str, error: str) -> None:
        self.failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "call_index": self.calls_made,
            "prompt_name": prompt_name,
            "error": error,
            "prompt_preview": prompt[:500],
            "output_preview": output[:1000],
        }
        with self.failure_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

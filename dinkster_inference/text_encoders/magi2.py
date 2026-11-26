from __future__ import annotations

from collections.abc import Iterable
import json
import os
from typing import Any, Optional

import torch
from tokenizers import Regex, Tokenizer, pre_tokenizers
from transformers import AutoTokenizer

from dinkster_inference import sd1_clip
import dinkster_inference.text_encoders.qwen35


_CJK_ISOLATED_REGEX = (
    r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF"
    r"\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
)


def _strip_empty(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: v
            for k, v in ((k, _strip_empty(v)) for k, v in obj.items())
            if v is not None and v != [] and v != {}
        }
    if isinstance(obj, list):
        return [_strip_empty(v) for v in obj if v is not None]
    return obj


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _pick(values: dict[str, Any], keys: Iterable[str]) -> list[tuple[str, Any]]:
    return [(k, values[k]) for k in keys if k in values and values[k] not in (None, [], {})]


def json_to_compact_markdown(raw_json: str | dict[str, Any]) -> str:
    obj = json.loads(raw_json.strip()) if isinstance(raw_json, str) else raw_json
    obj = _strip_empty(obj)
    if not isinstance(obj, dict) or "global_layer" not in obj:
        return raw_json if isinstance(raw_json, str) else json.dumps(raw_json, ensure_ascii=False)

    global_layer = _as_dict(obj.get("global_layer"))
    dynamic_layer = _as_dict(obj.get("dynamic_layer"))
    reference_layer = obj.get("reference_layer", [])

    lines: list[str] = []
    context = _one_line(global_layer.get("context"))
    description = _one_line(global_layer.get("description"))
    if context or description:
        lines.append(f"context: {context}" if context else "context")
        if description:
            lines.append(description)

    aesthetics = _as_dict(global_layer.get("aesthetics"))
    aesthetic_parts = [
        f"{label}={_one_line(aesthetics.get(key))}"
        for label, key in (("style", "style"), ("mood", "mood_atmosphere"), ("color", "color_scheme"))
        if _one_line(aesthetics.get(key))
    ]
    if aesthetic_parts:
        lines.append("aesthetics: " + "; ".join(aesthetic_parts))

    audio = _as_dict(global_layer.get("audio_baseline"))
    dialogue = _as_dict(audio.get("dialogue"))
    audio_parts = []
    language = _one_line(dialogue.get("language"))
    speakers = _as_list(dialogue.get("speaker_tags"))
    ambience = _one_line(audio.get("ambience"))
    if language:
        audio_parts.append(f"language={language}")
    if speakers:
        audio_parts.append("speakers=" + ",".join(map(_one_line, speakers)))
    if ambience:
        audio_parts.append(f"ambience={ambience}")
    if audio_parts:
        lines.append("audio: " + "; ".join(audio_parts))

    subjects = global_layer.get("alive_subjects_static")
    if isinstance(subjects, list) and subjects:
        lines.append("subjects:")
        for subject in subjects:
            if not isinstance(subject, dict):
                continue
            sid = _one_line(subject.get("subject_id"))
            desc = _one_line(subject.get("description"))
            position = _one_line(subject.get("position"))
            orientation = _one_line(subject.get("orientation"))
            attrs = [
                f"{k}={_one_line(v)}"
                for k, v in _pick(
                    _as_dict(subject.get("visual_attributes")),
                    ("gender", "age_group", "ethnicity", "clothing", "appearance_details"),
                )
                if _one_line(v)
            ]
            row = " - " + " | ".join(p for p in (sid, desc) if p)
            extra = "; ".join(p for p in (position, orientation) if p)
            if extra:
                row += f" ({extra})"
            if attrs:
                row += " :: " + "; ".join(attrs)
            lines.append(row)

    objects = global_layer.get("objects_static")
    if isinstance(objects, list) and objects:
        lines.append("objects:")
        for obj_item in objects:
            if not isinstance(obj_item, dict):
                continue
            oid = _one_line(obj_item.get("object_id"))
            desc = _one_line(obj_item.get("description"))
            shape = _one_line(obj_item.get("shape_and_color"))
            position = _one_line(obj_item.get("position"))
            row = " - " + " | ".join(p for p in (oid, desc) if p)
            details = "; ".join(p for p in (shape, position) if p)
            if details:
                row += " :: " + details
            lines.append(row)

    segments = dynamic_layer.get("timeline_segments")
    if isinstance(segments, list) and segments:
        lines.append("timeline:")
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            basic = _as_dict(segment.get("segment_basic_info"))
            ts = _one_line(basic.get("timestamp_range"))
            desc = _one_line(basic.get("segment_description"))
            head = f" - {ts}" if ts else " -"
            if desc:
                head += f" {desc}"
            lines.append(head.rstrip())

            seg_audio = _as_dict(segment.get("audio"))
            for dline in seg_audio.get("dialogue_lines", []) or []:
                if not isinstance(dline, dict):
                    continue
                speaker = _one_line(dline.get("speaker"))
                text = _one_line(dline.get("text"))
                dts = _one_line(dline.get("timestamp"))
                if text:
                    prefix = f"   - dialogue {speaker}: " if speaker else "   - dialogue: "
                    line = prefix + text + (f" ({dts})" if dts else "")
                    lines.append(line)

            for alive in segment.get("alive_subjects", []) or []:
                if not isinstance(alive, dict):
                    continue
                sid = _one_line(alive.get("subject_id"))
                action = _as_dict(alive.get("action"))
                parts = [
                    _one_line(action.get(k))
                    for k in ("primary_action", "interaction", "facial_expression")
                ]
                parts = [p for p in parts if p]
                if parts:
                    lines.append(f"   - action {sid}: " + "; ".join(parts))

            for obj_item in segment.get("objects", []) or []:
                if not isinstance(obj_item, dict):
                    continue
                oid = _one_line(obj_item.get("object_id"))
                state = _as_dict(obj_item.get("dynamic_state"))
                parts = [_one_line(state.get(k)) for k in ("state_change", "motion_detail")]
                parts = [p for p in parts if p]
                if parts:
                    lines.append(f"   - objects {oid}: " + "; ".join(parts))

    if isinstance(reference_layer, list):
        lines.extend(str(line) for line in reference_layer if str(line).strip())

    return "\n".join(line for line in lines if line.strip())


def normalize_magi2_prompt(prompt: str, ensure_figure_ids: Optional[list[int]] = None) -> str:
    if ensure_figure_ids:
        missing = [fid for fid in ensure_figure_ids if f"<Figure {fid}>" not in prompt]
        if missing:
            ref_lines = [f"The first frame refers to <Figure {fid}>" for fid in missing]
            try:
                parsed = json.loads(prompt)
                if not isinstance(parsed, dict):
                    raise ValueError("not dict")
                existing = list(parsed.get("reference_layer") or [])
                existing.extend(ref_lines)
                parsed["reference_layer"] = existing
                prompt = json.dumps(parsed, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError, ValueError):
                prompt = prompt + "reference_layer:" + "; ".join(ref_lines)

    try:
        parsed_json = json.loads(prompt)
        if isinstance(parsed_json, (dict, list)):
            return json_to_compact_markdown(parsed_json)
    except (json.JSONDecodeError, TypeError):
        pass
    return prompt


def _build_cjk_backend_tokenizer(tokenizer_json: Optional[str] = None) -> Tokenizer:
    tok_dir = os.path.join(
        os.path.dirname(os.path.realpath(dinkster_inference.text_encoders.qwen35.__file__)),
        "qwen35_tokenizer",
    )
    if tokenizer_json is not None:
        backend = Tokenizer.from_str(tokenizer_json)
    else:
        fast = AutoTokenizer.from_pretrained(tok_dir, padding_side="right", local_files_only=True)
        backend = fast.backend_tokenizer
    cjk_split = pre_tokenizers.Split(pattern=Regex(_CJK_ISOLATED_REGEX), behavior="isolated")
    orig_pre = backend.pre_tokenizer
    backend.pre_tokenizer = (
        pre_tokenizers.Sequence([cjk_split, orig_pre]) if orig_pre is not None else cjk_split
    )
    return backend


class Magi2SDTokenizer(dinkster_inference.text_encoders.qwen35.Qwen35Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}) -> None:
        super().__init__(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data,
            embedding_size=5120,
            embedding_key="qwen35_27b",
        )
        tok_json = tokenizer_data.get("qwen35_27b_tokenizer_json")
        self.fast_backend = _build_cjk_backend_tokenizer(tok_json)


class Magi2Tokenizer(sd1_clip.SD1Tokenizer):
    """MAGI-2 raw prompt tokenizer over Qwen3.5-27B (no chat template, isolated CJK pre-split)."""

    def __init__(self, embedding_directory=None, tokenizer_data={}) -> None:
        super().__init__(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data,
            name="qwen35_27b",
            tokenizer=Magi2SDTokenizer,
        )

    def tokenize_with_weights(
        self,
        text: str,
        return_word_ids: bool = False,
        max_length: int = 7000,
        magi2_figure_ids: Optional[list[int]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        sub = getattr(self, self.clip)
        normalized = normalize_magi2_prompt(text, ensure_figure_ids=magi2_figure_ids)
        if not normalized:
            # Empty unconditional prompt -> 0 tokens when uncond is empty, or single pad when needed
            return {
                self.clip: [[]],
                "magi2_txt_len": 0,
                "magi2_figure_spans": {},
            }

        encoded = sub.fast_backend.encode(normalized, add_special_tokens=False)
        ids = list(encoded.ids[:max_length])
        offsets = list(encoded.offsets[:max_length])

        figure_spans: dict[int, list[int]] = {}
        for fid in magi2_figure_ids or []:
            tag = f"<Figure {int(fid)}>"
            start_char = normalized.find(tag)
            if start_char == -1:
                figure_spans[int(fid)] = []
                continue
            end_char = start_char + len(tag)
            span = [
                idx
                for idx, (s, e) in enumerate(offsets)
                if not (s == 0 and e == 0 and idx != 0) and max(s, start_char) < min(e, end_char)
            ]
            figure_spans[int(fid)] = span

        pairs = [(tid, 1.0, idx) if return_word_ids else (tid, 1.0) for idx, tid in enumerate(ids)]
        return {
            self.clip: [pairs],
            "magi2_txt_len": len(ids),
            "magi2_figure_spans": figure_spans,
        }


class Magi2TEModel(sd1_clip.SD1ClipModel):
    """Qwen3.5-27B text encoder tapped at `layer_idx=-3` (`skip_layer=2`, layer 62 of 64 unnormed)."""

    def __init__(self, device="cpu", dtype=None, model_options={}, mtp: bool = False) -> None:
        clip_model = lambda **kw: dinkster_inference.text_encoders.qwen35.Qwen35ClipModel(
            **kw,
            layer="hidden",
            layer_idx=-3,
            attention_mask=True,
            model_type="qwen35_27b",
            mtp=mtp,
        )
        super().__init__(
            device=device,
            dtype=dtype,
            name="qwen35_27b",
            clip_model=clip_model,
            model_options=model_options,
        )

    def encode_token_weights(self, token_weight_pairs: dict[str, Any]):
        pairs_copy = dict(token_weight_pairs)
        txt_len = pairs_copy.pop("magi2_txt_len", None)
        figure_spans: dict[int, list[int]] = pairs_copy.pop("magi2_figure_spans", {}) or {}

        raw_seq = pairs_copy.get("qwen35_27b", [[]])[0]
        if len(raw_seq) == 0:
            # Empty unconditional feature [1, 0, 5120]
            empty = torch.zeros((1, 0, 5120), dtype=torch.float32)
            return empty, None, {"magi2_txt_len": 0, "magi2_figure_tokens": {}}

        out, pooled, extra = super().encode_token_weights(pairs_copy)
        extra = dict(extra) if isinstance(extra, dict) else {}
        extra.pop("attention_mask", None)

        actual_len = int(txt_len) if txt_len is not None else out.shape[1]
        out = out[:, :actual_len]

        figure_tokens: dict[int, torch.Tensor] = {}
        for fid, indices in figure_spans.items():
            valid_idx = [i for i in indices if 0 <= i < out.shape[1]]
            if valid_idx:
                figure_tokens[int(fid)] = out[0, valid_idx, :].mean(dim=0).clone()
            else:
                figure_tokens[int(fid)] = torch.zeros(
                    (out.shape[-1],), device=out.device, dtype=out.dtype
                )

        extra["magi2_txt_len"] = actual_len
        extra["magi2_figure_tokens"] = figure_tokens
        return out, pooled, extra


def te(
    dtype_llama=None,
    llama_quantization_metadata=None,
    mtp: bool = False,
    vocab_size: int = 248320,
    tokenizer_json: Optional[str] = None,
):
    config: dict[str, Any] = {"vocab_size": vocab_size}
    if tokenizer_json is not None:
        vocab = Tokenizer.from_str(tokenizer_json).get_vocab()
        config["stop_tokens"] = [vocab["<|endoftext|>"], vocab["<|im_end|>"]]

    class Magi2TEModel_(Magi2TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}) -> None:
            resolved_dtype = dtype_llama if dtype_llama is not None else dtype
            opts = model_options.copy()
            opts["qwen35_27b_model_config"] = {
                **config,
                **opts.get("qwen35_27b_model_config", {}),
            }
            if llama_quantization_metadata is not None:
                opts["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=resolved_dtype, model_options=opts, mtp=mtp)

    return Magi2TEModel_

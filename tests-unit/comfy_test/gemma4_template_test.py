"""Gemma4 chat template regression tests."""

import re

import pytest
import torch

from dinkster_inference.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import dinkster_inference.text_encoders.gemma4 as gemma4  # noqa: E402
import dinkster_inference.text_encoders.lt as lt  # noqa: E402

PROMPT = "describe a cute anime girl with fennec ears"
THOUGHT_BLOCK = "<|channel>thought\n<channel|>"

# E2B/E4B and 12B/31B ship different canonical chat templates: only the latter prime a
# closed thought block when thinking is off.
NO_PRIMING = [gemma4.Gemma4_E2B, gemma4.Gemma4_E4B]
PRIMING = [gemma4.Gemma4_31B, gemma4.Gemma4_12B]


class _CaptureTemplate:
    """Stands in for SDTokenizer.tokenize_with_weights so the built template is checked without model files."""
    llama_text = ""

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        self.llama_text = text
        return {}


def build_template(variant, **kwargs):
    prime = variant.tokenizer.tokenizer_class.prime_empty_thought
    probe = type("Probe", (gemma4.Gemma4_Tokenizer, _CaptureTemplate), {"prime_empty_thought": prime})()
    probe.tokenize_with_weights(PROMPT, **kwargs)
    return probe.llama_text


@pytest.mark.parametrize("variant", NO_PRIMING + PRIMING)
def test_thinking_enabled_only_asks_via_the_system_turn(variant):
    template = build_template(variant, skip_template=False, thinking=True)
    assert template == f"<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n{PROMPT}<turn|>\n<|turn>model\n"


@pytest.mark.parametrize("variant", NO_PRIMING)
def test_thinking_disabled_does_not_prime_a_thought_channel(variant):
    template = build_template(variant, skip_template=False, thinking=False)
    assert template == f"<|turn>user\n{PROMPT}<turn|>\n<|turn>model\n"
    assert "channel" not in template
    assert "<|think|>" not in template


@pytest.mark.parametrize("variant", PRIMING)
def test_thinking_disabled_primes_a_thought_channel(variant):
    template = build_template(variant, skip_template=False, thinking=False)
    assert template == f"<|turn>user\n{PROMPT}<turn|>\n<|turn>model\n{THOUGHT_BLOCK}"


@pytest.mark.parametrize("variant", NO_PRIMING + PRIMING)
@pytest.mark.parametrize("thinking", [False, True])
def test_skip_template_passes_text_through_unchanged(variant, thinking):
    assert build_template(variant, skip_template=True, thinking=thinking) == PROMPT


class _MediaTokens:
    """Tokenize media markers without requiring a downloaded vocabulary."""

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        self.llama_text = text
        ids = {"<image_soft_token>": 262144, "<|image|>": 258880,
               "<|video|>": 258884, "<|audio|>": 258881}
        parts = re.split(r"(<image_soft_token>|<\|image\|>|<\|video\|>|<\|audio\|>)", text)
        return [[(ids.get(part, 0), 1.0) for part in parts if part]]


class _Gemma4Media(gemma4.Gemma4_Tokenizer, _MediaTokens):
    pass


class _Gemma3Media(lt.Gemma3_Tokenizer, _MediaTokens):
    pass


def test_gemma4_multiple_media_preserve_order_and_separate_audio_lengths():
    probe = _Gemma4Media()
    images = [torch.full((1, 48, 48, 3), value) for value in (0.2, 0.8)]
    videos = [torch.tensor(values).view(-1, 1, 1, 1).expand(-1, 48, 48, 3)
              for values in ([0.1, 0.2, 0.3, 0.4, 0.5], [0.9, 1.0])]
    audios = [{"waveform": torch.zeros(1, 1, size), "sample_rate": rate}
              for size, rate in [(1600, 16000), (6400, 32000)]]
    tokens = probe.tokenize_with_weights(
        PROMPT, images=images, videos=videos, audios=audios,
        image=images[0], video=videos[0], audio=audios[0],
        skip_template=False, fps=2, max_soft_tokens=1,
    )
    embeds = [token[0] for token in tokens[0] if isinstance(token[0], dict)]
    assert [embed["type"] for embed in embeds] == ["image"] * 6 + ["audio"] * 2
    pixels = [embed["data"][0, 0, 0, 0].item() for embed in embeds[:6]]
    assert pixels == pytest.approx([51 / 255, 204 / 255, 25 / 255, 76 / 255, 127 / 255, 229 / 255])
    assert re.findall(r"\d\d:\d\d", probe.llama_text) == ["00:00", "00:01", "00:02", "00:00"]
    assert [part.split("<audio|>")[0].count("<|audio|>")
            for part in probe.llama_text.split("<|audio>")[1:]] == [3, 5]
    assert [embed["data"].shape for embed in embeds[6:]] == [(1, 10, 128), (1, 19, 128)]
    assert [embed["mask"].shape for embed in embeds[6:]] == [(1, 10), (1, 19)]
    # A later legacy call must not inherit a previous call's lists.
    legacy = probe.tokenize_with_weights(PROMPT, image=images[1], skip_template=False, max_soft_tokens=1)
    legacy_images = [token[0] for token in legacy[0] if isinstance(token[0], dict)]
    assert len(legacy_images) == 1
    torch.testing.assert_close(legacy_images[0]["data"], images[1])


@pytest.mark.parametrize("mode", ["list", "legacy", "list_precedence"])
def test_gemma3_image_lists_and_legacy_input(mode):
    probe = _Gemma3Media()
    images = [torch.full((1, 16, 32, 3), value) for value in (0.2, 0.8)]
    kwargs = {"images": images} if mode != "legacy" else {"image": images[1]}
    if mode == "list_precedence":
        kwargs["image"] = torch.zeros_like(images[0])
    tokens = probe.tokenize_with_weights(PROMPT, skip_template=False, **kwargs)
    embeds = [token[0] for token in tokens[0] if isinstance(token[0], dict)]
    expected = [0.8] if mode == "legacy" else [0.2, 0.8]
    assert probe.llama_text.count("<image_soft_token>") == len(expected)
    assert [embed["data"].mean().item() for embed in embeds] == pytest.approx(expected)

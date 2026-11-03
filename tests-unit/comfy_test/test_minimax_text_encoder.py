import pytest
import torch

from dinkster_inference.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from dinkster_inference import ops
from dinkster_inference.text_encoders import minimax


def test_text_connector_normalizes_layer_major_features_and_replaces_sink():
    connector = minimax.TextConnector(4, 2, dtype=torch.float32, operations=ops.manual_cast)
    with torch.no_grad():
        connector.input_mean.copy_(torch.tensor([1, 2, 3, 4]))
        connector.input_std.copy_(torch.tensor([1, 2, 4, 8]))
        connector.proj.weight.copy_(torch.tensor([[1, 2, 3, 4], [-2, 1, 0, 4]]))
        connector.output_mean.copy_(torch.tensor([10, -5]))
        connector.output_std.copy_(torch.tensor([2, 0.5]))
        connector.sink_embedding.copy_(torch.tensor([-7, 9]))
    hidden = torch.tensor([[[[2, 4], [5, 6], [7, 8]], [[7, 12], [11, 20], [15, 28]]]], dtype=torch.float64)
    # Tokens 1 and 2 flatten layers first: [5,6,11,20] and [7,8,15,28].
    expected = torch.tensor([[[-7, 9], [54, -4], [76, -3.5]]], dtype=torch.float64)
    output = connector(hidden)
    torch.testing.assert_close(output, expected)
    assert output.dtype == hidden.dtype


@pytest.mark.parametrize("image_model", [False, True])
def test_minimax_tokenizer_preserves_media_order_and_routes_model_key(image_model):
    cls = minimax.PH3ImageTokenizer if image_model else minimax.MiniMaxH3Tokenizer
    key = "qwen3vl_4b" if image_model else "qwen3vl_32b"
    tokenizer = cls.__new__(cls)
    tokenizer.clip = key

    class TextTokens:
        def tokenize_with_weights(self, text, **kwargs):
            return [[(ord(c), 1.0) for c in text]]

        def untokenize(self, tokens):
            return "".join(chr(token[0]) for token in tokens)

    setattr(tokenizer, key, TextTokens())
    images = [torch.full((1, 2, 3, 3), value) for value in (0.2, 0.8)]
    result = tokenizer.tokenize_with_weights("edit", images=images, return_word_ids=True)
    assert list(result) == [key]
    entries = result[key][0]
    text = "".join(chr(t[0]) for t in entries if isinstance(t[0], int) and t[0] < 128)
    assert text == "<Picture 1>: <Picture 2>: edit"
    assert [t[0] for t in entries if isinstance(t[0], int) and t[0] > 128] == [151652, 151653] * 2
    embeds = [t[0] for t in entries if isinstance(t[0], dict)]
    assert all(embed["data"] is image for embed, image in zip(embeds, images))
    assert all(t[2] == 0 for t in entries)
    assert tokenizer.tokenize_with_weights("") == {key: [[(151643, 1.0)]]}
    assert tokenizer.untokenize([(101, 1.0), (100, 1.0)]) == "ed"

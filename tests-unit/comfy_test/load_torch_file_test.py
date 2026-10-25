import ctypes
import pickle
import struct
from unittest import mock

import comfy_aimdo.model_mmap
import pytest
import safetensors.torch
import torch

from dinkster_inference import memory_management, storage, utils


class CPUModelMMAP:
    """Exercise the aimdo parser without initializing a GPU backend."""

    def __init__(self, path):
        with open(path, "rb") as file:
            self.buffer = ctypes.create_string_buffer(file.read())

    def get(self):
        return ctypes.addressof(self.buffer)

    def get_file_handle(self):
        return -1


@pytest.mark.parametrize("aimdo", [False, True])
@pytest.mark.parametrize("suffix", ["", ".safetensors", ".sft", ".ckpt", ".pt", ".bin"])
def test_safetensors_content_and_metadata(tmp_path, monkeypatch, aimdo, suffix):
    path = str(tmp_path / ("a" * 64 + suffix))
    expected = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    safetensors.torch.save_file({"weight": expected}, path, metadata={"source": "vault"})
    monkeypatch.setattr(memory_management, "aimdo_enabled", aimdo)
    monkeypatch.setattr(comfy_aimdo.model_mmap, "ModelMMAP", CPUModelMMAP)
    with mock.patch.object(storage, "annotate_state_dict", wraps=storage.annotate_state_dict) as annotate:
        state, metadata = utils.load_torch_file(path, return_metadata=True)
    assert metadata == {"source": "vault"}
    assert torch.equal(state["weight"], expected)
    assert state["weight"].device.type == "cpu"
    annotate.assert_called_once_with(state, path)
    assert torch.equal(utils.load_torch_file(path)["weight"], expected)


@pytest.mark.parametrize("zip_format", [False, True])
@pytest.mark.parametrize("suffix", ["", ".ckpt"])
def test_torch_checkpoints(tmp_path, monkeypatch, zip_format, suffix):
    path = str(tmp_path / ("b" * 64 + suffix))
    expected = torch.tensor([[1.0, -3.0], [7.0, 0.5]])
    torch.save({"state_dict": {"weight": expected}}, path, _use_new_zipfile_serialization=zip_format)
    monkeypatch.setattr(utils, "MMAP_TORCH_FILES", False)
    with mock.patch.object(storage, "annotate_state_dict", wraps=storage.annotate_state_dict) as annotate:
        state, metadata = utils.load_torch_file(path, return_metadata=True)
    assert metadata is None
    assert torch.equal(state["weight"], expected)
    annotate.assert_called_once_with(state, path)


@pytest.mark.parametrize("data", [
    struct.pack("<Q", 1024) + b"{",
    b"random bytes that are not a checkpoint",
    b"",
    b"\x01\x00\x00",
    struct.pack("<Q", 0) + b"{",
    struct.pack("<Q", 2) + b"[]",
])
def test_unrecognized_content_keeps_torch_error(tmp_path, monkeypatch, data):
    path = str(tmp_path / "invalid")
    (tmp_path / "invalid").write_bytes(data)
    monkeypatch.setattr(utils, "MMAP_TORCH_FILES", False)
    with pytest.raises((pickle.UnpicklingError, EOFError, RuntimeError, IndexError)) as direct:
        torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    with pytest.raises(type(direct.value)) as dispatched:
        utils.load_torch_file(path)
    assert str(dispatched.value) == str(direct.value)


def test_header_cap_keeps_torch_dispatch(tmp_path):
    path = tmp_path / "oversized"
    with path.open("wb") as file:
        file.write(struct.pack("<Q", 100_000_001) + b"{")
        file.truncate(100_000_009)
    with mock.patch.object(torch, "load", side_effect=RuntimeError("torch rejected header")):
        with pytest.raises(RuntimeError, match="torch rejected header"):
            utils.load_torch_file(str(path))


def test_truncated_suffixed_header_keeps_parser_error(tmp_path, monkeypatch):
    path = str(tmp_path / "truncated.safetensors")
    (tmp_path / "truncated.safetensors").write_bytes(struct.pack("<Q", 64) + b"{")
    monkeypatch.setattr(memory_management, "aimdo_enabled", False)
    with pytest.raises(safetensors.SafetensorError) as direct:
        safetensors.safe_open(path, framework="pt", device="cpu")
    with pytest.raises(type(direct.value)) as dispatched:
        utils.load_torch_file(path)
    assert str(dispatched.value) == str(direct.value)


def test_metadata_incomplete_message(tmp_path, monkeypatch):
    path = str(tmp_path / "truncated.safetensors")
    monkeypatch.setattr(memory_management, "aimdo_enabled", False)
    with mock.patch.object(safetensors, "safe_open", side_effect=Exception("MetadataIncompleteBuffer")):
        with pytest.raises(ValueError, match="corrupt/incomplete") as error:
            utils.load_torch_file(path)
    assert path in str(error.value)

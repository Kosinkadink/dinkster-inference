import ast
import os
import subprocess
import sys

import pytest

from tools.rewrite_upstream_imports import main, rewrite


def test_rewrite_preserves_external_gguf_metadata_key(tmp_path):
    source = tmp_path / "source.py"
    source.write_text(
        "import comfy\n"
        "module = comfy.sd\n"
        'target = "comfy.ldm.models.autoencoder.Encoder"\n'
        'field = "comfy.gguf.orig_shape.weight"\n'
    )

    assert rewrite(source)
    assert source.read_text() == (
        "import dinkster_inference\n"
        "module = dinkster_inference.sd\n"
        'target = "dinkster_inference.ldm.models.autoencoder.Encoder"\n'
        'field = "comfy.gguf.orig_shape.weight"\n'
    )


def test_rewrite_cli_skips_its_fixture_file(tmp_path, monkeypatch):
    source = tmp_path / "rewrite_upstream_imports_test.py"
    source.write_text("import comfy\n")
    monkeypatch.setattr(sys, "argv", ["rewrite_upstream_imports.py", str(source)])

    main()

    assert source.read_text() == "import comfy\n"


def test_rewrite_preserves_unicode_values_with_ascii_source(tmp_path):
    source = tmp_path / "source.py"
    original = (
        '# caf\u00e9\n'
        'value = "\u4f60\u597d"\n'
        'pattern = r"\u2018(.*?)\u2019"\n'
        'message = f"\u00e9 {value}"\n'
    )
    source.write_text(original)

    assert rewrite(source)
    updated = source.read_text()
    assert updated.isascii()
    assert ast.dump(ast.parse(updated)) == ast.dump(ast.parse(original))
    assert not rewrite(source)


@pytest.mark.parametrize("first", ["comfy", "dinkster_inference"])
def test_installed_alias_shares_modules_and_state(first):
    program = f"""
import importlib
import sys
cli = importlib.import_module('{first}.cli_args')
cli.args.cpu = True
for suffix in ('cli_args', 'model_patcher', 'samplers', 'sd', 'supported_models', 'ldm.modules.attention'):
    first = importlib.import_module('{first}.' + suffix)
    canonical = importlib.import_module('dinkster_inference.' + suffix)
    alias = importlib.import_module('comfy.' + suffix)
    assert first is canonical is alias, suffix
    assert sys.modules['comfy.' + suffix] is sys.modules['dinkster_inference.' + suffix]
    canonical.alias_state_probe = object()
    assert alias.alias_state_probe is canonical.alias_state_probe
assert (importlib.import_module('comfy').model_patcher.ModelPatcher is
        importlib.import_module('dinkster_inference.model_patcher').ModelPatcher)
assert importlib.import_module('comfy.supported_models').comfy is importlib.import_module('comfy')
"""
    subprocess.run([sys.executable, "-I", "-c", program], check=True,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})

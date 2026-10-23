import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.compare_workflows import compare_arrays


def test_output_difference_does_not_wrap_unsigned_pixels_or_cancel_signed_latents():
    assert compare_arrays(np.array([0, 255], dtype=np.uint8),
                          np.array([255, 1], dtype=np.uint8)) == 255
    assert compare_arrays(np.array([-3.0, 0.5], dtype=np.float32),
                          np.array([1.0, -0.25], dtype=np.float32)) == 4
    assert compare_arrays(np.array([[-3.0, 0.5]], dtype=np.float32),
                          np.array([[-3.0, 0.5]], dtype=np.float32)) == 0


@pytest.mark.parametrize("other", [
    np.array([[1, 2]], dtype=np.float32),
    np.array([1, 2], dtype=np.float64),
    np.array([1, np.nan], dtype=np.float32),
    np.array([1, np.inf], dtype=np.float32),
])
def test_incompatible_or_nonfinite_outputs_are_not_reported_as_parity(other):
    with pytest.raises(ValueError):
        compare_arrays(np.array([1, 2], dtype=np.float32), other)


def test_upstream_worker_selects_namespace_reference_over_installed_alias(tmp_path):
    reference = tmp_path / "reference" / "comfy"
    reference.mkdir(parents=True)
    (reference / "cli_args.py").write_text("from argparse import Namespace\nargs = Namespace()\n")
    (reference / "sd.py").write_text("raise RuntimeError('reference namespace selected')\n")
    script = Path(__file__).resolve().parents[1] / "tools/compare_workflows.py"
    process = subprocess.run([
        sys.executable, "-I", str(script), "--worker", "--side", "upstream", "--case", "sd15",
        "--upstream", str(reference.parent), "--output-dir", str(tmp_path / "result"),
    ], env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True)
    assert process.returncode != 0
    assert "RuntimeError: reference namespace selected" in process.stderr

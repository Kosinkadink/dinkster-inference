import os
import subprocess
import sys
import textwrap


def test_sam3d_and_mediapipe_import_on_cpu(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import importlib.abc
            import sys

            class NoApplicationMediaPipe(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'mediapipe', 'comfy_extras'}:
                        raise ImportError(f'Application dependency requested: {fullname}')

            sys.meta_path.insert(0, NoApplicationMediaPipe())
            import dinkster_inference.cli_args as cli
            cli.args.cpu = True
            import torch
            import dinkster_inference.ldm.sam3d_body.model.model as sam3d
            from dinkster_inference.mediapipe.face_landmarker import FaceLandmarker
            from dinkster_inference.mediapipe.face_geometry import transformation_matrix_from_detection

            assert sam3d.FaceLandmarker is FaceLandmarker
            assert callable(transformation_matrix_from_detection)
            model = FaceLandmarker(device='cpu', dtype=torch.float32, operations=torch.nn)
            assert model.detect_batch([]) == []
            assert not torch.cuda.is_initialized()
        """)],
        cwd=tmp_path,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

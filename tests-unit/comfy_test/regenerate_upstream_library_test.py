from pathlib import Path

from tools.regenerate_upstream_library import OMITTED_TESTS, regenerate


def test_seedvr2_internal_test_is_retained_whole():
    assert "test_seedvr2_internals.py" not in OMITTED_TESTS


def test_regenerate_carries_fork_changes_and_rewrites_upstream_scope(tmp_path):
    baseline = tmp_path / "baseline"
    upstream = tmp_path / "upstream"
    destination = tmp_path / "fork"
    (baseline / "comfy").mkdir(parents=True)
    (baseline / "comfy" / "library.py").write_text("import comfy\nvalue = 'base'\n")
    (upstream / "comfy").mkdir(parents=True)
    (upstream / "comfy" / "library.py").write_text("import comfy\nvalue = 'base'\nupstream = True\n")
    (upstream / "comfy_extras" / "mediapipe").mkdir(parents=True)
    (upstream / "comfy_extras" / "mediapipe" / "face.py").write_text("from comfy import sd\n")
    for name in ("comfy_test", "comfy_quant"):
        (baseline / "tests-unit" / name).mkdir(parents=True)
        (baseline / "tests-unit" / name / "upstream_test.py").write_text("import comfy\n")
        (upstream / "tests-unit" / name).mkdir(parents=True)
        (upstream / "tests-unit" / name / "upstream_test.py").write_text("import comfy\n")
    (upstream / "tests-unit" / "comfy_test" / "test_seedvr2_model.py").write_text("import comfy\n")
    (baseline / "node_helpers.py").write_text("import comfy\n")
    (upstream / "node_helpers.py").write_text("import comfy\n")
    (destination / "tests-unit" / "comfy_test").mkdir(parents=True)
    (destination / "tests-unit" / "comfy_test" / "upstream_sync_test.py").write_text("fixture = 'fork'\n")
    (destination / "tests-unit" / "comfy_test" / "upstream_test.py").write_text("import dinkster_inference\n")
    (destination / "node_helpers.py").write_text("import dinkster_inference\n")
    (destination / "application.py").write_text("keep = True\n")
    (destination / "dinkster_inference").mkdir()
    (destination / "dinkster_inference" / "library.py").write_text(
        "fork_only = True\nimport dinkster_inference\nvalue = 'base'\n"
    )
    (destination / "dinkster_inference" / "future_model.py").write_text("fork_only = True\n")
    for root in (baseline, upstream):
        examples = root / "comfy/comfy_types/examples"
        examples.mkdir(parents=True)
        (examples / "example_nodes.py").write_text(f"application = {str(root)!r}\n")
        (examples.parent / "README.md").write_text(str(root))
        (examples.parent / "node_typing.py").write_text(f"application = {str(root)!r}\n")
    regenerate(baseline, upstream, destination)

    assert not (destination / "dinkster_inference/comfy_types").exists()
    assert (destination / "dinkster_inference" / "library.py").read_text() == (
        "fork_only = True\nimport dinkster_inference\nvalue = 'base'\nupstream = True\n"
    )
    assert (destination / "dinkster_inference" / "mediapipe" / "face.py").read_text() == "from dinkster_inference import sd\n"
    assert (destination / "tests-unit" / "comfy_test" / "upstream_test.py").read_text() == "import dinkster_inference\n"
    assert not (destination / "tests-unit" / "comfy_test" / "test_seedvr2_model.py").exists()
    assert (destination / "tests-unit" / "comfy_test" / "upstream_sync_test.py").read_text() == "fixture = 'fork'\n"
    assert (destination / "node_helpers.py").read_text() == "import dinkster_inference\n"
    assert (destination / "application.py").read_text() == "keep = True\n"
    assert (destination / "dinkster_inference" / "future_model.py").read_text() == "fork_only = True\n"


def test_regenerate_stops_on_an_overlapping_fork_change(tmp_path):
    baseline = tmp_path / "baseline"
    upstream = tmp_path / "upstream"
    destination = tmp_path / "fork"
    for root, value in ((baseline, "value = 'base'\n"), (upstream, "value = 'upstream'\n")):
        (root / "comfy").mkdir(parents=True)
        (root / "comfy" / "library.py").write_text(value)
        (root / "node_helpers.py").write_text("pass\n")
    (destination / "dinkster_inference").mkdir(parents=True)
    (destination / "dinkster_inference" / "library.py").write_text("value = 'fork'\n")
    (destination / "node_helpers.py").write_text("pass\n")

    try:
        regenerate(baseline, upstream, destination)
    except SystemExit as error:
        assert "cannot carry fork change" in str(error)
    else:
        raise AssertionError("expected an overlapping fork change to stop regeneration")

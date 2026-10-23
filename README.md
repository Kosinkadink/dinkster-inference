# dinkster-inference

This distribution packages upstream ComfyUI's inference library as
`dinkster_inference`. The `comfy` import name aliases the same modules, not a
second instance of model-management state. ComfyUI's application, server and
node packages are not distributed here.

```bash
python -m pip install .
```

As in upstream ComfyUI, CPU callers select CPU execution before importing model
management or loaders:

```python
import dinkster_inference.cli_args as cli_args
cli_args.args.cpu = True
import comfy.sd
import dinkster_inference.sd
assert comfy.sd is dinkster_inference.sd
```

See [upstream provenance](docs/ledger.md) for the fork point and omitted
application tests. `python tools/check_upstream_parity.py` checks the library
and retained test files against that upstream commit. New upstream files use
`tools/rewrite_upstream_imports.py` for import renaming and ASCII source spelling;
escaped string values and non-ASCII tokenizer resource data retain their meaning.

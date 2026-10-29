"""Expose upstream import names without loading a second copy of library state."""

import importlib
import importlib.abc
import importlib.util
import sys

import dinkster_inference


class _AliasLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return importlib.import_module("dinkster_inference" + spec.name[len("comfy"):])

    def exec_module(self, module):
        # Upstream callers also access the package binding inside imported modules.
        if "dinkster_inference" in module.__dict__:
            module.comfy = module.dinkster_inference


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("comfy."):
            canonical = "dinkster_inference" + fullname[len("comfy"):]
            spec = importlib.util.find_spec(canonical)
            if spec is not None:
                return importlib.util.spec_from_loader(
                    fullname, _AliasLoader(), is_package=spec.submodule_search_locations is not None
                )
        return None


sys.meta_path.insert(0, _AliasFinder())
sys.modules[__name__] = dinkster_inference

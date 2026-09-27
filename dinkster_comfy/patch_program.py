from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Callable

import torch


def _tensor_descriptor(
    value: torch.Tensor, tensor_digests: dict[int, str]
) -> dict[str, object]:
    identity = id(value)
    digest = tensor_digests.get(identity)
    if digest is None:
        tensor = value.detach().cpu().contiguous()
        digest = hashlib.sha256(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        tensor_digests[identity] = digest
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": digest,
    }


def _descriptor(
    value: object,
    active: set[int] | None = None,
    tensor_digests: dict[int, str] | None = None,
) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("patch program values must be finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, torch.Tensor):
        if tensor_digests is None:
            tensor_digests = {}
        return {"tensor": _tensor_descriptor(value, tensor_digests)}

    if active is None:
        active = set()
    if tensor_digests is None:
        tensor_digests = {}
    identity = id(value)
    if identity in active:
        raise ValueError("patch program values must not contain cycles")
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            return {
                "mapping": [
                    [
                        _descriptor(key, active, tensor_digests),
                        _descriptor(item, active, tensor_digests),
                    ]
                    for key, item in sorted(
                        value.items(), key=lambda pair: repr(pair[0])
                    )
                ]
            }
        if isinstance(value, (set, frozenset)):
            items = [_descriptor(item, active, tensor_digests) for item in value]
            items.sort(key=lambda item: json.dumps(item, sort_keys=True))
            return {"set": items}
        if isinstance(value, (list, tuple)):
            return {
                "sequence": [
                    _descriptor(item, active, tensor_digests) for item in value
                ]
            }
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {
                    item.name: _descriptor(
                        getattr(value, item.name), active, tensor_digests
                    )
                    for item in dataclasses.fields(value)
                    if item.compare
                },
            }
        if isinstance(value, functools.partial):
            return {
                "partial": {
                    "function": _descriptor(value.func, active, tensor_digests),
                    "args": _descriptor(value.args, active, tensor_digests),
                    "keywords": _descriptor(value.keywords, active, tensor_digests),
                }
            }
        if (
            callable(value)
            and hasattr(value, "__module__")
            and hasattr(value, "__qualname__")
        ):
            return {"callable": f"{value.__module__}.{value.__qualname__}"}
        attributes = getattr(value, "__dict__", None)
        if attributes is not None:
            return {
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "attributes": _descriptor(attributes, active, tensor_digests),
            }
    finally:
        active.remove(identity)
    raise TypeError(
        f"unsupported patch program value: {type(value).__module__}.{type(value).__qualname__}"
    )


def _digest(value: object, tensor_digests: dict[int, str] | None = None) -> str:
    payload = json.dumps(
        _descriptor(value, tensor_digests=tensor_digests),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(b"dinkster.patch-program.v1\0" + payload).hexdigest()


@dataclass(frozen=True)
class PatchResource:
    identity: str
    value: object = field(compare=False, repr=False)

    @classmethod
    def bind(
        cls, value: object, tensor_digests: dict[int, str] | None = None
    ) -> PatchResource:
        return cls(identity=_digest(value, tensor_digests), value=value)


@dataclass(frozen=True)
class WeightDeltaEntry:
    target: str
    patch: PatchResource
    strength_patch: float
    strength_model: float
    offset: object = None
    function: object = field(default=None, compare=False, repr=False)
    function_identity: str | None = None

    @classmethod
    def create(
        cls,
        *,
        target: str,
        patch: object,
        strength_patch: float,
        strength_model: float,
        offset: object = None,
        function: object = None,
        tensor_digests: dict[int, str] | None = None,
    ) -> WeightDeltaEntry:
        strengths = {
            "strength_patch": float(strength_patch),
            "strength_model": float(strength_model),
        }
        for name, value in strengths.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        return cls(
            target=target,
            patch=PatchResource.bind(patch, tensor_digests),
            strength_patch=0.0
            if strengths["strength_patch"] == 0.0
            else strengths["strength_patch"],
            strength_model=0.0
            if strengths["strength_model"] == 0.0
            else strengths["strength_model"],
            offset=offset,
            function=function,
            function_identity=None
            if function is None
            else _digest(function, tensor_digests),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "kind": "weight_delta",
            "target": self.target,
            "patch": self.patch.identity,
            "strength_patch": self.strength_patch,
            "strength_model": self.strength_model,
            "offset": _descriptor(self.offset),
            "function": self.function_identity,
        }

    def runtime_tuple(self) -> tuple[object, object, object, object, object]:
        return (
            self.strength_patch,
            self.patch.value,
            self.strength_model,
            self.offset,
            self.function,
        )


@dataclass(frozen=True)
class ModuleInsertionEntry:
    namespace: str
    sites: tuple[str, ...]
    materialize_ref: str
    materialize: Callable = field(compare=False, repr=False)
    resources: PatchResource
    additional_models: tuple[object, ...] = field(
        default=(), compare=False, repr=False
    )
    additional_model_identities: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        namespace: str,
        sites: Iterable[str],
        materialize_ref: str,
        materialize: Callable,
        resources: object = None,
        additional_models: Iterable[object] = (),
        additional_model_identities: Iterable[str] = (),
    ) -> ModuleInsertionEntry:
        if not namespace:
            raise ValueError("module insertion namespace must not be empty")
        if not materialize_ref:
            raise ValueError("module insertion materialize_ref must not be empty")
        if not callable(materialize):
            raise TypeError("module insertion materialize must be callable")
        model_values = tuple(additional_models)
        model_identities = tuple(additional_model_identities)
        if len(model_values) != len(model_identities):
            raise ValueError(
                "each additional model must have a declared resource identity"
            )
        if any(not identity for identity in model_identities):
            raise ValueError("additional model identities must not be empty")
        return cls(
            namespace=namespace,
            sites=tuple(sites),
            materialize_ref=materialize_ref,
            materialize=materialize,
            resources=PatchResource.bind(resources),
            additional_models=model_values,
            additional_model_identities=model_identities,
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "kind": "module_insertion",
            "namespace": self.namespace,
            "sites": self.sites,
            "materialize": self.materialize_ref,
            "resources": self.resources.identity,
            "additional_models": self.additional_model_identities,
        }

    def validate_resources(self) -> None:
        if self.resources.identity != _digest(self.resources.value):
            raise RuntimeError(
                f"module insertion resources for '{self.namespace}' changed after binding"
            )

    def with_additional_models(
        self, models: Iterable[object]
    ) -> ModuleInsertionEntry:
        values = tuple(models)
        if len(values) != len(self.additional_model_identities):
            raise ValueError("replacement additional model count must match")
        return dataclasses.replace(self, additional_models=values)


PatchEntry = WeightDeltaEntry | ModuleInsertionEntry


@dataclass(frozen=True)
class PatchProgram:
    entries: tuple[PatchEntry, ...] = ()

    @property
    def digest(self) -> str:
        payload = json.dumps(
            [entry.descriptor() for entry in self.entries],
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(b"dinkster.patch-program.v1\0" + payload).hexdigest()

    def validate_resources(self, target: str | None = None) -> None:
        tensor_digests: dict[int, str] = {}
        for entry in self.entries:
            if isinstance(entry, ModuleInsertionEntry):
                if target is None:
                    entry.validate_resources()
                continue
            if target is not None and entry.target != target:
                continue
            if entry.patch.identity != _digest(entry.patch.value, tensor_digests):
                raise RuntimeError(
                    f"patch resource for '{entry.target}' changed after binding"
                )
            if entry.function is not None and entry.function_identity != _digest(
                entry.function, tensor_digests
            ):
                raise RuntimeError(
                    f"patch function for '{entry.target}' changed after binding"
                )

    def append_weight_delta(
        self,
        *,
        target: str,
        patch: object,
        strength_patch: float,
        strength_model: float,
        offset: object = None,
        function: object = None,
    ) -> PatchProgram:
        entry = WeightDeltaEntry.create(
            target=target,
            patch=patch,
            strength_patch=strength_patch,
            strength_model=strength_model,
            offset=offset,
            function=function,
        )
        return PatchProgram((*self.entries, entry))

    def extend_weight_deltas(
        self,
        entries: Iterable[tuple[str, object, float, float, object, object]],
    ) -> PatchProgram:
        tensor_digests: dict[int, str] = {}
        additions = tuple(
            WeightDeltaEntry.create(
                target=target,
                patch=patch,
                strength_patch=strength_patch,
                strength_model=strength_model,
                offset=offset,
                function=function,
                tensor_digests=tensor_digests,
            )
            for target, patch, strength_patch, strength_model, offset, function in entries
        )
        return PatchProgram((*self.entries, *additions))

    def weight_patches(
        self,
    ) -> dict[str, list[tuple[object, object, object, object, object]]]:
        patches: dict[str, list[tuple[object, object, object, object, object]]] = {}
        for entry in self.entries:
            if not isinstance(entry, WeightDeltaEntry):
                continue
            patches.setdefault(entry.target, []).append(entry.runtime_tuple())
        return patches

    def replace_module_insertions(
        self, namespace: str, entries: Iterable[ModuleInsertionEntry]
    ) -> PatchProgram:
        replacements = tuple(entries)
        if any(entry.namespace != namespace for entry in replacements):
            raise ValueError("module insertion namespace does not match replacement key")
        retained = tuple(
            entry
            for entry in self.entries
            if not (
                isinstance(entry, ModuleInsertionEntry)
                and entry.namespace == namespace
            )
        )
        return PatchProgram((*retained, *replacements))

    def module_insertions(
        self, namespace: str | None = None
    ) -> tuple[ModuleInsertionEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if isinstance(entry, ModuleInsertionEntry)
            and (namespace is None or entry.namespace == namespace)
        )

    def map_additional_models(
        self, transform: Callable[[object], object]
    ) -> PatchProgram:
        entries: list[PatchEntry] = []
        for entry in self.entries:
            if isinstance(entry, ModuleInsertionEntry) and entry.additional_models:
                entry = entry.with_additional_models(
                    transform(model) for model in entry.additional_models
                )
            entries.append(entry)
        return PatchProgram(tuple(entries))

    @classmethod
    def from_weight_patches(cls, patches: Mapping[str, list[tuple]]) -> PatchProgram:
        additions = []
        for target, entries in patches.items():
            for entry in entries:
                if len(entry) != 5:
                    raise ValueError("weight patch entries must contain five values")
                strength_patch, patch, strength_model, offset, function = entry
                additions.append(
                    (target, patch, strength_patch, strength_model, offset, function)
                )
        return cls().extend_weight_deltas(additions)

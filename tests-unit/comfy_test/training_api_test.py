"""Contract consumed by dinkster-training, checked before an inference pin moves."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from dinkster_inference.cli_args import args
from dinkster_inference.lora import model_lora_keys_unet
from dinkster_inference.model_base import BaseModel
from dinkster_inference.model_management import unload_all_models
from dinkster_inference.model_patcher import ModelPatcher
from dinkster_inference.patch_program import PatchProgram
from dinkster_inference.sd import load_checkpoint_guess_config
from dinkster_inference.weight_adapter import LoRAAdapter
from dinkster_inference.weight_adapter.lora import LoraDiff
import dinkster_inference.model_management as management


class PolicyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
        self.seen = []
        self.nested = None

    def forward(self, x, *, fail=False):
        self.seen.append((management.in_training, management.training_fp8_bwd))
        if self.nested is not None:
            self.nested(x.detach())
            self.seen.append((management.in_training, management.training_fp8_bwd))
        if fail:
            raise RuntimeError("forward failed")
        return (x * self.weight).sin()


def patcher(model):
    return ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))


def test_eight_training_symbols():
    assert isinstance(args.cpu, bool)
    assert issubclass(BaseModel, torch.nn.Module)
    assert callable(load_checkpoint_guess_config)
    assert callable(model_lora_keys_unet)
    assert callable(unload_all_models)
    assert callable(ModelPatcher.set_bypass_adapters)
    adapter = LoRAAdapter.create_train(torch.zeros(3, 2), rank=1, alpha=1.0)
    assert isinstance(adapter, LoraDiff)
    assert adapter.lora_down.weight.shape == (1, 2)
    assert adapter.lora_up.weight.shape == (3, 1)


def test_training_step_and_inference_job_no_leak():
    training = PolicyModel()
    inference = PolicyModel()
    train_patcher = patcher(training)
    infer_patcher = patcher(inference)
    train_patcher.set_training_settings(True, fp8_backward=True)
    train_patcher.set_gradient_checkpoints([""])
    train_patcher.patch_model(load_weights=False)
    infer_patcher.patch_model(load_weights=False)
    training.nested = inference
    optimizer = torch.optim.SGD(training.parameters(), lr=0.1)
    x = torch.tensor([0.25, 0.5])
    old_weight = training.weight.detach().clone()
    loss = training(x).square().sum()
    loss.backward()
    # Independently derived gradient of sum(sin(x*w)^2).
    expected_grad = x * torch.sin(2 * x * old_weight)
    torch.testing.assert_close(training.weight.grad, expected_grad)
    optimizer.step()
    torch.testing.assert_close(training.weight, old_weight - 0.1 * expected_grad)
    torch.testing.assert_close(inference(x), torch.sin(x * old_weight))
    assert len(training.seen) >= 4  # Initial forward and recomputation.
    assert all(policy == (True, True) for policy in training.seen)
    assert all(policy == (False, False) for policy in inference.seen)
    assert (management.in_training, management.training_fp8_bwd) == (False, False)
    with pytest.raises(RuntimeError, match="forward failed"):
        training(x, fail=True)
    assert not management.in_training
    train_patcher.unpatch_model(unpatch_weights=False)
    infer_patcher.unpatch_model(unpatch_weights=False)
    assert "forward" not in training.__dict__
    assert "forward" not in inference.__dict__


def test_policy_is_thread_local_and_restores_nested_scope():
    def execute(enabled):
        with management.training_settings_scope(enabled, enabled):
            with management.training_settings_scope(False, False):
                assert not management.in_training
            return management.in_training, management.training_fp8_bwd

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(execute, [True, False])) == [(True, True), (False, False)]
    assert not management.in_training


def test_program_identity_clone_and_validation():
    base = PatchProgram()
    trained = base.with_training_settings(True).with_gradient_checkpoints([""])
    trained.validate_resources()
    assert (
        len(
            {
                base.digest,
                trained.digest,
                trained.with_training_settings(True, True).digest,
                trained.with_gradient_checkpoints([]).digest,
            }
        )
        == 4
    )
    assert trained.clone_module_resources().digest == trained.digest
    assert not base.training_settings().enabled
    model_patcher = patcher(PolicyModel())
    model_patcher.set_training_settings(True)
    model_patcher.set_gradient_checkpoints([""])
    clone = model_patcher.clone()
    clone.set_training_settings(False)
    clone.set_gradient_checkpoints([])
    assert model_patcher.patch_program.training_settings().enabled
    assert not clone.patch_program.training_settings().enabled
    assert clone.patch_program.digest != model_patcher.patch_program.digest
    with pytest.raises(ValueError, match="FP8 backward"):
        base.with_training_settings(False, True)
    with pytest.raises(ValueError, match="unique"):
        base.with_gradient_checkpoints(["", ""])
    with pytest.raises(AttributeError):
        model_patcher.set_gradient_checkpoints(["missing"])


def test_direct_diffusion_forward_and_repeated_patch_restore():
    model = torch.nn.Module()
    model.diffusion_model = PolicyModel()
    model_patcher = patcher(model)
    model_patcher.set_training_settings(True)
    model_patcher.patch_model(load_weights=False)
    model_patcher.patch_model(load_weights=False)
    model.diffusion_model(torch.tensor([0.25, 0.5]))
    assert model.diffusion_model.seen == [(True, False)]
    model_patcher.unpatch_model(unpatch_weights=False)
    assert "forward" not in model.diffusion_model.__dict__


def test_checkpoint_requires_training():
    model_patcher = patcher(PolicyModel())
    model_patcher.set_gradient_checkpoints([""])
    with pytest.raises(ValueError, match="requires training"):
        model_patcher.patch_model(load_weights=False)
    assert "forward" not in model_patcher.model.__dict__


def test_shared_model_clone_switches_policy_without_nested_wrappers():
    model = PolicyModel()
    trained = patcher(model)
    trained.set_training_settings(True, True)
    trained.patch_model(load_weights=False)
    inference = trained.clone()
    inference.set_training_settings(False)
    inference.patch_model(load_weights=False)
    model(torch.tensor([0.25, 0.5]))
    assert model.seen == [(False, False)]
    trained.patch_model(load_weights=False)
    model(torch.tensor([0.25, 0.5]))
    assert model.seen[-1] == (True, True)
    inference.unpatch_model(unpatch_weights=False)
    assert "forward" not in model.__dict__


def test_policy_and_checkpoint_follow_adapter_injection_lifecycle():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3, bias=False))
    original = model[0].forward
    model_patcher = patcher(model)
    adapter = LoRAAdapter.create_train(model[0].weight, rank=1, alpha=1.0)
    model_patcher.set_bypass_adapters("training", {"0.weight": (adapter, 1.0)})
    model_patcher.set_training_settings(True)
    model_patcher.set_gradient_checkpoints(["0"])
    model_patcher.patch_model(load_weights=False)
    x = torch.tensor([[0.25, 0.5]])
    baseline = model(x).detach().clone()
    for _ in range(2):
        with model_patcher.use_ejected():
            assert model[0].forward == original
        model(x).sum().backward()
        assert adapter.lora_up.weight.grad is not None
        torch.testing.assert_close(model(x), baseline)
        adapter.lora_up.weight.grad = None
    model_patcher.unpatch_model(unpatch_weights=False)
    assert model[0].forward == original
    torch.testing.assert_close(model(x), baseline)

import copy
from typing import List

import torch

from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import Float16OptimizerWithFloat16Params
from megatron.legacy.model.module import MegatronModule


def _assert_all_parameters_zero_grads(gpt_model: GPTModel) -> None:
    for name, param in gpt_model.named_parameters():
        if param.grad is None:
            continue
        if not torch.allclose(param.grad, torch.zeros_like(param)):
            raise ValueError(f"Not all parameter gradients of layer:{name} are zeros : {param}")

    for name, child in gpt_model.named_children():
        _assert_all_parameters_zero_grads(child)


def _all_zeros_or_nans(tensor) -> bool:
    """
    Returns True if all values in the input tensor are either 0 or NaN, False otherwise.
    """
    nans_count = torch.sum(torch.isnan(tensor))
    zeros_count = torch.sum(torch.eq(tensor, 0))
    tmp_sum = nans_count + zeros_count
    total_elements = torch.numel(tensor)
    if total_elements != tmp_sum:
        return False
    return True


def _assert_all_parameters_zeros_or_nans(gpt_model: torch.nn.Module) -> None:
    """
    Recursively checks if all parameters in the given model are either 0 or NaN.
    Raises an AssertionError if any parameter contains values other than 0 or NaN.

    Args:
    model (torch.nn.Module): The PyTorch model to be checked.
    """
    for name, param in gpt_model.named_parameters():
        if not _all_zeros_or_nans(param):
            raise AssertionError(f"Parameter '{name}' contains values other than 0 or NaN.")

    for name, child in gpt_model.named_children():
        _assert_all_parameters_zeros_or_nans(child)


def reset_model_and_optimizer_grads(model, optimizer: Float16OptimizerWithFloat16Params,
                                    wanted_rank: int) -> None:
    if torch.distributed.get_rank() != wanted_rank:
        return

    gpt_model = _get_gpt_from_model(model)
    _reset_module_grads_to_zero(gpt_model)
    optimizer._copy_model_grads_to_main_grads()


def _reset_module_grads_to_zero(module: torch.nn.Module) -> None:
    for param in module.parameters():
        if param.grad is not None:
            param.grad.data.zero_()

    # Recursively reset grads of children modules
    for child in module.children():
        _reset_module_parameters_to_zero(child)


def reset_model_and_optimizer_weights(model: MegatronModule, optimizer: Float16OptimizerWithFloat16Params,
                                      wanted_rank: int) -> None:
    if torch.distributed.get_rank() != wanted_rank:
        return

    gpt_model = _get_gpt_from_model(model)
    _reset_module_parameters_to_zero(gpt_model)
    optimizer.reload_model_params()


def assert_model_params_and_grads_are_zero(model: MegatronModule, wanted_rank: int) -> None:
    if torch.distributed.get_rank() != wanted_rank:
        return

    gpt_model = _get_gpt_from_model(model)
    _assert_all_parameters_zeros_or_nans(gpt_model)
    _assert_all_parameters_zero_grads(gpt_model)


def _reset_module_parameters_to_zero(module: torch.nn.Module):
    for param in module.parameters():
        param.data.zero_()

    # Recursively reset parameters of children modules
    for child in module.children():
        _reset_module_parameters_to_zero(child)


def _reset_module_gradients_to_zero(module: torch.nn.Module):
    for param in module.parameters():
        if param.grad is not None:
            param.grad.data.zero_()

    # Recursively reset gradients of children modules
    for child in module.children():
        _reset_module_gradients_to_zero(child)


def _get_gpt_from_model(model: MegatronModule) -> GPTModel:
    gpt_model = model[0]._modules['module'].module
    print(f"@@@@@@@@@@{type(gpt_model)}@@@@@@@@@@@@@@@@")
    return gpt_model


def log_ZeroGPU(msg):
    print(f"ZeroGPU: {msg}")


def zero_a_and_check_model_b(model: List, optimizer):
    """
    1. Wait for both processes to arrive at the function.
    2. Save the state of Model B (including submodules).
    3. Perform actions on Model A.
    4. Ensure Model B's parameters and gradients (including submodules) did not change due to the actions on Model A.
    """
    # Synchronize both processes
    rank = torch.distributed.get_rank()
    torch.distributed.barrier()

    gpt_model = _get_gpt_from_model(model)
    # Save the initial state of Model B's weights and gradients (including submodules)
    if rank != 1:
        model_state_path = f"model_state_{rank}.pt"
        torch.save(gpt_model.state_dict(), model_state_path)

    torch.distributed.barrier()

    if rank == 1:
        reset_model_and_optimizer_weights(model, optimizer, wanted_rank=1)
        reset_model_and_optimizer_grads(model, optimizer, wanted_rank=1)

    torch.distributed.barrier()

    # Check if Model B's weights and gradients have changed (including submodules)
    if rank != 1:
        model_state_path = f"model_state_{rank}.pt"

        weights_changed, gradients_changed = _check_model_state(gpt_model, model_state_path)
        if weights_changed:
            print("Weights of Model B (including submodules) have changed after actions on Model A.")
        else:
            print("Weights of Model B (including submodules) have not changed after actions on Model A.")

        if gradients_changed:
            print("Gradients of Model B (including submodules) have changed after actions on Model A.")
        else:
            print("Gradients of Model B (including submodules) have not changed after actions on Model A.")


def _check_model_state(old_model: torch.nn.Module, filename: str):
    """
    Check if the weights and gradients of a model (including submodules) have changed compared to the state saved in the file.
    """
    new_state_dict = torch.load(filename)
    old_state_dict = old_model.state_dict()

    weights_changed = False
    for key in old_state_dict:
        if torch.any(new_state_dict[key] != old_state_dict[key]):
            weights_changed = True
            break

    gradients_changed = False
    for key in old_state_dict:
        if old_state_dict[key].grad is not None:
            if torch.any(new_state_dict[key].grad != old_state_dict[key].grad):
                gradients_changed = True
                break

    return weights_changed, gradients_changed

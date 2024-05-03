import os
from typing import List
import torch

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


def _get_gpt_from_model(model):
    gpt_model = model[0]._modules['module'].module
    return gpt_model


def _check_model_state(old_model: torch.nn.Module, filename: str):
    """
    Check if the weights and gradients of a model (including submodules) have changed compared to the state saved in the file.
    """
    new_state_dict = torch.load(filename)
    old_state_dict = old_model.state_dict()

    return new_state_dict.__str__() == old_state_dict.__str__()


def _assert_all_parameters_zero_grads(gpt_model) -> None:
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


def reset_model_and_optimizer_grads(model, optimizer) -> None:
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


def reset_model_and_optimizer_weights(model, optimizer) -> None:
    gpt_model = _get_gpt_from_model(model)
    _reset_module_parameters_to_zero(gpt_model)
    optimizer.reload_model_params()


def assert_model_params_and_grads_are_zero(model) -> None:
    gpt_model = _get_gpt_from_model(model)
    _assert_all_parameters_zeros_or_nans(gpt_model)
    _assert_all_parameters_zero_grads(gpt_model)


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

    torch.distributed.barrier()

    gpt_model = _get_gpt_from_model(model)
    # Save the initial state of Model B's weights and gradients (including submodules)
    if not is_faulty_gpu():
        model_state_path = f"model_state.pt"
        torch.save(gpt_model.state_dict(), model_state_path)

    torch.distributed.barrier()

    if is_faulty_gpu():
        reset_model_and_optimizer_weights(model, optimizer)
        reset_model_and_optimizer_grads(model, optimizer)

    torch.distributed.barrier()

    # Check if Model B's weights and gradients have changed (including submodules)
    if not is_faulty_gpu():
        model_state_path = f"model_state.pt"

        model_changed = _check_model_state(gpt_model, model_state_path)
        if model_changed:
            print("Weights/Gradients of Model B (including submodules) have changed after actions on Model A.")
        else:
            print("Weights/Gradients of Model B (including submodules) have not changed after actions on Model A.")


def is_faulty_gpu():
    current_gpu_rank = torch.distributed.get_rank()
    faulty_gpu_rank = get_faulty_gpu_rank()
    return current_gpu_rank == faulty_gpu_rank


def is_zero_gpu_on():
    return int(os.getenv("ZeroGPU_ON", "0"))


def get_failed_iteration():
    return int(os.getenv('FAILED_ITERATION'))


def set_zero_gpu_on():
    os.environ['ZeroGPU_ON'] = "1"


def get_faulty_gpu_rank():
    return int(os.getenv("FAILED_GPU"))

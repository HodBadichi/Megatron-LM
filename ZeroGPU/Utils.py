import torch

from megatron.core.optimizer import Float16OptimizerWithFloat16Params
from megatron.legacy.model.module import MegatronModule


def _assert_all_parameters_zero_grads(layer):
    '''

    Args:
        layer: type transformer_engine.pytorch.TransformerLayer
    '''
    for name, param in layer.named_parameters():
        if (param.grad is None):
            continue
        if not torch.allclose(param.grad, torch.zeros_like(param)):
            raise ValueError(f"Not all parameter gradients of layer:{name} are zeros : {param}")

    for name, child in layer.named_children():
        _assert_all_parameters_zero_grads(child)


def _all_zeros_or_nans(tensor):
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


def _assert_all_parameters_zeros_or_nans(index, layer):
    """
    Recursively checks if all parameters in the given model are either 0 or NaN.
    Raises an AssertionError if any parameter contains values other than 0 or NaN.

    Args:
    model (torch.nn.Module): The PyTorch model to be checked.
    """
    for name, param in layer.named_parameters():
        if not _all_zeros_or_nans(param):
            raise AssertionError(f"Parameter '{name}' contains values other than 0 or NaN. Index: {index}")

    for name, child in layer.named_children():
        _assert_all_parameters_zeros_or_nans(index, child)


def _get_gpt_layers_from_model(model: MegatronModule):
    # List of transformer_engine.pytorch.TransformerLayer
    gpt_layers = model[0].module._modules['module'].language_model.encoder.layers
    return gpt_layers


def reset_model_and_optimizer_grads(model: MegatronModule, optimizer: Float16OptimizerWithFloat16Params,
                                    wanted_rank: int):
    if torch.distributed.get_rank() != wanted_rank:
        return

    gpt_layers = _get_gpt_layers_from_model(model)
    for layer in gpt_layers:
        layer.zero_grad()
    optimizer._copy_model_grads_to_main_grads()


def reset_model_and_optimizer_weights(model: MegatronModule, optimizer: Float16OptimizerWithFloat16Params,
                                      wanted_rank: int):
    if torch.distributed.get_rank() != wanted_rank:
        return

    gpt_layers = _get_gpt_layers_from_model(model)
    for layer in gpt_layers:
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        else:
            for param in layer.parameters():
                param.data.zero_()
    optimizer.reload_model_params()


def assert_model_params_and_grads_are_zero(model: MegatronModule, wanted_rank: int):
    if torch.distributed.get_rank() != wanted_rank:
        return

    gpt_layers = _get_gpt_layers_from_model(model)
    for idx, layer in enumerate(gpt_layers):
        _assert_all_parameters_zeros_or_nans(idx, layer)
        _assert_all_parameters_zero_grads(layer)


def log_ZeroGPU(msg):
    print(f"ZeroGPU: {msg}")

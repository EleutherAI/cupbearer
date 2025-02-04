from typing import Callable, Dict, Tuple, Any, Union
from einops import rearrange, repeat
from collections import defaultdict
import torch
from torch import nn
from contextlib import contextmanager
import pdb

class _Finished(Exception):
    pass

def process_backward_zeros_transformer(noise: None, clean: torch.Tensor, grad: torch.Tensor, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replace clean activations with zeros and reshape grad correctly for transformers
    """
    direction = -clean

    if head_dim > 0:
        direction = rearrange(direction, 'batch seq (nh hd) -> batch seq nh hd', hd = head_dim)
        grad = rearrange(grad, 'batch seq (nh hd) -> batch seq nh hd', hd = head_dim)
        return direction, grad
    else:
        return direction, grad

def process_backward_transformer(noise: torch.Tensor, clean: torch.Tensor, grad: torch.Tensor, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replace clean activations with noise activations and reshape grad correctly for transformers
    """
    # Unsqueeze at sequence dimension
    noise = noise
    direction = noise - clean
    if head_dim > 0:
        direction = rearrange(direction, 'batch seq (nh hd) -> batch seq nh hd', hd = head_dim)
        grad = rearrange(grad, 'batch seq (nh hd) -> batch seq nh hd', hd = head_dim)
        return direction, grad
    else:
        return direction, grad

def process_backward_project_transformer(noise: Tuple[torch.Tensor, torch.Tensor], clean: torch.Tensor, grad: torch.Tensor, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orthogonally project clean activations to noise activations and reshape grad correctly for transformers
    """
    noise_vecs = rearrange(noise[0], 'noise_batch d -> noise_batch 1 d')
    noise_vals = rearrange(noise[1], 'noise_batch -> noise_batch 1 1')
    clean = rearrange(clean, 'batch seq d -> batch 1 seq d').type_as(noise_vecs)
    noise = (noise_vecs
            - rearrange(torch.linalg.vecdot(noise_vecs, clean), 'batch noise_batch seq -> batch noise_batch seq 1') * noise_vecs
            + (noise_vals * noise_vecs))
    
    # Unsqueeze at noise batch dimension
    direction = noise - clean

    if head_dim > 0:
        noise_batch_size = noise_vecs.shape[0]
        direction = rearrange(direction, 'batch noise_batch seq (nh hd) -> batch seq (noise_batch nh) hd', hd = head_dim)
        grad = rearrange(grad, 'batch seq (nh hd) -> batch seq nh hd', hd = head_dim)
        grad = repeat(grad, 'batch seq nh hd -> batch seq (noise_batch nh) hd', noise_batch = noise_batch_size)
        return direction, grad
    else:
        direction = rearrange(direction, 'batch noise_batch seq d -> batch seq noise_batch d')
        grad = rearrange(grad, 'batch seq d -> batch seq 1 d')
        return direction, grad

def process_backward_norm_transformer(noise: None, clean: torch.Tensor, grad: torch.Tensor, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orthogonally project clean activations to noise activations and reshape grad correctly for conv nets
    """
    if head_dim > 0:
        grad = rearrange(grad, 'batch seq (nh hd) -> batch seq nh hd', hd = head_dim)
    direction = grad
    return direction, grad

def process_backward_zeros_conv(noise: None, clean: torch.Tensor, grad: torch.Tensor, *args) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replace clean activations with zeros and reshape grad correctly for conv nets
    """
    return rearrange(-clean, 'batch maps h w -> batch maps (h w)'), rearrange(grad, 'batch maps h w -> batch maps (h w)')

def process_backward_conv(noise: torch.Tensor, clean: torch.Tensor, grad: torch.Tensor, *args) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orthogonally project clean activations to noise activations and reshape grad correctly for conv nets
    """
    # We treat the feature maps analogously to the sequence dimension in tranformers
    clean = rearrange(clean, 'batch maps h w -> batch maps (h w)')
    direction = noise - clean
    grad = rearrange(grad, 'batch maps (h w) -> batch maps h w')
    return direction, grad

def process_backward_project_conv(noise: Tuple[torch.Tensor, torch.Tensor], clean: torch.Tensor, grad: torch.Tensor, *args) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orthogonally project clean activations to noise activations and reshape grad correctly for conv nets
    """
    noise_vecs = rearrange(noise[0], 'noise_batch (h w) -> noise_batch 1 (h w)')
    noise_vals = rearrange(noise[1], 'noise_batch -> noise_batch 1 1')

    clean = rearrange(clean, 'batch maps h w -> batch 1 maps (h w)').type_as(noise_vecs)

    noise = (noise_vecs
            - rearrange(torch.linalg.vecdot(noise_vecs, clean), 'batch noise_batch maps -> batch noise_batch maps 1') * noise_vecs
            + (noise_vals * noise_vecs))
    
    direction = noise - clean
    grad = rearrange(grad, 'batch maps (h w) -> batch maps h w')
    return direction, grad

def get_tensor_process_fn(
    effect_capture_args: Dict[str, Any]
) -> Union[
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]],
    Callable[[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]],
    Callable[[None, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]
]:
    if effect_capture_args['model_type'] == 'transformer':
        if effect_capture_args['ablation'] == 'pcs':
            return process_backward_project_transformer
        elif effect_capture_args['ablation'] in ['mean', 'zeros']:
            return process_backward_transformer
        elif effect_capture_args['ablation'] == 'zeros':
            return process_backward_zeros_transformer
        elif effect_capture_args['ablation'] == 'grad_norm':
            return process_backward_norm_transformer
        else:
            raise ValueError(f"Invalid ablation {effect_capture_args['ablation']} for transformer")
    elif effect_capture_args['model_type'] == 'conv':
        if effect_capture_args['ablation'] == 'pcs':
            return process_backward_project_conv
        elif effect_capture_args['ablation'] in ['mean', 'zeros']:
            return process_backward_conv
        elif effect_capture_args['ablation'] == 'zeros':
            return process_backward_zeros_conv
        else:
            raise ValueError(f"Invalid ablation {effect_capture_args['ablation']} for conv")
    else:
        raise ValueError(f"Invalid model type {effect_capture_args['model_type']}")


@contextmanager
def prepare_model_for_effects(
    model: nn.Module,
    noise_acts: Dict[str, torch.Tensor] | Dict[str, Tuple[torch.Tensor, torch.Tensor]] | None,
    effect_capture_args: Dict[str, Any],
    head_dim: int = 0,
):
    effects = {}
    handles = []
    mod_to_clean = {}
    mod_to_noise = {}
    mod_to_name = {}
    names = list(noise_acts.keys())

    def bwd_hook(module: nn.Module, grad_input: tuple[torch.Tensor, ...] | torch.Tensor, grad_output: tuple[torch.Tensor, ...] | torch.Tensor):
        grad_input = grad_input[0] if isinstance(grad_input, tuple) else grad_input
        clean = mod_to_clean[module]
        direction, grad_input = get_tensor_process_fn(effect_capture_args)(mod_to_noise[module], clean, grad_input, head_dim)
        effect = torch.linalg.vecdot(direction, grad_input.type_as(direction))
        name = mod_to_name[module]
        effects[name] = effect.clone().detach()

    def fwd_hook(module: nn.Module, input: tuple[torch.Tensor, ...] | torch.Tensor, output: tuple[torch.Tensor, ...] | torch.Tensor):
        output = output[0] if isinstance(output, tuple) else output
        mod_to_clean[module] = output.clone().detach()

    for name, module in model.named_modules():
        mod_to_name[module] = name
        for path, noise in noise_acts.items():
            if not name == path:
                continue
            handles.append(module.register_full_backward_hook(bwd_hook))
            handles.append(module.register_forward_hook(fwd_hook))
            mod_to_noise[module] = noise_acts[name]

    try:
        yield effects
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad()

def get_effects(
    *args,
    model: nn.Module,
    noise_acts: Dict[str, torch.Tensor] | Dict[str, Tuple[torch.Tensor, torch.Tensor]] | None,
    output_func: Callable[[torch.Tensor], torch.Tensor],
    effect_capture_args: Dict[str, Any],
    head_dim: int = 0,
    **kwargs
) -> dict[str, torch.Tensor]:
    """Get the approximate effects of ablating model activations with noise_acts for the given inputs
    using attribution patching.

    Args:
        model: The model to get the effects from.
        names: The names of the modules to get the effects from.
        noise_acts: A dictionary of noise activations for attribution patching.
        output_func: A function that takes the output of the model and reduces
            to a (batch_size, ) shaped tensor.
        tensor_process_fns: A dictionary of functions that transform tensors in a manner appropriate for the particular model
        head_dim: The size of attention heads (if applicable).
        *args: Arguments to pass to the model.
        **kwargs: Keyword arguments to pass to the model.

    Returns:
        A dictionary mapping the names of the modules to the attribution effects.
    """
    with prepare_model_for_effects(model, noise_acts, effect_capture_args, head_dim) as effects:
        with torch.enable_grad():
                out = model(*args, **kwargs)
                out = output_func(out, *args, 'out')
                assert out.ndim == 1, "output_func should reduce to a 1D tensor"
                out.backward(torch.ones_like(out))
    return effects

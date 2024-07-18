from typing import Callable, Dict, Tuple, Any, Union
from einops import rearrange, repeat
from collections import defaultdict
import torch
from torch import nn

class _Finished(Exception):
    pass


def process_backward_zeros_transformer(noise: None, clean: torch.Tensor, grad_output: torch.Tensor, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replace clean activations with zeros and reshape grad_output correctly for transformers
    """
    direction = -clean

    if head_dim > 0:
        direction = rearrange(direction, 'batch seq (nh hd) -> batch seq nh hd', hd = head_dim)
        grad_output = rearrange(grad_output, 'batch seq (nh hd) -> batch seq nh hd', hd = head_dim)
        return direction, grad_output
    else:
        return direction, grad_output

def process_backward_transformer(noise: torch.Tensor, clean: torch.Tensor, grad_output: torch.Tensor, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replace clean activations with noise activations and reshape grad_output correctly for transformers
    """
    # Unsqueeze at sequence dimension
    noise = noise
    direction = noise - clean
    if head_dim > 0:
        direction = rearrange(direction, 'batch seq (nh hd) -> batch seq nh hd', hd = head_dim)
        grad_output = rearrange(grad_output, 'batch seq (nh hd) -> batch seq nh hd', hd = head_dim)
        return direction, grad_output
    else:
        return direction, grad_output

def process_backward_project_transformer(noise: Tuple[torch.Tensor, torch.Tensor], clean: torch.Tensor, grad_output: torch.Tensor, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orthogonally project clean activations to noise activations and reshape grad_output correctly for transformers
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
        grad_output = rearrange(grad_output, 'batch seq (nh hd) -> batch seq nh hd', hd = head_dim)
        grad_output = repeat(grad_output, 'batch seq nh hd -> batch seq (noise_batch nh) hd', noise_batch = noise_batch_size)
        return direction, grad_output
    else:
        direction = rearrange(direction, 'batch noise_batch seq d -> batch seq noise_batch d')
        grad_output = rearrange(grad_output, 'batch seq d -> batch seq 1 d')
        return direction, grad_output

def process_backward_zeros_conv(noise: None, clean: torch.Tensor, grad_output: torch.Tensor, *args) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replace clean activations with zeros and reshape grad_output correctly for conv nets
    """
    return rearrange(-clean, 'batch maps h w -> batch maps (h w)'), rearrange(grad_output, 'batch maps h w -> batch maps (h w)')

def process_backward_conv(noise: torch.Tensor, clean: torch.Tensor, grad_output: torch.Tensor, *args) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orthogonally project clean activations to noise activations and reshape grad_output correctly for conv nets
    """
    # We treat the feature maps analogously to the sequence dimension in tranformers
    clean = rearrange(clean, 'batch maps h w -> batch maps (h w)')
    direction = noise - clean
    grad_output = rearrange(grad_output, 'batch maps (h w) -> batch maps h w')
    return direction, grad_output

def process_backward_project_conv(noise: Tuple[torch.Tensor, torch.Tensor], clean: torch.Tensor, grad_output: torch.Tensor, *args) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orthogonally project clean activations to noise activations and reshape grad_output correctly for conv nets
    """
    noise_vecs = rearrange(noise[0], 'noise_batch (h w) -> noise_batch 1 (h w)')
    noise_vals = rearrange(noise[1], 'noise_batch -> noise_batch 1 1')

    clean = rearrange(clean, 'batch maps h w -> batch 1 maps (h w)').type_as(noise_vecs)

    noise = (noise_vecs
            - rearrange(torch.linalg.vecdot(noise_vecs, clean), 'batch noise_batch maps -> batch noise_batch maps 1') * noise_vecs
            + (noise_vals * noise_vecs))
    
    direction = noise - clean
    grad_output = rearrange(grad_output, 'batch maps (h w) -> batch maps h w')
    return direction, grad_output

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
    effects = {}
    handles = []
    mod_to_clean = {}
    mod_to_noise = {}
    mod_to_name = {}
    names = list(noise_acts.keys())

    try:
        def bwd_hook(module: nn.Module, grad_input: tuple[torch.Tensor, ...] | torch.Tensor, grad_output: tuple[torch.Tensor, ...] | torch.Tensor):
            grad_output = grad_output[0] if isinstance(grad_output, tuple) else grad_output

            clean = mod_to_clean.pop(module)

            direction, grad_output = get_tensor_process_fn(effect_capture_args)(mod_to_noise[module], clean, grad_output, head_dim)
            effect = torch.linalg.vecdot(direction, grad_output.type_as(direction))
            name = mod_to_name[module]
            effects[name] = effect.clone().detach()

            if set(names).issubset(effects.keys()):
                raise _Finished()

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

        with torch.enable_grad():
            try:
                out = model(*args, **kwargs)
                out = output_func(out)
                assert out.ndim == 1, "output_func should reduce to a 1D tensor"
                out.backward(torch.ones_like(out))
            except _Finished:
                pass

    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad()
    return effects

def get_edge_effects(
    *args,
    model: nn.Module,
    noise_acts: Dict[str, torch.Tensor] | Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    output_func: Callable[[torch.Tensor], torch.Tensor],
    head_dim: int = 0,
    n_layers: int = 4,
    **kwargs
) -> dict[str, torch.Tensor]:
    """
    Only supports Mistral 7B
    """
    effects = defaultdict(list)
    handles = []
    mod_to_clean = {}
    mod_to_noise = {}
    mod_to_name = {}
    mod_to_parents = {}

    def get_layer(name: str) -> float:
        return next((int(part) for part in name.split('.') if part.isdigit()), float('inf'))

    def bwd_hook(module: nn.Module, grad_input: tuple[torch.Tensor, ...] | torch.Tensor, grad_output: tuple[torch.Tensor, ...] | torch.Tensor):
        if isinstance(grad_input, tuple):
            grad_input, *_ = grad_input
        
        target_name = mod_to_name[module]
        head_target = head_dim > 0 and target_name.endswith('self_attn')
  
        for original_module in mod_to_parents[module]:
            original_name = mod_to_name[original_module]
            clean = mod_to_clean[original_module]
            noise = mod_to_noise[original_name]
            
            if original_name.endswith('self_attn'):
                assert head_dim > 0, "head_dim must be greater than 0 if source is attention head"
                grad_input = rearrange(grad_input, 'batch seq (nh hd) -> batch seq nh hd', hd = head_dim)
                clean = rearrange(clean, 'batch seq (nh hd) -> batch seq nh hd', hd=head_dim)
                noise = rearrange(noise, '(nh hd) -> nh hd', hd=head_dim)
                n_heads = noise.shape[-2]

                # Compute effects for each source head
                effects_list = []
                if head_target:
                    for i in range(n_heads):
                        direction = noise[i, :] - clean[..., i, :]
                        # One copy for each head target
                        direction = rearrange(direction, 'batch seq hd -> batch seq 1 hd')
                        effect = torch.linalg.vecdot(direction, grad_input.type_as(direction))
                        effects_list.append(effect)                
                    # Concatenate effects
                    effect = rearrange(effects_list, 'nh_source batch seq nh_target -> batch seq (nh_source nh_target)')
                else:
                    direction = noise - clean
                    effect = torch.linalg.vecdot(direction, grad_input.type_as(direction))
            else:
                # mlp -> mlp
                direction = noise - clean
                effect = rearrange(torch.linalg.vecdot(direction, grad_input.type_as(direction)), 'batch seq -> batch seq 1')
            
            effects[original_name].append(effect.clone().detach())

    def fwd_hook(module: nn.Module, input: tuple[torch.Tensor, ...] | torch.Tensor, output: tuple[torch.Tensor, ...] | torch.Tensor):
        output = output[0] if isinstance(output, tuple) else output

        mod_to_clean[module] = output.detach()

    modules = list(model.named_modules())

    for i, (name, module) in enumerate(modules):
        mod_to_name[module] = name
        mod_to_parents[module] = []
        current_layer = get_layer(name)
        
        for j in range(0, i):
            parent_name, parent_module = modules[j]
            parent_layer = get_layer(parent_name)
            if current_layer - parent_layer <= n_layers and parent_name in noise_acts and (name.endswith('mlp') or name.endswith('self_attn')):
                mod_to_parents[module].append(parent_module)
        
        if name in noise_acts:
            handles.append(module.register_forward_hook(fwd_hook))
            mod_to_noise[name] = noise_acts[name]
        
        if mod_to_parents[module]:
            handles.append(module.register_full_backward_hook(bwd_hook))

    try:
        with torch.enable_grad():
            out = model(*args, **kwargs)
            out = output_func(out)
            assert out.ndim == 1, "output_func should reduce to a 1D tensor"
            out.backward(torch.ones_like(out))
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad()

    return {k: torch.cat(v, dim=-1) for k, v in effects.items()}

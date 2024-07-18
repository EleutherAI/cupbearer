import torch
from cupbearer.tasks import Task


def get_last_token_activation_function_for_task(task: Task, randperm: bool = False, proj_dim: int = 2**8):
    """Get the activation function for the last token of each input."""
    def get_activation_at_last_token(
        activation: torch.Tensor, 
        inputs: list[str], 
        name: str
    ):
        """Get the activations at the last token of each input."""
        # The activation should be (batch, sequence, residual dimension)
        assert activation.ndim == 3, activation.shape
        batch_size = len(inputs)

        # Tokenize the inputs to know how many tokens there are. It's a bit unfortunate
        # that we're doing this twice (once here, once in the model), but not a huge deal.
        tokens = task.model.tokenize(inputs, **task.model.tokenize_kwargs)
        last_non_padding_index = tokens["attention_mask"].sum(dim=1) - 1

        acts = activation[range(batch_size), last_non_padding_index, :]

        if randperm:
            indices = torch.randperm(acts.size(-1))[:proj_dim]
            acts = acts[:, indices]

        return acts
    
    return get_activation_at_last_token
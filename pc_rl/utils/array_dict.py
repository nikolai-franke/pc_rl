import torch
from parllel import ArrayDict
from torch import Tensor


def dict_to_batched_data(
    array_dict: ArrayDict[Tensor],
) -> tuple[Tensor, Tensor]:
    pos, ptr = array_dict["pos"], array_dict["ptr"]
    num_nodes = ptr[1:] - ptr[:-1]
    batch = torch.repeat_interleave(
        torch.arange(len(num_nodes), device=num_nodes.device),
        repeats=num_nodes,
    )

    return pos, batch

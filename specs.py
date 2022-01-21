import dataclasses

import torch

T = torch.Tensor


@dataclasses.dataclass
class Transition:
    state: T
    action: T
    reward: T
    next_state: T
    done: T
    logits: T

    def as_dict(self):
        return dataclasses.asdict(self)
from typing import Tuple

import torch
from rlmeta.core import remote
from rlmeta.core.model import RemotableModel

from models.models import get_model


class AtariPPOModel(RemotableModel):

    def __init__(self, task_id: str, observation_space: Tuple[int, ...], action_dim: int) -> None:
        super().__init__()
        self.model = get_model(task_id)(observation_space, action_space=action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p, v = self.model(obs)
        return p, v

    # This parameters should be equal to the number of active env at any given time
    @remote.remote_method(batch_size=4 * 8)
    def act(self, obs: torch.Tensor, deterministic_policy: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device

        with torch.inference_mode():
            x = obs.to(device)
            logits, v = self.forward(x)
            action = logits.softmax(dim=-1).multinomial(1, replacement=True)
            action = torch.where(deterministic_policy.to(device), logits.argmax(dim=-1, keepdim=True), action)

        return action.cpu(), logits.cpu(), v.cpu()

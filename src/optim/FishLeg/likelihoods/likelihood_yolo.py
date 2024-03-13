import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
import torch.nn as nn

from .likelihood_base import FishLikelihoodBase

__all__ = [
    "Likelihood_yolo",
]


class Likelihood_yolo(FishLikelihoodBase):
    r"""
    The Bernoulli likelihood used for classification.
    Using the standard Normal CDF :math:`\Phi(x)`) and the identity
    :math:`\Phi(-x) = 1-\Phi(x)`, we can write the likelihood as:

    .. math::
        p(y|f(x))=\Phi(yf(x))

    """

    def __init__(self, model: nn.Module, version: str, device: str = "cpu") -> None:
        self.device = device
        self.model = model
        self.version = version
        self.compute_loss = None
        self.init = 0

    def nll(self, preds: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        preds = self._get_preds(preds)

        if self.init == 0:
            exec(f"from .yolo.{self.version} import ComputeLoss") # Import correct version of YOLO ComputeLoss()
            exec("self.compute_loss = ComputeLoss(self.model)")
            n += 1
        loss, _ = self.compute_loss(preds, observations) # Computes loss using YOLO loss function
        return loss

    def draw(self, preds: torch.Tensor) -> torch.Tensor:
        preds = self._get_preds(preds)

        # Sample bounding box
        sigma = 0.2 # Some small amount
        sample = Normal(torch.tensor([0.0]), torch.tensor([sigma]))
        preds[..., :4] = preds[..., :4] + sample

    
    def _get_preds(self, preds: torch.Tensor) -> torch.Tensor:
        if self.version == "v5":
            return preds[1]
        return preds


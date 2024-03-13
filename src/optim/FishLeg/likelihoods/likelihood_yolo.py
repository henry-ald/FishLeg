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
        
        exec(f"from .yolo.{self.version} import ComputeLoss") # Import correct version of YOLO ComputeLoss()
        exec("self.compute_loss = ComputeLoss(self.model)") 

    def nll(self, preds: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        loss, _ = self.compute_loss(preds, observations) # Computes loss using YOLO loss function
        return loss

    def draw(self, preds: torch.Tensor) -> torch.Tensor:
    
        # Sample bounding box
        sigma = 0.3 # Some small amount
        sample = Normal(torch.tensor([0.0]), torch.tensor([sigma]))
        preds[..., :4] = preds[..., :4] + sample

        # Sample object confidence
        objpred = torch.index_select(preds, 4, 4)
        sample = Bernoulli(objpred).sample()
        preds[..., 4] = sample
        
        # Sample class confidences
        clsidx = torch.tensor([i for i in range(5, preds.shape[4])]).to(self.device) # 1-D tensor of indices corresponding to confidences in preds[0]
        clspred = torch.index_select(preds[0], 2, clsidx) # get the confidences
        sample = Bernoulli(clspred).sample()
        preds[..., 5:] = sample
        return preds


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

    def __init__(self, model: nn.Module, version, device: str = "cpu") -> None:
        self.device = device
        self.imported_functions = version
        self.compute_loss = self.imported_functions.ComputeLoss(model)

    def nll(self, preds: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        loss, _ = self.compute_loss(preds[1], observations) # Computes loss using YOLO loss function
        return loss

    def draw(self, preds: torch.Tensor) -> torch.Tensor:
        #preds = self._get_preds(preds)

        if self.imp.de_parallel(self.model).model[-1] == len(preds): # i.e. for v7 output format
            for i, pi in enumerate(preds):
                preds[i] = self._sample(preds[i])
        else: # i.e. for v5 output format
            preds = self._sample(preds[0])
        
        preds = self.imported_functions.non_max_suppression(preds)

        for i, _ in enumerate(preds):
            idx = torch.tensor([2,3,4,5,0,1], device=self.device) # send final two cols to first two, shift first 4 along
            idx = idx.repeat(preds[i].shape[0], 1) # preds[i].shape[0] occurences in the first dim, 1 occurence in the second dim
            new = torch.zeros(preds[i].shape, device=self.device)
            new.scatter_(1, idx, preds[i])
            preds[i] = new
            preds[i][:, 2:] = self.imported_functions.xyxy2xywhn(preds[i][:, 2:]) # Convert coords from xyxy to xywh normalised by img size
        
        preds = torch.cat(preds, dim=0)
        return preds
    
    def _sample(self, preds: torch.Tensor) -> torch.Tensor:
        
        # Sample bounding box
        sigma = 0.2 # Some small amount
        sample = Normal(torch.tensor([0.0]), torch.tensor([sigma])).sample().to(self.device)
        preds[..., :4] = preds[..., :4] + sample

        # Sample object confidence
        fin = preds.dim() - 1 # Index of last dimension -> classifications etc
        objpred = torch.index_select(preds, fin, torch.tensor([4], device=self.device))
        sample = Bernoulli(logits=objpred).sample().to(self.device)
        preds[..., 4] = sample.squeeze()
        
        # Sample class confidences
        clsidx = torch.tensor([i for i in range(5, preds.shape[-1])]).to(self.device) # 1-D tensor of indices corresponding to confidences in preds[0]
        clspred = torch.index_select(preds, fin, clsidx) # get the confidences
        sample = Bernoulli(logits=clspred).sample().to(self.device)
        preds[..., 5:] = sample

        return preds
    
    def _get_preds(self, preds: torch.Tensor) -> torch.Tensor:
        if self.version == "v5":
            return preds[1]
        return preds


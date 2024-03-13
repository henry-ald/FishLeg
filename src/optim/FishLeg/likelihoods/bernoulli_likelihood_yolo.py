import torch
from torch.distributions.bernoulli import Bernoulli

from .likelihood_base import FishLikelihoodBase

__all__ = [
    "BernoulliLikelihood_yolo",
]


class BernoulliLikelihood_yolo(FishLikelihoodBase):
    r"""
    The Bernoulli likelihood used for classification.
    Using the standard Normal CDF :math:`\Phi(x)`) and the identity
    :math:`\Phi(-x) = 1-\Phi(x)`, we can write the likelihood as:

    .. math::
        p(y|f(x))=\Phi(yf(x))

    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def nll(self, preds: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        preds = self._get_preds(preds)
        bce = torch.sum(preds * (1.0 - observations) + torch.nn.Softplus()(-preds))
        return bce / preds.shape[0]

    def draw(self, preds: torch.Tensor) -> torch.Tensor:
        preds = self._get_preds(preds)
        return Bernoulli(logits=preds).sample()

    def _get_preds(self, preds: torch.Tensor) -> torch.Tensor:
        # preds is a list of length 2
        # preds[0] is a 3-D tensor: [batch_size; sum of (no. anchor sizes * no. of grid rows * no. of grid cols ); [x ,y , width, height, obj_score, cls_score]]
        # we want the confidences from the third axis of preds[0]. this is of length no. of classes
        # preds[1] provides the same info, except in the form of 3 5-D tensors that indexes preds[0] by grid position
        indices = torch.tensor([i for i in range(5, preds[0].shape[2])]).to(self.device) # 1-D tensor of indices corresponding to confidences in preds[0]
        preds = torch.index_select(preds[0], 2, indices) # get the confidences
        return preds


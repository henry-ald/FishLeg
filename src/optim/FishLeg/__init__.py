from .fishleg import FishLeg
from .layers import *
from .likelihoods import *
from .utils import *

FISH_LAYERS = {"linear": FishLinear}
FISH_LIKELIHOODS = {
    "gaussian": GaussianLikelihood,
    "bernoulli": BernoulliLikelihood,
    "bernoulli_yolo": BernoulliLikelihood_yolo,
    "softmax": SoftMaxLikelihood,
    "yolo": Likelihood_yolo
}

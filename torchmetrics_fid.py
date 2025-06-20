# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn import Module
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_info, rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import (_SCIPY_AVAILABLE,
                                            _TORCH_FIDELITY_AVAILABLE)

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*Metric `FrechetInceptionDistance` will save all extracted features in buffer.*",
    category=UserWarning
)

if _TORCH_FIDELITY_AVAILABLE:
    from torch_fidelity.feature_extractor_inceptionv3 import \
        FeatureExtractorInceptionV3
else:

    class FeatureExtractorInceptionV3(Module):  # type: ignore
        pass

    __doctest_skip__ = ["FrechetInceptionDistance", "FID"]


if _SCIPY_AVAILABLE:
    import scipy


class NoTrainInceptionV3(FeatureExtractorInceptionV3):
    def __init__(
        self,
        name: str,
        features_list: List[str],
        feature_extractor_weights_path: Optional[str] = None,
    ) -> None:
        super().__init__(name, features_list, feature_extractor_weights_path)
        # put into evaluation mode
        self.eval()

    def train(self, mode: bool) -> "NoTrainInceptionV3":
        """the inception network should not be able to be switched away from evaluation mode."""
        return super().train(False)

    def forward(self, x: Tensor) -> Tensor:
        out = super().forward(x)
        return out[0].reshape(x.shape[0], -1)


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    All credit to `Square Root of a Positive Definite Matrix`_
    """

    @staticmethod
    def forward(ctx: Any, input_data: Tensor) -> Tensor:
        # TODO: update whenever pytorch gets an matrix square root function
        # Issue: https://github.com/pytorch/pytorch/issues/9983
        m = input_data.detach().cpu().numpy().astype(np.float_)
        scipy_res, _ = scipy.linalg.sqrtm(m, disp=False)
        sqrtm = torch.from_numpy(scipy_res.real).to(input_data)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


def _compute_fid(
    mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor, eps: float = 1e-6
) -> Tensor:
    r"""
    Adjusted version of `Fid Score`_

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant - used if sigma_1 @ sigma_2 matrix is singular

    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2

    covmean = sqrtm(sigma1.mm(sigma2))
    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        rank_zero_info(
            f"FID calculation produces singular product; adding {eps} to diagonal of covariance estimates"
        )
        offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
        covmean = sqrtm((sigma1 + offset).mm(sigma2 + offset))

    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


class FrechetInceptionDistance(Metric):
    r"""
    Calculates Fréchet inception distance (FID_) which is used to access the quality of generated images. Given by

    .. math::
        FID = |\mu - \mu_w| + tr(\Sigma + \Sigma_w - 2(\Sigma \Sigma_w)^{\frac{1}{2}})

    where :math:`\mathcal{N}(\mu, \Sigma)` is the multivariate normal distribution estimated from Inception v3 [1]
    features calculated on real life images and :math:`\mathcal{N}(\mu_w, \Sigma_w)` is the multivariate normal
    distribution estimated from Inception v3 features calculated on generated (fake) images. The metric was
    originally proposed in [1].

    Using the default feature extraction (Inception v3 using the original weights from [2]), the input is
    expected to be mini-batches of 3-channel RGB images of shape (``3 x H x W``) with dtype uint8. All images
    will be resized to 299 x 299 which is the size of the original training data. The boolian flag ``real``
    determines if the images should update the statistics of the real distribution or the fake distribution.

    .. note:: using this metrics requires you to have ``scipy`` install. Either install as ``pip install
        torchmetrics[image]`` or ``pip install scipy``

    .. note:: using this metric with the default feature extractor requires that ``torch-fidelity``
        is installed. Either install as ``pip install torchmetrics[image]`` or
        ``pip install torch-fidelity``

    Args:
        feature:
            Either an integer or ``nn.Module``:

            - an integer will indicate the inceptionv3 feature layer to choose. Can be one of the following:
              64, 192, 768, 2048
            - an ``nn.Module`` for using a custom feature extractor. Expects that its forward method returns
              an ``[N,d]`` matrix where ``N`` is the batch size and ``d`` is the feature size.

        reset_real_features: Whether to also reset the real features. Since in many cases the real dataset does not
            change, the features can cached them to avoid recomputing them which is costly. Set this to ``False`` if
            your dataset does not change.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    References:
        [1] Rethinking the Inception Architecture for Computer Vision
        Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
        https://arxiv.org/abs/1512.00567

        [2] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium,
        Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter
        https://arxiv.org/abs/1706.08500

    Raises:
        ValueError:
            If ``feature`` is set to an ``int`` (default settings) and ``torch-fidelity`` is not installed
        ValueError:
            If ``feature`` is set to an ``int`` not in [64, 192, 768, 2048]
        TypeError:
            If ``feature`` is not an ``str``, ``int`` or ``torch.nn.Module``
        ValueError:
            If ``reset_real_features`` is not an ``bool``

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(123)
        >>> from torchmetrics.image.fid import FrechetInceptionDistance
        >>> fid = FrechetInceptionDistance(feature=64)
        >>> # generate two slightly overlapping image intensity distributions
        >>> imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
        >>> imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
        >>> fid.update(imgs_dist1, real=True)
        >>> fid.update(imgs_dist2, real=False)
        >>> fid.compute()
        tensor(12.7202)

    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    real_features: List[Tensor]
    fake_features: List[Tensor]

    def __init__(
        self,
        feature: Union[int, Module] = 2048,
        reset_real_features: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        rank_zero_warn(
            "Metric `FrechetInceptionDistance` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
        )

        if isinstance(feature, int):
            if not _TORCH_FIDELITY_AVAILABLE:
                raise ModuleNotFoundError(
                    "FrechetInceptionDistance metric requires that `Torch-fidelity` is installed."
                    " Either install as `pip install torchmetrics[image]` or `pip install torch-fidelity`."
                )
            valid_int_input = [64, 192, 768, 2048]
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}."
                )

            self.inception = NoTrainInceptionV3(
                name="inception-v3-compat", features_list=[str(feature)]
            )
        elif isinstance(feature, Module):
            self.inception = feature
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        self.add_state("real_features", [], dist_reduce_fx=None)
        self.add_state("fake_features", [], dist_reduce_fx=None)

    def update(self, imgs: Tensor, real: bool, features=None) -> None:  # type: ignore
        """Update the state with extracted features.

        Args:
            imgs: tensor with images feed to the feature extractor
            real: bool indicating if ``imgs`` belong to the real or the fake distribution
        """
        if features is None:
            features = self.inception(imgs)

        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        real_features = dim_zero_cat(self.real_features)
        fake_features = dim_zero_cat(self.fake_features)
        # computation is extremely sensitive so it needs to happen in double precision
        orig_dtype = real_features.dtype
        real_features = real_features.double()
        fake_features = fake_features.double()

        # calculate mean and covariance
        n = real_features.shape[0]
        m = fake_features.shape[0]
        mean1 = real_features.mean(dim=0)
        mean2 = fake_features.mean(dim=0)
        diff1 = real_features - mean1
        diff2 = fake_features - mean2
        cov1 = 1.0 / (n - 1) * diff1.t().mm(diff1)
        cov2 = 1.0 / (m - 1) * diff2.t().mm(diff2)

        # compute fid
        return _compute_fid(mean1, cov1, mean2, cov2).to(orig_dtype)

    def reset(self) -> None:
        if not self.reset_real_features:
            # remove temporarily to avoid resetting
            value = self._defaults.pop("real_features")
            super().reset()
            self._defaults["real_features"] = value
        else:
            super().reset()

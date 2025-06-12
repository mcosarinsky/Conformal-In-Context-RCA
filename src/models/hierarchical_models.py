import torch
from typing import Optional
from abc import ABC, abstractmethod
from src.utils.custom_types import SamplerType, OutputDictType
from src.models.network_blocks import gauss_sampler

from src.losses import (
    HierarchicalKLLoss,
    HierarchicalReconstructionLoss,
    multinoulli_loss_2d,
    KL_two_gauss_with_diag_cov,
)


class AbstractHierarchicalProbabilisticModel(ABC, torch.nn.Module):
    """
    Base class for all conditional hierarchical models.
    Can be extended for segmentation or other use cases.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _aggregate_levels(self, level_outputs: OutputDictType) -> OutputDictType:
        pass

    def reconstruct_from_z(
        self, z: OutputDictType, x: Optional[torch.Tensor] = None
    ) -> OutputDictType:
        level_outputs = self.likelihood(z, x)
        return self._aggregate_levels(level_outputs)

    def predict_output_samples(self, x: torch.Tensor, N: int = 1) -> torch.Tensor:
        bs = x.shape[0]
        xb = torch.vstack([x for _ in range(N)])
        _, _, z = self.prior(xb)
        agg = self.reconstruct_from_z(z, xb)
        yb_hat = agg[0]
        yb_hat = yb_hat.view([N, bs] + list(yb_hat.shape[1:]))
        return yb_hat.transpose(0, 1)

    def predict(self, x: torch.Tensor, N: int = 1) -> torch.Tensor:
        return torch.mean(self.predict_output_samples(x, N), dim=1)

    def posterior_samples(
        self, x: torch.Tensor, y: torch.Tensor, N: int = 1
    ) -> OutputDictType:
        bs = x.shape[0]
        xb = torch.vstack([x for _ in range(N)])
        yb = torch.vstack([y for _ in range(N)])
        _, _, z = self.posterior(xb, yb)
        return {l: zl.view([N, bs] + list(zl.shape[1:])) for l, zl in z.items()}

    def posterior_outputs(
        self, x: torch.Tensor, y: torch.Tensor, N: int = 1
    ) -> torch.Tensor:
        bs = x.shape[0]
        xb = torch.vstack([x for _ in range(N)])
        yb = torch.vstack([y for _ in range(N)])
        _, _, z = self.posterior(xb, yb)
        agg = self.reconstruct_from_z(z)
        yb_hat = agg[0]
        yb_hat = yb_hat.view([N, bs] + list(yb_hat.shape[1:]))
        return yb_hat.transpose(0, 1)

    def output_level_predictions(self, x: torch.Tensor):
        _, _, z = self.prior(x)
        return self.likelihood(z)

    @staticmethod
    def samples_to_list(yb: torch.Tensor) -> list[torch.Tensor]:
        return [yb[i] for i in range(yb.shape[0])]


class AbstractHierarchicalProbabilisticSegmentationModel(
    AbstractHierarchicalProbabilisticModel
):
    """
    Base class for all hierarchical probabilistic segmentation models
    such as Probabilistic U-Net and PHIseg.
    """

    def __init__(self):
        super().__init__()

    def _aggregate_levels(
        self,
        level_outputs: OutputDictType,
    ) -> OutputDictType:
        assert list(level_outputs.keys()) == list(range(self.latent_levels))

        combined_outputs = {
            self.latent_levels - 1: level_outputs[self.latent_levels - 1]
        }

        for l in reversed(range(self.latent_levels - 1)):
            combined_outputs[l] = combined_outputs[l + 1] + level_outputs[l]

        return combined_outputs


class PHISeg(AbstractHierarchicalProbabilisticSegmentationModel):
    """
    The actual PHISeg model architecture and loss configuration.
    """

    def __init__(
        self,
        total_levels: int,
        latent_levels: int,
        zdim: int,
        num_classes: int,
        beta: float = 1.0,
        input_channels: int = 1,
    ) -> None:
        super().__init__()

        from src.models.phiseg import PHISegPrior, PHISegPosterior, PHISegLikelihood

        sampler: SamplerType = gauss_sampler

        self.latent_levels = latent_levels
        self.total_levels = total_levels
        self.num_classes = num_classes
        self.beta = beta

        self.prior = PHISegPrior(
            sampler=sampler,
            total_levels=total_levels,
            latent_levels=latent_levels,
            zdim=zdim,
            input_channels=input_channels,
        )
        self.posterior = PHISegPosterior(
            sampler=sampler,
            total_levels=total_levels,
            latent_levels=latent_levels,
            zdim=zdim,
            num_classes=num_classes,
            input_channels=input_channels,
        )
        self.likelihood = PHISegLikelihood(
            total_levels=total_levels,
            latent_levels=latent_levels,
            zdim=zdim,
            num_classes=num_classes,
        )

        kl_weight_dict = {l: 4.0**l for l in range(latent_levels)}
        recon_weight_dict = {l: 1.0 for l in range(latent_levels)}

        self.hierarchical_kl_loss = HierarchicalKLLoss(
            KL_divergence=KL_two_gauss_with_diag_cov, weight_dict=kl_weight_dict
        )

        self.hierarchical_recon_loss = HierarchicalReconstructionLoss(
            reconstruction_loss=multinoulli_loss_2d, weight_dict=recon_weight_dict
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, zs = self.prior(x)
        ys = self.likelihood(zs)
        return self._aggregate_levels(ys)[0]
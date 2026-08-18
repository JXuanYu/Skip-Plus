import torch
import torch.nn as nn


class STE_Ceil(torch.autograd.Function):
    """Straight-Through Estimator for Ceil"""
    @staticmethod
    def forward(ctx, x_in):
        return torch.ceil(x_in)

    @staticmethod
    def backward(ctx, g):
        return g  # Straight-through gradient


class STE_Round(torch.autograd.Function):
    """Straight-Through Estimator for Round"""
    @staticmethod
    def forward(ctx, x_in):
        return torch.round(x_in)

    @staticmethod
    def backward(ctx, g):
        return g


class STE_Min(torch.autograd.Function):
    """Straight-Through Estimator for Min"""
    @staticmethod
    def forward(ctx, x_in1, x_in2, x_in3=None):
        def to_tensor(val, ref):
            return val if isinstance(val, torch.Tensor) else torch.tensor(val, dtype=ref.dtype, device=ref.device)

        t1 = to_tensor(x_in1, x_in2)
        t2 = to_tensor(x_in2, x_in2)
        t3 = to_tensor(x_in3 if x_in3 is not None else float('inf'), x_in2)

        return torch.min(torch.min(t1, t2), t3)

    @staticmethod
    def backward(ctx, g):
        return None, g, g


ste_ceil = STE_Ceil.apply
ste_round = STE_Round.apply
ste_min = STE_Min.apply

class CompressRate(nn.Module):
    """Differentiable module for learning the token compression ratio."""
    def __init__(self, patch_number=196, granularity=1, temperature=1.0):
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.patch_number = patch_number
        self.class_token_num = 1
        self.temperature = float(temperature)

        # Learnable logit for the token keep ratio.
        initial_logit = self._ratio_to_logit(torch.tensor(0.9))
        self.ratio_logit = nn.Parameter(initial_logit, requires_grad=True)

        self.kept_token_number = patch_number + self.class_token_num
        self.update_kept_token_number()

    def _ratio_to_logit(self, ratio):
        """Convert keep ratio to logit."""
        ratio = torch.clamp(ratio, 0.1, 0.9)
        return torch.log(ratio / (1.0 - ratio))

    def _logit_to_ratio(self, logit):
        """Convert logit to keep ratio."""
        return torch.sigmoid(logit) * 0.8 + 0.1

    def update_kept_token_number(self):
        """Update the number of kept tokens."""
        ratio = self._logit_to_ratio(self.ratio_logit)

        if self.training:
            expected_kept = ratio * self.patch_number
            kept_num = ste_round(expected_kept) + self.class_token_num
            self.kept_token_number = int(kept_num)
            return kept_num
        else:
            kept_num = round(ratio.item() * self.patch_number) + self.class_token_num
            self.kept_token_number = kept_num
            return kept_num

    def get_token_probability(self):
        """Compute token keep probabilities with a temperature sigmoid rank mask."""
        ratio = self._logit_to_ratio(self.ratio_logit)
        expected_patch_kept = ratio * self.patch_number

        # Patch tokens are assumed sorted by importance, ranked from 1 to patch_number.
        # +0.5 aligns the sigmoid 0.5 threshold with round(expected_patch_kept).
        patch_ranks = torch.arange(
            1,
            self.patch_number + 1,
            device=self.ratio_logit.device,
            dtype=self.ratio_logit.dtype,
        )
        patch_prob = torch.sigmoid(
            (expected_patch_kept - patch_ranks + 0.5) / self.temperature
        )

        # Always keep the CLS token.
        cls_prob = torch.ones(
            self.class_token_num,
            device=self.ratio_logit.device,
            dtype=self.ratio_logit.dtype,
        )
        return torch.cat([cls_prob, patch_prob], dim=0)

    def get_token_mask(self, token_number=None):
        """Generate a hard-forward, soft-backward STE token mask."""
        token_prob = self.get_token_probability()
        mask = torch.ones_like(token_prob)

        end_idx = int(token_number) if token_number else len(mask)
        mask[int(self.kept_token_number):end_idx] = 0

        return mask - token_prob.detach() + token_prob
